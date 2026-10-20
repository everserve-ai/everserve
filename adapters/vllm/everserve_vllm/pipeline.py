# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""vLLM's pipeline traffic over the fabric family's current group.

vLLM moves two things between pipeline stages, both through torch's process groups: the
intermediate tensors of a step (``isend_tensor_dict`` / ``irecv_tensor_dict`` on the PP group,
metadata over its CPU group and tensors over its device group), and the sampled tokens the last
stage broadcasts back each step (``PPHandler``, over a sibling device group it makes at init).
Neither can follow a change: a torch group is formed once over the world, it cannot drop a dead
rank, and a spare that never joined the world has none. Here both go over the fabric family's
current group instead (:class:`~everserve.comm.collective.GroupFamily`): the tensors through
:mod:`everserve.comm.exchange`, the broadcast through the group's own; from bootstrap on, on
survivors and spares alike, so a change swaps the group underneath and nothing else. Before this
process is bootstrapped (the engine's warm-up) vLLM's own path runs, as it does on the stock
engine.

Positions: the family's group over a column holds the column's members in stage order, so a
stage is its rank in the group, and vLLM's ``dst``/``src`` (ranks within the PP group) are
positions already; the broadcast's source, an engine rank, is mapped to its stage.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterator
from typing import Any

import torch

from everserve.comm.collective import LocalStandIn
from everserve.comm.exchange import Exchange, Layout, packable
from everserve.comm.runtime import CHANGE_ABORT

log = logging.getLogger(__name__)

PIPELINE_METHODS = (
    "send_tensor_dict",
    "recv_tensor_dict",
    "isend_tensor_dict",
    "irecv_tensor_dict",
)
_ORIGINALS: dict[str, Any] = {}
_ORIGINAL_BROADCAST: Any = None

EXCHANGE = Exchange()
"""This process's transfers between stages: its depth is the engine's stage count, set at
bootstrap (``worker.bootstrap``); its counts are the run report's."""


class Done:
    """A handle for an exchange that is complete on the stream: nothing to wait for."""

    def wait(self) -> None:
        """Nothing: the receive is ordered on the stream before any use of its tensors."""


class MemberSetChanged(RuntimeError):
    """A pipeline transfer refused because a loss refused the family's current group.

    Its message carries :data:`~everserve.comm.runtime.CHANGE_ABORT`: the error reaches the
    engine's executor as text through vLLM's worker RPC, from any step method that moves
    pipeline traffic (``execute_model``, and ``sample_tokens`` through vLLM's sampled-token
    broadcast), and the executor ends the step as the loss it is.
    """

    def __init__(self) -> None:
        """The refusal's one message."""
        super().__init__(f"{CHANGE_ABORT}: the pipeline refused its group until the boundary")


class FamilyBroadcastGroup:
    """What ``PPHandler`` holds in place of its sibling device group.

    A broadcast on it goes over the fabric family's current group; before this process is
    bootstrapped, over ``original`` (the sibling torch group, where one could be made).
    """

    def __init__(self, original: Any) -> None:
        """``original`` is vLLM's sibling group, or None where there is no torch world."""
        self.original = original


def family_comm() -> tuple[Any, Any] | None:
    """The fabric family's current communicator and the layout, once this process is bootstrapped.

    None before that, and None while the family's current group is a stand-in (a spare before
    its join, which sends nothing).

    Raises:
        MemberSetChanged: once a loss has refused the current group, until the boundary swaps it.
    """
    from everserve_vllm import worker

    state = worker.STATE
    if state is None or state.collective.fabric is None or state.collective.layout is None:
        return None
    current = state.collective.fabric.current
    if isinstance(current, LocalStandIn):
        return None
    if getattr(current, "refused", False):
        # a loss refused the group (everserve_vllm.collective_op) and its communicator is being
        # released; a call into it would take the process down, so the step fails here instead
        # and the change at the boundary swaps the family to its pre-formed group
        raise MemberSetChanged()
    comm = getattr(current, "group", None)
    return (comm, state.collective.layout) if comm is not None else None


def _refused(group: Any) -> None:
    """Raise once ``group`` was refused: read by the exchange after its wait, before the enqueue.

    The group is the one the call found, whatever the family holds by then: a refusal of it
    while the call waited out the group's previous transfer (which an abort ends, the peer
    being dead) must keep the enqueue off its freed communicator.
    """
    if getattr(group, "refused", False):
        raise MemberSetChanged()


RUNNER: Any = None
"""This worker's model runner, set at bootstrap: the step's layout comes from it."""
_SCHEDULER_OUTPUT: Any = None
"""The step's scheduler output, kept by the hook on the worker's ``execute_model``."""
_LOCAL: Any = None
"""The runner of a stage running alone (:func:`local_stage`), or None."""


@contextlib.contextmanager
def local_stage(runner: Any) -> Iterator[None]:
    """Run this stage alone for the block.

    A receive yields the step's intermediate tensors freshly made at the runner's layout, a
    send and the sampled-token broadcast are dropped. For a spare's kernel warm-up on a stage
    past the first (A38): vLLM drives the worker's ``execute_model`` with synthetic steps, and
    on such a stage every step would wait on the stage before, which a spare has no peer for;
    alone, the step computes on zeros and the runner's kernels compile as they would on a
    serving rank. The step's size comes from the scheduler output the call is wrapped in
    (:func:`stepping`).
    """
    global _LOCAL, RUNNER
    before = _LOCAL, RUNNER
    _LOCAL, RUNNER = runner, runner
    try:
        yield
    finally:
        _LOCAL, RUNNER = before


def stepping(execute_model: Any) -> Any:
    """``execute_model`` with its scheduler output kept for the step (:func:`layout_for_step`)."""

    def execute(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        global _SCHEDULER_OUTPUT
        _SCHEDULER_OUTPUT = scheduler_output
        try:
            return execute_model(scheduler_output, *args, **kwargs)
        finally:
            _SCHEDULER_OUTPUT = None

    return execute


def _local_recv() -> dict[str, Any]:
    """The step's intermediate tensors made here, zeros at the runner's layout.

    Raises:
        RuntimeError: the stage cannot name the step's layout (no runner, no scheduler output,
            no preallocated intermediate tensors, or the padding raised).
    """
    layout = layout_for_step()
    if layout is None:
        raise RuntimeError(
            "a stage running alone cannot name its step's layout: the runner's intermediate "
            "tensors, the scheduler output or the padding are missing"
        )
    device = getattr(_LOCAL, "device", None)
    return {key: torch.zeros(shape, dtype=dtype, device=device) for key, dtype, shape in layout}


def _reason(why: str, exc_info: bool = False) -> None:
    """Log once why this stage cannot name the step's layout."""
    log.warning("no packed layout for this step: %s", why, exc_info=exc_info)


def set_runner(runner: Any) -> None:
    """Name this worker's model runner (:func:`layout_for_step` reads it)."""
    global RUNNER
    RUNNER = runner


def hook_worker(worker: Any) -> None:
    """Keep each step's scheduler output across the worker's ``execute_model`` (hook H2).

    The receiving stage posts its receive before the runner has built the batch, so the step's
    token count must come from the scheduler output the worker was handed; the hook keeps it
    for :func:`layout_for_step` and clears it after the step.
    """
    execute_model = stepping(worker.execute_model)
    execute_model.__name__ = "execute_model"
    worker.execute_model = execute_model


def _v2_dispatch() -> tuple[Any, Any]:
    """The padding rule as model runner v2 imports it: the uniform-decode count and the dispatch.

    ``get_uniform_decode_token_count`` and ``dispatch_cg_and_sync_dp``, vLLM's own.
    """
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
    from vllm.v1.worker.utils import get_uniform_decode_token_count

    return get_uniform_decode_token_count, dispatch_cg_and_sync_dp


def padded_tokens(runner: Any, so: Any) -> int:
    """The step's token count after model runner v2's padding, ahead of its receive (A39).

    The runner computes it inside ``execute_model``, after the receive, from the batch it
    gathers (``gather_batch_req_state``: per request, whether it is still prefilling and
    whether a prefilling request's one-token chunk runs as a decode) and the graph dispatch
    (``dispatch_cg_and_sync_dp``). The same two facts are read here from the scheduler output
    the hook keeps and the runner's request state as it stands before the step: a new request's
    prefill length is its prefill token list, a cached request's the runner's; its computed
    count is the scheduler output's. Then vLLM's own two functions give the padded count, so a
    stage pads exactly as its runner will. Speculative tokens, LoRA, micro-batching and
    data-parallel sync are not in play on this adapter's engines; a runner without v2's state
    raises, and the step goes by the header path.

    Raises:
        AttributeError: a runner without model runner v2's request state or graph manager.
    """
    uniform_count, dispatch = _v2_dispatch()
    states = runner.req_states
    scheduled = so.num_scheduled_tokens
    num_reqs, num_toks = len(scheduled), int(so.total_num_scheduled_tokens)
    max_query_len: int | None = max(scheduled.values())
    prefilling: dict[str, tuple[bool, int]] = {}  # request -> (is prefilling, computed)
    for r in getattr(so, "scheduled_new_reqs", None) or []:
        tokens = r.prefill_token_ids
        prefill_len = len(tokens) if tokens is not None else r.prompt_len
        prefilling[r.req_id] = (r.num_computed_tokens < prefill_len, r.num_computed_tokens)
    cached = getattr(so, "scheduled_cached_reqs", None)
    for rid, computed in zip(
        getattr(cached, "req_ids", None) or [],
        getattr(cached, "num_computed_tokens", None) or [],
        strict=True,
    ):
        prefill_len = int(states.prefill_len.np[states.req_id_to_index[rid]])
        prefilling[rid] = (computed < prefill_len, computed)
    has_prefill = any(p for p, _ in prefilling.values())
    # a prefilling request's one-token chunk past its first token runs as a decode
    eligible = not has_prefill or all(
        not prefilling[rid][0] or (n == 1 and prefilling[rid][1] > 0)
        for rid, n in scheduled.items()
    )
    uniform = uniform_count(num_reqs, num_toks, max_query_len, eligible)
    if has_prefill:
        max_query_len = None  # as the runner: a varlen decode graph admits no prefill
    desc, _ = dispatch(
        runner.cudagraph_manager,
        num_reqs,
        num_toks,
        uniform,
        runner.dp_size,
        runner.dp_rank,
        max_query_len=max_query_len,
        need_eager=False,
        num_active_loras=0,
        parallel_config=runner.parallel_config,
        allow_ubatching=getattr(runner, "ubatch_runner", None) is not None,
        uniform_decode=uniform == runner.decode_query_len,
    )
    return max(num_toks, int(desc.num_tokens))


def layout_for_step() -> Layout | None:
    """The intermediate tensors this stage receives this step: keys, dtype and ``[n, hidden]``.

    ``n`` is the step's token count after the runner's padding, computed as the runner will
    for its own batch from the same scheduler output (:func:`padded_tokens`); the keys and
    dtype are the runner's preallocated intermediate tensors
    (``make_empty_intermediate_tensors``). None when any of that is missing, and the header
    path carries the step.
    """
    so, runner = _SCHEDULER_OUTPUT, RUNNER
    if so is None or runner is None:
        _reason("no scheduler output" if so is None else "no runner")
        return None
    pre = getattr(runner, "intermediate_tensors", None)
    tensors = getattr(pre, "tensors", None)
    if not tensors:
        _reason(f"the runner's intermediate tensors are {pre!r}")
        return None
    if not getattr(so, "total_num_scheduled_tokens", 0):
        _reason("the step schedules no tokens")
        return None
    try:
        n = padded_tokens(runner, so)
    except Exception:  # noqa: BLE001 - the header path carries a step the layout cannot name
        _reason("the runner's padding raised", exc_info=True)
        return None
    layout: Layout = []
    for key, t in tensors.items():
        if t.dim() != 2 or n > t.shape[0]:
            _reason(f"{key} is {tuple(t.shape)} for {n} padded tokens")
            return None
        layout.append((key, t.dtype, (n, int(t.shape[1]))))
    return layout


def _family() -> tuple[Any, Any, Any] | None:
    """:func:`family_comm`'s communicator and layout, and the refusal check bound to its group."""
    from everserve_vllm import worker

    found = family_comm()
    if found is None:
        return None
    comm, layout = found
    assert worker.STATE is not None and worker.STATE.collective.fabric is not None
    group = worker.STATE.collective.fabric.current
    return comm, layout, lambda: _refused(group)


def _in_step() -> bool:
    """Whether this process is inside a scheduled step (the hook saw its scheduler output).

    The switch both stages take alike: inside a step the sender packs and the receiver names
    the layout; outside one (vLLM's own warm-ups and profiling drive the runner directly) both
    use the header path. The first stage has no intermediate tensors of its own, so the
    sender cannot name a layout; the step is the shared fact.
    """
    return _SCHEDULER_OUTPUT is not None


def _send(comm: Any, dst: int, tensors: Any, check: Any) -> None:
    """One packed operation inside a scheduled step; the self-describing header outside one.

    The route is a fact about the transfer, not a setting: inside a step the receiver names the
    layout from its own runner (:func:`layout_for_step`) and expects one packed operation;
    outside one (vLLM's profiling run and kernel warm-ups drive the runner with no scheduler
    output) it cannot, and the header path describes the tensors (ledger A33).

    Raises:
        RuntimeError: a step's tensors that cannot go as one buffer: a header send against the
            receiver's packed receive would hang both stages.
    """
    if not _in_step():
        EXCHANGE.send_named(comm, dst, tensors, comm.device, check)
        return
    if not packable(tensors):
        raise RuntimeError(
            "a scheduled step's pipeline tensors are not one dtype and one shape: "
            f"{ {k: (t.dtype, tuple(t.shape)) for k, t in tensors.items()} }"
        )
    EXCHANGE.send_packed(comm, dst, tensors, comm.device, check)


def _recv(comm: Any, src: int, check: Any) -> dict[str, Any]:
    """One packed operation by the step's layout inside a scheduled step; the header outside.

    Raises:
        RuntimeError: inside a step whose layout this stage cannot name: the sender packed,
            and a header receive against a packed send would hang both stages.
    """
    if not _in_step():
        return EXCHANGE.recv_named(comm, src, comm.device, check)
    layout = layout_for_step()
    if layout is None:
        raise RuntimeError(
            "a scheduled step whose layout this stage cannot name: the runner's "
            "intermediate tensors or its padding are missing"
        )
    return EXCHANGE.recv_packed(comm, src, layout, comm.device, check)


def install(group_cls: Any) -> None:
    """Route the pipeline methods of PP groups and the sampled-token broadcast as the module says.

    The registry's A33, A34, A50 and A51 (:data:`everserve_vllm.patches.PIPELINE`), put on
    ``group_cls``, vLLM's ``GroupCoordinator``. Idempotent.
    """
    from everserve_vllm import patches

    patches.PIPELINE.apply(group_cls)


def _is_pipeline(group: Any) -> bool:
    return group.unique_name.startswith("pp") and group.world_size > 1


def routed(group_cls: Any) -> tuple[dict[str, Any], Any]:
    """The pipeline methods and the broadcast as the module says, over ``group_cls``'s own.

    The originals are kept (:data:`_ORIGINALS`, :data:`_ORIGINAL_BROADCAST`) for the paths that
    fall through to them and for :func:`release`; the four transfers report a failure's peers
    (:func:`_reporting`). Returns the methods by name and the broadcast; the registry puts them
    on.
    """
    global _ORIGINAL_BROADCAST
    for name in PIPELINE_METHODS:
        _ORIGINALS[name] = getattr(group_cls, name)
    _ORIGINALS["make_sibling_device_group"] = group_cls.make_sibling_device_group
    _ORIGINAL_BROADCAST = torch.distributed.broadcast

    def alone(self: Any) -> bool:
        return _LOCAL is not None and _is_pipeline(self)

    def send_tensor_dict(self: Any, tensor_dict: Any, dst: int | None = None, **kw: Any) -> Any:
        if alone(self):
            return None
        found = _family() if _is_pipeline(self) else None
        if found is None:
            return _ORIGINALS["send_tensor_dict"](self, tensor_dict, dst, **kw)
        comm, _, check = found
        _send(comm, self.rank_in_group + 1 if dst is None else dst, tensor_dict, check)
        return None

    def recv_tensor_dict(self: Any, src: int | None = None, **kw: Any) -> Any:
        if alone(self):
            return _local_recv()
        found = _family() if _is_pipeline(self) else None
        if found is None:
            return _ORIGINALS["recv_tensor_dict"](self, src, **kw)
        comm, _, check = found
        return _recv(comm, self.rank_in_group - 1 if src is None else src, check)

    def isend_tensor_dict(self: Any, tensor_dict: Any, dst: int | None = None, **kw: Any) -> Any:
        if alone(self):
            return [Done()]
        found = _family() if _is_pipeline(self) else None
        if found is None:
            return _ORIGINALS["isend_tensor_dict"](self, tensor_dict, dst, **kw)
        comm, _, check = found
        _send(comm, self.rank_in_group + 1 if dst is None else dst, tensor_dict, check)
        return [Done()]

    def irecv_tensor_dict(self: Any, src: int | None = None, **kw: Any) -> Any:
        if alone(self):
            return _local_recv(), [Done()], []
        found = _family() if _is_pipeline(self) else None
        if found is None:
            return _ORIGINALS["irecv_tensor_dict"](self, src, **kw)
        comm, _, check = found
        got = _recv(comm, self.rank_in_group - 1 if src is None else src, check)
        return got, [Done()], []

    def make_sibling_device_group(self: Any, group_desc: str | None = None) -> Any:
        if not _is_pipeline(self):
            return _ORIGINALS["make_sibling_device_group"](self, group_desc)
        original = None
        if torch.distributed.is_initialized():
            original = _ORIGINALS["make_sibling_device_group"](self, group_desc)
        return FamilyBroadcastGroup(original)

    def broadcast(tensor: torch.Tensor, src: int = 0, group: Any = None, **kw: Any) -> Any:
        if not isinstance(group, FamilyBroadcastGroup):
            return _ORIGINAL_BROADCAST(tensor, src, group=group, **kw)
        if _LOCAL is not None:
            return None  # a stage alone: the tensor stays as the source made it
        found = _family()
        if found is None:
            if group.original is None:
                raise RuntimeError("a pipeline broadcast before this process joined any group")
            return _ORIGINAL_BROADCAST(tensor, src, group=group.original, **kw)
        comm, layout, check = found
        with EXCHANGE.one_in_flight(comm, comm.device, check):
            comm.broadcast(tensor, layout.stage(src))
        return None

    methods: dict[str, Any] = {}
    for name, fn in (
        ("send_tensor_dict", send_tensor_dict),
        ("recv_tensor_dict", recv_tensor_dict),
        ("isend_tensor_dict", isend_tensor_dict),
        ("irecv_tensor_dict", irecv_tensor_dict),
        ("make_sibling_device_group", make_sibling_device_group),
    ):
        fn.__name__ = name
        fn.__doc__ = _ORIGINALS[name].__doc__
        methods[name] = _reporting(fn) if name in PIPELINE_METHODS else fn
    return methods, broadcast


def release() -> tuple[dict[str, Any], Any]:
    """The originals :func:`routed` kept, by name, and the broadcast; the record cleared.

    The registry's restore puts them back; the next :func:`routed` records afresh.
    """
    global _ORIGINAL_BROADCAST
    originals, broadcast = dict(_ORIGINALS), _ORIGINAL_BROADCAST
    _ORIGINALS.clear()
    _ORIGINAL_BROADCAST = None
    return originals, broadcast


def _reporting(fn: Any) -> Any:
    """``fn`` with a failure naming the column's peers lost before it propagates (A51).

    The peer's sockets close when its process dies, and NCCL's send or receive fails about two
    seconds later. The worker reports the peers of its column through its daemon link, the
    second signal after the coordinator's own word (decision 0023), which the coordinator folds
    if it already knows. The failure then propagates as vLLM delivers it: a failed step, which
    the executor's change ends.

    A transfer refused because a change already refused the group (:class:`MemberSetChanged`)
    names no loss: the change names its own, and the peers are alive; reported, a rank's live
    peers would be marked lost.
    """

    @functools.wraps(fn)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except MemberSetChanged:
            raise
        except Exception as e:
            if _is_pipeline(self):
                _report_peers_lost(f"{fn.__name__}: {type(e).__name__}: {e}")
            raise

    return guarded


def _report_peers_lost(why: str) -> None:
    """Report this process's pipeline peers lost, if it has a daemon link and a layout."""
    from everserve_vllm import collective_op, worker
    from everserve_vllm.losses import pipeline_peers

    c = collective_op.collective()
    state = worker.STATE
    if c is None or c.layout is None or state is None:
        return
    peers = pipeline_peers(c.layout, c.members, c.me)
    if peers:
        _, epoch = state.runtime.identity()
        log.warning("pipeline peers %s lost: %s; reported at epoch %d", peers, why[:200], epoch)
        state.link.report_lost(peers, f"pipeline peers of rank {c.me}: {why[:200]}", epoch)
