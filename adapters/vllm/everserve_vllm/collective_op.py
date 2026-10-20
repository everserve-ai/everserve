# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The graph-safe all-reduce: vLLM's TP all-reduce through the process's ``Collective``.

The collective is :class:`~everserve.comm.collective.Collective`, one per process.

vLLM's ``GroupCoordinator.all_reduce`` for the TP group is replaced (the registry's A12,
:mod:`everserve_vllm.patches`) by the custom op ``everserve::all_reduce_into(inp, out,
group_name)``, which reduces in place into an ``out``
allocated in the traced code and calls the process-global :class:`VllmCollective`:

* decode-size tensors go to the slot collective (:mod:`everserve.comm.slotkernel`, the kernel in
  ``core/csrc/slot.cu``); FULL CUDA graphs capture that kernel, which reads its membership from
  the region on every replay, so a captured graph follows a membership change without recapture;
* larger tensors (prefill, mixed batches) go to the current group, resolved at call time. The op
  is a PIECEWISE split point, so those calls run eagerly and no graph ever holds a communicator.

``compilation_config()`` keeps stock's FULL_AND_PIECEWISE, restates the split points and sets
Inductor's deterministic mode (:data:`INDUCTOR`, decision 0029).

A membership change has two halves (:class:`VllmCollective`, decision 0025). On the daemon
link's thread the runtime refuses the current set: the slot's abort word, and every NCCL handle
this process can reach (the live group's, the fabric family's, vLLM's own) is pointed at a
refusing stand-in, so a step blocked on a dead peer fails in Python and nothing calls into those
communicators again. Their release (``ncclCommAbort``, and torch's abort of its groups) starts at
the same moment, each on its own thread, and the change is staged and acknowledged without
waiting for any of it. At the next step boundary, with this process's stream idle,
:meth:`VllmCollective.apply_staged` waits for each release up to a bound, leaves a slow one
behind as leaked, and points the collective, vLLM's TP group, the fabric family and the slot at
the new set. Applying it earlier would send the rest of the refused step's collectives to a set
the spare has not joined yet.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from everserve.comm.collective import Collective, GroupFamily
from everserve.comm.cover import FABRIC
from everserve.comm.group import ShrinkableGroup
from everserve.comm.handle import LocalStandIn
from everserve.comm.layout import ParallelLayout, Scope
from everserve.comm.shadow import (
    Member,
    MemberSet,
    is_replacement,
    replaced_rank,
    replacement,
    set_key,
)
from everserve.comm.slot import SlotCollective
from everserve_vllm.groups import Releases, abort_of
from everserve_vllm.settings import current as settings

log = logging.getLogger(__name__)

OP = "everserve::all_reduce_into"

_LIB = None
_ORIG_ALL_REDUCE = None
_COLLECTIVE: VllmCollective | None = None
_LOCK = threading.Lock()


@dataclass
class _Staged:
    members: MemberSet
    group: ShrinkableGroup | LocalStandIn
    epoch: int


class VllmCollective(Collective):
    """A :class:`Collective` whose change is applied at the step boundary.

    Args (beyond :class:`Collective`):
        spare_row: the slot row the spare uses; a member set's replacement id maps to it.
        on_apply: called with the new group after a staged change is applied (the worker points
            vLLM's TP group at it there).
        layout: the engine's layout; which members of a set this process's groups span. None
            is one stage: the live group is the whole engine, and slot rows are rank ids.
        fabric: the pipeline's groups (:class:`~everserve.comm.collective.GroupFamily`), for an
            engine with more than one stage; they move at the same boundary.
        on_apply_fabric: called with the fabric's new group after a staged change is applied
            (the worker points vLLM's PP group at it there).
    """

    def __init__(
        self,
        me: Member,
        live: ShrinkableGroup | LocalStandIn,
        shadows: Mapping[MemberSet, dist.ProcessGroup | ShrinkableGroup],
        slot: SlotCollective | None = None,
        spare_row: int = 4,
        on_apply: Any = None,
        layout: ParallelLayout | None = None,
        fabric: GroupFamily | None = None,
        on_apply_fabric: Any = None,
        spare_rows_at_columns: bool = False,
        stands_for: Sequence[Member] = (),
    ) -> None:
        """See :class:`Collective`; the backend and store are unused (groups are pre-formed).

        ``spare_rows_at_columns`` is a spare node's: every replacement sits at its column there.
        """
        super().__init__(
            me,
            live,
            shadows,
            slot,
            backend="nccl",
            store=None,
            fabric=fabric,
            stands_for=stands_for,
        )
        self.spare_row = spare_row
        self.spare_rows_at_columns = spare_rows_at_columns
        self.on_apply = on_apply
        self.on_apply_fabric = on_apply_fabric
        self.layout = layout
        self.staged: _Staged | None = None
        self.releases = Releases()
        """The refused groups' releases, finished at the boundary (decision 0025)."""
        self.release_bound_s: float | None = None
        """How long a release may run before it counts as leaked: the policy's, set at bootstrap;
        None is :data:`everserve_vllm.groups.RELEASE_BOUND_S`."""

    def rows_of(self, members: MemberSet) -> list[int]:
        """Slot rows for a member set: this stage's members' columns, the spare's row in its place.

        The slot region is a node's (one stage's): a member's row is its column, and a spare's
        the row kept for it on an engine node, or its column on a spare node, where the region
        holds the stage the spares serve. Without a layout the engine is one stage and a rank's
        row is its id.
        """
        if self.layout is None:
            return [self.spare_row if is_replacement(m) else int(m) for m in members]
        stage = self.layout.tensor_members(members, self.me)
        if self.spare_rows_at_columns:
            return [self.layout.column(replaced_rank(m) if is_replacement(m) else m) for m in stage]
        return [self.spare_row if is_replacement(m) else self.layout.column(m) for m in stage]

    def keep(self, scope: str, members: MemberSet, group: Any) -> None:
        """Hold a group a cover plan formed; with a stage of one, the stage's stand-in beside it.

        A plan names only groups of two or more (decision 0035). A change to a set still needs
        its stage-scope entry, and on a stage of one member that is a local stand-in: held here
        with the set's pipeline group, as the policy's bootstrap held it (:func:`stand_ins`).
        """
        super().keep(scope, members, group)
        if scope != FABRIC or self.layout is None:
            return
        with self._lock:
            members = tuple(self._renamed.get(m, m) for m in members)
            if (
                members not in self._shadows
                and len(self.layout.tensor_members(members, self.me)) == 1
            ):
                self._shadows[members] = LocalStandIn()

    def drop(self, scope: str, members: MemberSet) -> Any:
        """Let go of a group; a pipeline group's stage stand-in goes with it."""
        group = super().drop(scope, members)
        if scope == FABRIC:
            with self._lock:
                if isinstance(self._shadows.get(tuple(members)), LocalStandIn):
                    del self._shadows[tuple(members)]
        return group

    def take_identity(self, me: Member) -> list[tuple[str, MemberSet, Any]]:
        """Take ``me`` (the core's); the groups set aside wait for :meth:`release_set_aside`.

        The sets this spare formed as another position's spare are refused now, so nothing runs
        on them, and released once the take is done: their ``ncclCommAbort`` (a dead peer's
        among them) can hold the device while it runs, and the take's own device work, the
        shard's build and the warm-up, must not wait behind it (started at the take, the
        releases hold the worker's build 46 s).
        """
        dropped = super().take_identity(me)
        self._set_aside = [
            (f"{scope} {set_key(members)}", getattr(group, "refuse", None))
            for scope, members, group in dropped
        ]
        self._set_aside = [(name, refuse() if refuse else None) for name, refuse in self._set_aside]
        return dropped

    def release_set_aside(self) -> int:
        """Start the releases of the groups set aside at the take (decision 0025); how many.

        Reaped at the next boundary with the change's own and listed released or leaked like
        them, named ``"<scope> <set>"``.
        """
        pending, self._set_aside = getattr(self, "_set_aside", []), []
        for name, run in pending:
            self.releases.start(name, run)
        return len(pending)

    def refuse(self) -> None:
        """Refuse every group with a dead peer now, and start their release beside the change.

        The slot's abort word; the live group, the fabric family's current group and every
        handle vLLM holds pointed at a refusing stand-in; then one ``ncclCommAbort`` per
        communicator and torch's abort of its groups, each on a release thread. Returns at once:
        the acknowledgement does not wait for a release, the boundary does, bounded
        (decision 0025). Not called for a planned departure: the runtime refuses nothing for a
        change whose every lost rank departs on notice (protocol 9's ``Change.departing``).
        """
        super().refuse()
        self.releases.reap(self.release_bound_s)  # an earlier change's releases, by now
        own = {"live": self.current, "fabric": self.fabric.current if self.fabric else None}
        for name, group in own.items():
            refuse = getattr(group, "refuse", None)
            if refuse is not None:
                self.releases.start(name, refuse())
        # vLLM's handles to the same communicators are refused too; the rest are released here
        held = tuple(getattr(g, "group", None) for g in own.values())
        for name, comm in _refuse_engine_groups(skip=held):
            self.releases.start(name, abort_of(comm))
        self.releases.start("torch", _torch_abort())

    def change_to_group(
        self, members: MemberSet, group: ShrinkableGroup | LocalStandIn, epoch: int
    ) -> None:
        """Stage the change; :meth:`apply_staged` performs it at the step boundary.

        Raises:
            KeyError: if the fabric has no pre-formed group for ``members``; nothing is staged.
        """
        if self.fabric is not None and not self.fabric.has(members):
            raise KeyError(f"no pre-formed fabric group for {set_key(tuple(members))}")
        with self._lock:
            self.staged = _Staged(tuple(members), group, epoch)
        log.info("membership %s staged at epoch %d", set_key(tuple(members)), epoch)

    def serving(self) -> set[tuple[str, MemberSet]]:
        """The groups served on, and the staged change's: the boundary is about to take them."""
        out = super().serving()
        with self._lock:
            staged = self.staged
        if staged is not None:
            scopes = ("domain", FABRIC) if self.fabric is not None else ("domain",)
            out |= {(scope, staged.members) for scope in scopes}
        return out

    def apply_staged(self) -> dict[str, Any] | None:
        """At a step boundary: move to the staged set. Returns what moved, or None.

        The slot's counters and words are zeroed and its row's order written from the staged
        set; the executor clears the abort words of the rows whose process is gone.
        """
        with self._lock:
            staged, self.staged = self.staged, None
        if staged is None:
            return None
        group = staged.group
        with self._lock:
            self.current = group
            self.members = staged.members
            self.epoch = staged.epoch
            self.handle.swap(group.group if isinstance(group, ShrinkableGroup) else group)
        rows = self.rows_of(staged.members)
        if self.slot is not None:
            self.slot.resume()
            self.slot.set_members(rows, staged.epoch)
        if self.on_apply is not None:
            self.on_apply(group)
        if self.fabric is not None:
            fabric_group = self.fabric.swap(staged.members)
            if self.on_apply_fabric is not None:
                self.on_apply_fabric(fabric_group)
        # after the swap, never before it: the boundary waits for no release (decision 0025)
        self.releases.reap(self.release_bound_s)
        return {"members": set_key(staged.members), "rows": rows, "epoch": staged.epoch}


def _refuse_engine_groups(skip: tuple[Any, ...] = ()) -> list[tuple[str, Any]]:
    """Point every vLLM group at a :class:`~everserve_vllm.groups.RefusedComm`.

    An aborted handle marked disabled sends vLLM to its torch process group, which still holds
    the dead rank; a refusing stand-in makes the interrupted step fail fast instead. Returns the
    replaced handles other than those in ``skip`` (the live TP handle and the fabric family's,
    released by their own groups).
    """
    try:
        from vllm.distributed import parallel_state as ps
    except ImportError:
        return []
    from everserve_vllm.groups import RefusedComm, unguarded

    rest = []
    for name in ("_TP", "_EP", "_DP", "_PP", "_WORLD"):
        dc = getattr(getattr(ps, name, None), "device_communicator", None)
        pn = getattr(dc, "pynccl_comm", None)
        if pn is None or isinstance(pn, RefusedComm):
            continue
        pn = unguarded(pn)  # the communicator itself: compared and aborted, never its guard
        dc.pynccl_comm = RefusedComm(pn)
        if (
            all(pn is not s for s in skip)
            and getattr(pn, "comm", None) is not None
            and not pn.disabled
        ):
            rest.append((name, pn))
    return rest


def _torch_abort() -> Callable[[], None] | None:
    """Torch's abort of its process groups, for a release thread; None without a torch world."""
    try:
        from torch.distributed import distributed_c10d as c10d
    except ImportError:
        return None
    return c10d._abort_process_group if c10d.is_initialized() else None


def collective() -> VllmCollective | None:
    """This process's collective, or None before its first collective."""
    return _COLLECTIVE


def ensure(group_name: str = "tp:0") -> VllmCollective:
    """This process's collective, made now if no collective has run yet.

    The op makes it at the first graph-safe all-reduce; an engine without one (tensor
    parallelism of one, where the model reduces nothing) never calls the op, and bootstrap
    needs the collective all the same: the slot region and its watch, the fabric family for the
    pipeline, the runtime. ``group_name`` is vLLM's name for the tensor-parallel group.
    """
    global _COLLECTIVE
    if _COLLECTIVE is None:
        with _LOCK:
            if _COLLECTIVE is None:
                _COLLECTIVE = _make(group_name)
    return _COLLECTIVE


def standing(eligible: Sequence[int], group_name: str = "tp:0") -> VllmCollective:
    """A collective for one engine of a pool spare's, standing for ``eligible`` (decision 0031).

    A pool spare stands for a position of each of several engines of one shape at once, each
    with its own collective: their member sets name the same ranks, so one collective cannot
    hold them. None is the process's (:func:`ensure`) until a take adopts one (:func:`adopt`).
    """
    return _make(group_name, eligible)


def adopt(c: VllmCollective) -> None:
    """Make ``c`` the process's collective: a pool spare's, for the engine whose position it took.

    Raises:
        RuntimeError: when the process already has another collective.
    """
    global _COLLECTIVE
    with _LOCK:
        if _COLLECTIVE is not None and _COLLECTIVE is not c:
            raise RuntimeError("this process's collective is another engine's")
        _COLLECTIVE = c


def _make(group_name: str, eligible: Sequence[int] | None = None) -> VllmCollective:
    """Build the process collective at the first collective (the profiling run).

    Member ids are the engine's ranks. The live group is vLLM's TP communicator over this
    stage's ranks; in the spare it is a local stand-in until it joins. With more than one stage
    the pipeline's communicator is the fabric family's live group, swapped at the same boundary.
    The slot is this node's region at this process's row, its column in the stage.
    """
    from vllm.distributed import parallel_state as ps

    g = ps._groups[group_name]()
    if g is None:
        raise ValueError(f"group {group_name} is destroyed")
    s = settings()
    spare = s.spare
    layout = ParallelLayout(*s.layout) if s.layout else ParallelLayout(g.world_size)
    # a spare's id in member sets names the rank it stands in for, so a set that replaces
    # several ranks at once names each spare
    eligible = list(s.spare_any if eligible is None else eligible)
    if spare and eligible:
        me: Member = replacement(eligible[0])
    elif spare:
        if s.spare_rank is None:
            raise RuntimeError("a spare launched without the rank it stands for")
        me = replacement(s.spare_rank)
    else:
        me = int(g.rank)
    live: ShrinkableGroup | LocalStandIn
    fabric: GroupFamily | None = None
    if spare:
        live = LocalStandIn()
        if layout.pipeline_parallel > 1:
            fabric = GroupFamily(Scope.FABRIC, LocalStandIn())
    else:
        from everserve_vllm.groups import PyNcclGroup

        # a tensor-parallel group of one has no communicator in vLLM: nothing to reduce, and
        # the stand-in says so; the slot region still has its row, for the watch
        dc = g.device_communicator
        live = (
            PyNcclGroup.over(dc.pynccl_comm, list(g.ranks))
            if g.world_size > 1 and dc is not None
            else LocalStandIn()
        )
        if layout.pipeline_parallel > 1:
            pp = ps.get_pp_group()
            fabric = GroupFamily(
                Scope.FABRIC,
                PyNcclGroup.over(pp.device_communicator.pynccl_comm, list(pp.ranks)),
            )
    slot = None
    path = s.region
    if path and not (spare and eligible):  # a spare for any position makes its slot at the take
        from everserve.comm.slotkernel import Region, SlotKernel

        row = s.row if s.row is not None else layout.column(me)
        slot = SlotKernel(Region(path), row, torch.cuda.current_device())
    c = VllmCollective(
        me,
        live,
        {},
        slot,
        spare_row=layout.tensor_parallel,
        layout=layout,
        fabric=fabric,
        spare_rows_at_columns=s.spare_node,
        stands_for=[replacement(r) for r in eligible[1:]] if spare else (),
    )
    c.members = tuple(range(layout.world))  # the spare's too: its sets are over this world
    return c


def take(rank: int, row: int) -> int:
    """A spare standing for any position takes ``rank``'s (decision 0030); the sets set aside.

    Its collective becomes ``rank``'s replacement, keeping that position's pre-formed sets, and
    its slot is made at ``row`` of this node's region, before the attachment captures graphs
    that run the slot kernel. The environment then names the rank as a preloaded spare's does.
    """
    c = ensure()
    dropped = len(c.take_identity(replacement(rank)))
    from everserve_vllm import settings as process_settings

    path = process_settings.update(spare_rank=rank, row=row).region
    if path and c.slot is None:
        from everserve.comm.slotkernel import Region, SlotKernel

        c.slot = SlotKernel(Region(path), row, torch.cuda.current_device())
    return dropped


def all_reduce_into(c: Collective, inp: torch.Tensor, out: torch.Tensor) -> None:
    """The op's body, separate from vLLM: the collective decides slot or group by size."""
    c.all_reduce_into(inp, out)


def _op(inp: torch.Tensor, out: torch.Tensor, group_name: str) -> None:
    all_reduce_into(ensure(group_name), inp, out)


def _op_fake(inp: torch.Tensor, out: torch.Tensor, group_name: str) -> None:
    return None


def register_op() -> None:
    """Register ``everserve::all_reduce_into`` into our library once.

    Our own library, not a patch on vLLM: it stays registered when A12 is taken off.
    """
    global _LIB
    if _LIB is not None:
        return
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    _LIB = Library("everserve", "FRAGMENT")
    direct_register_custom_op(
        "all_reduce_into", _op, mutates_args=["out"], fake_impl=_op_fake, target_lib=_LIB
    )


def routed_all_reduce(original: Any) -> Any:
    """``GroupCoordinator.all_reduce`` through the op on a TP group of more than one.

    ``original`` (kept in :data:`_ORIG_ALL_REDUCE`, for :func:`release_all_reduce`) runs on
    every other group. The registry puts it on (A12, :data:`everserve_vllm.patches.ALL_REDUCE`).
    """
    global _ORIG_ALL_REDUCE
    _ORIG_ALL_REDUCE = original

    def all_reduce(self: Any, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1 or not self.unique_name.startswith("tp"):
            return _ORIG_ALL_REDUCE(self, input_)
        out = torch.empty_like(input_)
        torch.ops.everserve.all_reduce_into(input_, out, self.unique_name)
        return out

    return all_reduce


def release_all_reduce() -> Any:
    """The original :func:`routed_all_reduce` kept; the record cleared."""
    global _ORIG_ALL_REDUCE
    original, _ORIG_ALL_REDUCE = _ORIG_ALL_REDUCE, None
    return original


def compilation_config(guarantee: Any = None) -> dict[str, Any]:
    """Split at attention, the KV-cache update and the graph-safe all-reduce; keep FULL graphs.

    A user-supplied ``splitting_ops`` replaces vLLM's default list, so the attention split points
    are restated from its class attribute. Inductor's deterministic settings (:data:`INDUCTOR`)
    follow the engine's guarantee class (decision 0030): an exact engine compiles with them, a
    continuous one with vLLM's defaults, its numerics stated rather than promised identical.
    ``guarantee`` is the policy's ``Engine.guarantee``; None is exact, the class an engine has
    unless its policy says otherwise.
    """
    from vllm.config.compilation import CompilationConfig

    from everserve.policy import Guarantee

    ops = list(CompilationConfig._attention_ops) + [
        "vllm::unified_kv_cache_update",
        "vllm::unified_mla_kv_cache_update",
        OP,
    ]
    out: dict[str, Any] = {"splitting_ops": ops}
    if guarantee in (None, Guarantee.EXACT):
        out["inductor_compile_config"] = dict(INDUCTOR)
    return out


INDUCTOR: dict[str, Any] = {
    "deterministic": True,
    "combo_kernels": False,
    "benchmark_combo_kernel": False,
}
"""Inductor's settings for every engine the adapter launches (decision 0029).

``deterministic`` takes timing out of the compiled kernels: Inductor's autotuner then refuses to
benchmark reductions, persistent reductions and scans (their tile and warp choices shift
numerics) and picks their one configuration by rule, so two launches of an engine, on one node
or across the nodes a staged cache reaches, run the same kernels. Pointwise kernels still
benchmark; their configuration cannot change a result. Without it a launch that autotunes one
reduction fresh lands in another rounding family (about one launch in nine, and a spare on its
own node); the staged cache and the register check of 0029 catch such a launch, this keeps it
from happening.

Combo kernels are off with it: vLLM turns them on with their benchmark whenever neither is set
(``CompilationConfig.__post_init__``), and that benchmark is one deterministic mode bans, so a
compile with both raises at the scheduler's combo-kernel pass and every engine fails its first
compile. The throughput arms carry their cost.
"""
