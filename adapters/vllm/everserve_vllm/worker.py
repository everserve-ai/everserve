# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Per-worker bootstrap and the step-boundary half of a membership change.

After the engine has loaded, allocated KV and captured its graphs, the executor calls
:func:`bootstrap` on every serving worker (and the spare calls :func:`start` on itself). Each
process then holds, in :data:`STATE`:

* its :class:`~everserve_vllm.collective_op.VllmCollective` (made at the first collective) with
  the pre-formed sets added: under a cover plan (decision 0035) the groups each plan names, formed
  in the background by :data:`COVER` (:func:`everserve_vllm.groups.form_planned`); otherwise the
  policy's (:func:`everserve_vllm.groups.form_shadow_groups` over
  :func:`everserve.policy.sets_to_preform`: a replacement set per launched spare), at bootstrap;
* an :class:`AdapterRuntime` governing it;
* a :class:`~everserve.state.mirror.MirrorCommittedKV` over a ``MirrorClient`` to the node's
  ``everserve-store``, attached to the connector, with the runtime's attempt callbacks dropping
  uncommitted rows and its commit callbacks making a step durable;
* a :class:`~everserve.contract.daemon.DaemonConnection` to ``everserve-controller``; the daemon's
  ``Change`` arrives there and becomes ``Runtime.on_change(lost, outcome)``;
* a :class:`~everserve.comm.slotwatch.SlotWatch` over its slot row, reporting through that link a
  member the slot kernel waits for whose process is exiting, or which stays late past
  ``Bootstrap.hang_s`` (decision 0021).

Both the daemon socket and the mirror socket come from the daemon's published record
(:class:`everserve.discovery.FileDiscovery`).

:func:`boundary` is the RPC survivors run between steps once a change is committed: apply the
runtime's pending change, move the collective and vLLM's TP group to the new set, attach to the
executor's new broadcast ring.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from everserve.comm.collective import LocalStandIn
from everserve.comm.cover import Cover, LinkReports, plan_from
from everserve.comm.layout import ParallelLayout
from everserve.comm.runtime import Runtime
from everserve.comm.shadow import Member as MemberId
from everserve.comm.shadow import (
    MemberSet,
    check_formable,
    rendezvous_base_port,
    rendezvous_plan,
    replacement,
    set_key,
)
from everserve.comm.slotwatch import SlotWatch
from everserve.contract.compile_cache import (
    artifact_digest,
    rank_digest,
    rank_manifest,
    share,
    write_manifest,
)
from everserve.contract.compile_cache import digest as cache_digest
from everserve.contract.daemon import EXIT_STALE, DaemonConnection
from everserve.contract.signals import (
    CompileCache,
    CoverPlan,
    EnginePlacement,
    Member,
    Role,
    SlotChoice,
    SoftwareIdentity,
)
from everserve.discovery import DaemonRecord, FileDiscovery, local_daemon
from everserve.policy import Engine, Policy, sets_to_preform
from everserve.state.kvcopy import PagedKV
from everserve.state.mirror import MirrorClient, MirrorCommittedKV
from everserve_vllm import collective_op, firststep, pipeline
from everserve_vllm.collective_op import VllmCollective
from everserve_vllm.groups import RELEASE_BOUND_S

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bootstrap:
    """What every process of one engine is told at bootstrap.

    Attributes:
        engine: the engine's name in the policy and at the daemon.
        node: the node the executor and the spares it launched are on; its daemon record is
            ``<discovery_dir>/<node>.json``.
        nodes: each rank's node, in rank order, from the policy's engine entry: a worker
            registers with its own node's daemon and mirrors into its own node's store.
        discovery_dir: where daemons publish their records.
        world: serving ranks.
        sets: the member sets to pre-form, in the global order every process uses.
        layout: the engine's layout: which members of a set each process's groups span.
        hosts: each member's node address, where a group whose first member it is
            rendezvouses (:func:`rendezvous_hosts`).
        base_port: set ``i``'s stage group rendezvouses on ``base_port + 2i``, its column
            group on ``base_port + 2i + 1``.
        event_dir: where each runtime writes ``rank<N>.json``, or empty for none.
        hang_s: how long a live member may stay late in the slot collective before the slot
            watch reports it hung.
        release_bound_s: how long a boundary waits for a refused group's release before
            leaving it behind (decision 0025): the policy's engine entry's
            ``releaseBoundMs``, or :data:`everserve_vllm.groups.RELEASE_BOUND_S`.

    The pipeline exchange keeps as many transfers in flight per communicator as the engine has
    stages (:attr:`exchange_depth`), and a scheduled step's tensors cross it packed
    (:mod:`everserve_vllm.pipeline`); neither is configured.
    """

    engine: str
    node: str
    nodes: tuple[str, ...]
    discovery_dir: str
    world: int
    sets: tuple[MemberSet, ...]
    layout: ParallelLayout
    hosts: Mapping[MemberId, str]
    base_port: int
    event_dir: str = ""
    hang_s: float = 10.0
    release_bound_s: float = RELEASE_BOUND_S
    cover_planned: bool = False
    """The coordinator plans the groups this engine pre-forms (decision 0035): ``sets`` is empty,
    and each process forms what the plans its daemon forwards name."""
    cover_form_timeout_s: float = 30.0
    """How long a formation of a planned group may run (the policy's ``cover.formTimeoutMs``)."""

    @property
    def exchange_depth(self) -> int:
        """Transfers the exchange keeps in flight per communicator: one per pipeline stage.

        At a depth of the stage count the packed exchange matches stock vLLM's overlapped
        pipeline sends.
        """
        return self.layout.pipeline_parallel


class AdapterRuntime(Runtime):
    """The runtime with the adapter's cover wired to its changes and plans."""

    def on_cover_plan(self, plan: CoverPlan) -> None:
        """A cover plan its daemon forwarded (decision 0035): this process's cover applies it."""
        cover_plan(plan)

    def on_change(self, lost: Sequence[int], *args: Any, **kw: Any) -> None:
        """No group is formed between a change and its commit; the boundary resumes the cover."""
        global _CHANGE_LOST
        if COVER is not None:
            COVER.pause()
            _CHANGE_LOST = tuple(lost)
        super().on_change(lost, *args, **kw)

    def on_displaced(self, epoch: int, why: str) -> None:
        """A later commit no longer has this process serve its rank: it leaves.

        Where the node ends released processes (the node role's region), it leaves as a rank
        leaving on notice does: its link closed, then its row marked released, which the node
        reads and ends it on, with no loss reported. A node left holding nothing ends, and its
        pod asks anew. Elsewhere it ends itself, with
        :data:`~everserve.contract.daemon.EXIT_STALE`. Its rank serves elsewhere, so there is
        nothing of it to drain.
        """
        log.warning("leaving: displaced at epoch %d (%s)", epoch, why)
        self._trace("displaced", epoch=epoch, detail=why)
        slot = STATE.collective.slot if STATE is not None else None
        if slot is not None and hasattr(slot, "region") and slot.region.ends_released():
            from everserve_vllm.vacate import release_position

            release_position(slot.region, slot.row, STATE.link)
            return
        os._exit(EXIT_STALE)


@dataclass
class WorkerState:
    """This process's adapter objects."""

    rank: int
    collective: VllmCollective
    runtime: AdapterRuntime
    kv: MirrorCommittedKV
    link: DaemonConnection
    formed: dict[str, float] = field(default_factory=dict)
    watch: SlotWatch | None = None


STATE: WorkerState | None = None

STEP_METHODS = ("execute_model", "sample_tokens")
"""The RPCs a step is made of: while one is unanswered the step is in flight, and the workers'
slot watch reads its members' processes when a step stalls (decision 0021)."""


def head_slots(kv_caches: dict[str, torch.Tensor]) -> int:
    """KV head slots per layer on this rank, from the paged layout."""
    t = next(iter(kv_caches.values()))
    paged = PagedKV.of(t)
    return t.shape[1] if paged.layout.value == "combined" else t.shape[-2]


def start(
    rank: int,
    device: torch.device,
    cfg: Bootstrap,
    role: Role = Role.WORKER,
    runtime_cls: type[AdapterRuntime] = AdapterRuntime,
    slot_choice: SlotChoice | None = None,
    compile_cache: CompileCache | None = None,
    identity: SoftwareIdentity | None = None,
) -> dict[str, Any]:
    """Build this process's adapter objects and connect them. Returns timings for the run log.

    ``compile_cache`` is the pair this process loaded and warmed with (:func:`compiled_with`),
    carried in its registration (decision 0029), as is ``identity``, what it runs (0034).

    ``role`` and ``runtime_cls`` are the spare's (``SPARE``, its restoring runtime) in the spare;
    ``slot_choice`` is a spare's where it created its node's region.

    Raises:
        RuntimeError: without a worker-side connector, or when the node's daemon has not
            published its record.
    """
    t0 = time.time()
    torch.cuda.set_device(device)
    # made at the first graph-safe all-reduce, or here for an engine that reduces nothing (TP1)
    c = collective_op.ensure()
    c.release_bound_s = cfg.release_bound_s
    pipeline.EXCHANGE.in_flight = max(1, cfg.exchange_depth)
    formed = _form(c, cfg, c.me, device)
    t1 = time.time()
    # a worker is on its rank's node; a spare is on the node that launched it
    node = cfg.nodes[rank] if role is Role.WORKER and rank < len(cfg.nodes) else cfg.node
    record = _record(cfg, node)

    def link_for(runtime: AdapterRuntime) -> DaemonConnection:
        return DaemonConnection(
            record.address,
            cfg.engine,
            Member(node, device.index or 0),
            runtime,
            role=role,
            slot=slot_choice,
            compile_cache=compile_cache,
            identity=identity,
        )

    if cfg.cover_planned:
        # before the link registers: a plan may follow its registration at once
        c.name_process(node, device.index or 0)
    out = _bind(rank, device, cfg, runtime_cls, c, formed, record, link_for)
    if cfg.cover_planned:
        assert STATE is not None
        start_cover(c, cfg, STATE.link, device, node)
    return {
        "rank": rank,
        "role": role.value,
        "shadow_s": round(t1 - t0, 3),
        "formed": formed,
        **out,
        "bootstrap_s": round(time.time() - t0, 3),
    }


def _form(c: VllmCollective, cfg: Bootstrap, me: Member, device: torch.device) -> dict[str, float]:
    """Form and keep every group of ``cfg.sets`` that ``me`` is in, in both scopes."""
    from everserve_vllm.groups import form_shadow_groups

    layout = cfg.layout
    domain = rendezvous_plan(
        cfg.sets, me, lambda s: layout.tensor_members(s, me), cfg.hosts, cfg.base_port, 0, 2
    )
    shadows, formed = form_shadow_groups(me, domain, device)
    stand_ins(shadows, cfg.sets, me, lambda s: layout.tensor_members(s, me))
    c.preform(shadows)
    if c.fabric is not None:
        fabric = rendezvous_plan(
            cfg.sets, me, lambda s: layout.pipeline_members(s, me), cfg.hosts, cfg.base_port, 1, 2
        )
        pipeline, formed_pipeline = form_shadow_groups(me, fabric, device)
        stand_ins(pipeline, cfg.sets, me, lambda s: layout.pipeline_members(s, me))
        c.fabric.preform(pipeline)
        formed.update(formed_pipeline)
    return formed


def _record(cfg: Bootstrap, node: str) -> DaemonRecord:
    record = local_daemon(node, cfg.discovery_dir)
    if record is None:
        raise RuntimeError(f"no daemon record for node {node} in {cfg.discovery_dir}")
    return record


def _bind(
    rank: int,
    device: torch.device,
    cfg: Bootstrap,
    runtime_cls: type[AdapterRuntime],
    c: VllmCollective,
    formed: dict[str, float],
    record: DaemonRecord,
    link_for: Callable[[AdapterRuntime], DaemonConnection],
) -> dict[str, Any]:
    """This process as ``rank``: its runtime, its KV at the node's store, its link and watch."""
    global STATE
    from everserve_vllm import connector

    conn = connector.WORKER
    if conn is None:
        raise RuntimeError("no worker-side KV connector in this process")
    layout = cfg.layout
    runtime = runtime_cls(
        rank,
        cfg.world,
        device,
        c,
        Path(cfg.event_dir) if cfg.event_dir else None,
        # the executor applies the change on every survivor at one boundary and the scheduler
        # replays from the dead rank's committed counts, so survivors cannot split on a step;
        # the runtime's reconciliation collective would also run before the boundary moves the
        # collective, on the old set
        reconcile_on_recover=False,
    )
    heads = head_slots(conn.kv_caches)
    kv = MirrorCommittedKV(
        MirrorClient(record.mirror_address),
        cfg.engine,
        rank,
        units=lambda: [(h, (rank, h)) for h in range(heads)],
    )
    kv.register(conn.kv_caches)
    conn.lifecycle.attach(kv, runtime)
    link = link_for(runtime)
    watch = None
    slot = c.slot
    if slot is not None and hasattr(slot, "region"):
        watch = SlotWatch(
            slot.region,
            slot.row,
            report=lambda ranks, why, epoch: link.report_lost(ranks, why, epoch),
            epoch=lambda: runtime.identity()[1],
            progress=lambda: conn.lifecycle.ended,
            hang_s=cfg.hang_s,
            first_rank=layout.stage(rank) * layout.tensor_parallel,
        ).start()
    STATE = WorkerState(rank, c, runtime, kv, link, formed, watch)
    c.releases.on_leak = link.report_leak  # a release left behind is reported (0025, amended)
    return {"heads": heads, "layout": kv.paged.layout.value if kv.paged else None}


class Standing:
    """A spare that stands for any position of an engine before a change gives it one.

    Decision 0030: the process holds no shard, no KV and no rank. It has formed the replacement
    sets of every position it is eligible for and registered with its daemon as standing for
    any; this is the runtime its link applies changes to until then. A change that names it in
    ``take`` calls :meth:`on_take` first, which hands that take to ``take`` (the process's main
    thread builds and warms the shard there and binds the rank's runtime and KV) and becomes
    that runtime's stand-in: every later call goes to it, the change's own first.
    """

    def __init__(self, node: str, device: int, take: Callable[[Any], AdapterRuntime]) -> None:
        """Stand on ``device`` of ``node``; ``take(t)`` binds ``t.rank``, returns its runtime."""
        self.node = node
        self.device = device
        self._take = take
        self.target: AdapterRuntime | None = None
        self.taking: int | None = None
        self.orig_rank = 0  # unused on the wire for a spare standing for any position
        self.epoch = 0

    def on_take(self, takes: Sequence[Any]) -> None:
        """Take the position a change names for this node and device, if it names one.

        The same take delivered again (a coordinator that restarted before the commit
        re-delivers the change it logged) is the take already made, and does nothing.

        Raises:
            RuntimeError: a take of another rank after one, or the binding's own failure.
        """
        mine = [t for t in takes if t.node == self.node and t.device == self.device]
        if not mine:
            return
        if self.target is not None and self.target.orig_rank == mine[0].rank:
            return  # re-delivered: this spare holds the rank already
        if self.target is not None:
            raise RuntimeError(
                f"this spare already took rank {self.target.orig_rank}; a change gave it "
                f"rank {mine[0].rank}"
            )
        self.taking = mine[0].rank
        self.target = self._take(mine[0])

    def identity(self) -> tuple[int, int]:
        """The taken rank's identity once taken; the rank being taken while it binds."""
        if self.target is not None:
            return self.target.identity()
        return (self.taking if self.taking is not None else self.orig_rank), self.epoch

    def __getattr__(self, name: str) -> Any:
        """Before a take there is no runtime to answer; after, the taken rank's answers."""
        target = self.__dict__.get("target")
        if target is None:
            raise AttributeError(f"a standing spare has no {name} before it takes a position")
        return getattr(target, name)

    def on_cover_plan(self, plan: CoverPlan) -> None:
        """A cover plan, before a take and after alike: the process's cover applies it."""
        cover_plan(plan)

    def on_change(self, lost: Sequence[int], **kw: Any) -> None:
        """A change before any take is not this spare's; after, the taken runtime's."""
        if self.target is not None:
            self.target.on_change(lost, **kw)


def stand(
    device: torch.device,
    cfg: Bootstrap,
    take: Callable[[Any], AdapterRuntime],
    slot_choice: SlotChoice | None = None,
    identity: SoftwareIdentity | None = None,
    collective: VllmCollective | None = None,
) -> tuple[Standing, dict[str, Any]]:
    """Stand as a spare for any of the positions the process's collective names (decision 0030).

    Forms the replacement sets of every one of them, then registers with this node's daemon as
    standing for any, with no compile cache (it reports its pair at the join, once it has
    attached). Returns the standing runtime and the formation's record.
    """
    t0 = time.time()
    torch.cuda.set_device(device)
    # the kernels loaded now, not at the take: the slot collective made at the take builds on them,
    # and a build there sits on the change's path (48 s)
    from everserve import kernels

    kernels.load()
    # a pool spare's engine stands on a collective of its own (decision 0031)
    c = collective if collective is not None else collective_op.ensure()
    c.release_bound_s = cfg.release_bound_s
    pipeline.EXCHANGE.in_flight = max(1, cfg.exchange_depth)
    formed: dict[str, float] = {}
    for me in c.stands_for:
        formed.update(_form(c, cfg, me, device))
    t1 = time.time()
    record = _record(cfg, cfg.node)
    if cfg.cover_planned:
        c.name_process(cfg.node, device.index or 0)
    standing = Standing(cfg.node, device.index or 0, take)
    standing.link = DaemonConnection(
        record.address,
        cfg.engine,
        Member(cfg.node, device.index or 0),
        standing,
        role=Role.SPARE,
        slot=slot_choice,
        any_rank=True,
        identity=identity,
    )
    standing.collective, standing.formed, standing.record = c, formed, record
    if cfg.cover_planned:
        start_cover(c, cfg, standing.link, device)
    return standing, {
        "stands_for": list(c.stands_for),
        "sets": len(c.shadow_sets),
        "shadow_s": round(t1 - t0, 3),
        "formed": formed,
    }


def bind_taken(
    standing: Standing,
    rank: int,
    device: torch.device,
    cfg: Bootstrap,
    runtime_cls: type[AdapterRuntime],
) -> AdapterRuntime:
    """Bind the rank a standing spare took: its runtime and KV, over the link it registered on."""
    _bind(
        rank,
        device,
        cfg,
        runtime_cls,
        standing.collective,
        standing.formed,
        standing.record,
        lambda runtime: standing.link,
    )
    assert STATE is not None
    return STATE.runtime


def stand_ins(
    shadows: dict[MemberSet, Any],
    sets: Sequence[MemberSet],
    me: MemberId,
    select: Callable[[MemberSet], Sequence[MemberId]],
) -> None:
    """A stand-in for every set of ``sets`` whose group for ``me`` is ``me`` alone.

    The rendezvous forms no group of one (tensor or pipeline parallelism of one), and a change
    to such a set must still find its group: the stand-in says there are no peers, and the
    collective keeps it as it keeps a spare's before the join. Sets without ``me`` are other
    processes' to form.
    """
    for s in sets:
        if s not in shadows and me in s and len(select(s)) == 1:
            shadows[s] = LocalStandIn()


def rendezvous_hosts(
    engine: Engine,
    records: Mapping[str, DaemonRecord | None],
    spare_nodes: Mapping[int, str],
) -> dict[MemberId, str]:
    """Each member's node address, where a group it leads rendezvouses.

    A rank's address is its node's, from the node's daemon record; a spare's is the node that
    holds it, ``spare_nodes[rank]``. ``records`` holds each node's record, None where none is
    published, which is an error here: a group whose first member is on that node could not
    rendezvous.

    Raises:
        RuntimeError: for a node without a daemon record.
    """
    hosts: dict[MemberId, str] = {}
    for rank, m in enumerate(engine.members):
        rec = records.get(m.node)
        if rec is None:
            raise RuntimeError(f"no daemon record for node {m.node}, which holds rank {rank}")
        hosts[rank] = rec.host
    for r, node in spare_nodes.items():
        rec = records.get(node)
        if rec is None:
            raise RuntimeError(f"no daemon record for node {node}, which holds the spare for {r}")
        hosts[replacement(r)] = rec.host
    return hosts


def eligible_ranks(engine: Engine, node: str) -> list[int]:
    """The positions a spare on ``node`` that stands for any position can take (decision 0030).

    Every rank of the engine not served on ``node``: a spare cannot replace a rank on its own
    node, which a loss of the node takes with it.
    """
    return [r for r, m in enumerate(engine.members) if m.node != node]


def spare_nodes_of(engine: Engine) -> dict[int, str]:
    """Each rank with a spare, and the node that holds it, from the policy's spare slots.

    A slot for one rank stands for it; a slot without a rank stands for every rank it is
    eligible for (:func:`eligible_ranks`), so every process pre-forms each of those replacement
    sets with the spare's node as the spare's host. A rank with a slot of its own keeps it.
    """
    out = {s.rank: s.node for s in engine.spares if s.rank is not None}
    for s in engine.spares:
        if s.rank is None:
            for r in eligible_ranks(engine, s.node):
                out.setdefault(r, s.node)
    return out


def bootstrap_from_policy(
    policy: Policy,
    engine: str,
    node: str,
    discovery_dir: str,
    spare_nodes: Mapping[int, str],
    event_dir: str = "",
    hang_s: float = 10.0,
) -> Bootstrap:
    """What every process of ``engine`` pre-forms and connects to, from the policy and discovery.

    The executor computes it for its workers and sends it to them; a spare a daemon launched
    computes the same for itself (decision 0024), since nobody sends it anything before the
    join: the member sets from the policy's outcomes and the spares it places, each member's
    host from the nodes' records, the rendezvous ports from the engine's name. For an engine the
    policy pins (bare metal, a rig); one placed by what registers bootstraps from its placement
    (:func:`bootstrap_from_placement`).

    Raises:
        RuntimeError: for a node without a daemon record.
        ValueError: for a set no process could play (``check_formable``), or an engine whose
            policy pins no members.
    """
    spec = policy.engine(engine)
    if not spec.members:
        # placed by what registers (decision 0041): its placement is the control plane's
        raise ValueError(f"engine {engine!r} pins no members: its placement comes from the link")
    discovery = FileDiscovery(discovery_dir)
    nodes = {m.node for m in spec.members} | set(spare_nodes.values()) | {node}
    hosts = rendezvous_hosts(spec, {n: discovery.daemon(n) for n in nodes}, spare_nodes)
    return _bootstrap(
        policy,
        engine,
        node,
        tuple(m.node for m in spec.members),
        hosts,
        sorted(spare_nodes),
        discovery_dir,
        event_dir,
        hang_s,
    )


def bootstrap_from_placement(
    policy: Policy,
    engine: str,
    node: str,
    placement: EnginePlacement,
    event_dir: str = "",
    hang_s: float = 10.0,
) -> Bootstrap:
    """What every process of ``engine`` pre-forms and connects to, from its placement.

    The control plane's placement (protocol 11), for an engine placed by what registers, which
    pins no members. Each rank's node and host are its placement's; each standing spare's host
    is where its rank's replacement rendezvouses, a spare for any position standing for every
    rank not on its own node (as :func:`spare_nodes_of`), a spare for one rank keeping it.

    Raises:
        ValueError: for a placement that does not place every rank of the engine's world once,
            with its host, or a set no process could play (``check_formable``).
    """
    spec = policy.engine(engine)
    by_rank = {p.rank: p for p in placement.members}
    world = spec.world_size
    if len(placement.members) != len(by_rank) or set(by_rank) != set(range(world)):
        raise ValueError(
            f"engine {engine!r}'s placement at epoch {placement.epoch} places ranks "
            f"{sorted(p.rank for p in placement.members)}, not each of its {world} once"
        )
    unaddressed = [p.rank for p in placement.members if not p.host]
    unaddressed += [f"spare on {s.node}" for s in placement.spares if not s.host]
    if unaddressed:
        raise ValueError(f"engine {engine!r}'s placement has no host for {unaddressed}")
    nodes = tuple(by_rank[r].node for r in range(world))
    hosts: dict[MemberId, str] = {r: by_rank[r].host for r in range(world)}
    spare_hosts = {s.for_rank: s.host for s in placement.spares if s.for_rank is not None}
    for s in placement.spares:
        if s.for_rank is None:
            for r in range(world):
                if nodes[r] != s.node:
                    spare_hosts.setdefault(r, s.host)
    hosts.update({replacement(r): h for r, h in spare_hosts.items()})
    return _bootstrap(
        policy, engine, node, nodes, hosts, sorted(spare_hosts), "", event_dir, hang_s
    )


def _bootstrap(
    policy: Policy,
    engine: str,
    node: str,
    nodes: tuple[str, ...],
    hosts: Mapping[MemberId, str],
    spare_ranks: list[int],
    discovery_dir: str,
    event_dir: str,
    hang_s: float,
) -> Bootstrap:
    """The bootstrap for a placement however it was found: the sets, the layout, the ports."""
    spec = policy.engine(engine)
    sets: tuple[MemberSet, ...] = ()
    if not policy.cover_planned:  # under a cover plan the plans name every group (0035)
        sets = tuple(sets_to_preform(policy, spec, spare_ranks))
        check_formable(sets, spec.world_size, spare_ranks)
    return Bootstrap(
        engine=engine,
        node=node,
        nodes=nodes,
        discovery_dir=discovery_dir,
        world=spec.world_size,
        sets=sets,
        layout=ParallelLayout(spec.tensor_parallel, spec.pipeline_parallel),
        hosts=hosts,
        base_port=rendezvous_base_port(engine),
        event_dir=event_dir,
        hang_s=hang_s,
        release_bound_s=(
            spec.release_bound_ms / 1000.0 if spec.release_bound_ms is not None else RELEASE_BOUND_S
        ),
        cover_planned=policy.cover_planned,
        cover_form_timeout_s=policy.cover_form_timeout_ms / 1000.0,
    )


def point_tp_group_at(group: Any) -> None:
    """Point vLLM's TP group at ``group``'s communicator (sampler all-gather, eager calls).

    The live group object is kept; only its device communicator's NCCL handle changes, so
    nothing is re-initialised.
    """
    from vllm.distributed import parallel_state as ps

    from everserve_vllm.groups import UncapturedComm

    comm = getattr(group, "group", None)
    dc = getattr(ps._TP, "device_communicator", None)
    if comm is not None and dc is not None:
        dc.pynccl_comm = UncapturedComm(comm)


def point_pp_group_at(group: Any) -> None:
    """Point vLLM's PP group at ``group``'s communicator: the pipeline send and receive.

    As :func:`point_tp_group_at`: the group object is kept, its device communicator's NCCL
    handle changes, and the send and receive, which run outside the captured graphs, read the
    handle at every call.
    """
    from vllm.distributed import parallel_state as ps

    from everserve_vllm.groups import UncapturedComm

    comm = getattr(group, "group", None)
    dc = getattr(getattr(ps, "_PP", None), "device_communicator", None)
    if comm is not None and dc is not None:
        dc.pynccl_comm = UncapturedComm(comm)


def guard_engine_groups(worker: Any = None) -> list[str]:
    """Put :class:`~everserve_vllm.groups.UncapturedComm` on every NCCL handle vLLM holds.

    Before vLLM captures its graphs (the executor calls it on every worker ahead of
    ``compile_or_warm_up_model``, a spare ahead of its own warm-up): a capture that reaches NCCL
    then fails the start instead of leaving a graph that would hang an abort. Returns the groups
    guarded. ``worker`` is the RPC's worker, unused.
    """
    from vllm.distributed import parallel_state as ps

    from everserve_vllm.groups import RefusedComm, UncapturedComm

    guarded = []
    for name in ("_TP", "_PP", "_EP", "_DP", "_WORLD"):
        dc = getattr(getattr(ps, name, None), "device_communicator", None)
        pn = getattr(dc, "pynccl_comm", None)
        if pn is None or isinstance(pn, RefusedComm | UncapturedComm):
            continue
        dc.pynccl_comm = UncapturedComm(pn)
        guarded.append(name)
    log.info("NCCL handles guarded against a graph's capture: %s", ", ".join(guarded) or "none")
    return guarded


def take_rows(
    layout: ParallelLayout, first_rank: int, slot_rows: list[int], took: Iterable[int]
) -> list[int]:
    """The executor's record of its node's order after a change; returns the rows that left.

    ``took`` are the ranks the change took. ``slot_rows`` is a row per column of the stage whose
    first rank is ``first_rank``, updated in place: a taken position of this stage is taken by a
    spare on this node (decision 0022), whose row, the one past the columns, takes the position
    in the sum (decision 0012). Each member writes its own row's order at the boundary; the
    executor keeps this record to know which rows left, and reports it. Positions of other
    stages are other nodes' regions'.
    """
    left = []
    for r in took:
        if layout.stage(r) == layout.stage(first_rank):
            left.append(slot_rows[layout.column(r)])
            slot_rows[layout.column(r)] = layout.tensor_parallel
    return left


def mark_steps(worker: Any) -> None:
    """Keep this process's step word in the slot region while it runs a step RPC.

    The word is this node's part of the stall gate (decision 0021): set when a step RPC is
    received, cleared and counted when it is answered, so a node without the engine knows a step
    is in flight, and a member that dies inside a step leaves the step in flight until the change
    clears its word. ``worker`` is the RPC target vLLM's worker loop dispatches to; the step
    methods are wrapped on it. Without a slot region there is no word to keep.
    """
    assert STATE is not None
    slot = STATE.collective.slot
    if slot is None or not hasattr(slot, "region"):
        return
    region, row = slot.region, slot.row
    for name in STEP_METHODS:
        method = getattr(worker, name)

        def marked(*args: Any, _method: Any = method, **kwargs: Any) -> Any:
            region.begin_step(row)
            try:
                return _method(*args, **kwargs)
            finally:
                region.end_step(row)

        setattr(worker, name, marked)


def _compiled_under(model: Any) -> tuple[str, str]:
    """The (cache_dir, local_cache_dir) the compiled module under ``model`` keeps, else empty."""
    modules = [model]
    walk = getattr(model, "modules", None)
    if callable(walk):
        try:
            modules += list(walk())
        except Exception:  # noqa: BLE001 - a module tree that cannot be walked holds nothing
            pass
    for m in modules:
        cc = getattr(m, "compilation_config", None)
        cache_dir = getattr(cc, "cache_dir", "") or ""
        local = getattr(cc, "local_cache_dir", None) or getattr(m, "_aot_cache_dir", None) or ""
        if cache_dir or local:
            return str(cache_dir), str(local)
    return "", ""


def _artifact_of(cache_dir: str, local: str, model: Any) -> str:
    """The AOT artifact directory a process runs with, whichever path compiled it.

    A process that loaded the artifact names it on ``local_cache_dir``; one that compiled
    fresh has ``cache_dir`` set and ``local_cache_dir`` pointing into the key's directory,
    while the artifact it saved is named on the compiled module (``_aot_cache_dir``). One
    engine, one artifact, so the key is its hash on every rank: a rank that compiled fresh and
    reported the key directory's name instead would be held against the artifact's hash the
    other ranks report.
    """
    for candidate in (local, getattr(model, "_aot_cache_dir", None) if model is not None else None):
        if candidate and "torch_aot_compile" in str(candidate):
            return _artifact_root(str(candidate))
    if model is not None:
        for m in [model] + (
            list(model.modules()) if callable(getattr(model, "modules", None)) else []
        ):
            c = getattr(m, "_aot_cache_dir", None)
            if c and "torch_aot_compile" in str(c):
                return _artifact_root(str(c))
    return ""


def _artifact_root(path: str) -> str:
    """The artifact's hash directory, ``…/torch_aot_compile/<hash>``, for a path at or inside it.

    vLLM 0.31 saves each rank's share under the hash, ``torch_aot_compile/<hash>/rank_<r>_<dp>``,
    and names that rank directory as the artifact a process saved or loaded. The hash is the
    engine's one key (decision 0029: every stage shares it, as stock vLLM's stages do), so the
    key is the hash directory's name, never the rank's.
    """
    parts = Path(path).parts
    if "torch_aot_compile" not in parts:
        return path
    i = parts.index("torch_aot_compile")
    return str(Path(*parts[: i + 2])) if i + 1 < len(parts) else path


def _rank_pair(artifact: str, rank: int, publish: bool) -> CompileCache:
    """Rank ``rank``'s pair; with ``publish``, its manifest written beside it (decision 0043).

    The serving process of a position publishes: its node's store serves exactly the manifest's
    files to the node of the position's spare, so the process shares them with the node's group
    first (vLLM wrote them for itself alone) and writes the manifest last. A set that cannot be
    published leaves this node no holder of the rank's set, which a spare waiting on it reports;
    the pair stands.
    """
    if not publish:
        return rank_digest(artifact, rank)
    m = rank_manifest(artifact, rank)
    try:
        share(artifact, m)
        write_manifest(artifact, m)
    except OSError as e:
        log.warning("compile cache: rank %d's manifest not written under %s: %s", rank, artifact, e)
    return m.pair


def compiled_with(
    vllm_config: Any, model: Any = None, rank: int | None = None, publish: bool = False
) -> CompileCache | None:
    """The (key, digest) of the compile cache this process loaded (decision 0029, A40).

    vLLM keeps the key's directory on ``compilation_config.cache_dir``
    (``torch_compile_cache/<key>``) and the AOT artifact it loaded on
    ``compilation_config.local_cache_dir``, derived at load from the configuration and the
    model's code; nothing on disk links the two. The digest is over both
    (:func:`everserve.contract.compile_cache.digest`), so two processes report one pair only when
    they run the same compiled kernels, autotune results included. None when the engine compiled
    nothing (``cache_dir`` unset) or the directory is gone.
    """
    cc = getattr(vllm_config, "compilation_config", None)
    cache_dir = getattr(cc, "cache_dir", "") or ""
    local = getattr(cc, "local_cache_dir", None) or ""
    if not cache_dir and not local and model is not None:
        # the compiled module keeps the configuration it compiled under and the artifact it
        # loaded (decorators.py: ``compilation_config``, ``_aot_cache_dir``); vLLM decorates
        # the backbone, a submodule of the model the runner holds, and a process whose
        # registering config is another object than the backbone's (a spare's) reads them off
        # the module tree
        cache_dir, local = _compiled_under(model)
    if not cache_dir and not local:
        log.warning(
            "compile cache: no cache_dir or local_cache_dir on the configuration or the model"
        )
    artifact = _artifact_of(cache_dir, local, model)
    try:
        if artifact:
            # the artifact's hash is the engine's one key on every rank; the digest the rank's
            # share of it
            if rank is not None and (Path(artifact) / f"rank_{rank}_0").is_dir():
                pair = _rank_pair(artifact, rank, publish)
            else:
                pair = artifact_digest(artifact)
        elif cache_dir:
            key_dir = Path(cache_dir)
            pair = cache_digest(key_dir.parent, key_dir.name)
        elif local:
            # vLLM's AOT path names only the artifact it loaded (decorators.py sets
            # local_cache_dir and never cache_dir): the artifact's hash is the identity, and
            # the digest is this rank's share of it (its subtree and the kernels it names), so
            # a spare's pair can equal its rank's on a node staged with every rank's share
            if rank is not None and (Path(local) / f"rank_{rank}_0").is_dir():
                pair = rank_digest(local, rank)
            else:
                pair = artifact_digest(local)
        else:
            return None
    except FileNotFoundError:
        log.warning(
            "compile cache %s is not on disk; registering without a pair", cache_dir or local
        )
        return None
    return CompileCache(key=pair.key, digest=pair.digest)


def held_identity(vllm_config: Any) -> SoftwareIdentity:
    """The identity of this process as it holds its position: its KV format read off its caches."""
    from everserve_vllm import connector
    from everserve_vllm.identity import software_identity

    conn = connector.WORKER
    return software_identity(vllm_config, conn.kv_caches if conn is not None else None)


COVER: Cover | None = None
"""This process's cover (decision 0035), once it is held to its engine's plans: its engine's, or
for a pool spare the engine whose position it took (:func:`adopt_cover`)."""
COVERS: dict[str, Cover] = {}
"""Every engine's cover this process holds: one, or a pool spare's one per engine (0031)."""
_EARLY_PLANS: dict[str, CoverPlan] = {}
_COVER_LOCK = threading.Lock()
_AFTER_COMMIT: list[Callable[[], None]] = []
"""Run once at the next commit's boundary: a pool spare's withdrawal from its other engines."""
_CHANGE_LOST: tuple[int, ...] = ()
"""The ranks the change in flight lost: the commit renames their takers."""


def cover_plan(plan: CoverPlan) -> None:
    """A plan its daemon forwarded: applied by the cover, or kept until the cover exists.

    The daemon sends the newest plan right after a registration, which can come before the
    process has built its cover.
    """
    with _COVER_LOCK:
        cover = COVERS.get(plan.engine)
        if cover is None:
            early = _EARLY_PLANS.get(plan.engine)
            if early is None or plan.generation > early.generation:
                _EARLY_PLANS[plan.engine] = plan
            return
    cover.apply(plan_from(plan))


def start_cover(
    c: VllmCollective,
    cfg: Bootstrap,
    link: DaemonConnection,
    device: torch.device,
    node: str = "",
) -> Cover:
    """Hold this process to its engine's cover plans; a plan already received is applied."""
    global COVER
    from everserve_vllm.groups import form_planned

    stream = torch.cuda.Stream(device=device)

    def mine(g: Any) -> bool:
        return any(m in g.spans for m in c.stands_for)

    def form(g: Any, generation: int) -> Any:
        me = next(m for m in c.stands_for if m in g.spans)
        return form_planned(
            g, me, device, stream, link, cfg.engine, generation, cfg.cover_form_timeout_s
        )

    def release(scope: str, spans: MemberSet, group: Any) -> None:
        refuse = getattr(group, "refuse", None)
        c.releases.start(f"cover {scope} {set_key(spans)}", refuse() if refuse else None)

    # the run's record of what each plan did here, by this process's device
    record = (
        Path(cfg.event_dir) / f"cover_{cfg.engine}_{node or cfg.node}_{device.index or 0}.json"
        if cfg.event_dir
        else None
    )

    def stuck(name: str, run_s: float) -> None:
        # a formation past its deadline holds a thread as a leaked release does: one count, one
        # report (decision 0025, amended)
        c.releases.leaked += 1
        link.report_leak(name, run_s, c.releases.leaked)

    cover = Cover(
        mine,
        c,
        form,
        release,
        LinkReports(cfg.engine, lambda: c.me, link),
        record=record,
        stuck_after_s=cfg.cover_form_timeout_s,
        on_stuck=stuck,
    )
    with _COVER_LOCK:
        COVERS[cfg.engine] = cover
        if COVER is None:
            COVER = cover
        early = _EARLY_PLANS.pop(cfg.engine, None)
    if early is not None:
        cover.apply(plan_from(early))
    return cover


def adopt_cover(engine: str) -> None:
    """A pool spare took a position of ``engine``: its cover is the process's from now."""
    global COVER
    with _COVER_LOCK:
        COVER = COVERS[engine]


def on_next_commit(run: Callable[[], None]) -> None:
    """Run ``run`` once, at the boundary of the next commit this process applies."""
    _AFTER_COMMIT.append(run)


def withdraw(engine: str, c: VllmCollective, link: Any) -> int:
    """A pool spare leaves ``engine``'s pool: its registration closed, its cover and groups let go.

    The coordinator reads the closed registration as a withdrawal, not a loss, since this
    process holds another engine's position at a committed epoch (decision 0031). Returns how
    many groups were released.
    """
    link.close()
    with _COVER_LOCK:
        cover = COVERS.pop(engine, None)
        _EARLY_PLANS.pop(engine, None)
    if cover is not None:
        cover.close()
    # a group several sets need is held under each: released once
    groups = {id(g): (scope, members, g) for (scope, members), g in c.held().items()}
    for scope, members, group in groups.values():
        refuse = getattr(group, "refuse", None)
        c.releases.start(f"withdrawn {scope} {set_key(members)}", refuse() if refuse else None)
    log.info("withdrew from %s: %d groups released", engine, len(groups))
    return len(groups)


def after_commit(c: VllmCollective, lost: Sequence[int]) -> int:
    """At the commit's boundary: takers renamed, stale groups released, formation resumed.

    Returns how many stale groups were set aside for release.
    """
    stale = c.rename_at_commit(lost)
    still = {id(g) for g in c.held().values()}
    released: set[int] = set()
    for scope, members, group in stale:
        # a group several sets need goes with the last of them: a stale set's may serve another
        if id(group) in still or id(group) in released:
            continue
        released.add(id(group))
        if COVER is not None:
            COVER.forget(group)
        refuse = getattr(group, "refuse", None)
        c.releases.start(f"stale {scope} {set_key(members)}", refuse() if refuse else None)
    if COVER is not None:
        # a plan made for the placement this commit ended forms nothing
        COVER.resume(c.epoch)
    return len(stale)


def bootstrap(worker: Any, cfg: Bootstrap) -> dict[str, Any]:
    """RPC target on every serving worker; see :func:`start`."""
    w = worker.worker
    model = getattr(getattr(w, "model_runner", None), "model", None)
    out = start(
        w.rank,
        w.device,
        cfg,
        compile_cache=compiled_with(w.vllm_config, model, w.rank, publish=True),
        identity=held_identity(w.vllm_config),
    )
    assert STATE is not None
    STATE.collective.on_apply = point_tp_group_at
    STATE.collective.on_apply_fabric = point_pp_group_at
    mark_steps(worker)
    pipeline.set_runner(w.model_runner)
    pipeline.hook_worker(w)
    return out


def apply_change() -> dict[str, Any]:
    """At a step boundary: apply the runtime's pending change and move the collective."""
    assert STATE is not None
    t0 = time.time()
    torch.cuda.synchronize()
    applied = STATE.runtime.at_boundary()
    moved = STATE.collective.apply_staged()
    if COVER is not None and (applied or moved):
        after_commit(STATE.collective, _CHANGE_LOST)
    if applied or moved:
        while _AFTER_COMMIT:
            _AFTER_COMMIT.pop(0)()
    STATE.runtime.dump()
    firststep.arm()  # what the next steps spend their time on, for the run's report
    return {
        "rank": STATE.rank,
        "applied": applied,
        "moved": moved,
        "released": list(STATE.collective.releases.done),
        "releasing": STATE.collective.releases.running(),
        "leaked": STATE.collective.releases.leaked,
        "boundary_s": round(time.time() - t0, 4),
        # when this process entered the boundary and when each release of the old groups
        # started and returned, on this node's clock: the join's ring_ready leg reads ~0.25 s
        # exactly on the joins where the old fabric group's abort returns inside the boundary's
        # window, and these say whether the survivor reached the boundary late or the executor
        # waited on something else
        "t_entered": t0,
        "release_times": STATE.collective.releases.times(),
    }


def resume_requests(worker: Any, requests: list[Any]) -> dict[str, Any]:
    """RPC target on every worker after a change: re-add each running request from ``requests``.

    The model runner drops whatever state it held for the request (the survivors' state ran
    into the aborted step; the spare has none) and adds it back with the whole token list, the
    blocks, and the committed count as its computed length (decision 0013). Every rank then
    holds the same state, and a one-token step from it runs as a decode on the captured graph.
    """
    from types import SimpleNamespace

    t0 = time.time()
    runner = worker.worker.model_runner
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=requests))
    shaped = decode_shaped(runner, requests)
    return {
        "rank": worker.worker.rank,
        "requests": len(requests),
        "computed": {r.req_id: r.num_computed_tokens for r in requests},
        "decode_shaped": shaped,
        "resume_s": round(time.time() - t0, 4),
    }


def decode_shaped(runner: Any, requests: list[Any]) -> list[str]:
    """Make the step after the resume a decode for every request past its prompt (A11c).

    The runner re-adds a request from its whole token list, so its prefill length is that list's
    length, one past the committed count, and the first step back runs as a one-token chunked
    prefill: the same graph bucket as a decode, a different attention path for a length-1 query
    flagged prefill, and different rounding in the K/V row written at the committed position,
    on every rank. The control computes that position as a decode, and that row's seam alone
    flips a session's tokens after a resume whose restored state matches the source's row for
    row.

    Here, for a request whose committed count covers its prompt, the runner's prefill length is
    set to the committed count (so nothing is prefilling) and the token at the committed
    position, which a decode step reads from the runner's last-sampled buffer rather than the
    token list, is placed there; the token list and the committed count stay as the control's.
    Returns the ids shaped; a runner without model runner v2's request state keeps vLLM's shape.
    """
    states = getattr(runner, "req_states", None)
    index = getattr(states, "req_id_to_index", None)
    prefill_len = getattr(states, "prefill_len", None)
    last = getattr(states, "last_sampled_tokens", None)
    if index is None or prefill_len is None or last is None:
        log.info("resume keeps vLLM's step shape: no v2 request state on this runner")
        return []
    shaped = []
    for r in requests:
        computed = r.num_computed_tokens
        tokens = r.prefill_token_ids
        if tokens is None or computed < r.prompt_len or computed >= len(tokens):
            continue
        idx = index[r.req_id]
        prefill_len.np[idx] = computed
        last[idx, 0] = int(tokens[computed])
        shaped.append(r.req_id)
    if shaped:
        # the add's staged writes already copied the prefill lengths to the device-visible
        # buffer the step reads (apply_staged_writes); the step's input token for the resumed
        # position comes from the last-sampled buffer only where the device-side length says
        # the position is past the prefill, so the host array is copied again, the same call
        prefill_len.copy_to_uva()
    return shaped


def _worker_proc() -> Any:
    """The ``WorkerProc`` executing the current RPC (``_execute_worker_rpc``).

    The nearest caller frame whose ``self`` owns the broadcast ring, however deep the RPC target.
    """
    frame = sys._getframe(1)
    while frame is not None:
        me = frame.f_locals.get("self")
        if me is not None and hasattr(me, "rpc_broadcast_mq") and hasattr(me, "worker"):
            return me
        frame = frame.f_back
    raise RuntimeError("not inside a WorkerProc RPC")


def boundary(worker: Any, mq_handle: Any, tag: str) -> dict[str, Any]:
    """RPC target on survivors, sent over the old ring: apply the change, attach to the new ring."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    proc = _worker_proc()
    out = apply_change()
    t0 = time.time()
    proc.rpc_broadcast_mq = MessageQueue.create_from_handle(mq_handle, worker.worker.rank)
    proc.rpc_broadcast_mq.wait_until_ready()
    return {"tag": tag, **out, "ring_s": round(time.time() - t0, 4)}


def boundary_except(
    worker: Any, mq_handle: Any, tag: str, leaving: int | Sequence[int]
) -> dict[str, Any] | None:
    """:func:`boundary` on every rank but the ``leaving`` ones.

    A rank that leaves on notice is still alive and reads the old ring; it must neither apply
    nor attach. It drains its in-flight commits to the store instead and answers, so the executor
    can let the spare catch up on them. A node's ranks leave together (decision 0022). After the
    drain it releases itself (:func:`everserve_vllm.vacate.release_position`): its daemon link
    closed and its row marked, which its node's executor reads and ends the process on.
    """
    leaving = (leaving,) if isinstance(leaving, int) else tuple(leaving)
    if worker.worker.rank in leaving:
        # the leaving rank stops stepping here; its last commits must be in the store before the
        # spare's catch-up restore reads them
        t0 = time.time()
        assert STATE is not None
        STATE.kv.drain()
        drained_s = round(time.time() - t0, 4)
        slot = STATE.collective.slot
        # only where this node's executor ends released processes: the node role's
        released = slot is not None and hasattr(slot, "region") and slot.region.ends_released()
        if released:
            from everserve_vllm.vacate import release_position

            release_position(slot.region, slot.row, STATE.link)
        return {
            "tag": tag,
            "rank": worker.worker.rank,
            "drained_s": drained_s,
            "released": released,
        }
    return boundary(worker, mq_handle, tag)


def stop_watch(worker: Any) -> None:
    """RPC target at shutdown: stop the slot watch before the executor stops the workers."""
    if STATE is not None and STATE.watch is not None:
        STATE.watch.stop()


def stats(worker: Any) -> dict[str, Any]:
    """RPC target: this worker's commit times and runtime events."""
    from everserve_vllm import connector

    conn = connector.WORKER
    kv = conn.lifecycle.kv if conn is not None else None
    slot = STATE.collective.slot if STATE is not None else None
    releases = None
    if STATE is not None:
        STATE.collective.releases.reap(STATE.collective.release_bound_s)
        releases = {
            "done": list(STATE.collective.releases.done),
            "running": STATE.collective.releases.running(),
            "leaked": STATE.collective.releases.leaked,
        }
    return {
        "rank": worker.worker.rank,
        "releases": releases,
        "exchange_ops": dict(pipeline.EXCHANGE.stats),
        "commit": conn.lifecycle.report() if conn is not None else None,
        "paths": {
            # which kernels this run measured; a failed build falls back with one log line
            "gather": getattr(kv, "gather_path", None),
            "slot": slot.variant if slot is not None else None,
        },
        "events": [e.kind for e in STATE.runtime.events] if STATE is not None else [],
        "slot_watch": (
            {"reported": STATE.watch.reported, "slow": STATE.watch.slow}
            if STATE is not None and STATE.watch is not None
            else None
        ),
        "first_steps": firststep.report(),
    }
