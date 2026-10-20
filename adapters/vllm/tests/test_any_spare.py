# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A spare that stands for any position (decision 0030): which ranks it stands for, its standing
runtime before and after a take, the daemon link handing it the take before the change, the
collective taking one identity, and the hand-off from the link's thread to the main thread."""

import threading
from types import SimpleNamespace

import pytest

from everserve.comm.collective import Collective
from everserve.comm.handle import LocalStandIn
from everserve.comm.shadow import replacement
from everserve.contract.signals import Role, Take


def _engine(spares):
    members = [SimpleNamespace(node="node-a"), SimpleNamespace(node="node-b")]
    return SimpleNamespace(members=members, spares=spares, tensor_parallel=1, pipeline_parallel=2)


def test_a_rankless_slot_stands_for_every_rank_not_on_its_node():
    from everserve_vllm import worker

    any_c = SimpleNamespace(node="node-c", device=0, rank=None)
    assert worker.eligible_ranks(_engine([any_c]), "node-c") == [0, 1]
    assert worker.eligible_ranks(_engine([any_c]), "node-b") == [0]
    assert worker.spare_nodes_of(_engine([any_c])) == {0: "node-c", 1: "node-c"}
    # a rank with a slot of its own keeps it
    own = SimpleNamespace(node="node-d", device=0, rank=1)
    assert worker.spare_nodes_of(_engine([any_c, own])) == {0: "node-c", 1: "node-d"}
    assert worker.spare_nodes_of(_engine([own])) == {1: "node-d"}


def test_the_standing_runtime_takes_the_position_named_for_its_device_and_then_is_that_rank():
    from everserve_vllm import worker

    taken = []
    target = SimpleNamespace(
        orig_rank=1, identity=lambda: (1, 4), on_change=lambda lost, **kw: taken.append(lost)
    )

    def take(t):
        assert standing.identity() == (t.rank, 0)  # the ack names the rank being taken
        taken.append(t.rank)
        return target

    standing = worker.Standing("node-c", 0, take)
    assert standing.identity() == (0, 0)
    with pytest.raises(AttributeError, match="before it takes a position"):
        _ = standing.restore_info
    standing.on_change([1])  # a change that is not its own does nothing before a take
    standing.on_take([Take(0, "node-c", 1), Take(1, "node-d", 0)])  # someone else's take
    assert standing.target is None and taken == []
    standing.on_take([Take(1, "node-c", 0)])
    assert standing.target is target and standing.identity() == (1, 4)
    standing.on_change([1], outcome="replace")
    assert taken == [1, [1]]
    target.restore_info = {"rows": 3}
    assert standing.restore_info == {"rows": 3}  # the taken runtime answers from here on
    # a coordinator that restarted before the commit re-delivers the change it logged: the same
    # take is the one already made, and nothing is taken twice
    standing.on_take([Take(1, "node-c", 0)])
    assert standing.target is target and taken == [1, [1]]
    with pytest.raises(RuntimeError, match="already took rank 1"):
        standing.on_take([Take(0, "node-c", 0)])


def test_the_daemon_link_hands_the_take_over_first_and_acknowledges_as_the_rank_taken():
    from everserve.contract.daemon import DaemonConnection
    from everserve.contract.signals import Change, Outcome

    calls, sent = [], []

    class Runtime:
        took_epoch = None  # standing: has taken nothing yet
        rank = 0

        def on_take(self, takes):
            calls.append(("take", takes))
            self.rank = takes[0].rank

        def on_change(self, lost, **kw):
            calls.append(("change", list(lost), kw["outcome"]))

        def identity(self):
            return self.rank, 7

    link = DaemonConnection.__new__(DaemonConnection)
    link._runtime, link._role, link._staged = Runtime(), Role.SPARE, set()
    link._send = sent.append
    change = Change(epoch=7, lost=(1,), outcome=Outcome.REPLACE, take=(Take(1, "node-c", 0),))
    link._apply(change)
    assert calls == [("take", (Take(1, "node-c", 0),)), ("change", [1], Outcome.REPLACE)]
    ((ack,),) = [(s.ack,) for s in sent]
    assert (ack.rank, ack.epoch, ack.error) == (1, 7, "")
    # a runtime standing for its own position (a preloaded spare, a worker) answers the take
    # with nothing and gets the change as before
    plain = SimpleNamespace(
        took_epoch=None,
        on_take=lambda takes: calls.append(("plain-take", len(takes))),
        on_change=lambda lost, **kw: calls.append(("plain", list(lost))),
        identity=lambda: (3, 7),
    )
    link._runtime = plain
    link._apply(change)
    assert calls[-1] == ("plain", [1]) and sent[-1].ack.rank == 3


def test_the_collective_takes_one_identity_and_names_the_rank_and_row(monkeypatch):
    from everserve_vllm import collective_op, settings

    c = Collective(
        replacement(0),
        LocalStandIn(),
        {(replacement(0), 1): LocalStandIn(), (0, replacement(1)): LocalStandIn()},
        stands_for=(replacement(1),),
    )
    monkeypatch.setattr(collective_op, "ensure", lambda group_name="tp:0": c)
    monkeypatch.setattr(settings, "_CURRENT", settings.Settings(spare=True, spare_rank=0, row=9))
    assert collective_op.take(1, 0) == 1
    assert c.me == replacement(1) and c.shadow_sets == [(0, replacement(1))] and c.slot is None
    assert (settings.current().spare_rank, settings.current().row) == (1, 0)


def _stand_fakes(monkeypatch, attach):
    from everserve_vllm import collective_op, spare, spare_main, worker

    log = []
    standing = SimpleNamespace(link="the link")
    taken = {"stood": threading.Event()}

    def stand(device, cfg, take, slot_choice, identity):
        taken["take"] = take
        taken["stood"].set()
        log.append(("stand", device, slot_choice, identity))
        return standing, {"stands_for": [replacement(0), replacement(1)]}

    monkeypatch.setattr(worker, "stand", stand)
    monkeypatch.setattr(
        collective_op, "take", lambda rank, row: log.append(("ids", rank, row)) or 1
    )
    monkeypatch.setattr(spare, "attach", attach)
    monkeypatch.setattr(worker, "compiled_with", lambda config, model, rank: ("pair", rank))
    monkeypatch.setattr(
        worker,
        "bind_taken",
        lambda st, rank, device, cfg, cls: log.append(("bind", rank, cls.__name__)) or "runtime",
    )
    w = SimpleNamespace(device="cuda:0", vllm_config="cfg", model_runner=None)
    proc = SimpleNamespace(worker=SimpleNamespace(worker=w))
    spec = _engine([SimpleNamespace(node="node-c", device=0, rank=None)])

    def load(rank):  # the worker is built at the take, as the rank taken
        log.append(("load", rank))
        return proc

    def placeholder():
        log.append(("placeholder",))
        return "cuda:0"

    return spare_main, (load, placeholder), spec, log, taken


def test_a_take_runs_on_the_main_thread_while_the_link_waits_and_returns_the_rank_s_runtime(
    monkeypatch,
):
    from everserve_vllm import spare

    def attach(proc, engine, rank, kv, shard):
        log.append(("attach", rank, shard))
        return spare.Attachment(engine, rank, 1, 0.01, 2.0, 1.8, 0.1, 1.4, 0.003)

    spare_main, proc, spec, log, taken = _stand_fakes(monkeypatch, attach)
    got = {}

    def link_thread():
        if not taken["stood"].wait(5):
            return  # it never stood: the test fails on what the link got, not in a hang
        got["runtime"] = taken["take"](Take(1, "node-c", 0))  # the link's thread, in the change

    t = threading.Thread(target=link_thread, daemon=True)
    t.start()
    rank, link, pair, built, record = spare_main.stand_and_take(
        *proc,
        "rig",
        spec,
        "node-c",
        "cfg",
        "copy",
        lambda r: "calls",
        "slot",
        identity="what it runs",
    )
    t.join(5)
    assert (rank, link, pair) == (1, "the link", ("pair", 1)) and got["runtime"] == "runtime"
    assert built.worker.worker.vllm_config == "cfg"
    # the take's record rides in the join's answer into the change record
    assert record["rank"] == 1 and record["weights_s"] == 0.1 and record["kv_config_s"] == 1.4
    assert {"take_s", "identity_s", "build_s"} <= set(record)
    assert log == [
        ("placeholder",),  # standing: no worker, the groups at a placeholder rank
        ("stand", "cuda:0", "slot", "what it runs"),  # registered with its identity
        ("ids", 1, 0),  # node-c serves nothing: the rank's column
        ("load", 1),  # built at the take: a worker built as rank 0 cannot serve rank 1's stage
        ("attach", 1, "copy"),
        ("bind", 1, "SpareRuntime"),
    ]


def test_a_take_that_fails_fails_the_change_on_the_link_and_the_process_alike(monkeypatch):
    def attach(proc, engine, rank, kv, shard):
        raise MemoryError("no room for the shard")

    spare_main, proc, spec, log, taken = _stand_fakes(monkeypatch, attach)
    got = {}

    def link_thread():
        if not taken["stood"].wait(5):
            return  # it never stood: the test fails on what the link got, not in a hang
        try:
            taken["take"](Take(0, "node-c", 0))
        except RuntimeError as e:
            got["error"] = str(e)

    t = threading.Thread(target=link_thread, daemon=True)
    t.start()
    with pytest.raises(MemoryError):
        spare_main.stand_and_take(*proc, "rig", spec, "node-c", "cfg", "copy", lambda w: 0, None)
    t.join(5)
    assert got["error"] == "taking rank 0 failed: MemoryError: no room for the shard"


def test_the_slot_row_is_the_rank_s_column_on_a_spare_node_and_past_the_stage_on_an_engine_node():
    from everserve_vllm import spare_main

    spec = SimpleNamespace(
        members=[SimpleNamespace(node="node-a")] * 2 + [SimpleNamespace(node="node-b")] * 2,
        tensor_parallel=2,
        pipeline_parallel=2,
    )
    assert [spare_main.spare_row(spec, "node-c", r) for r in range(4)] == [0, 1, 0, 1]
    assert spare_main.spare_row(spec, "node-a", 3) == 2


def test_the_sets_set_aside_at_a_take_are_refused_then_released_after_it(monkeypatch):
    """Decision 0025 for the other positions' sets: refused at the take, aborted on release
    threads, reaped and listed released like the change's own groups."""
    import time

    from everserve.comm.collective import GroupFamily
    from everserve.comm.layout import Scope
    from everserve_vllm.collective_op import VllmCollective

    aborted = []

    class Group:
        refused = False

        def __init__(self, name):
            self.name = name

        def refuse(self):
            self.refused = True
            return lambda: aborted.append(self.name)

    r0, r1 = (replacement(0), 1), (0, replacement(1))
    fabric = GroupFamily(Scope.FABRIC, LocalStandIn(), {r0: Group("fabric R0"), r1: LocalStandIn()})
    c = VllmCollective(
        replacement(0),
        LocalStandIn(),
        {r0: Group("domain R0"), r1: LocalStandIn()},
        fabric=fabric,
        stands_for=(replacement(1),),
    )
    dropped = c.take_identity(replacement(1))
    assert [scope for scope, _, _ in dropped] == ["domain", "fabric"]
    assert all(g.refused for _, _, g in dropped)  # refused at the take: nothing runs on them
    assert aborted == [] and c.releases.running() == []  # released only after the take
    assert c.release_set_aside() == 2 and c.release_set_aside() == 0
    for _ in range(200):
        if not c.releases.running():
            break
        time.sleep(0.005)
    done = c.releases.reap(bound_s=5.0)
    assert sorted(aborted) == ["domain R0", "fabric R0"]
    assert sorted(r.split(" ")[0] for r in done) == ["domain", "fabric"] and c.releases.leaked == 0
    assert c.shadow_sets == [r1] and fabric.shadow_sets == [r1]


def test_a_take_whose_spare_compiled_other_kernels_than_the_position_fails_naming_both(monkeypatch):
    """Decision 0029 held at the take for a spare that stood for any position: the change's take
    carries the position's pair; a spare that attached with another fails the change."""
    from everserve.contract.signals import CompileCache
    from everserve_vllm import spare_main

    spare_main.check_pair(1, None, CompileCache("k", "d"))  # the position never reported one
    spare_main.check_pair(1, CompileCache("k", "d"), CompileCache("k", "d"))
    with pytest.raises(
        RuntimeError, match="key k digest d; this spare attached with key k digest e"
    ):
        spare_main.check_pair(1, CompileCache("k", "d"), CompileCache("k", "e"))

    from everserve_vllm import spare

    spare_main_mod, proc, spec, log, taken = _stand_fakes(
        monkeypatch,
        lambda proc, engine, rank, kv, shard: spare.Attachment(engine, rank, 1, 0.0, 0.0),
    )
    from everserve_vllm import worker

    monkeypatch.setattr(worker, "compiled_with", lambda c, m, r: CompileCache("k", "other"))
    got = {}

    def link_thread():
        if not taken["stood"].wait(5):
            return  # it never stood: the test fails on what the link got, not in a hang
        t = Take(1, "node-c", 0, compile_cache=CompileCache("k", "d"))  # the wire's own
        try:
            taken["take"](t)
        except RuntimeError as e:
            got["error"] = str(e)

    th = threading.Thread(target=link_thread, daemon=True)
    th.start()
    with pytest.raises(RuntimeError, match="compile pair"):
        spare_main_mod.stand_and_take(
            *proc, "rig", spec, "node-c", "cfg", "copy", lambda w: 0, None
        )
    th.join(5)
    assert "digest other" in got["error"] and not any(e[0] == "bind" for e in log)


def test_the_spare_waits_for_every_node_s_daemon_record_before_its_rendezvous():
    from everserve_vllm import spare_main

    published = {"node-a": "rec", "node-c": "rec"}
    asked = []

    class Discovery:
        def daemon(self, node):
            asked.append(node)
            if len(asked) > 4:
                published["node-b"] = "rec"  # node-b's daemon starts late
            return published.get(node)

    spare_main.wait_for_records(Discovery(), ["node-a", "node-b", "node-c"], poll_s=0.0)
    assert "node-b" in published and asked.count("node-b") >= 2
    stopped = []
    spare_main.wait_for_records(
        type("Never", (), {"daemon": lambda self, n: None})(),
        ["node-x"],
        stop=lambda: stopped.append(1) or len(stopped) > 2,
        poll_s=0.0,
    )
    assert len(stopped) == 3


def test_a_taken_rank_s_commits_continue_from_the_restored_watermark(monkeypatch):
    """The first commit after a take is at the watermark plus one, never step 0."""
    import sys

    import everserve_vllm
    from everserve_vllm import spare
    from everserve_vllm.lifecycle import Lifecycle

    committed = []
    life = Lifecycle()
    life.kv = SimpleNamespace(
        commit=lambda epoch, step, counts: committed.append((epoch, step)), fence=lambda s: None
    )
    life.runtime = SimpleNamespace(epoch=1)
    fake = SimpleNamespace(WORKER=SimpleNamespace(lifecycle=life))  # the connector needs vLLM
    monkeypatch.setitem(sys.modules, "everserve_vllm.connector", fake)
    monkeypatch.setattr(everserve_vllm, "connector", fake, raising=False)
    rt = SimpleNamespace(restore_info={"watermark": 1329})
    assert spare.continue_rank_steps(rt) == 1330
    life.commit()
    life.commit()
    assert committed == [(1, 1330), (1, 1331)]
    assert spare.continue_rank_steps(SimpleNamespace(restore_info=None)) is None


def test_a_change_re_delivered_at_the_epoch_the_spare_took_restores_nothing_again(monkeypatch):
    """A restarted coordinator re-delivers the change it logged before its commit; the spare that
    took the position at that epoch acknowledges again without staging or restoring twice."""
    from everserve.outcome import Outcome
    from everserve_vllm import spare
    from everserve_vllm.worker import AdapterRuntime

    applied = []
    monkeypatch.setattr(AdapterRuntime, "on_change", lambda self, *a, **kw: applied.append(a))
    rt = spare.SpareRuntime.__new__(spare.SpareRuntime)
    rt.orig_rank, rt.took_epoch = 1, 2
    rt.on_change([1], None, Outcome.REPLACE, 2, ())
    assert applied == []  # neither staged again nor restored again
