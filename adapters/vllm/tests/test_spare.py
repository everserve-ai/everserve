# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The spare's move table."""

from types import SimpleNamespace
from typing import Any

from everserve_vllm.spare import moves


def test_the_spare_takes_every_unit_of_its_rank_into_the_same_head_slot():
    assert moves(2, 1) == [((2, 0), 0, 0)]
    assert moves(1, 3) == [((1, 0), 0, 0), ((1, 1), 1, 1), ((1, 2), 2, 2)]
    assert moves(0, 0) == []


def test_a_tp_group_of_one_leaves_the_spares_stub_alone(monkeypatch):
    """A LocalStandIn has no communicator: the stub's local collectives stay as they are."""
    import sys
    import types

    from everserve.comm.handle import LocalStandIn
    from everserve_vllm.spare import _point_stub_at

    before = object()
    ps = types.ModuleType("parallel_state")
    ps._TP = before
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = ps
    vllm = types.ModuleType("vllm")
    vllm.distributed = distributed
    stand_ins = {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": ps,
    }
    for name, mod in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _point_stub_at(LocalStandIn())
    assert ps._TP is before


def test_the_spares_stub_groups_follow_the_layout(monkeypatch):
    """TP1 x PP2, rank 1: the TP group is one rank (its row), the PP group its column at stage 1."""
    import sys
    import types

    from everserve_vllm import spare

    ps = types.ModuleType("parallel_state")

    class GroupCoordinator:
        pass

    ps.GroupCoordinator = GroupCoordinator
    ps._groups = {}
    ps._PP = None
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = ps
    vllm = types.ModuleType("vllm")
    vllm.distributed = distributed
    stand_ins = {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": ps,
    }
    for name, mod in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(spare, "_take_serving_tactics", lambda: None)
    monkeypatch.setattr(spare, "_warm_pipeline_stage", lambda: None)
    from everserve_vllm import settings

    monkeypatch.setattr(settings, "_CURRENT", settings.Settings(layout=(1, 2)))
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    spare._install_stubs(config, rank=1, local_rank=0)
    assert (ps._TP.ranks, ps._TP.world_size, ps._TP.rank_in_group) == ([1], 1, 0)
    assert (ps._PP.ranks, ps._PP.world_size, ps._PP.rank_in_group) == ([0, 1], 2, 1)
    assert ps._WORLD.world_size == 2 and ps._EP.ranks == [1]


def _fake_vllm(monkeypatch, stage, stages=2):
    """``parallel_state`` with a PP group at ``stage`` and ``gpu_worker`` carrying the two warm-up
    entry points, recording what they ran under."""
    import sys
    import types

    from everserve_vllm import pipeline

    ran = []

    def kernel_warmup(worker):
        ran.append(("kernel_warmup", pipeline._LOCAL is worker.model_runner))

    def warmup_kernels(model_runner, execute_model, sample_tokens):
        execute_model("so")
        ran.append(("warmup_kernels", pipeline._LOCAL is model_runner, pipeline._SCHEDULER_OUTPUT))

    gw = types.ModuleType("gpu_worker")
    gw.kernel_warmup, gw.warmup_kernels = kernel_warmup, warmup_kernels
    ps = types.ModuleType("parallel_state")
    ps.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=stage == 0, rank_in_group=stage, world_size=stages
    )
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.parallel_state": ps,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": gw,
    }
    mods["vllm"].distributed = mods["vllm.distributed"]
    mods["vllm.distributed"].parallel_state = ps
    mods["vllm.v1"].worker = mods["vllm.v1.worker"]
    mods["vllm.v1.worker"].gpu_worker = gw
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return gw, ran


def test_past_the_first_stage_the_warm_ups_run_on_the_stage_alone_with_the_step_kept(monkeypatch):
    from everserve_vllm import pipeline, spare

    gw, ran = _fake_vllm(monkeypatch, stage=1)
    spare._warm_pipeline_stage()
    runner = object()
    seen = []
    gw.kernel_warmup(SimpleNamespace(model_runner=runner))
    gw.warmup_kernels(runner, lambda so: seen.append((so, pipeline._SCHEDULER_OUTPUT)), None)
    assert ran == [("kernel_warmup", True), ("warmup_kernels", True, None)]
    assert seen == [("so", "so")]  # the step's scheduler output kept for the receive's layout
    assert pipeline._LOCAL is None and pipeline.RUNNER is None


def test_on_the_first_stage_of_a_pipeline_the_warm_ups_run_alone_too(monkeypatch):
    """Its steps send to the stage after, which a spare has no peer for: the send is dropped."""
    from everserve_vllm import pipeline, spare

    gw, ran = _fake_vllm(monkeypatch, stage=0)
    spare._warm_pipeline_stage()
    runner = object()
    gw.warmup_kernels(runner, lambda so: None, None)
    assert ran == [("warmup_kernels", True, None)] and pipeline._LOCAL is None


def test_an_engine_of_one_stage_warms_up_as_vllm_runs_it(monkeypatch):
    from everserve_vllm import pipeline, spare

    gw, ran = _fake_vllm(monkeypatch, stage=0, stages=1)
    spare._warm_pipeline_stage()
    gw.warmup_kernels("runner", lambda so: None, None)
    assert ran == [("warmup_kernels", False, None)] and pipeline._LOCAL is None


def _calls(world: int = 2) -> Any:
    """A core's start-up record holding only the configurations, one per rank of ``world``."""
    from everserve_vllm import replay

    configs = [SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=1024)])] * world
    return replay.Calls((("initialize_from_config", (configs,), {}),))


def test_a_spare_registers_for_the_position_it_attached_to_and_no_other(monkeypatch):
    """The warm-up is a property of the attachment (engine, rank): ``attach`` warms for it and
    ``register`` registers for exactly that, refusing another engine's bootstrap."""
    import pytest

    from everserve_vllm import replay, spare, worker

    monkeypatch.setattr(spare, "_warm", lambda w: (1.25, 3.5))
    monkeypatch.setattr(replay, "free_bytes", lambda: 1 << 30)
    _fake_vllm(monkeypatch, stage=1)
    config = SimpleNamespace(cache_config=SimpleNamespace())
    inner = SimpleNamespace(
        device="cuda:0", vllm_config=config, model_runner=SimpleNamespace(model="model")
    )
    proc = SimpleNamespace(
        worker=SimpleNamespace(initialize_from_config=lambda c: None, worker=inner)
    )
    attachment = spare.attach(proc, "rig", 1, _calls())
    assert attachment.engine == "rig" and attachment.rank == 1 and attachment.stage == 1
    assert (attachment.sizing_s, attachment.warm_s) == (1.25, 3.5)
    assert set(attachment.kv_config_parts) == {"initialize_from_config_s"}
    started = []
    monkeypatch.setattr(
        worker, "start", lambda rank, device, cfg, **kw: started.append((rank, device, cfg, kw))
    )
    monkeypatch.setattr(worker, "compiled_with", lambda config, model, rank: (config, model, rank))
    monkeypatch.setattr(worker, "held_identity", lambda config: ("identity", config))
    with pytest.raises(RuntimeError, match="does not register for 'other'"):
        spare.register(attachment, proc, SimpleNamespace(engine="other"))
    assert started == []
    spare.register(attachment, proc, SimpleNamespace(engine="rig"), slot_choice="slot")
    ((rank, device, cfg, kw),) = started
    assert (rank, device, cfg.engine) == (1, "cuda:0", "rig")
    assert kw["role"].name == "SPARE" and kw["runtime_cls"] is spare.SpareRuntime
    assert kw["slot_choice"] == "slot" and kw["compile_cache"] == (config, "model", 1)
    assert kw["identity"] == ("identity", config)  # what it runs, its KV format read off its caches


def test_a_resident_spare_builds_the_position_s_model_from_the_copy_at_the_attachment(monkeypatch):
    """With a shard the attachment points the stand-in groups at the rank, builds the model
    (vLLM's ``load_model``, timed as ``shard_copy_s``), then reads the KV configuration off the
    built worker and warms as any attachment does; without one nothing is built."""
    from everserve_vllm import replay, spare

    monkeypatch.setattr(spare, "_warm", lambda w: (0.5, 2.0))
    monkeypatch.setattr(replay, "free_bytes", lambda: 1 << 30)
    _fake_vllm(monkeypatch, stage=1)
    pointed, built, configured = [], [], []
    monkeypatch.setattr(
        spare, "point_groups", lambda config, rank, local_rank: pointed.append((rank, local_rank))
    )
    inner = SimpleNamespace(
        device="cuda:0",
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace()),
        local_rank=0,
        rank=0,
    )
    proc = SimpleNamespace(
        rank=0,
        worker=SimpleNamespace(
            initialize_from_config=configured.append,
            load_model=lambda: built.append(len(pointed)),
            worker=inner,
        ),
    )
    calls = _calls(world=4)
    attachment = spare.attach(proc, "rig", 3, calls, shard="the copy")
    assert pointed == [(3, 0)] and built == [1]  # pointed at the rank, then built
    # the core's list, as its workers got it: the wrapper takes its own rank's from it
    assert (proc.rank, inner.rank) == (3, 3) and configured == [calls.calls[-1][1][0]]
    assert attachment.engine == "rig" and attachment.rank == 3 and attachment.stage == 1
    assert attachment.shard_copy_s >= 0 and (attachment.sizing_s, attachment.warm_s) == (0.5, 2.0)
    plain = spare.attach(proc, "rig", 1, calls)
    assert built == [1] and len(configured) == 2 and plain.shard_copy_s == 0.0


def test_a_resident_spare_s_worker_is_built_with_no_model_and_the_loader_set(monkeypatch):
    """``load(resident=True)`` registers the resident loader as the configuration's load format
    and builds the worker with vLLM's ``load_model`` a no-op for the build only; a preloaded
    spare's worker loads its model as it is built."""
    import sys
    import types

    from everserve_vllm import resident, spare

    built, loaded, installed = [], [], []

    class Worker:
        def load_model(self, **kw):
            loaded.append(kw)

    class WorkerProc:
        def __init__(self, vllm_config, local_rank, rank, **kw):
            self.worker = Worker()
            built.append((rank, local_rank, sorted(kw)))
            self.worker.load_model()  # as vLLM's __init__ does after init_device
            self._init_message_queues(None, None)

    gw = types.ModuleType("gpu_worker")
    gw.Worker, gw.init_worker_distributed_environment = Worker, "vllm's"
    mp = types.ModuleType("multiproc_executor")
    mp.WorkerProc = WorkerProc
    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": gw,
        "vllm.v1.executor": types.ModuleType("vllm.v1.executor"),
        "vllm.v1.executor.multiproc_executor": mp,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(resident, "install", lambda: installed.append(True))
    cfg = SimpleNamespace(load_config=SimpleNamespace(load_format="auto"))
    proc = spare.load(cfg, 1, "lock", local_rank=0, resident=True)
    assert built == [
        (
            1,
            0,
            [
                "distributed_init_method",
                "input_shm_handle",
                "is_driver_worker",
                "shared_worker_lock",
            ],
        )
    ]
    assert loaded == [] and installed == [True]  # nothing loaded at the build
    assert cfg.load_config.load_format == resident.LOAD_FORMAT
    assert proc.rpc_broadcast_mq is None and proc.worker_response_mq is None
    assert gw.init_worker_distributed_environment == "vllm's"
    proc.worker.load_model()  # the attachment's call, after the build: vLLM's own
    assert loaded == [{}]
    cfg2 = SimpleNamespace(load_config=SimpleNamespace(load_format="auto"))
    spare.load(cfg2, 0, "lock")
    assert loaded == [{}, {}] and cfg2.load_config.load_format == "auto"


def test_the_restore_waits_until_the_rank_s_rows_are_held_and_caught_up_at_this_node(monkeypatch):
    """A shared spare's node gets the lost rank's rows staged onto it before the change; the
    restore reads them once the store holds them and their carrier is caught up, and a wait past
    the bound fails the change naming the state seen (the acknowledgement carries it)."""
    import pytest

    from everserve_vllm import spare, worker

    states = iter([(False, 0, 0, False), (True, 1, 40, False), (True, 1, 41, True)])
    calls = []

    class KV:
        layers = [("layer", None)]
        restored_counts = {"r": 7}
        restored_rows = 12

        def state(self, rank):
            calls.append(("state", rank))
            return next(states)

        def restore(self, dead, moves, since_step=0):
            calls.append(("restore", dead, len(moves), since_step))
            return 41

    kv = KV()
    staged = spare.wait_staged(kv, 1, timeout_s=5.0, poll_s=0.0)
    assert staged["epoch"] == 1 and staged["step"] == 41 and staged["polls"] == 3
    assert calls == [("state", 1)] * 3
    never = type("Never", (), {"state": lambda self, rank: (True, 2, 9, False)})()
    with pytest.raises(TimeoutError, match="held=True epoch=2 step=9 caught_up=False"):
        spare.wait_staged(never, 3, timeout_s=0.01, poll_s=0.0)

    calls.clear()
    monkeypatch.setattr(worker, "STATE", SimpleNamespace(kv=kv))
    monkeypatch.setattr(worker, "head_slots", lambda layers: 4)
    monkeypatch.setattr(spare.AdapterRuntime, "on_change", lambda self, *a, **kw: None)
    rt = spare.SpareRuntime.__new__(spare.SpareRuntime)
    rt.orig_rank, rt.epoch = (
        1,
        1,
    )  # the epoch the change applied (AdapterRuntime.on_change, stubbed)
    kv.state = lambda rank: (True, 1, 41, True)
    rt.on_change([1], outcome=spare.Outcome.REPLACE)
    assert calls == [("restore", 1, 4, 0)]
    info = rt.restore_info
    assert info["staged"]["step"] == 41 and info["watermark"] == 41 and info["rows"] == 12
    assert info["counts"] == {"r": 7} and info["heads"] == 4 and info["dead"] == 1


def test_a_spare_that_took_its_position_at_an_abandoned_change_stops(monkeypatch):
    """Decision 0032: the membership it took a position in will never commit; its daemon
    relaunches it standing. A spare still standing, or one that took at another epoch, stays."""
    from everserve_vllm import spare

    sent = []
    monkeypatch.setattr(spare.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    rt = spare.SpareRuntime.__new__(spare.SpareRuntime)  # bare: no collective, no event log
    rt.trace, rt._event_log = [], None
    rt.on_abandoned(3, "rank 0 departed")  # standing: took nothing
    rt.took_epoch = 2
    rt.on_abandoned(3, "rank 0 departed")  # took at another epoch
    assert sent == []
    rt.took_epoch = 3
    rt.on_abandoned(3, "rank 0 departed")
    assert sent == [(spare.os.getpid(), spare.signal.SIGTERM)]


def test_the_warm_up_s_parts_count_each_second_once():
    """The graph capture's own dummy runs are the capture's, so the parts sum to the warm-up
    (3.4 to 5.3 s a take) and no second is counted twice."""
    import time

    from everserve_vllm import spare

    spare.WARM_PARTS.clear()
    dummy = spare._timing("compile_sizes_s", lambda: time.sleep(0.02))
    capture = spare._timing("capture_s", lambda: (dummy(), dummy(), time.sleep(0.02)))
    dummy()
    capture()
    assert set(spare.WARM_PARTS) == {"compile_sizes_s", "capture_s"}
    assert 0.02 <= spare.WARM_PARTS["compile_sizes_s"] < 0.04  # the one outside the capture
    assert spare.WARM_PARTS["capture_s"] >= 0.06  # its own two dummy runs included
