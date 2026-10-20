# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""An undecided process's link against a fake daemon: it asks for its stage, is told, and stays;
any link writes its stacks when the daemon asks."""

import threading
import time

import pytest
from link_doubles import FakeDaemon

from everserve.contract.daemon import DaemonClient, Refused, UndecidedConnection
from everserve.contract.signals import (
    Abandoned,
    DumpStacks,
    EnginePlacement,
    EnginePolicy,
    Goodbye,
    InstancePhase,
    InstanceState,
    Member,
    RankPlacement,
    Register,
    Role,
    RoleAssigned,
    SoftwareIdentity,
    StepConfig,
    StopCause,
    from_message,
    to_message,
)
from everserve.proto import signals_pb2 as pb
from everserve.wire import read_frame, write_frame


def asking(daemon, stage):
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("reg", daemon.accept()))
    t.start()
    link = UndecidedConnection(daemon.path, "k", stage, Member("gpu-3", 0), connect_timeout_s=5)
    t.join(5)
    return link, got["reg"]


def test_an_undecided_process_asks_for_its_stage_is_told_and_stays_silent():
    daemon = FakeDaemon()
    link, reg = asking(daemon, (8, 9, 10, 11))
    assert isinstance(reg, Register)
    assert (reg.role, reg.rank, reg.stage) == (Role.UNDECIDED, 8, (8, 9, 10, 11))
    assert reg.member == Member("gpu-3", 0)
    with pytest.raises(TimeoutError):
        link.assigned(timeout_s=0.3)  # nothing until the daemon answers
    daemon.send(RoleAssigned("k", 8, 0, Role.WORKER, (8, 9, 10, 11)))
    told = link.assigned(timeout_s=5)
    assert (told.role, told.epoch, told.stage) == (Role.WORKER, 0, (8, 9, 10, 11))
    assert told.refusal == ""  # a worker, not a refusal read as one (its role defaults to WORKER)
    assert not link.closed.is_set()  # the connection is the stage's liveness
    link.close()
    assert daemon.ended()  # and it said nothing on it


UNKNOWN = b"\xf8\x03\x01"  # a Signal whose only field (63) is of a kind added after this build


def test_a_frame_of_a_kind_this_build_does_not_know_is_passed_over_on_every_link():
    daemon = FakeDaemon()
    link, _ = asking(daemon, (4,))
    write_frame(daemon.conn, UNKNOWN)  # before the answer, as a newer daemon's first frame
    daemon.send(RoleAssigned("k", 4, 0, Role.WORKER, (4,)))
    assert link.assigned(timeout_s=5).role is Role.WORKER
    assert not link.closed.is_set()
    link.close()

    client_daemon = FakeDaemon()
    t = threading.Thread(target=client_daemon.accept)
    t.start()
    client = DaemonClient(client_daemon.path, "k", Member("n", 0), connect_timeout_s=5)
    t.join(5)
    write_frame(client_daemon.conn, UNKNOWN)
    client_daemon.send(Abandoned(epoch=3, detail="after"))
    assert client.abandoned.get(timeout=5).epoch == 3  # the link read on past it
    client.close()


def test_a_change_with_an_outcome_this_build_does_not_know_is_passed_over():
    """An outcome added after this build (99, say) reaching a process that predates it ends
    neither the change's decoding as an empty one nor the link: the reader passes it over and
    reads on."""
    from everserve.contract.signals import UnknownSignal, from_message
    from everserve.proto import signals_pb2 as pb

    head = pb.Signal(change=pb.Change(outcome=99, epoch=2))
    with pytest.raises(UnknownSignal):
        from_message(pb.Signal.FromString(head.SerializeToString()))
    with pytest.raises(ValueError, match="change without an outcome"):
        from_message(pb.Signal(change=pb.Change(epoch=2)))  # unset: a malformed change

    daemon = FakeDaemon()
    t = threading.Thread(target=daemon.accept)
    t.start()
    client = DaemonClient(daemon.path, "k", Member("n", 0), connect_timeout_s=5)
    t.join(5)
    write_frame(daemon.conn, head.SerializeToString())
    daemon.send(Abandoned(epoch=3, detail="after"))
    assert client.abandoned.get(timeout=5).epoch == 3  # the link read on past it
    client.close()


def test_a_connection_ended_before_an_answer_is_not_an_answer():
    daemon = FakeDaemon()
    link, _ = asking(daemon, (4,))
    daemon.conn.close()
    with pytest.raises(ConnectionError):
        link.assigned(timeout_s=5)


def test_an_undecided_registration_names_a_rank():
    with pytest.raises(ValueError):
        UndecidedConnection("/nonexistent", "k", (), Member("gpu-3", 0))


def test_every_link_writes_its_stacks_to_its_own_log_when_the_daemon_asks(capfd):
    daemon = FakeDaemon()
    link, _ = asking(daemon, (4, 5))
    daemon.send(DumpStacks(reason="asked by test"))
    daemon.send(RoleAssigned("k", 4, 1, Role.SPARE, (4, 5)))
    assert link.assigned(timeout_s=5).role is Role.SPARE  # read after the dump: it is written
    err = capfd.readouterr().err
    assert "_listen" in err  # the link's own thread, in the dump
    link.close()

    client_daemon = FakeDaemon()
    t = threading.Thread(target=client_daemon.accept)
    t.start()
    client = DaemonClient(client_daemon.path, "k", Member("n", 0), connect_timeout_s=5)
    t.join(5)
    client_daemon.send(DumpStacks(reason="the head's turn"))
    client_daemon.conn.close()
    assert client.closed.wait(5)
    assert "_listen" in capfd.readouterr().err


def policy(version, guarantee="GUARANTEE_EXACT"):
    import json

    from google.protobuf import json_format

    from everserve.proto import policy_pb2

    doc = {
        "version": version,
        "outcomesInOrder": ["OUTCOME_REPLACE"],
        "spares": {"loading": "LOADING_PRELOADED"},
        "cover": {"enabled": True},
        "engines": [{"name": "k", "tensorParallel": 8, "guarantee": guarantee}],
    }
    return EnginePolicy(
        "k", json_format.Parse(json.dumps(doc), policy_pb2.Policy()).SerializeToString()
    )


def test_the_engine_s_policy_comes_on_the_link_before_the_answer_and_the_last_is_kept():
    daemon = FakeDaemon()
    link, _ = asking(daemon, (8,))
    assert link.policy is None
    daemon.send(policy("v1"))
    daemon.send(RoleAssigned("k", 8, 0, Role.WORKER, (8,)))
    assert link.engine_policy(timeout_s=5).version == "v1"
    assert link.assigned(timeout_s=5).role is Role.WORKER
    daemon.send(policy("v2", "GUARANTEE_CONTINUOUS"))  # changed, or the daemon relinked
    daemon.send(EnginePolicy("k", b"\xff\xff not a policy"))  # refused: the last is kept
    daemon.send(DumpStacks())  # read after both: they were taken
    deadline = time.monotonic() + 5
    while link.policy.version != "v2" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert link.policy.version == "v2"
    assert link.policy.engine("k").world_size == 8
    link.close()


def test_a_link_that_ends_before_the_policy_never_gives_one():
    daemon = FakeDaemon()
    link, _ = asking(daemon, (8,))
    with pytest.raises(TimeoutError):
        link.engine_policy(timeout_s=0.3)
    daemon.conn.close()
    with pytest.raises(ConnectionError):
        link.engine_policy(timeout_s=5)


def test_a_client_hears_its_engine_s_policy_too():
    daemon = FakeDaemon()
    t = threading.Thread(target=daemon.accept)
    t.start()
    client = DaemonClient(daemon.path, "k", Member("n", 0), connect_timeout_s=5, head=True)
    t.join(5)
    daemon.send(policy("v3"))
    assert client.engine_policy(timeout_s=5).version == "v3"
    client.close()


def placed(epoch):
    return EnginePlacement("k", epoch, (RankPlacement(0, "gpu-0", None, "10.0.0.1"),))


def test_a_waiting_process_hears_where_the_engine_runs_and_keeps_the_latest(monkeypatch):
    monkeypatch.setenv("EVERSERVE_HOST_IP", "10.0.0.9")
    daemon = FakeDaemon()
    link, reg = asking(daemon, (0,))
    assert reg.host == "10.0.0.9"  # its address, where its groups rendezvous
    with pytest.raises(TimeoutError, match="no placement"):
        link.engine_placement(timeout_s=0.3)
    daemon.send(placed(2))
    assert link.engine_placement(timeout_s=5).epoch == 2
    daemon.send(placed(1))  # an older one, out of order: the later is kept
    daemon.send(placed(3))
    daemon.send(DumpStacks())
    deadline = time.monotonic() + 5
    while link.placement.epoch != 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert link.placement.epoch == 3
    link.close()


def test_a_launching_process_hands_its_children_the_policy_and_placement_it_heard():
    import io

    from everserve.contract.daemon import hand_over, handed_over

    sig = policy("v4")
    handed = hand_over(sig, placed(5))
    got_policy, got_placement = handed_over(io.BytesIO(handed))
    assert (got_policy.version, got_placement) == ("v4", placed(5))
    from everserve.wire import frame

    backwards = b"".join(frame(to_message(m).SerializeToString()) for m in (placed(5), sig))
    with pytest.raises(ValueError, match="not the engine's policy and placement"):
        handed_over(io.BytesIO(backwards))
    with pytest.raises(EOFError):
        handed_over(io.BytesIO(handed[:-3]))  # a launcher that died mid-write


def test_a_kind_the_schema_names_but_the_codec_does_not_decode_never_ends_a_link():
    """The instance state (protocol 11) once killed every link: its kind was in the schema."""
    daemon = FakeDaemon()
    link, _ = asking(daemon, (2, 3))
    ack = pb.Signal(ack=pb.Ack(rank=2, epoch=1)).SerializeToString()  # no link decodes an ack
    write_frame(daemon.conn, ack)
    daemon.send(InstanceState("k", InstancePhase.RECOVERING, 3, 4, 7, "rank 2 replaced"))
    daemon.send(RoleAssigned("k", 2, 0, Role.SPARE, (2, 3)))
    assert link.assigned(timeout_s=5).role is Role.SPARE  # read on past both
    assert not link.closed.is_set()
    st = link.state
    assert (st.phase, st.positions_held, st.world, st.epoch) == (InstancePhase.RECOVERING, 3, 4, 7)
    link.close()


def test_a_refused_ask_is_raised_never_read_as_a_role_and_its_goodbye_names_why():
    daemon = FakeDaemon()
    link, _ = asking(daemon, (4,))
    why = 'max_num_seqs is "128" here, "256" on the engine'
    daemon.send(RoleAssigned("k", 4, 0, Role.WORKER, (4,), refusal=why))  # WORKER: the default
    with pytest.raises(Refused) as refused:
        link.assigned(timeout_s=5)
    assert refused.value.refusal == why
    link.refused(refused.value)
    bye = from_message(pb.Signal.FromString(read_frame(daemon.conn)))
    assert bye == Goodbye(cause=StopCause.REFUSED, detail=why)
    assert daemon.ended()


def test_the_ask_carries_what_the_process_steps_with():
    daemon = FakeDaemon()
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("reg", daemon.accept()))
    t.start()
    step = StepConfig("d1g35t", {"max_num_seqs": "256", "vllm.scheduler": "b08c"})
    identity = SoftwareIdentity("vllm", "v0.31.0", "c0ffee", step=step)
    link = UndecidedConnection(
        daemon.path, "k", (4,), Member("gpu-3", 0), connect_timeout_s=5, identity=identity
    )
    t.join(5)
    assert got["reg"].identity.step == step
    link.close()


def test_a_spare_asks_for_its_rank_s_compiled_set_and_is_told_it_is_in_place_or_why_not():
    from everserve.contract.daemon import CacheUnavailable
    from everserve.contract.signals import CacheReady, CacheRefused, CacheWanted

    daemon = FakeDaemon()
    link, _ = asking(daemon, (2, 3))
    daemon.send(RoleAssigned("k", 2, 0, Role.SPARE, (2, 3)))
    assert link.assigned(timeout_s=5).role is Role.SPARE
    got = {}

    def answer(*replies):
        asked = from_message(pb.Signal.FromString(read_frame(daemon.conn)))
        got.setdefault("asks", []).append(asked)
        for r in replies:
            daemon.send(r)

    t = threading.Thread(target=answer, args=(CacheReady("k", 2, "e" * 64, "d2"),))
    t.start()
    ready = link.cache(2, timeout_s=5)
    t.join(5)
    assert got["asks"] == [CacheWanted("k", 2, 0)]
    assert (ready.key, ready.digest) == ("e" * 64, "d2")
    # an answer for another rank than the one asked is not this one's
    stray = CacheReady("k", 7, "e" * 64, "d7")
    t = threading.Thread(target=answer, args=(stray, CacheRefused("k", 3, "no holder in 600 s")))
    t.start()
    with pytest.raises(CacheUnavailable, match="no holder"):
        link.cache(3, timeout_s=5)
    t.join(5)
    link.close()
