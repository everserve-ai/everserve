# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
import pytest

from everserve.contract.signals import (
    Change,
    CompileCache,
    DumpStacks,
    EnginePlacement,
    EnginePolicy,
    InstancePhase,
    InstanceState,
    Join,
    KvFormat,
    Member,
    MembershipCommitted,
    Notice,
    ParallelWidths,
    RankJoining,
    RankLost,
    RankPlacement,
    Reason,
    Register,
    Role,
    RoleAssigned,
    SlotChoice,
    SoftwareIdentity,
    SparePlacement,
    Take,
    from_message,
    to_message,
)
from everserve.outcome import Outcome
from everserve.proto import signals_pb2


def test_membership_rejects_rank_outside_world():
    with pytest.raises(ValueError):
        MembershipCommitted(epoch=1, active_ranks=(0, 1, 4), world_size=4)


def test_membership_rejects_rank_both_active_and_lost():
    with pytest.raises(ValueError):
        MembershipCommitted(epoch=1, active_ranks=(0, 1), world_size=2, lost=(1,))


def test_membership_accepts_a_shrink():
    m = MembershipCommitted(epoch=2, active_ranks=(0, 1, 2), world_size=4, lost=(3,), pause_ms=2560)
    assert m.active_ranks == (0, 1, 2)
    assert m.lost == (3,)


def test_defaults():
    assert Notice(rank=3, deadline_ms=1_000).reason is Reason.NOTICE
    assert RankLost(ranks=(3,)).reason is Reason.FAULT


@pytest.mark.parametrize(
    "sig",
    [
        Notice(rank=3, deadline_ms=1_700_000_000_000, reason=Reason.NOTICE),
        RankLost(ranks=(2,), reason=Reason.FAULT, detail="sentinel"),
        RankLost(ranks=(2,), reason=Reason.OPERATOR),
        RankLost(ranks=(2,), detail="slot row 0: no words from rank 2", epoch=3),
        RankJoining(rank=2, member=Member(node="n2", device=5)),
        MembershipCommitted(
            epoch=3, active_ranks=(0, 1, 3), world_size=4, pause_ms=2560, lost=(2,)
        ),
        Change(lost=(2,), outcome=Outcome.REPLACE, epoch=4),
        Register(engine="k3-a", rank=5, member=Member(node="n2", device=5)),
        Register(engine="k3-a", rank=2, member=Member(node="n1", device=4), role=Role.SPARE),
        Register(engine="k3-a", rank=0, member=Member(node="n1", device=0), role=Role.CLIENT),
        Register(
            engine="k3-a",
            rank=0,
            member=Member(node="n1", device=0),
            role=Role.CLIENT,
            slot=SlotChoice("host", "probe", "no answer from devices [2] in 15.0s", 15.0),
        ),
        Register(
            engine="k3-a",
            rank=1,
            member=Member(node="n2", device=1),
            compile_cache=CompileCache(key="760834052d", digest="3ae869f0"),
        ),
        Register(
            engine="k3-a",
            rank=0,
            member=Member(node="n3", device=2),
            role=Role.SPARE,
            any_rank=True,
        ),
        Register(
            engine="k3-a",
            rank=1,
            member=Member(node="n2", device=1),
            identity=SoftwareIdentity(
                engine="vllm",
                engine_commit="6e517b15c",
                adapter_commit="a9c4391" + "0" * 33,
                torch_version="2.9.0",
                cuda_version="12.9",
                driver_version="580.65.06",
                nccl_version="2.27.3",
                model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                model_revision="f50dbad2",
                checkpoint_digest="ab" * 32,
                quantization="compressed-tensors",
                model_dtype="bfloat16",
                kv_format=KvFormat("bfloat16", 16, "cd" * 32),
                widths=ParallelWidths(tensor=1, pipeline=2),
                serves_changes=True,
            ),
        ),
        Register(  # a spare for any position: no layout hash until it takes one
            engine="k3-a",
            rank=0,
            member=Member(node="n3", device=2),
            role=Role.SPARE,
            any_rank=True,
            identity=SoftwareIdentity(
                "vllm", "6e517b15c", "a9c4391", adapter_dirty=True, kv_format=KvFormat("fp8", 16)
            ),
        ),
        Change(lost=(1,), outcome=Outcome.REPLACE, epoch=2, take=(Take(1, "n3", 2),)),
        Change(
            lost=(1,),
            outcome=Outcome.REPLACE,
            epoch=2,
            take=(Take(1, "n3", 2, CompileCache(key="760834052d", digest="e7b25807")),),
        ),
        # protocol 9: the ranks leaving on notice, and the whole committed membership
        Change(lost=(1, 2), outcome=Outcome.REPLACE, epoch=3, departing=(1,)),
        MembershipCommitted(
            epoch=3,
            active_ranks=(0, 1, 2),
            world_size=3,
            members=(
                RankPlacement(0, "node-a", 0),
                RankPlacement(1, "node-d", 0),
                RankPlacement(2, "node-e", 1),
            ),
        ),
        # protocol 11: an undecided process asks for its stage and is told; a stack dump is asked
        # on the link
        Register(
            engine="k3-a",
            rank=8,
            member=Member(node="gpu-3", device=0),
            role=Role.UNDECIDED,
            stage=(8, 9, 10, 11, 12, 13, 14, 15),
        ),
        RoleAssigned(engine="k3-a", rank=8, epoch=0, role=Role.WORKER, stage=(8, 9, 10, 11)),
        RoleAssigned(engine="k3-a", rank=8, epoch=4, role=Role.SPARE, stage=(8, 9)),
        DumpStacks(reason="the step at epoch 4 outran its bound"),
        EnginePolicy(engine="k3-a", policy=b"\n\x02v1"),
        InstanceState("k3-a", InstancePhase.DEGRADED, 14, 16, 9, "rank 3 lost under SHRINK"),
        # the placement at bring-up: a rank placed only by its assignment has no device yet, and
        # a spare stands for one rank or for any
        EnginePlacement(
            engine="k3-a",
            epoch=0,
            members=(
                RankPlacement(0, "gpu-0", 0, "10.0.0.1"),
                RankPlacement(1, "gpu-1", None, "10.0.0.2"),
            ),
            spares=(
                SparePlacement(1, "gpu-2", 0, "10.0.0.3"),
                SparePlacement(None, "gpu-3", 1, "10.0.0.4"),
            ),
        ),
        Register(
            engine="k3-a",
            rank=4,
            member=Member(node="gpu-1", device=0),
            role=Role.SPARE,
            host="10.0.0.2",
        ),
    ],
)
def test_round_trip_through_the_wire_schema(sig):
    wire = to_message(sig).SerializeToString()
    back = from_message(signals_pb2.Signal.FromString(wire))
    assert back == sig


def test_empty_or_reasonless_messages_are_rejected():
    with pytest.raises(ValueError):
        from_message(signals_pb2.Signal())
    with pytest.raises(ValueError):
        from_message(signals_pb2.Signal(rank_lost=signals_pb2.RankLost(ranks=(1,))))
    with pytest.raises(ValueError):
        from_message(signals_pb2.Signal(change=signals_pb2.Change(lost=[1], epoch=1)))


def test_a_join_round_trips_with_its_bytes():
    sig = Join(engine="e1", epoch=3, rank=4, payload=bytes(range(256)))
    assert from_message(to_message(sig)) == sig


def test_a_register_carries_the_oldest_protocol_its_sender_serves():
    """Decision 0040: a link opens when each side serves the other, so a hello says the range."""
    from everserve.contract.signals import PROTOCOL_OLDEST, PROTOCOL_VERSION

    reg = Register("e1", 2, Member("n1", 0), Role.WORKER)
    assert (reg.protocol, reg.protocol_oldest) == (PROTOCOL_VERSION, PROTOCOL_OLDEST)
    msg = to_message(reg)
    assert msg.register.protocol_oldest == PROTOCOL_OLDEST and from_message(msg) == reg
    older = signals_pb2.Signal()
    older.CopyFrom(msg)
    older.register.ClearField("protocol_oldest")  # a build before 0040 sends none
    assert from_message(older).protocol_oldest == 0


def test_a_rank_s_device_absent_is_none_never_device_0():
    msg = signals_pb2.Signal(
        engine_placement=signals_pb2.EnginePlacement(
            engine="k",
            members=[
                signals_pb2.RankPlacement(rank=0, node="a", device=0, host="h"),
                signals_pb2.RankPlacement(rank=1, node="b", host="h"),
            ],
        )
    )
    placed = from_message(signals_pb2.Signal.FromString(msg.SerializeToString()))
    assert [p.device for p in placed.members] == [0, None]


def test_a_kind_the_codec_does_not_decode_is_told_apart_from_an_empty_signal():
    from everserve.contract.signals import UnknownSignal

    with pytest.raises(UnknownSignal):
        from_message(signals_pb2.Signal(ack=signals_pb2.Ack(rank=1, epoch=2)))
    with pytest.raises(ValueError, match="empty signal") as e:
        from_message(signals_pb2.Signal())
    assert not isinstance(e.value, UnknownSignal)
