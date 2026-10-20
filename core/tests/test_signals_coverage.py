# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every field of every signal survives the hand-written codec: decoded, encoded again, equal.

The signals are the contract between the adapter (Python, these dataclasses) and the control
plane (Rust, generated from the same proto). A field the proto has and the dataclasses lack is
dropped on decode and sent as its default, which no test of either side alone can see (a head
change lost for ``SoftwareIdentity.takes_head_change``, a commit's ``head_node`` dropped the same
way). Each message here is filled in every field from the
schema itself, as Kubernetes' round-trip tests fill every field of its types
(``apimachinery/pkg/api/apitesting/roundtrip``), so a field added to the proto and not to the
codec fails here, on the first run.
"""

from __future__ import annotations

import pytest
from google.protobuf.descriptor import Descriptor, FieldDescriptor

from everserve.contract.signals import from_message, to_message
from everserve.proto import signals_pb2 as pb

SIGNAL_KINDS = [f.name for f in pb.Signal.DESCRIPTOR.oneofs_by_name["kind"].fields]

VALUES: dict[tuple[str, str], object] = {
    # a membership names ranks inside its world, which the dataclass checks
    ("MembershipCommitted", "world_size"): 4,
    ("MembershipCommitted", "active_ranks"): [0, 1],
    ("MembershipCommitted", "lost"): [3],
    ("InstanceState", "world"): 4,
}
"""Values a field must take for its message to be a valid one; any other is filled generically."""


def fill(msg, depth: int = 0) -> None:
    """Set every field of ``msg``, and every field of each message it holds, to a non-default
    value: one element of a repeated field, one entry of a map, a oneof's every member in turn
    (the last one stays, the schema allowing one)."""
    d: Descriptor = msg.DESCRIPTOR
    for f in d.fields:
        key = (d.name, f.name)
        if key in VALUES:
            value = VALUES[key]
            if f.is_repeated:
                getattr(msg, f.name).extend(value)
            else:
                setattr(msg, f.name, value)
            continue
        if f.message_type is not None and f.message_type.GetOptions().map_entry:
            entry = f.message_type
            k = _scalar(entry.fields_by_name["key"], f.name)
            v_field = entry.fields_by_name["value"]
            if v_field.message_type is not None:
                fill(getattr(msg, f.name)[k], depth + 1)
            else:
                getattr(msg, f.name)[k] = _scalar(v_field, f.name)
        elif f.message_type is not None:
            if depth > 4:
                continue
            if f.is_repeated:
                fill(getattr(msg, f.name).add(), depth + 1)
            else:
                fill(getattr(msg, f.name), depth + 1)
        elif f.is_repeated:
            getattr(msg, f.name).append(_scalar(f, f.name))
        else:
            setattr(msg, f.name, _scalar(f, f.name))


def _scalar(f: FieldDescriptor, name: str) -> object:
    """A non-default value of ``f``'s type, distinct per field name where the type allows."""
    t = f.type
    if t == FieldDescriptor.TYPE_ENUM:
        return f.enum_type.values[-1].number  # the newest value: the one a codec forgets
    if t == FieldDescriptor.TYPE_STRING:
        return f"{name}-value"
    if t == FieldDescriptor.TYPE_BYTES:
        return f"{name}-bytes".encode()
    if t == FieldDescriptor.TYPE_BOOL:
        return True
    if t in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
        return 1.5
    return 1 + len(name) % 3  # small: ranks and epochs stay inside a world of 4


NOT_DECODED = {
    "ack": "only sent, built as protobuf by DaemonConnection._apply (the daemon stamps its time)",
    "stacks_dumped": "the answer to DumpStacks, which no process builds yet (DumpStacks says so)",
}
"""The kinds no process of the adapter receives, and why; every other kind round-trips."""


@pytest.mark.parametrize("kind", [k for k in SIGNAL_KINDS if k not in NOT_DECODED])
def test_every_field_of_a_signal_survives_the_codec(kind):
    sent = pb.Signal()
    fill(getattr(sent, kind))
    got = to_message(from_message(pb.Signal.FromString(sent.SerializeToString())))
    assert got == sent, f"{kind}: the codec changed or dropped a field:\n{sent}\n!=\n{got}"


@pytest.mark.parametrize("kind", sorted(NOT_DECODED))
def test_a_kind_listed_as_not_decoded_is_not(kind):
    """The list cannot go stale: a kind given a decoder leaves it, and is held to every field."""
    from everserve.contract.signals import UnknownSignal

    sent = pb.Signal()
    fill(getattr(sent, kind))
    with pytest.raises(UnknownSignal):
        from_message(sent)


def test_a_retired_rank_s_notice_is_heard_not_the_end_of_its_link():
    """Decision 0025's retirement reaches a process as a notice of reason RETIRED; before this
    codec knew the reason, its decoding raised and ended the worker's link."""
    from everserve.contract.signals import Notice, Reason

    notice = pb.Signal(notice=pb.Notice(rank=2, deadline_ms=5, reason=pb.REASON_RETIRED))
    assert from_message(notice) == Notice(rank=2, deadline_ms=5, reason=Reason.RETIRED)


def test_every_signal_dataclass_is_a_kind_of_the_registry():
    """A dataclass added to the ``Signal`` union and not to the registry is caught here, not
    at the first ``to_message`` of one."""
    import typing

    from everserve.contract import signals

    assert {k.type for k in signals._KINDS} == set(typing.get_args(signals.Signal))
    assert {k.field for k in signals._KINDS} == set(SIGNAL_KINDS) - set(NOT_DECODED)
