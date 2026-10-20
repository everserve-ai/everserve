# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from google.protobuf import json_format

from everserve.comm.shadow import is_replacement, replacement
from everserve.policy import Loading, Outcome, choose_outcome, from_message, load, sets_to_preform
from everserve.proto import policy_pb2

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "policy-pilot.json"


def test_pilot_policy_loads():
    p = load(EXAMPLE)
    assert p.version == "pilot-1"
    assert p.outcomes_in_order == (Outcome.REPLACE, Outcome.REBUILD)
    assert p.spares_per_engine == 1 and p.spare_loading is Loading.PRELOADED
    e = p.engine("k3-a")
    assert e.world_size == 16 and e.members[8].node == "n2"
    with pytest.raises(KeyError):
        p.engine("nope")


def test_choose_outcome_takes_the_first_whose_preconditions_hold():
    p = load(EXAMPLE)
    assert choose_outcome(p, spare_ready=True, placement_independent_model=False) is Outcome.REPLACE
    assert (
        choose_outcome(p, spare_ready=False, placement_independent_model=False) is Outcome.REBUILD
    )
    shrink_first = p.__class__(
        **{**p.__dict__, "outcomes_in_order": (Outcome.SHRINK, Outcome.REPLACE)}
    )
    assert (
        choose_outcome(shrink_first, spare_ready=True, placement_independent_model=True)
        is Outcome.SHRINK
    )
    assert (
        choose_outcome(shrink_first, spare_ready=True, placement_independent_model=False)
        is Outcome.REPLACE
    )
    assert (
        choose_outcome(shrink_first, spare_ready=False, placement_independent_model=False)
        is Outcome.REBUILD
    )


def test_sets_to_preform_follow_the_allowed_outcomes_and_the_launched_spares():
    p = load(EXAMPLE)
    e = p.engine("k3-a")
    # no spare launched: nothing to pre-form (no SHRINK in the pilot policy)
    assert sets_to_preform(p, e) == []
    # one spare for rank 3: its replacement set and no other; a set for a spare that does not
    # exist could never form
    sets = sets_to_preform(p, e, spare_ranks=[3])
    assert len(sets) == 1 and len(sets[0]) == 16
    assert sets[0][3] == replacement(3)
    assert sum(is_replacement(m) for m in sets[0]) == 1
    # a node's four spares: four sets, in rank order, whatever order they were given in
    sets = sets_to_preform(p, e, spare_ranks=[9, 8, 11, 10, 8])
    ids = [next(m for m in s if is_replacement(m)) for s in sets]
    assert ids == [replacement(r) for r in (8, 9, 10, 11)]
    with pytest.raises(ValueError, match="outside"):
        sets_to_preform(p, e, spare_ranks=[16])
    with_shrink = p.__class__(**{**p.__dict__, "outcomes_in_order": (Outcome.SHRINK,)})
    assert len(sets_to_preform(with_shrink, e)) == 16  # survivor sets only
    assert all(len(s) == 15 for s in sets_to_preform(with_shrink, e))
    with pytest.raises(ValueError, match="without REPLACE"):
        sets_to_preform(with_shrink, e, spare_ranks=[0])


def test_invalid_documents_are_rejected():
    msg = json_format.Parse(
        '{"version":"x","spares":{"loading":"LOADING_PRELOADED"}}', policy_pb2.Policy()
    )
    with pytest.raises(ValueError, match="no outcome"):
        from_message(msg)
    msg = json_format.Parse(
        '{"version":"x","outcomesInOrder":["OUTCOME_REBUILD"]}', policy_pb2.Policy()
    )
    with pytest.raises(ValueError, match="loading"):
        from_message(msg)


def test_an_engine_s_spare_nodes_are_in_the_view():
    two = load(EXAMPLE.with_name("policy-two-node.json"))
    assert two.engines[0].spare_nodes == ("node-c",)
    assert two.engines[0].pipeline_parallel == 2 and two.engines[0].world_size == 8
    assert load(EXAMPLE).engines[0].spare_nodes == ()


def test_an_engine_is_exact_unless_the_policy_says_continuous(tmp_path):
    """Decision 0030: the guarantee class is the engine's, exact by default."""
    import json

    from everserve.policy import Guarantee

    resident = EXAMPLE.with_name("policy-two-node-resident.json")
    assert load(resident).engines[0].guarantee is Guarantee.EXACT
    doc = json.loads(resident.read_text())
    doc["engines"][0]["guarantee"] = "GUARANTEE_CONTINUOUS"
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(doc))
    assert load(p).engines[0].guarantee is Guarantee.CONTINUOUS


def test_replica_nodes_are_named_or_default_to_the_siblings():
    """Decision 0030: the per-position shape names its spare node; the resident shape names
    none, and the coordinator targets each rank's siblings."""
    assert load(EXAMPLE.with_name("policy-two-node.json")).engines[0].replica_nodes == ("node-c",)
    assert load(EXAMPLE.with_name("policy-two-node-resident.json")).engines[0].replica_nodes == ()


def test_the_daemons_spares_and_their_launch_are_in_the_view():
    managed = load(EXAMPLE.with_name("policy-two-node-managed.json"))
    e = managed.engines[0]
    assert e.spare_ranks == (4, 5, 6, 7) and e.spare_node(5) == "node-c" and e.spare_node(1) is None
    assert e.spare_launch is not None and e.spare_launch.command[0] == "sh"
    assert e.spare_launch.env["EVERSERVE_SPARE_RANK"] == "{rank}"
    assert not e.fabric_collectives
    assert load(EXAMPLE).engines[0].spare_launch is None  # the engine launches its own


def test_the_release_bound_is_the_engines_or_none(tmp_path):
    """Decision 0025's bound is the operator's: in the policy when given, None otherwise."""
    import json

    managed = EXAMPLE.with_name("policy-two-node-managed.json")
    base = json.loads(managed.read_text())
    assert load(managed).engines[0].release_bound_ms == 10000
    del base["engines"][0]["releaseBoundMs"]
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(base))
    assert load(p).engines[0].release_bound_ms is None


def _engine(cover=True, **engine):
    import json

    doc = {
        "version": "v",
        "outcomesInOrder": ["OUTCOME_REPLACE"],
        "spares": {"loading": "LOADING_PRELOADED"},
        "engines": [{"name": "k", **engine}],
        "cover": {"enabled": cover},
    }
    return from_message(json_format.Parse(json.dumps(doc), policy_pb2.Policy())).engines[0]


def test_an_engine_s_world_is_its_shape_and_its_members_an_optional_pin():
    """Decision 0036: the policy states the shape; the placement is status unless pinned."""
    e = _engine(tensorParallel=8, pipelineParallel=2, dataParallel=2, headOnDemand=True)
    assert (e.world_size, e.members, e.head_on_demand) == (32, (), True)
    one = _engine(tensorParallel=4)  # an unset width counts as one
    assert (one.pipeline_parallel, one.data_parallel, one.world_size) == (1, 1, 4)
    assert not one.head_on_demand
    pinned = [{"node": "a", "device": d} for d in range(4)]
    assert len(_engine(tensorParallel=2, pipelineParallel=2, members=pinned).members) == 4
    with pytest.raises(ValueError, match="pins 4 members for a world of 8"):
        _engine(tensorParallel=8, members=pinned)


def test_an_engine_its_processes_place_is_replaced_only_under_cover():
    """Its spares stand after bring-up: only the sets cover plans as they stand replace a rank."""
    with pytest.raises(ValueError, match="pins no members.*enable cover"):
        _engine(cover=False, tensorParallel=8)
    pinned = [{"node": "a", "device": d} for d in range(8)]
    assert _engine(cover=False, tensorParallel=8, members=pinned).world_size == 8


def test_a_cover_formation_timeout_is_the_policy_s_or_thirty_seconds():
    """The coordinator and every member read the same deadline for a planned group (0035)."""
    base = (
        '{"version":"x","outcomesInOrder":["OUTCOME_REPLACE"],'
        '"spares":{"loading":"LOADING_PRELOADED"},"cover":{"enabled":true%s}}'
    )
    unset = json_format.Parse(base % "", policy_pb2.Policy())
    given = json_format.Parse(base % ',"formTimeoutMs":12000', policy_pb2.Policy())
    assert from_message(unset).cover_form_timeout_ms == 30_000
    assert from_message(given).cover_form_timeout_ms == 12_000
