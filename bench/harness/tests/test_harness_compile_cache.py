# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The compile cache prediction (decision 0029) as one figure, its failing check named; and the
reads that name an autotune result by its path, so the compiled driver helper a take rebuilds
(55,832 bytes in the compile cache) does not fail it."""

import json

import written_since
from summary import compile_cache, engine_down, engine_status

B = {"caches": [{"key": "13aba33f7a", "digest": "59ee"}, {"key": "57f7aebbdc", "digest": "a303"}]}
HELPER = (
    "/home/everserve/.cache/vllm/torch_compile_cache/x/inductor_cache/triton/0/MIH4X/cuda_utils.so"
)


def ready(consistent, pair=("02376ca3", "b8d71a31"), rank=1):
    """An engine's status: its consistency and ``rank``'s registered pair."""
    members = (
        []
        if pair is None
        else [{"rank": rank, "compileCache": dict(zip(("key", "digest"), pair, strict=True))}]
    )
    return {"compileCacheConsistent": consistent, "members": members}


def test_the_cache_held_when_every_check_reads_true():
    """The spare registered its position's pair, and the whole-node digests differ by the Triton
    helper cache_sync built in place, which is reported and not scored."""
    written = written_since.summarise([(HELPER, 55_832)], 4, 40)
    with_helper = {"caches": [*B["caches"], {"key": "13aba33f7a", "digest": "helper"}]}
    f = compile_cache(ready(True), ready(True), [1], written, B, with_helper)
    assert f["value"] is True and f["failing"] == [] and f["staging_digests_equal"] is False
    assert f["pairs"]["1"]["position"] == f["pairs"]["1"]["spare"]
    assert written["named"] == [{"path": HELPER, "bytes": 55_832}] and written["autotune"] == []


def test_each_failing_check_is_named():
    tuned = written_since.summarise(
        [
            (
                "/home/everserve/.cache/flashinfer/flashinfer_autotune_cache/autotune_configs.json",
                10,
            ),
            ("/home/everserve/.triton/cache/AB/kernel.autotune.json", 20),
            ("/home/everserve/.cache/vllm/x/inductor_cache/ab/kernel.best_config", 30),
        ],
        4,
        40,
    )
    assert len(tuned["autotune"]) == 3
    f = compile_cache(ready(False), ready(True, ("02376ca3", "ffff")), [1], tuned)
    assert f["value"] is False
    assert f["failing"] == ["readiness_consistent", "spare_holds_position_s", "no_autotune_written"]


def test_a_check_with_nothing_to_read_is_left_out_and_none_read_is_no_figure():
    # a read between the claim and the commit: no rank 1 to compare, so not read
    f = compile_cache(ready(True), ready(True, None), [1], {"files": 0})
    assert f["value"] is True and f["checks"]["spare_holds_position_s"] is None
    assert compile_cache(None, None, [], None) is None
    assert (
        compile_cache(ready(True), ready(True), [], None)["checks"]["spare_holds_position_s"]
        is None
    )


def test_the_engine_is_down_when_a_read_in_the_trial_says_so_and_not_at_the_drill_s_end():
    """Decision 0032: the head's orderly goodbye at the drill's end reads down, stopped."""
    up, stopped = {"down": None}, {"down": {"cause": "stopped", "detail": ""}}
    lost = {"down": {"cause": "lost", "detail": "connection ended without a goodbye"}}
    failed = {"down": {"cause": "change_failed", "detail": "rank 0 departed"}}
    assert engine_down(up, up, stopped)["value"] is False
    assert engine_down(up, lost, lost)["counted"] == ["after", "during"]
    assert engine_down(up, up, failed)["value"] is True
    assert engine_down(stopped, up, up)["counted"] == ["before"]
    assert engine_down(None, None, None) is None


def test_a_saved_engine_read_is_read_for_its_status(tmp_path):
    saved = tmp_path / "readiness_before.json"
    saved.write_text(json.dumps({"metadata": {"name": "rig"}, "spec": {}, "status": ready(True)}))
    assert engine_status(str(saved)) == ready(True)
    assert engine_status(str(tmp_path / "absent.json")) is None
    assert engine_status("") is None
