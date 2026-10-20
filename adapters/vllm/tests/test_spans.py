# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The claim-to-commit breakdown from the coordinator's moments."""

from types import SimpleNamespace

from everserve_vllm.spans import claim_to_commit


def test_the_four_moments_give_three_spans_and_the_way_to_the_executor():
    us = 1_700_000_000_000_000
    committed = SimpleNamespace(
        reported_at_us=us,
        delivered_at_us=us + 20_000,
        acknowledged_at_us=us + 120_000,
        committed_at_us=us + 125_000,
    )
    out = claim_to_commit(committed, heard_at=1_700_000_000.130)
    assert out is not None
    assert out["deliver_s"] == 0.02
    assert out["acknowledgements_s"] == 0.1
    assert out["commit_s"] == 0.005
    assert out["heard_after_commit_s"] == 0.005
    assert out["coordinator"]["reported_at"] == 1_700_000_000.0


def test_a_commit_without_stamps_gives_no_breakdown():
    assert claim_to_commit(SimpleNamespace(), heard_at=1.0) is None
    old = SimpleNamespace(
        reported_at_us=0, delivered_at_us=0, acknowledged_at_us=0, committed_at_us=0
    )
    assert claim_to_commit(old, heard_at=1.0) is None
