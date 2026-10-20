# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Dynamo proof's verdict is the coordinator's record, not the streams finishing."""

import json
import subprocess
import sys
from pathlib import Path

VERDICT = Path(__file__).parents[1] / "k8s" / "proof_verdict.py"

# a refused change's record, cut down: the spare's set planned and not formed, the change refused,
# the head's loss uncovered (no spare), so the Covered condition False whatever rank 1's set does
PROOF_13 = {
    "status": {
        "conditions": [{"type": "Covered", "status": "False", "message": "2 losses"}],
        "cover": {
            "generation": 2,
            "epoch": 0,
            "sets": [
                {"set": "0-1", "purpose": "current", "covered": True, "formed": 1, "groups": 1},
                {
                    "set": "0-R1@node-c:0",
                    "purpose": "REPLACE of ranks [1]",
                    "covered": False,
                    "formed": 0,
                    "groups": 1,
                },
            ],
        },
        "lastChange": None,
    }
}


def run(tmp_path: Path, record: dict, check: str, rank: int = 1) -> tuple[int, str]:
    path = tmp_path / "engines-proof.json"
    path.write_text(json.dumps(record))
    p = subprocess.run(
        [sys.executable, str(VERDICT), check, str(path), str(rank)],
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout.strip()


def formed(record: dict) -> dict:
    out = json.loads(json.dumps(record))
    out["status"]["cover"]["sets"][1].update(covered=True, formed=1)
    return out


def test_a_loss_whose_set_has_not_formed_is_not_covered(tmp_path):
    assert run(tmp_path, PROOF_13, "covered") == (
        1,
        "plan 2 at epoch 0: 0-R1@node-c:0: 0 of 1 groups formed",
    )


def test_a_formed_set_covers_its_rank_while_the_heads_loss_is_uncovered(tmp_path):
    code, line = run(tmp_path, formed(PROOF_13), "covered")
    assert code == 0, line


def test_a_rank_with_no_set_planned_is_not_covered(tmp_path):
    assert run(tmp_path, formed(PROOF_13), "covered", rank=0) == (
        1,
        "plan 2 at epoch 0: no set planned for the loss of rank 0",
    )


def test_streams_that_finish_without_a_committed_change_do_not_pass(tmp_path):
    assert run(tmp_path, PROOF_13, "replaced") == (1, "no change committed")


def test_a_replace_of_the_lost_rank_passes(tmp_path):
    record = formed(PROOF_13)
    record["status"]["lastChange"] = {
        "epoch": 1,
        "outcome": "replace",
        "lost": [1],
        "cover": "through covered set 0-R1@node-c:0",
    }
    assert run(tmp_path, record, "replaced") == (
        0,
        "epoch 1: replace of ranks [1] through covered set 0-R1@node-c:0",
    )


def test_a_shrink_or_another_ranks_change_does_not_pass(tmp_path):
    record = formed(PROOF_13)
    record["status"]["lastChange"] = {"epoch": 1, "outcome": "shrink", "lost": [1], "cover": ""}
    assert run(tmp_path, record, "replaced")[0] == 1
    record["status"]["lastChange"].update(outcome="replace", lost=[0])
    assert run(tmp_path, record, "replaced")[0] == 1


def test_a_replace_at_another_epoch_than_the_one_expected_does_not_pass(tmp_path):
    """Each loss of a run commits its own epoch: the second loss's verdict is not the first's."""
    record = formed(PROOF_13)
    record["status"]["lastChange"] = {"epoch": 1, "outcome": "replace", "lost": [1], "cover": ""}
    path = tmp_path / "engines-proof.json"
    path.write_text(json.dumps(record))
    one = [sys.executable, str(VERDICT), "replaced", str(path), "1"]
    assert subprocess.run([*one, "1"], capture_output=True, check=False).returncode == 0
    p = subprocess.run([*one, "2"], capture_output=True, text=True, check=False)
    assert p.returncode == 1 and p.stdout.strip().endswith("epoch 2 expected")


def test_the_holder_of_a_rank_is_the_node_the_record_places_it_on(tmp_path):
    """After a REPLACE the rank is the spare's: the driver finds the pod to take by the record."""
    record = formed(PROOF_13)
    record["status"]["members"] = [
        {"rank": 0, "node": "node-a"},
        {"rank": 1, "node": "node-c"},
    ]
    assert run(tmp_path, record, "holder") == (0, "node-c")
    assert run(tmp_path, record, "holder", rank=2) == (1, "no member holds rank 2")


def test_the_engine_reads_covered_only_when_its_condition_is_true(tmp_path):
    """A loss on the head's node leaves Covered's list, so this engine shape reads Covered True
    once rank 1's set formed; a record that still counts the head's loss reads False."""
    record = formed(PROOF_13)
    assert run(tmp_path, record, "engine-covered")[0] == 1  # the record's own: "2 losses"
    record["status"]["conditions"] = [
        {
            "type": "Covered",
            "status": "True",
            "reason": "EveryLossCovered",
            "message": "the head's node holds [0], whose loss restarts the engine",
        }
    ]
    code, line = run(tmp_path, record, "engine-covered")
    assert code == 0 and line.startswith("Covered True (EveryLossCovered)")
    record["status"]["conditions"] = []
    assert run(tmp_path, record, "engine-covered") == (1, "no Covered condition")


def test_a_missing_record_is_a_failure_that_says_why(tmp_path):
    p = subprocess.run(
        [sys.executable, str(VERDICT), "replaced", str(tmp_path / "none.json"), "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 1 and p.stdout.startswith("no record:")


def check(tmp_path: Path, record: dict, *args: str) -> tuple[int, str]:
    path = tmp_path / "engines-proof.json"
    path.write_text(json.dumps(record))
    p = subprocess.run(
        [sys.executable, str(VERDICT), args[0], str(path), *args[1:]],
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout.strip()


def test_an_engine_serves_once_every_position_is_held_not_once_its_head_registers(tmp_path):
    """A head registers before its ranks join it: a record read then holds no member."""
    starting = {
        "status": {
            "head": {"node": "gpu-a"},
            "members": [],
            "worldSize": 0,
            "conditions": [{"type": "Serving", "status": "False", "reason": "NoMembers"}],
        }
    }
    code, line = check(tmp_path, starting, "serving")
    assert code == 1 and "0 of 0 positions held" in line
    members = [{"rank": 0, "node": "gpu-a"}, {"rank": 1, "node": "gpu-b"}]
    serving = {
        "status": {
            "head": {"node": "gpu-a"},
            "members": members,
            "worldSize": 2,
            "conditions": [{"type": "Serving", "status": "True", "reason": "Members"}],
        }
    }
    assert check(tmp_path, serving, "serving")[0] == 0
    one = json.loads(json.dumps(serving))
    one["status"]["members"] = members[:1]
    assert check(tmp_path, one, "serving")[0] == 1
    headless = json.loads(json.dumps(serving))
    headless["status"]["head"] = None
    assert check(tmp_path, headless, "serving")[0] == 1


def test_the_loss_must_fall_inside_every_stream(tmp_path):
    """A REPLACE under load leaves one mark, a gap at the loss; streams that finished before
    the node was taken, or started after, show nothing of it and do not pass."""
    load = {
        "per_stream": {
            "0": {"first": 100.0, "last": 140.0, "gap_s": 0.6, "gap_at": 120.0},
            "1": {"first": 101.0, "last": 141.0, "gap_s": 0.5, "gap_at": 120.2},
        }
    }
    path = tmp_path / "load-1.json"
    path.write_text(json.dumps(load))
    run_inside = lambda at: subprocess.run(  # noqa: E731 - a one-line helper for the cases
        [sys.executable, str(VERDICT), "inside", str(path), str(at)],
        capture_output=True,
        text=True,
        check=False,
    )
    p = run_inside(120.0)
    assert p.returncode == 0 and "inside 2 of 2 streams" in p.stdout
    p = run_inside(150.0)  # the streams had finished
    assert p.returncode == 1 and "outside: 0" in p.stdout and "1:" in p.stdout
    p = run_inside(99.0)  # taken before any token
    assert p.returncode == 1 and "outside" in p.stdout
    # a stream whose worst gap came before the loss: the loss was not what it shows
    load["per_stream"]["1"].update(gap_at=105.0, gap_s=0.5)
    path.write_text(json.dumps(load))
    p = run_inside(120.0)
    assert p.returncode == 1 and "a worst gap before the loss: 1:" in p.stdout
    # no record
    p = subprocess.run(
        [sys.executable, str(VERDICT), "inside", str(tmp_path / "none.json"), "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 1 and "no load record" in p.stdout
