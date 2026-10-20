# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The committed run records obey the rule bench/runs/README.md states."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "harness"))

import record  # noqa: E402 - the harness's modules import by name from their own directory


@pytest.mark.parametrize(
    ("relative", "size", "held"),
    [
        ("RESULT.md", 10_000_000, True),
        ("results.json", 10_000_000, True),
        ("run/attempt3/verdict.json", 100, True),
        ("run/run.log", record.LOG_BYTES, True),
        ("run/run.log", record.LOG_BYTES + 1, False),
        ("run/engines-proof-loss-1.json", 1000, True),
        ("run/engines-proof-loss-1.json", record.SMALL_BYTES + 1, False),
        ("run/arms/stock/trial1/summary.json", 10_000_000, True),
        ("run/arms/stock/trial1/result.json", 100, False),
        ("run/arms/node_fault/trial1/backup_rank0_holder.json", 100, False),
        ("run/arms/node_fault/trial1/a/out/control_1.json", 100, False),
        ("run/logs/serving-head.log", 100, False),
        ("run/pods.txt.bak", 100, False),
        ("run/kill-1.armed", 100, False),
        ("run/actions.jsonl", 10_000_000, True),
        ("data/topology.txt", 100, False),
    ],
)
def test_the_rule_keeps_the_evidence_and_drops_the_dumps(relative, size, held):
    assert record.kept(relative, size) is held


def test_scrub_replaces_what_names_the_account_and_keeps_the_rest():
    text = (
        "cd /Users/someone/workspace/everserve/.claude/worktrees/fix && ls /Users/someone/x\n"
        "kubeconfig at /private/var/folders/qw/abc123/T//everserve-proof-secrets.XYZ/kubeconfig\n"
        "image cr.eu-north1.nebius.cloud/e00abcdefghij12345/everserve:dev-1"
        " in project-e00abcdefghij1234567, nodes under /home/everserve\n"
        '{"public_ip": "89.169.1.2", "private_ip": "10.0.0.11"}'
        " ssh root@195.242.28.140 pod 10.1.2.3\n"
        "torch 2.9.0 cuda 13.0.3.0 nccl 2.27.3 at http://127.0.0.1:8000 and https://1.2.3.4:6443\n"
    )
    out = record.scrub(text)
    assert out == (
        "cd <tree> && ls $HOME/x\n"
        "kubeconfig at $TMPDIR/everserve-proof-secrets.XYZ/kubeconfig\n"
        "image cr.eu-north1.nebius.cloud/<registry>/everserve:dev-1"
        " in project-<id>, nodes under /home/everserve\n"
        '{"public_ip": "<public-ip>", "private_ip": "10.0.0.11"}'
        " ssh root@<public-ip> pod 10.1.2.3\n"
        "torch 2.9.0 cuda 13.0.3.0 nccl 2.27.3 at http://127.0.0.1:8000 and https://<public-ip>:6443\n"
    )
    assert record.unscrubbed(out) == []


def test_curate_removes_the_dumps_and_scrubs_in_place(tmp_path):
    (tmp_path / "RESULT.md").write_text("ran from /Users/someone/workspace/everserve\n")
    (tmp_path / "run" / "out").mkdir(parents=True)
    (tmp_path / "run" / "out" / "control_1.json").write_text("{}")
    (tmp_path / "run" / "run.log").write_text("fine\n")
    removed = record.curate(tmp_path)
    assert removed == ["run/out/control_1.json"]
    assert not (tmp_path / "run" / "out").exists()
    assert (tmp_path / "RESULT.md").read_text() == "ran from <tree>\n"
    assert record.check(tmp_path) == []


def test_every_tracked_record_obeys_the_rule():
    """The tree holds what the README says it holds: a dump or an account's name fails here."""
    runs = ROOT / "bench" / "runs"
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "bench/runs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    relatives = [t[len("bench/runs/") :] for t in tracked if t]
    if not relatives:
        pytest.skip("not a git checkout: nothing is tracked")
    broken = record.check(runs, relatives)
    assert broken == [], "\n".join(broken[:40])
