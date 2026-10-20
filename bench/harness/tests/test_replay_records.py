# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Every read the Dynamo proof's harness acts on, replayed against records real runs saved.

Each case is a saved record and the verdict that matches what that run did. A harness that reads
a record early stops a run that should have gone on, which hand-made records do not catch. A
read the harness acts on gets a case here against a record that holds it (and a commit adding a
read with no record that holds it says so).

The records are `bench/runs/2026-10-15-dynamo-proof-gated-join`'s, copied into ``fixtures/``:
two REPLACEs of rank 1, at epochs 1 and 2, each read right after its loss.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
K8S = ROOT / "bench" / "harness" / "k8s"
VERDICT = K8S / "proof_verdict.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

G1 = FIXTURES / "proof-25-engines-loss-1.json"
G2 = FIXTURES / "proof-25-engines-loss-2.json"

CASES = [
    # (record, check, args, holds, words the line must contain)
    (G1, "serving", [], True, "2 of 2 positions held"),
    (G2, "serving", [], True, "2 of 2 positions held"),
    # the head holds rank 0 throughout; rank 1 is on the spare that took it, another each loss
    (G1, "holder", ["0"], True, "computeinstance-e00h8q579gcdmej8a6"),
    (G1, "holder", ["1"], True, "computeinstance-e00jh0j1ryxae786kr"),
    (G2, "holder", ["1"], True, "computeinstance-e00gdq11amv5ep4ja7"),
    # read right after each loss: the next set for rank 1 not yet planned, so not covered
    (G1, "covered", ["1"], False, "plan 3 at epoch 1: no set planned for the loss of rank 1"),
    (G2, "covered", ["1"], False, "plan 5 at epoch 2: no set planned for the loss of rank 1"),
    (G2, "engine-covered", ["1"], False, "the head's node holds [0], whose loss restarts"),
    # each loss a REPLACE of rank 1 at its own epoch, and only that one
    (G1, "replaced", ["1", "1"], True, "epoch 1: replace of ranks [1]"),
    (G1, "replaced", ["1", "2"], False, "epoch 2 expected"),
    (G2, "replaced", ["1", "2"], True, "epoch 2: replace of ranks [1]"),
]


@pytest.mark.parametrize(
    "record,check,args,holds,words",
    CASES,
    ids=[f"{c[0].parent.parent.name}/{c[0].parent.name}:{c[1]}{c[2]}" for c in CASES],
)
def test_a_saved_record_reads_as_its_run_went(record, check, args, holds, words):
    p = subprocess.run(
        [sys.executable, str(VERDICT), check, str(record), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (p.returncode == 0) is holds, p.stdout
    assert words in p.stdout, p.stdout


def test_members_that_differ_stop_the_run_at_bring_up_naming_the_field(tmp_path):
    """No saved record carries the condition the coordinator reports for members running
    different software, as the check came after the runs: this is the first loss's record with
    that condition added in the coordinator's words."""
    record = json.loads(G1.read_text())
    record["status"]["conditions"].append({
        "type": "Degraded", "status": "True", "reason": "MembersDiffer",
        "message": "rank 1 and rank 0 run different software: checkpoint_digest is "
                   '"984d63…" here, "b18fa9…" on the position',
    })  # fmt: skip
    path = tmp_path / "engines-proof.json"
    path.write_text(json.dumps(record))
    p = subprocess.run(
        [sys.executable, str(VERDICT), "serving", str(path)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert p.returncode == 1 and "MembersDiffer: rank 1 and rank 0" in p.stdout, p.stdout
