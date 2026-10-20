# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The verdict on a loss arm's result: complete, and exact against the control arm.

A loss arm counts only when a change was committed and resumed, the head saw the loss, and
every session's tokens match the control arm's for the whole length. A replaced-but-diverged
arm (a spare serving its stage from empty rows, every session diverging at the first token after
the resume) can never read green.

    accept.py RESULT.json CONTROL.json

Prints one line and exits 0 on green, 1 otherwise.
"""

import json
import sys


def prefix(a: list[int], b: list[int]) -> int:
    """How many leading tokens ``a`` and ``b`` share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def verdict(result: dict, control: dict) -> tuple[bool, str]:
    """``(green, line)`` for ``result`` against ``control`` (the drill's records)."""
    changes = result.get("changes") or []
    events = [e["kind"] for e in result.get("executor_events") or []]
    complete = bool(changes) and all(c.get("n_replayed") is not None for c in changes)
    complete = complete and "rank_lost" in events
    mine, theirs = result.get("tokens") or [], control.get("tokens") or []
    sessions = [(prefix(a, b), len(a), len(b)) for a, b in zip(mine, theirs, strict=False)]
    exact = bool(sessions) and len(mine) == len(theirs)
    exact = exact and all(n == la == lb for n, la, lb in sessions)
    line = (
        f"loss arm {'complete' if complete else 'INCOMPLETE'}, "
        f"{'exact' if exact else 'DIVERGED'} vs control: changes {len(changes)}, events {events}, "
        f"identical prefix per session {[n for n, _, _ in sessions]} "
        f"of {[la for _, la, _ in sessions]}"
    )
    return complete and exact, line


def agreement(result: dict, control: dict) -> tuple[bool, str]:
    """``(green, line)`` for two runs with no change: every session's tokens the same.

    The loss arms' "exact" is attributable to the change only on an engine that repeats itself:
    two runs of the same sessions with no fault, on the same nodes, must agree token for token
    (a kernel chosen by timing, or a reduction order that varies run to run, shows here first).
    """
    changes = result.get("changes") or []
    mine, theirs = result.get("tokens") or [], control.get("tokens") or []
    sessions = [(prefix(a, b), len(a), len(b)) for a, b in zip(mine, theirs, strict=False)]
    same = bool(sessions) and len(mine) == len(theirs) and not changes
    same = same and all(n == la == lb for n, la, lb in sessions)
    line = (
        f"repeat {'AGREES' if same else 'DIFFERS'} with the first run: changes {len(changes)}, "
        f"identical prefix per session {[n for n, _, _ in sessions]} "
        f"of {[la for _, la, _ in sessions]}"
    )
    return same, line


def main(argv: list[str]) -> int:
    """The entry point: the result and the control, by path."""
    result, control = (json.load(open(p)) for p in argv[1:3])
    green, line = verdict(result, control)
    print(line)
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
