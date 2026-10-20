# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The claim-to-commit breakdown of a change, from the coordinator's four moments.

A committed membership carries when the coordinator took the report that moved the change,
delivered the change to the nodes, received the last acknowledgement, and committed, on its own
clock (``everserve.contract.signals.MembershipCommitted``, microseconds since the Unix epoch). The
executor adds when the commit reached it on its clock. The spans between them are the run
record's breakdown: deliver, acknowledgements, commit, and the commit's way to the executor,
which includes the two clocks' offset (the harness measures it: ``clockoffset.py``).
"""

from __future__ import annotations

from typing import Any


def claim_to_commit(committed: Any, heard_at: float) -> dict[str, Any] | None:
    """The spans of a change from its report to the executor hearing the commit, in seconds.

    None when the coordinator stamped nothing (a build before the stamps: every moment zero).
    ``heard_at`` is the executor's ``time.time()`` at the commit's arrival.
    """
    moments = {
        name: getattr(committed, f"{name}_at_us", 0) / 1e6
        for name in ("reported", "delivered", "acknowledged", "committed")
    }
    if not all(moments.values()):
        return None
    return {
        "coordinator": {f"{k}_at": round(v, 6) for k, v in moments.items()},
        "deliver_s": round(moments["delivered"] - moments["reported"], 6),
        "acknowledgements_s": round(moments["acknowledged"] - moments["delivered"], 6),
        "commit_s": round(moments["committed"] - moments["acknowledged"], 6),
        "heard_after_commit_s": round(heard_at - moments["committed"], 6),
    }
