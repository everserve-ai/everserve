# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A sample of one rank's rows in one store, as digests, to score a backup against its holder.

    backup_rows.py SOCKET ENGINE RANK --tables N --table-stride K --slots M --slot-stride J
        -> one JSON line: the rank's epoch and step at this store, and per sampled row its
           written step and a digest of its bytes

Run at the store of the node that holds the rank and at its replica's store; ``--compare`` then
scores the replica (decision 0030's backup after a take): of the rows the holder has written at
or before the replica's step, the share the replica holds with the same bytes. A second loss
restores from the replica, so this is what that restore would read.

    backup_rows.py --compare HOLDER.json REPLICA.json   -> {"coverage": ..., ...}
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sample(a: argparse.Namespace) -> dict:
    """The sampled rows of the rank at this store."""
    from everserve.state.mirror import MirrorClient

    client = MirrorClient(a.socket, connect_timeout_s=10.0)
    out: dict = {"engine": a.engine, "rank": a.rank, "rows": {}}
    try:
        held, epoch, step, caught_up = client.state(a.engine, a.rank)
        out.update(held=held, epoch=epoch, step=step, caught_up=caught_up)
        slots = list(range(0, a.slots, a.slot_stride))
        for table in range(0, a.tables, a.table_stride):
            _e, _s, rows = client.rows(a.engine, a.rank, table, slots)
            for slot, written, data in rows:
                if written:
                    digest = hashlib.blake2b(data, digest_size=8).hexdigest()
                    out["rows"][f"{table}:{slot}"] = [written, digest]
    finally:
        client.close()
    return out


def compare(holder: dict, replica: dict) -> dict:
    """Score ``replica`` against ``holder`` at one point: the lesser of their steps.

    Every row the holder wrote at or before that step must be on the replica with the same
    bytes; a replica behind the holder is scored where both stand, never by the replica's step
    alone, which a replica missing rows would pass. Read both with the engine quiescent (the
    battery reads them after the arm's sessions finish) and the two steps agree.
    """
    upto = min(holder.get("step", 0), replica.get("step", 0))
    due = {k: v for k, v in holder["rows"].items() if v[0] <= upto}
    same = sum(1 for k, (_w, d) in due.items() if replica["rows"].get(k, [0, None])[1] == d)
    return {
        "rank": holder["rank"],
        "holder": {k: holder.get(k) for k in ("epoch", "step", "held", "caught_up")},
        "replica": {k: replica.get(k) for k in ("epoch", "step", "held", "caught_up")},
        "scored_at_step": upto,
        "rows_due": len(due),
        "rows_matching": same,
        "coverage": round(same / len(due), 6) if due else None,
    }


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--compare", nargs=2, metavar=("HOLDER", "REPLICA"))
    ap.add_argument("socket", nargs="?")
    ap.add_argument("engine", nargs="?")
    ap.add_argument("rank", nargs="?", type=int)
    ap.add_argument("--tables", type=int, default=8)
    ap.add_argument("--table-stride", type=int, default=1)
    ap.add_argument("--slots", type=int, default=22400)
    ap.add_argument("--slot-stride", type=int, default=16)
    a = ap.parse_args(argv[1:])
    if a.compare:
        h, r = (json.loads(Path(p).read_text()) for p in a.compare)
        print(json.dumps(compare(h, r)))
        return 0
    print(json.dumps(sample(a)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
