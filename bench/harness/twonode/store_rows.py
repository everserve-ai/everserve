# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Rows of a store, read through its rows query, for a row-level diff across copies.

    store_rows.py SOCKET ENGINE RANK TABLE:SLOT [TABLE:SLOT ...]   -> one JSON line

``MirrorClient.rows`` reads the rows under the store's lock and returns the rank's epoch and
step and each slot's ``written_step`` with its bytes: the source's, the replica's,
the kill-time snapshot's and the spare's copies of one row can then be diffed byte for byte,
and the step that wrote each copy compared.
"""

import json
import sys
from collections import defaultdict

from everserve.state.mirror import MirrorClient


def main(argv: list[str]) -> int:
    """The entry point."""
    sock, engine, rank, items = argv[1], argv[2], int(argv[3]), argv[4:]
    by_table: dict[int, list[int]] = defaultdict(list)
    for item in items:
        table, slot = item.split(":")
        by_table[int(table)].append(int(slot))
    client = MirrorClient(sock, connect_timeout_s=10.0)
    out: dict = {"socket": sock, "engine": engine, "rank": rank, "rows": {}}
    try:
        for table, slots in by_table.items():
            epoch, step, rows = client.rows(engine, rank, table, slots)
            out["epoch"], out["step"] = epoch, step
            for slot, written_step, data in rows:
                out["rows"][f"{table}@{slot}"] = {"written_step": written_step, "hex": data.hex()}
    except Exception as e:  # noqa: BLE001 - the record names the failure
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        client.close()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
