# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The raw bytes of rows from a kill-time KV file (restore_check.save_rows), as hex.

kv_rows.py FILE.pt LAYER:HEAD@SLOT [LAYER:HEAD@SLOT ...]   -> one JSON line
"""

import json
import sys

import torch

path = sys.argv[1]
items = sys.argv[2:]
d = torch.load(path, weights_only=True)  # tensors and plain containers only, never code
pos = {s: i for i, s in enumerate(d["slots"])}
out = {}
for it in items:
    key, slot = it.split("@")
    i = pos[int(slot)]
    out[it] = d["rows"][key][:, i].contiguous().view(torch.uint8).numpy().tobytes().hex()
print(json.dumps(out))
