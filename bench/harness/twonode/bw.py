# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Bandwidth between two pods: NCCL over the private network, one GPU each.

Usage: python bw.py RANK MASTER_IP   (rank 0 on the head, rank 1 on the other node)
"""

import datetime
import sys
import time

import torch
import torch.distributed as dist

rank, master = int(sys.argv[1]), sys.argv[2]
print(f"rank {rank}: cuda {torch.cuda.is_available()}, joining tcp://{master}:29600", flush=True)
dist.init_process_group(
    "nccl",
    init_method=f"tcp://{master}:29600",
    rank=rank,
    world_size=2,
    timeout=datetime.timedelta(seconds=120),
)
print(f"rank {rank}: joined", flush=True)
dev = torch.device("cuda", 0)
sizes = {
    "decode_pp_send_128KB": 65536,
    "prefill_pp_send_268MB": 134_217_728,
    "replication_burst_650MB": 325_000_000,
}
for name, n in sizes.items():
    x = torch.ones(n, dtype=torch.bfloat16, device=dev)
    for _ in range(2):  # warm
        if rank == 0:
            dist.send(x, 1)
        else:
            dist.recv(x, 0)
    torch.cuda.synchronize()
    dist.barrier()
    reps = 20 if n < 1_000_000 else 3
    t0 = time.perf_counter()
    for _ in range(reps):
        if rank == 0:
            dist.send(x, 1)
        else:
            dist.recv(x, 0)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    if rank == 1:
        print(f"{name}: {dt * 1e3:.2f} ms per send, {n * 2 / dt / 1e9:.2f} GB/s", flush=True)
    dist.barrier()
dist.destroy_process_group()
