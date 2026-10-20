# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A crash from inside a pod: every process serving a rank killed with SIGKILL, no notice given.

Run inside the pod (``kubectl exec ... python3 - < killer.py``). It prints the moment of the kill
on the pod's clock and each process it killed, the proof's record of the loss.

The processes serving a rank, by the command line each shows: vLLM's workers by the title vLLM
gives them, and a spare that took a rank, which serves in its own process.

Two processes are never killed: this one, whose own command line holds the names it matches (a
killer that matches itself dies part way), and the container's PID 1, which SIGKILL sent from
inside the container does not reach. A PID 1 that matches is reported, so the record says what was
out of reach.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Iterable

# the names in pieces: this file's own text must not match them when it is a command line
NAMES: tuple[bytes, ...] = (b"VLLM::" + b"Worker", b"everserve_vllm." + b"spare_main")


def select(processes: Iterable[tuple[int, bytes]], me: int) -> tuple[list[int], list[int]]:
    """The pids among ``processes`` (pid, command line) that serve a rank.

    Those to kill, and those that match but are out of reach (the container's PID 1). ``me`` is
    never chosen.
    """
    kill, unreachable = [], []
    for pid, cmd in processes:
        if pid == me or not any(n in cmd for n in NAMES):
            continue
        (unreachable if pid == 1 else kill).append(pid)
    return sorted(kill), sorted(unreachable)


def processes() -> list[tuple[int, bytes]]:
    """Every process this one can read: its pid and command line, NULs as in /proc."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            out.append((int(p), open(f"/proc/{p}/cmdline", "rb").read()))
        except OSError:
            continue
    return out


def first_word(cmd: bytes) -> str:
    """A command line's program, for the record."""
    return cmd.split(bytes(1))[0].decode(errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Kill this pod's processes serving a rank and print each, with the moment."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    table = dict(processes())
    kill, unreachable = select(table.items(), os.getpid())
    print("kill at", int(time.time() * 1000), "ms on the pod clock", flush=True)
    for pid in kill:
        try:
            os.kill(pid, signal.SIGKILL)
            print("killed", pid, first_word(table[pid]), flush=True)
        except OSError as e:
            print("not killed", pid, first_word(table[pid]), e, flush=True)
    for pid in unreachable:
        print("out of reach", pid, first_word(table[pid]), "(the container's PID 1)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
