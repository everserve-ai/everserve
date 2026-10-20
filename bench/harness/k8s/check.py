# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What the Kubernetes test asserts of its probes' records (stage_probe.py's JSON lines).

    check.py decided  LOGDIR --world 4 --nnodes 2      every stage served once, the rest stand
    check.py placed   LOGDIR --world 4                 every probe heard every rank placed, by host
    check.py reserved LOGDIR --pod probe-nodes-1 --stage 2,3   that pod was told to serve the stage
    check.py stood    LOGDIR --pod probe-nodes-3 --stage 2,3   that pod was told to stand for it
    check.py refused  LOGDIR --pod probe-odd-0              that pod was refused for its step
    check.py exited   LOGDIR --pod probe-odd-0 --code 2     its container ended so (<pod>.pod.json)

LOGDIR holds one ``<pod>.log`` per probe (``kubectl logs``). Each check prints what it read and
exits 1 on the first claim that fails, naming it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def records(logdir: Path) -> dict[str, dict]:
    """Each pod's record: the last value of every key it printed."""
    out: dict[str, dict] = {}
    for f in sorted(logdir.glob("*.log")):
        got: dict = {}
        for line in f.read_text().splitlines():
            try:
                got.update(json.loads(line))
            except ValueError:
                continue  # a line that is not the probe's (a Python warning)
        out[f.stem] = got
    return out


def fail(claim: str, seen: object) -> None:
    """Say which claim failed and what was seen, and exit 1."""
    print(json.dumps({"failed": claim, "seen": seen}, indent=1))
    sys.exit(1)


def admitted(recs: dict[str, dict]) -> dict[str, dict]:
    """The probes the step guard let in: a refused probe is no part of the engine (assertion 5)."""
    return {p: r for p, r in recs.items() if "refused" not in r}


def decided(recs: dict[str, dict], world: int, nnodes: int) -> None:
    """Every stage has exactly one probe told to serve it; every other probe of it stands."""
    recs = admitted(recs)
    per = world // nnodes
    for n in range(nnodes):
        stage = list(range(n * per, (n + 1) * per))
        askers = {p: r for p, r in recs.items() if r.get("asked") == stage}
        workers = [p for p, r in askers.items() if r.get("told") == "worker"]
        if not askers:
            fail(f"a probe asked for stage {stage}", recs)
        if len(workers) != 1:
            fail(
                f"exactly one probe serves stage {stage}",
                {p: r.get("told") for p, r in askers.items()},
            )
        others = [r.get("told") for p, r in askers.items() if p not in workers]
        if any(t != "spare" for t in others):
            fail(f"every other probe of stage {stage} stands", others)
    print(json.dumps({"decided": {p: [r.get("asked"), r.get("told")] for p, r in recs.items()}}))


def placed(recs: dict[str, dict], world: int) -> None:
    """Every probe heard the same placement, with every rank and the host of its serving probe."""
    recs = admitted(recs)
    serving = {}
    for r in recs.values():
        if r.get("told") == "worker":
            for rank in r["asked"]:
                serving[str(rank)] = r["host"]
    seen = {p: r.get("members") for p, r in recs.items()}
    for p, members in seen.items():
        if not members or sorted(members, key=int) != [str(k) for k in range(world)]:
            fail(f"{p} heard every rank of {world} placed", members)
        hosts = {rank: m[1] for rank, m in members.items()}
        if hosts != serving:
            fail(f"{p}'s placement names each rank's serving probe's host", [hosts, serving])
    print(json.dumps({"placed": serving}))


def reserved(recs: dict[str, dict], pod: str, stage: list[int]) -> None:
    """``pod`` was told to serve ``stage``: a stage released by its server's end is given again."""
    r = recs.get(pod, {})
    if r.get("asked") != stage or r.get("told") != "worker":
        fail(f"{pod} was told to serve {stage}", r)
    print(json.dumps({"reserved": pod, "stage": stage}))


def stood(recs: dict[str, dict], pod: str, stage: list[int]) -> None:
    """``pod`` was told to stand for ``stage``: its server is remembered, a restart included."""
    r = recs.get(pod, {})
    if r.get("asked") != stage or r.get("told") != "spare":
        fail(f"{pod} was told to stand for {stage}", r)
    print(json.dumps({"stood": pod, "stage": stage}))


def refused(recs: dict[str, dict], pod: str) -> None:
    """``pod`` was refused for its step config, and told nothing: the step guard at the ask.

    The step guard names what differs against the engine's (``capability.rs``,
    ``step_difference``): here the probe's one field, ``probe is "other" here, "probe" on the
    engine``. A guarantee's refusal says "in the policy" instead.
    """
    r = recs.get(pod, {})
    why = str(r.get("refused", ""))
    if 'probe is "other" here' not in why or "on the engine" not in why or "told" in r:
        fail(f"{pod} was refused for its step config before it was told anything", r)
    print(json.dumps({"refused": pod, "why": r["refused"]}))


def exited(logdir: Path, pod: str, code: int) -> None:
    """``pod``'s container ended with ``code``, as its captured ``<pod>.pod.json`` reports it.

    The pod restarts as any pod does, so its last end is in ``lastState`` once it runs again, and
    in ``state`` while it waits to.
    """
    path = logdir / f"{pod}.pod.json"
    try:
        status = json.loads(path.read_text()).get("status", {})
    except (OSError, ValueError) as e:
        fail(f"{pod}'s pod was captured", str(e))
    ends = [
        s.get(k, {}).get("terminated", {}).get("exitCode")
        for s in status.get("containerStatuses", [])
        for k in ("state", "lastState")
    ]
    if code not in ends:
        fail(f"{pod} exited {code}", ends)
    print(json.dumps({"exited": pod, "code": code}))


def main() -> None:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "check", choices=["decided", "placed", "reserved", "stood", "refused", "exited"]
    )
    ap.add_argument("--code", type=int, default=0)
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--nnodes", type=int, default=2)
    ap.add_argument("--pod", default="")
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    if a.check == "exited":
        exited(a.logdir, a.pod, a.code)
        return
    recs = records(a.logdir)
    if a.check == "decided":
        decided(recs, a.world, a.nnodes)
    elif a.check == "placed":
        placed(recs, a.world)
    elif a.check == "reserved":
        reserved(recs, a.pod, [int(x) for x in a.stage.split(",")])
    elif a.check == "refused":
        refused(recs, a.pod)
    else:
        stood(recs, a.pod, [int(x) for x in a.stage.split(",")])


if __name__ == "__main__":
    main()
