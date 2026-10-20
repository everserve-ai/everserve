# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A node pod of an instance without its engine: asks what it is, reports what it is told, stays.

The Kubernetes test without GPUs runs this in each pod of a group (a StatefulSet standing for a
framework's group, its ordinal the slot) under the pod contract. It registers with its node's daemon
undecided for its stage, as ``everserve_vllm.node_main --slot`` does, and prints one JSON line per
thing it hears: the engine's policy, the daemon's answer, and each placement. It then stays, silent,
its connection the stage's liveness, until it is stopped; it never registers a worker, so a stage
it was told to serve stays assigned to it and is released when it ends.

    python stage_probe.py --engine k --world 4 --nnodes 2 --slot $(POD_INDEX)
    python stage_probe.py --engine k --world 4 --nnodes 2 --node-rank 0     # the head's node
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time


def say(**what: object) -> None:
    """One line of the probe's record, stamped."""
    print(json.dumps({"t": round(time.time(), 3), **what}), flush=True)


def identity(step: str):  # noqa: ANN201 - the contract's type, imported where the pod has it
    """What a probe registers as: no engine of its own, one step config every probe shares."""
    import hashlib

    from everserve.contract.signals import SoftwareIdentity, StepConfig

    digest = hashlib.sha256(f"probe={step}".encode()).hexdigest()
    guarantee = os.environ.get(
        "EVERSERVE_GUARANTEE", ""
    )  # the pod states it, as an engine pod does
    step_config = StepConfig(digest, {"probe": step})
    return SoftwareIdentity("probe", "", "", step=step_config, guarantee=guarantee)


def main() -> int:
    """Ask, report, stay."""
    from everserve.contract.daemon import EXIT_REFUSED, Refused, UndecidedConnection
    from everserve.contract.signals import Member
    from everserve.discovery import local_daemon
    from everserve.podcontract import NODE_ENV

    ap = argparse.ArgumentParser(prog="stage_probe")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--world", type=int, required=True, help="the engine's ranks")
    ap.add_argument("--nnodes", type=int, required=True, help="the nodes the engine spans")
    placed = ap.add_mutually_exclusive_group(required=True)
    placed.add_argument("--slot", type=int, help="this pod's index in its group")
    placed.add_argument("--node-rank", type=int, help="the node rank asked for (the head's: 0)")
    ap.add_argument(
        "--step",
        default="probe",
        help="what this probe says it steps with: every probe of an engine says the same, and one "
        "that says another is refused by the step guard",
    )
    a = ap.parse_args()
    node = os.environ[NODE_ENV]
    per = a.world // a.nnodes
    # as node_main --slot: the head is node 0; slots ask for the other nodes in turn
    if a.node_rank is not None:
        node_rank = a.node_rank
    else:
        node_rank = 1 + a.slot % (a.nnodes - 1) if a.nnodes > 1 else 0
    stage = tuple(range(node_rank * per, (node_rank + 1) * per))
    record = local_daemon(node)
    if record is None:
        say(error=f"no daemon for node {node}")
        return 1
    link = UndecidedConnection(
        record.address, a.engine, stage, Member(node, 0), identity=identity(a.step)
    )
    say(asked=list(stage), node=node, slot=a.slot, host=record.host)
    # the answer first: a refused probe is closed on without the policy
    try:
        told = link.assigned(timeout_s=120)
    except Refused as refused:
        say(refused=refused.refusal)
        link.refused(refused)
        return EXIT_REFUSED
    policy = link.engine_policy(timeout_s=120)
    say(policy=policy.version, world=policy.engine(a.engine).world_size)
    say(told=told.role.value, epoch=told.epoch, stage=list(told.stage))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    seen = None
    while not stop.wait(1.0) and not link.closed.is_set():
        placed = link.placement
        if placed is not None and placed is not seen:
            seen = placed
            say(
                placement=placed.epoch,
                members={p.rank: [p.node, p.host] for p in placed.members},
                spares=[[s.for_rank, s.node, s.host] for s in placed.spares],
            )
    say(ended="stopped" if stop.is_set() else "the daemon closed the link")
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
