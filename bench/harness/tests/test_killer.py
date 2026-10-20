# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The in-pod killer chooses exactly the processes serving a rank, never itself or the container's
PID 1, which SIGKILL from inside does not reach."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "killer", Path(__file__).resolve().parents[1] / "k8s" / "killer.py"
)
killer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(killer)

NUL = b"\x00"


def cmd(*words: str) -> bytes:
    return NUL.join(w.encode() for w in words) + NUL


# a head pod: Dynamo's worker as PID 1, its engine core, and rank 0's worker beside them
HEAD = [
    (1, cmd("python3", "-m", "dynamo.vllm", "--model", "Qwen/Qwen2.5-0.5B-Instruct")),
    (198, cmd("VLLM::EngineCore")),
    (258, cmd("VLLM::Worker_PP0")),
    (400, cmd("python3", "-")),
]
# a node pod: its node process, rank 1's worker, a spare that took a rank
NODE = [
    (1, cmd("python3", "-m", "everserve_vllm.node_main")),
    (176, cmd("VLLM::Worker_PP1")),
    (220, cmd("python3", "-m", "everserve_vllm.spare_main", "--rank", "1")),
]


def test_a_crash_kills_the_workers_and_a_spare_serving_and_nothing_else():
    assert killer.select(NODE, me=999) == ([176, 220], [])
    assert killer.select(HEAD, me=999) == ([258], [])


def test_a_pid_1_serving_a_rank_is_named_out_of_reach():
    spare_as_pid_1 = [(1, cmd("python3", "-m", "everserve_vllm.spare_main", "--rank", "1"))]
    assert killer.select(spare_as_pid_1, me=999) == ([], [1])


def test_the_killer_never_chooses_itself_whatever_its_command_line_holds():
    me = (500, cmd("python3", "-c", "VLLM::Worker everserve_vllm.spare_main"))
    assert killer.select([*NODE, me], me=500)[0] == [176, 220]


def test_the_killers_own_text_holds_no_name_it_matches():
    text = (Path(killer.__file__)).read_bytes()
    assert not any(n in text for n in killer.NAMES)


FIXTURES = Path(__file__).parent / "fixtures"


def recorded(name: str) -> list[tuple[int, bytes]]:
    """A ``ps -eo pid,ppid,args`` capture as (pid, command line), NULs between args as in /proc."""
    out = []
    for line in (FIXTURES / name).read_text().splitlines()[1:]:
        pid, _ppid, *args = line.split()
        out.append((int(pid), bytes(1).join(a.encode() for a in args)))
    return out


def test_on_proof_25s_head_pod_a_crash_kills_the_worker_and_leaves_the_engine_core():
    assert killer.select(recorded("proof-25-head-ps.txt"), me=656) == ([394], [])


def test_on_proof_25s_node_pod_a_crash_kills_the_worker_and_not_the_node_main():
    assert killer.select(recorded("proof-25-node-ps.txt"), me=531) == ([246], [])
