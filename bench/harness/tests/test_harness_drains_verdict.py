# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The drains smoke's verdict: a record of a run that did what the pre-registration predicts
passes every line, and each record that breaks one prediction fails that line and no other."""

import json

import drains_verdict as dv
import pytest

MAX = 40
T0 = "2026-10-12T16:00:{:02d}.000000Z"
COLOR = "\x1b[2m{}\x1b[0m \x1b[32m INFO\x1b[0m everserve_controller::coord: {}"


def at(s: int) -> str:
    return T0.format(s)


def coord_line(s: int, text: str) -> str:
    return COLOR.format(at(s), text)


def write(root, rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content if isinstance(content, str) else json.dumps(content))


DEADLINE_B_MS = int(dv.when(at(30)) * 1000)
DEADLINE_C_MS = int(dv.when(at(59)) * 1000)


def uncovered(losses):
    return {
        "type": "Covered",
        "status": "False",
        "reason": "LossUncovered",
        "message": f"{len(losses)} losses without a formed set (0 planned and forming): "
        + str(losses).replace(" ", ""),
    }


def cover(epoch, covered):
    sets = [{"purpose": f"REPLACE of ranks [{r}]", "covered": c} for r, c in covered.items()]
    return {"epoch": epoch, "sets": sets}


def meta(name, generation=0, written_by=""):
    return {
        "name": name,
        "generation": generation,
        "resourceVersion": "1.1",
        "writtenBy": written_by,
    }


def readiness(**over):
    """A ``GET /v1/engines/rig`` read (decision 0042): the engine's readiness under ``status``."""
    r = {
        "engine": "rig",
        "members": [
            {"rank": 0, "node": "node-a"},
            {"rank": 1, "node": "node-b"},
            {"rank": 2, "node": "node-c"},
        ],
        "spares": [
            {"rank": 1, "node": "node-d", "state": "warm"},
            {"rank": 2, "node": "node-e", "state": "warm"},
        ],
        "nodes": [
            {"node": "node-b", "outcome": "replace", "reclaimable": True},
            {"node": "node-c", "outcome": "replace", "reclaimable": True},
        ],
        "conditions": [uncovered([[0]])],
        "cover": cover(0, {1: True, 2: True}),
        "headConflict": None,
        "down": None,
    }
    r.update(over)
    return {"metadata": meta("rig"), "spec": {}, "status": r}


def node(name, unschedulable, *, deadline=None, draining=None, waits="", generation=1, by=""):
    """A Node object: what a PUT answers and a ``GET /v1/nodes`` lists."""
    spec = {"unschedulable": unschedulable, "reason": "nodes.sh"}
    if deadline is not None:
        spec["drain"] = {"deadlineMs": deadline}
    cond = {
        "type": "Draining",
        "status": "True" if draining else "False",
        "reason": draining or "NotDraining",
        "message": waits,
        "lastTransitionMs": 1,
    }
    status = {"linked": True, "devices": [], "engines": [], "conditions": [cond]}
    return {"metadata": meta(name, generation, by), "spec": spec, "status": status}


def nodes(*items):
    return {"metadata": {"resourceVersion": "1.1"}, "items": list(items)}


def answer(body, code):
    return json.dumps(body) + f"\nHTTP {code}\n"


WAITS = "2 positions would move at once, the policy allows 1"


def metrics(warm=2, conflicts=0, down=0):
    return (
        f'everserve_spares{{engine="rig",state="warm"}} {warm}\n'
        f'everserve_engine_down{{engine="rig"}} {down}\n'
        + (f'everserve_head_conflicts_total{{engine="rig"}} {conflicts}\n' if conflicts else "")
    )


@pytest.fixture
def run(tmp_path):
    """The record of a run that went as predicted."""
    passed = "".join(f"PASSED adapters/vllm/tests/test_node.py::{t}\n" for t in dv.GUARD_TESTS)
    write(tmp_path, "setup/pin_guards_a.txt", passed + "12 passed in 9.1s\n")
    write(
        tmp_path,
        "collect/a/out/daemon.log",
        coord_line(1, "head of rig registered (pid 10, launcher 10)") + "\n",
    )
    for n, rank in dv.DRAINED.items():
        write(
            tmp_path,
            f"collect/{n}/out/daemon.log",
            coord_line(1, "client of rig registered") + "\n",
        )
        write(
            tmp_path,
            f"collect/{n}/out/headless.log",
            f"2026-10-12 16:01:00,000 node released worker {rank} ended: "
            "{'epoch': 1, 'at': 1.0, 'how': 'killed', 'exitcode': -9, 'end_s': 0.4}\n"
            "2026-10-12 16:01:01,000 node this node holds nothing of rig: released {}\n"
            "==== node_main exited 0\n",
        )
        write(tmp_path, f"reads/final_procs_{n}.txt", "")
    write(
        tmp_path,
        "collect/c/out/daemon.log",
        coord_line(1, "client of rig registered")
        + "\n"
        + coord_line(42, "rig: notice for rank 2 at 1 ms")
        + "\n",
    )
    write(
        tmp_path,
        "collect/a/out/coord.log",
        "\n".join(
            [
                f"==== {at(0)[:19]}Z launch",
                coord_line(
                    6,
                    "node-c: its drain waits: 2 positions would move at once, the policy allows 1",
                ),
                f"==== {at(20)[:19]}Z launch",
                coord_line(21, "spec store: file /workspace/run/spec.json, 2 resources"),
                coord_line(35, "rig: membership committed at epoch 1 (Replace, lost [1])"),
                coord_line(41, "rig: drain of node-c releases ranks [2] at 1 ms"),
                coord_line(41, "rig: notice for rank 2 at 1 ms"),
                coord_line(55, "rig: membership committed at epoch 2 (Replace, lost [2])"),
            ]
        )
        + "\n",
    )
    final = readiness(
        members=[
            {"rank": 0, "node": "node-a"},
            {"rank": 1, "node": "node-d"},
            {"rank": 2, "node": "node-e"},
        ]
    )
    cordoned = readiness(
        spares=[
            {"rank": 1, "node": "node-d", "state": "cordoned"},
            {"rank": 2, "node": "node-e", "state": "cordoned"},
        ],
        nodes=[
            {"node": "node-b", "outcome": "rebuild", "reclaimable": False},
            {"node": "node-c", "outcome": "rebuild", "reclaimable": False},
        ],
        conditions=[uncovered([[0], [1], [2]])],
        cover=cover(0, {}),
    )
    for phase, ready, warm in [
        ("registered", readiness(), 2),
        ("cordoned", cordoned, 0),
        ("uncordoned", readiness(), 2),
        ("recovered", readiness(), 2),
        ("covered_after_b", readiness(cover=cover(1, {2: True})), 1),
        ("draining", readiness(), 2),
        ("final", final, 0),
    ]:
        write(tmp_path, f"reads/{phase}_readiness.json", ready)
        write(tmp_path, f"reads/{phase}_metrics.txt", metrics(warm=warm))
    spare_nodes = list(dv.SPARE_NODE.values())
    write(tmp_path, "reads/cordoned_nodes.json", nodes(*(node(n, True) for n in spare_nodes)))
    write(tmp_path, "reads/uncordoned_nodes.json", nodes(*(node(n, False) for n in spare_nodes)))
    write(
        tmp_path,
        "reads/draining_nodes.json",
        nodes(
            node("node-b", True, deadline=DEADLINE_B_MS, draining="Scheduled"),
            node("node-c", True, deadline=DEADLINE_C_MS, draining="Waiting", waits=WAITS),
        ),
    )
    by = "operator api: driver"
    for n in spare_nodes:
        x = n.removeprefix("node-")
        write(tmp_path, f"reads/cordon_{x}.txt", answer(node(n, True, by=by), 200))
        write(
            tmp_path,
            f"reads/uncordon_{x}.txt",
            answer(node(n, False, generation=2, by=by), 200),
        )
    write(
        tmp_path,
        "reads/drain_b.txt",
        answer(node("node-b", True, deadline=DEADLINE_B_MS, draining="Scheduled", by=by), 200),
    )
    write(
        tmp_path,
        "reads/drain_c.txt",
        answer(
            node("node-c", True, deadline=DEADLINE_C_MS, draining="Waiting", waits=WAITS, by=by),
            200,
        ),
    )
    refused = {
        "code": 409,
        "reason": "DrainUnderway",
        "message": "node-b: its drain is underway (the notices went out); it cannot be moved",
    }
    write(tmp_path, "reads/drain_b_again.txt", answer(refused, 409))
    write(
        tmp_path,
        "actions.jsonl",
        "\n".join(
            json.dumps({"action": a, "t": dv.when(at(3)), "rc": 0})
            for a in [
                "read_registered",
                "cordon_d",
                "cordon_e",
                "uncordon_d",
                "uncordon_e",
                "drain_b",
                "drain_c",
                "drain_b_again",
                "coord_restart",
                "covered_after_b",
                "drill_end",
            ]
        ),
    )
    write(tmp_path, "collect/a/out/drains.json", {"tokens": [[1] * MAX] * 8, "changes": [{}, {}]})
    return tmp_path


def scored(root):
    return dv.verdict(dv.Record(root, MAX))


def test_a_run_that_went_as_predicted_passes_every_line(run):
    v = scored(run)
    assert v["verdict"] == "pass", {
        k: p["why"] for k, p in v["predictions"].items() if not p["pass"]
    }


def _fails_only(root, name):
    v = scored(root)
    failed = [k for k, p in v["predictions"].items() if not p["pass"]]
    assert v["verdict"] == "fail" and failed == [name], failed


def test_a_skipped_guard_fails_h1(run):
    write(
        run, "setup/pin_guards_a.txt", (run / "setup/pin_guards_a.txt").read_text() + "1 skipped\n"
    )
    _fails_only(run, "H1")


def test_a_second_head_on_node_b_fails_h2(run):
    write(
        run,
        "collect/b/out/daemon.log",
        coord_line(1, "head of rig registered (pid 9, launcher 9)") + "\n",
    )
    _fails_only(run, "H2")


def test_a_counted_conflict_fails_h3(run):
    write(run, "reads/final_metrics.txt", metrics(warm=0, conflicts=1))
    _fails_only(run, "H3")


def test_a_worker_that_lingered_past_its_bound_fails_r2(run):
    text = (run / "collect/b/out/headless.log").read_text().replace("'end_s': 0.4", "'end_s': 9.2")
    write(run, "collect/b/out/headless.log", text)
    _fails_only(run, "R2")


def test_a_worker_ended_as_overdue_rather_than_released_at_its_boundary_fails_r1(run):
    text = (run / "collect/c/out/headless.log").read_text()
    text = text.replace("'end_s': 0.4}", "'end_s': 0.4, 'overdue': True}")
    write(run, "collect/c/out/headless.log", text)
    _fails_only(run, "R1")


def test_a_released_exit_reported_as_a_loss_fails_r3(run):
    log = run / "collect/c/out/headless.log"
    write(
        run,
        "collect/c/out/headless.log",
        log.read_text()
        + "2026-10-12 16:01:02,000 node worker 2 is exiting; reported to the daemon\n",
    )
    _fails_only(run, "R3")


def test_a_node_role_that_did_not_exit_cleanly_fails_r4(run):
    log = run / "collect/b/out/headless.log"
    write(run, "collect/b/out/headless.log", log.read_text().replace("exited 0", "exited 1"))
    _fails_only(run, "R4")


def test_a_short_session_fails_r5(run):
    write(
        run,
        "collect/a/out/drains.json",
        {"tokens": [[1] * MAX] * 7 + [[1] * 3], "changes": [{}, {}]},
    )
    _fails_only(run, "R5")


def test_spares_still_counted_warm_while_cordoned_fail_n1(run):
    write(run, "reads/cordoned_metrics.txt", metrics(warm=2))
    _fails_only(run, "N1")


def test_node_c_s_notice_before_node_b_s_commit_fails_n2(run):
    log = (run / "collect/a/out/coord.log").read_text()
    early = coord_line(41, "rig: notice for rank 2 at 1 ms")
    write(
        run,
        "collect/a/out/coord.log",
        log.replace(early, coord_line(10, "rig: notice for rank 2 at 1 ms")),
    )
    _fails_only(run, "N2")


def test_node_c_s_drain_answered_as_scheduled_rather_than_waiting_fails_n2(run):
    body = node("node-c", True, deadline=DEADLINE_C_MS, draining="Scheduled")
    write(run, "reads/drain_c.txt", answer(body, 200))
    _fails_only(run, "N2")


def test_node_c_listed_as_scheduled_while_node_b_s_change_runs_fails_n2(run):
    write(
        run,
        "reads/draining_nodes.json",
        nodes(
            node("node-b", True, deadline=DEADLINE_B_MS, draining="Scheduled"),
            node("node-c", True, deadline=DEADLINE_C_MS, draining="Scheduled"),
        ),
    )
    _fails_only(run, "N2")


def test_a_moved_deadline_fails_n3(run):
    moved = node("node-b", True, deadline=DEADLINE_B_MS + 60000, draining="Scheduled")
    write(run, "reads/drain_b_again.txt", answer(moved, 200))
    _fails_only(run, "N3")


def test_a_second_drain_refused_for_a_stale_version_not_the_drain_fails_n3(run):
    stale = {"code": 409, "reason": "Conflict", "message": "node-b changed since version 1.1"}
    write(run, "reads/drain_b_again.txt", answer(stale, 409))
    _fails_only(run, "N3")


def test_a_held_deadline_that_moved_though_refused_fails_n3(run):
    write(
        run,
        "reads/draining_nodes.json",
        nodes(
            node("node-b", True, deadline=DEADLINE_B_MS + 60000, draining="Scheduled"),
            node("node-c", True, deadline=DEADLINE_C_MS, draining="Waiting", waits=WAITS),
        ),
    )
    _fails_only(run, "N3")


def test_node_b_s_change_long_after_its_deadline_fails_n4(run):
    # node-b's deadline at 5 s, as asked and as held: its commit at 35 s is 30 s late
    early = int(dv.when(at(5)) * 1000)
    for rel in ("reads/drain_b.txt", "reads/draining_nodes.json"):
        write(run, rel, (run / rel).read_text().replace(str(DEADLINE_B_MS), str(early)))
    _fails_only(run, "N4")


def test_an_engine_down_in_any_read_fails_n5(run):
    write(run, "reads/draining_metrics.txt", metrics(down=1))
    _fails_only(run, "N5")


def test_one_spare_node_left_uncordoned_fails_n1(run):
    write(run, "reads/cordon_e.txt", "node-e: not found\nHTTP 404\n")
    _fails_only(run, "N1")


def test_a_position_held_on_the_other_spare_s_node_fails_r3(run):
    final = json.loads((run / "reads/final_readiness.json").read_text())
    final["status"]["members"][2]["node"] = "node-d"
    write(run, "reads/final_readiness.json", final)
    _fails_only(run, "R3")


def test_a_warm_up_s_reported_exit_before_the_run_s_launch_does_not_fail_r3(run):
    log = (run / "collect/c/out/headless.log").read_text()
    warm = "2026-10-12 15:59:00,000 node worker 2 is exiting; reported to the daemon\n"
    write(run, "collect/c/out/headless.log", warm + f"==== {at(0)[:19]}Z launch\n" + log)
    assert scored(run)["verdict"] == "pass"


def test_the_restart_is_the_coordinator_s_last_launch_after_a_warm_up(run):
    log = (run / "collect/a/out/coord.log").read_text()
    write(run, "collect/a/out/coord.log", "==== 2026-10-12T15:50:00Z launch\n" + log)
    assert scored(run)["verdict"] == "pass"


def test_an_engine_down_only_in_the_coordinator_s_log_fails_n5(run):
    log = (run / "collect/a/out/coord.log").read_text()
    down = coord_line(50, "rig: down: its head failed: the change at epoch 2 was abandoned") + "\n"
    write(run, "collect/a/out/coord.log", log + down)
    _fails_only(run, "N5")


def test_cover_not_formed_again_after_the_uncordon_fails_c1(run):
    write(run, "reads/recovered_readiness.json", readiness(cover=cover(0, {1: True, 2: False})))
    v = scored(run)
    assert [k for k, p in v["predictions"].items() if not p["pass"]] == ["C1"]


def test_rank_2_s_set_covered_only_after_node_c_s_deadline_fails_c2(run):
    late = int(dv.when(at(1)) * 1000)  # node-c's deadline before the covered read at 3 s
    body = node("node-c", True, deadline=late, draining="Waiting", waits=WAITS)
    write(run, "reads/drain_c.txt", answer(body, 200))
    _fails_only(run, "C2")


def test_a_head_that_stopped_in_order_at_the_end_is_not_the_engine_down(run):
    stopped = {"atMs": 1, "cause": "stopped", "detail": ""}
    final = json.loads((run / "reads/final_readiness.json").read_text())
    final["status"]["down"] = stopped
    write(run, "reads/final_readiness.json", final)
    write(run, "reads/final_metrics.txt", metrics(warm=0, down=1))
    log = (run / "collect/a/out/coord.log").read_text()
    write(
        run, "collect/a/out/coord.log", log + coord_line(58, "rig: down: its head stopped: ") + "\n"
    )
    assert scored(run)["verdict"] == "pass"


def test_a_head_that_failed_is_the_engine_down(run):
    log = (run / "collect/a/out/coord.log").read_text()
    failed = coord_line(50, "rig: down: its head failed: the change at epoch 2 was abandoned")
    write(run, "collect/a/out/coord.log", log + failed + "\n")
    _fails_only(run, "N5")
