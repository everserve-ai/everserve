# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process's part of the cover plan (decision 0035): what it forms, keeps, releases, reports."""

import threading
import time

from everserve.comm.cover import (
    DOMAIN,
    FABRIC,
    Cover,
    Plan,
    PlannedGroup,
    PlannedSet,
    lost_members,
)


class Held:
    """A collective's groups, by scope and member set, and the set it serves on."""

    def __init__(self, groups=None, serving=()):
        self.groups = dict(groups or {})
        self._serving = set(serving)
        self.renamed = {}

    def held(self):
        return dict(self.groups)

    def serving(self):
        return set(self._serving)

    def keep(self, scope, members, group):
        self.groups[(scope, members)] = group

    def drop(self, scope, members):
        return self.groups.pop((scope, members), None)

    def renames(self):
        return dict(self.renamed)


class Reports:
    def __init__(self):
        self.lines = []

    def formed(self, generation, group, seconds):
        self.lines.append(("formed", generation, group.key, seconds > 0))

    def failed(self, generation, group, error):
        self.lines.append(("failed", generation, group.key, error))

    def released(self, generation, scope, members):
        self.lines.append(("released", generation, (scope, members)))


def group(members, scope=DOMAIN, spans=None):
    """A set and the group of ``scope`` it needs: the group over ``spans``, the set's members
    unless said."""
    spans = tuple(members if spans is None else spans)
    return tuple(members), PlannedGroup(scope, spans, spans[0])


def plan(engine, generation, epoch, placement=None, entries=()):
    """A plan of ``entries``, (set, group) pairs: each group once, every set naming its own."""
    groups, sets = [], []
    for members, g in entries:
        if g not in groups:
            groups.append(g)
        sets.append(PlannedSet(members, (groups.index(g),)))
    return Plan(engine, generation, epoch, dict(placement or {}), tuple(groups), tuple(sets))


ME = 1
LIVE = (0, 1, 2)
SPARE = (0, "R1@node-d:0", 2)  # a set I am not in: rank 1's loss, node-d's spare in its place
LOSS_2 = (0, 1, "R2@node-e:0")


def mine(g):
    return ME in g.spans


def settle(cover, count, reports, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(reports.lines) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"{len(reports.lines)} reports of {count}: {reports.lines}")


def made(formed):
    def form(g, generation=None):
        formed.append(g.key)
        return f"comm {g.key}"

    return form


def test_a_plan_s_own_groups_are_formed_the_live_one_kept_and_others_not_mine_ignored():
    held = Held({(DOMAIN, LIVE): "live"}, serving={(DOMAIN, LIVE)})
    formed, released, reports = [], [], Reports()
    c = Cover(mine, held, made(formed), lambda *a: released.append(a), reports)
    p = plan(
        "rig", 1, 0, {}, (group(LIVE), group(SPARE, spans=(0, "R1@node-d:0", 2)), group(LOSS_2))
    )
    assert c.apply(p)
    settle(c, 2, reports)
    assert reports.lines[0] == ("formed", 1, (DOMAIN, LIVE), False)  # kept: reported, no time
    assert reports.lines[1] == ("formed", 1, (DOMAIN, LOSS_2), True)
    assert (
        formed == [(DOMAIN, LOSS_2)] and held.groups[(DOMAIN, LOSS_2)] == f"comm {(DOMAIN, LOSS_2)}"
    )
    assert released == []
    c.close()


def test_a_generation_at_or_below_the_one_applied_is_ignored():
    reports = Reports()
    c = Cover(mine, Held(), made([]), lambda *a: None, reports)
    assert c.apply(plan("rig", 3, 0))
    assert not c.apply(plan("rig", 3, 0)) and not c.apply(plan("rig", 2, 0))
    assert c.generation == 3
    c.close()


def test_a_held_group_the_plan_drops_is_released_but_never_the_one_served_on():
    held = Held({(DOMAIN, LIVE): "live", (DOMAIN, LOSS_2): "old"}, serving={(DOMAIN, LIVE)})
    released, reports = [], Reports()
    c = Cover(mine, held, made([]), lambda *a: released.append(a), reports)
    c.apply(plan("rig", 5, 1, {}, ()))  # a plan naming nothing of mine, by mistake
    assert released == [(DOMAIN, LOSS_2, "old")] and (DOMAIN, LIVE) in held.groups
    kinds = [(r[0], r[2]) for r in reports.lines]
    assert ("released", (DOMAIN, LOSS_2)) in kinds and ("failed", (DOMAIN, LIVE)) in kinds
    c.close()


def test_kept_groups_are_reported_again_at_every_new_generation():
    held = Held({(DOMAIN, LIVE): "live"}, serving={(DOMAIN, LIVE)})
    reports = Reports()
    c = Cover(mine, held, made([]), lambda *a: None, reports)
    c.apply(plan("rig", 1, 0, {}, (group(LIVE),)))
    c.apply(plan("rig", 2, 0, {}, (group(LIVE),)))
    assert reports.lines == [
        ("formed", 1, (DOMAIN, LIVE), False),
        ("formed", 2, (DOMAIN, LIVE), False),
    ]
    c.close()


def test_nothing_is_formed_between_a_change_and_its_commit():
    formed, reports = [], Reports()
    c = Cover(mine, Held(), made(formed), lambda *a: None, reports)
    c.pause()
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2),)))
    time.sleep(0.2)
    assert formed == []
    c.resume()
    settle(c, 1, reports)
    assert formed == [(DOMAIN, LOSS_2)]
    c.close()


def test_a_group_whose_plan_was_superseded_before_it_formed_is_not_formed():
    formed, reports = [], Reports()
    c = Cover(mine, Held(), made(formed), lambda *a: None, reports)
    c.pause()
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2),)))
    c.apply(plan("rig", 2, 0, {}, ()))
    c.resume()
    time.sleep(0.3)
    assert formed == [] and not any(e["kind"].startswith("formed") for e in c.events)
    c.close()


def test_a_group_dropped_while_it_formed_is_released_once_formed():
    gate, formed, released, reports = threading.Event(), [], [], Reports()

    def slow(g, generation=None):
        gate.wait(5)
        formed.append(g.key)
        return "comm"

    held = Held()
    c = Cover(mine, held, slow, lambda *a: released.append(a), reports)
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2),)))
    time.sleep(0.1)  # forming now
    c.apply(plan("rig", 2, 0, {}, ()))
    gate.set()
    settle(c, 1, reports)
    assert released == [(DOMAIN, LOSS_2, "comm")] and held.groups == {}
    assert reports.lines == [("released", 2, (DOMAIN, LOSS_2))]
    c.close()


def test_a_group_named_by_two_plans_before_it_formed_is_formed_once():
    gate, formed, reports = threading.Event(), [], Reports()

    def slow(g, generation=None):
        gate.wait(5)
        formed.append(g.key)
        return "comm"

    c = Cover(mine, Held(), slow, lambda *a: None, reports)
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2),)))
    c.apply(plan("rig", 2, 0, {}, (group(LOSS_2),)))
    gate.set()
    settle(c, 1, reports)
    time.sleep(0.2)
    assert formed == [(DOMAIN, LOSS_2)]
    assert reports.lines == [("formed", 2, (DOMAIN, LOSS_2), True)]  # at the newest generation
    c.close()


def test_a_failed_formation_is_reported_once_and_not_tried_again_by_the_process():
    tries, reports = [], Reports()

    def broken(g, generation=None):
        tries.append(g.key)
        raise TimeoutError("store timed out")

    c = Cover(mine, Held(), broken, lambda *a: None, reports)
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2, scope=FABRIC),)))
    settle(c, 1, reports)
    time.sleep(0.2)
    assert tries == [(FABRIC, LOSS_2)]
    assert reports.lines == [("failed", 1, (FABRIC, LOSS_2), "TimeoutError: store timed out")]
    c.close()


def test_a_held_set_naming_a_moved_position_as_a_rank_is_stale_and_a_spare_s_is_not():
    assert lost_members((0, 1, "R2@node-e:0"), [1])  # rank 1's old process is in it
    assert not lost_members((0, "R1@node-d:0", 2), [1])  # the taker, or another spare: not stale
    assert not lost_members((0, 2), [1])


def test_a_plan_at_a_newer_epoch_lets_go_of_groups_naming_a_moved_position():
    # a spare standing for any position: it holds (0, 1, R2) and runs no commit of its own
    stale, kept = (0, 1, "R2@node-e:0"), (0, "R1@node-e:0", 2)
    held = Held({(DOMAIN, stale): "with old rank 1", (DOMAIN, kept): "rank 1's cover"})
    released, reports = [], Reports()
    c = Cover(
        lambda g: {"R1@node-e:0", "R2@node-e:0"} & set(g.spans),
        held,
        made([]),
        lambda *a: released.append(a),
        reports,
    )
    a = {0: "node-a:0", 1: "node-b:0", 2: "node-c:0"}
    c.apply(plan("rig", 1, 0, a, (group(stale), group(kept))))
    moved = {**a, 1: "node-d:0"}  # rank 1 moved to node-d at epoch 1
    c.apply(plan("rig", 2, 1, moved, (group(stale),)))  # the same key, a new holder of rank 1
    assert released == [(DOMAIN, stale, "with old rank 1"), (DOMAIN, kept, "rank 1's cover")]
    assert ("formed", 2, (DOMAIN, stale), False) not in reports.lines  # never reported as kept
    c.close()


def test_a_plan_at_the_same_epoch_keeps_what_it_names():
    g1 = (0, 1, "R2@node-e:0")
    held = Held({(DOMAIN, g1): "comm"})
    released, reports = [], Reports()
    c = Cover(lambda g: True, held, made([]), lambda *a: released.append(a), reports)
    a = {0: "node-a:0", 1: "node-b:0", 2: "node-c:0"}
    c.apply(plan("rig", 1, 0, a, (group(g1),)))
    c.apply(plan("rig", 2, 0, a, (group(g1),)))
    assert released == [] and reports.lines[-1] == ("formed", 2, (DOMAIN, g1), False)
    c.close()


def test_a_plan_during_a_change_releases_nothing_until_its_commit():
    # the change moves to (0, R1@node-d:0, 2); a plan arriving before the boundary drops it
    target = (0, "R1@node-d:0", 2)
    held = Held({(DOMAIN, LIVE): "live", (DOMAIN, target): "target"}, serving={(DOMAIN, LIVE)})
    released, reports = [], Reports()
    c = Cover(lambda g: True, held, made([]), lambda *a: released.append(a), reports)
    c.pause()
    assert c.apply(plan("rig", 4, 1, {}, (group(LIVE),)))
    assert released == [] and reports.lines == [] and (DOMAIN, target) in held.groups
    # the boundary moves to it and the commit renames it: it serves now
    held.groups = {(DOMAIN, LIVE): "target"}
    c.resume()
    assert released == [] and reports.lines == [("formed", 4, (DOMAIN, LIVE), False)]
    c.close()


def test_the_newest_plan_at_the_commit_lets_go_of_what_it_drops():
    stale = (0, 1, "R2@node-e:0")
    held = Held({(DOMAIN, LIVE): "live", (DOMAIN, stale): "old"}, serving={(DOMAIN, LIVE)})
    released, reports = [], Reports()
    c = Cover(lambda g: True, held, made([]), lambda *a: released.append(a), reports)
    a = {0: "a", 1: "b", 2: "c"}
    c.apply(plan("rig", 1, 0, a, (group(LIVE), group(stale))))
    c.pause()
    c.apply(plan("rig", 2, 1, {**a, 1: "d"}, (group(LIVE), group(stale))))  # rank 1 moved
    assert released == []
    c.resume()  # the moved position's old group goes, and the plan's is formed anew
    assert released == [(DOMAIN, stale, "old")]
    settle(c, 5, reports)
    assert ("formed", 2, (DOMAIN, stale), True) in reports.lines
    c.close()


def test_after_a_commit_a_plan_for_the_epoch_it_ended_forms_nothing_until_the_next_plan():
    """The plan made at the change's start is still the newest when the commit resumes the
    cover; applied, rank 0 would form the set the commit has just made serving."""
    target = (0, "R1@node-d:0", 2)
    formed, released, reports = [], [], Reports()
    held = Held({(DOMAIN, LIVE): "live", (DOMAIN, target): "target"}, serving={(DOMAIN, LIVE)})
    c = Cover(lambda g: True, held, made(formed), lambda *a: released.append(a), reports)
    c.apply(plan("rig", 2, 0, {}, (group(LIVE), group(target))))
    settle(c, 2, reports)
    c.pause()
    # the boundary: the change moved to target, which serves now under the live set's key, its
    # spare renamed to the rank it took; the old live group, naming the lost rank, set aside and
    # released by the commit (worker.after_commit)
    held.groups = {(DOMAIN, LIVE): "target"}
    held.renamed = {"R1@node-d:0": 1}
    assert c.forget("live")
    reports.lines.clear()
    c.resume(1)
    time.sleep(0.2)
    assert formed == [] and released == [] and reports.lines == []
    # the coordinator's plan for epoch 1 is applied in full
    c.apply(plan("rig", 3, 1, {}, (group(LIVE),)))
    assert reports.lines == [("formed", 3, (DOMAIN, LIVE), False)] and formed == []
    c.close()


def test_an_abandoned_change_resumes_the_plan_it_paused():
    formed, reports = [], Reports()
    c = Cover(mine, Held(), made(formed), lambda *a: None, reports)
    c.apply(plan("rig", 1, 0, {}, ()))
    c.pause()
    c.apply(plan("rig", 2, 0, {}, (group(LOSS_2),)))
    c.resume()  # abandoned: nothing committed, the epoch-0 plan still stands
    settle(c, 1, reports)
    assert formed == [(DOMAIN, LOSS_2)]
    c.close()


def test_the_cover_s_events_are_written_to_its_record_as_they_happen(tmp_path):
    import json

    record = tmp_path / "events" / "cover_rig_node-a_0.json"
    reports = Reports()
    c = Cover(mine, Held(), made([]), lambda *a: None, reports, record=record)
    c.apply(plan("rig", 1, 0, {}, (group(LOSS_2),)))
    settle(c, 1, reports)
    events = json.loads(record.read_text())
    assert [(e["kind"], e["generation"], e["spans"]) for e in events] == [
        ("formed", 1, "0-1-R2@node-e:0")
    ]
    c.close()


def test_a_group_two_sets_need_is_formed_once_and_held_under_both():
    """Decision 0035, amended: a group is its scope and spans, whatever sets need it. Two loss
    sets that share rank 1's stage group (0, 1) form it once."""
    shared = PlannedGroup(DOMAIN, (0, 1), 0)
    a, b = (0, 1, "R2@node-e:0"), (0, 1, "R2@node-f:0")
    held, formed, reports = Held(), [], Reports()
    c = Cover(mine, held, made(formed), lambda *x: None, reports)
    c.apply(
        Plan(
            "rig",
            1,
            0,
            {},
            (shared,),
            (PlannedSet(a, (0,)), PlannedSet(b, (0,))),
        )
    )
    settle(c, 1, reports)
    assert formed == [(DOMAIN, (0, 1))]  # once
    assert held.groups[(DOMAIN, a)] is held.groups[(DOMAIN, b)]  # the same group under both
    assert reports.lines == [("formed", 1, (DOMAIN, (0, 1)), True)]
    c.close()


def test_a_shared_group_goes_only_with_the_last_set_that_needs_it():
    shared = PlannedGroup(DOMAIN, (0, 1), 0)
    a, b = (0, 1, "R2@node-e:0"), (0, 1, "R2@node-f:0")
    held, released, reports = Held(), [], Reports()
    c = Cover(mine, held, made([]), lambda *x: released.append(x), reports)
    c.apply(Plan("rig", 1, 0, {}, (shared,), (PlannedSet(a, (0,)), PlannedSet(b, (0,)))))
    settle(c, 1, reports)
    group = held.groups[(DOMAIN, a)]
    # node-e's spare is gone: its set leaves the plan, the group stays for node-f's
    c.apply(Plan("rig", 2, 0, {}, (shared,), (PlannedSet(b, (0,)),)))
    assert (DOMAIN, a) not in held.groups and held.groups[(DOMAIN, b)] is group
    assert released == []
    assert reports.lines[-1] == ("formed", 2, (DOMAIN, (0, 1)), False)  # kept
    # no set needs it any more: released, once, by its spans
    c.apply(Plan("rig", 3, 0, {}, (), ()))
    assert released == [(DOMAIN, (0, 1), group)] and held.groups == {}
    assert reports.lines[-1] == ("released", 3, (DOMAIN, (0, 1)))
    c.close()


def test_a_formation_past_its_deadline_is_reported_once_and_a_prompt_one_never():
    """A member that died after the go and before it connected leaves NCCL's blocking init
    waiting for good: the formation's thread is reported as left behind, once (0025)."""
    stuck_gate, stuck = threading.Event(), []

    def form(g, generation=None):
        if g.spans == (0, 1):
            stuck_gate.wait(5)  # a join whose peer never comes
        return f"comm {g.key}"

    held, reports = Held(), Reports()
    c = Cover(
        mine,
        held,
        form,
        lambda *x: None,
        reports,
        stuck_after_s=0.2,
        on_stuck=lambda name, run_s: stuck.append((name, run_s)),
    )
    c.apply(
        Plan(
            "rig",
            1,
            0,
            {},
            (PlannedGroup(DOMAIN, (0, 1), 0), PlannedGroup(DOMAIN, (0, 2), 0)),
            (PlannedSet((0, 1, "R2@node-e:0"), (0,)), PlannedSet((0, "R1@node-d:0", 2), (1,))),
        )
    )
    deadline = time.monotonic() + 3
    while not stuck and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)  # time for a second report, which must not come
    assert [name for name, _ in stuck] == [f"formation of {DOMAIN} 0-1"]
    assert stuck[0][1] >= 0.2
    stuck_gate.set()
    c.close()
