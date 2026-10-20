# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The communicator guard: shared holders overlap, the exclusive side waits for them and keeps
new ones out, and a communicator has one guard for as long as it lives."""

import threading
import time

from everserve.comm.guard import Guard, guard_of


class Comm:
    """Something weak-referenceable, as vLLM's communicator is."""


def test_shared_holders_overlap_and_the_exclusive_side_waits_for_them():
    g = Guard()
    inside = threading.Event()
    leave = threading.Event()

    def call():
        with g.shared():
            inside.set()
            leave.wait()

    t = threading.Thread(target=call)
    t.start()
    inside.wait()
    with g.shared():  # a second shared holder is not kept out by the first
        pass
    got = []

    def release():
        with g.exclusive():
            got.append(time.perf_counter())

    r = threading.Thread(target=release)
    r.start()
    time.sleep(0.2)
    assert not got, "the exclusive side ran under a shared holder"
    t0 = time.perf_counter()
    leave.set()
    r.join(timeout=2)
    assert got and got[0] >= t0
    t.join()


def test_the_exclusive_side_keeps_new_shared_holders_out_until_it_ends():
    g = Guard()
    held = threading.Event()
    done = threading.Event()

    def release():
        with g.exclusive():
            held.set()
            done.wait()

    r = threading.Thread(target=release)
    r.start()
    held.wait()
    entered = []

    def call():
        with g.shared():
            entered.append(time.perf_counter())

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.2)
    assert not entered, "a shared holder got in under the exclusive side"
    done.set()
    t.join(timeout=2)
    r.join()
    assert entered


def test_a_communicator_has_one_guard_for_its_life():
    a, b = Comm(), Comm()
    assert guard_of(a) is guard_of(a) and guard_of(a) is not guard_of(b)
