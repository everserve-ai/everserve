# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A daemon's spare and its node's slot region: made by the executor on an engine node, by the
spare itself on a node that only holds spares."""

from pathlib import Path

from everserve.comm.slotkernel import Region
from everserve.policy import load
from everserve_vllm.spare_main import prepare_region

POLICY = Path(__file__).parents[3] / "bench/harness/policies/three-node-tp1pp2-managed.json"


def test_on_a_spare_node_the_spare_creates_the_region_and_takes_its_column(tmp_path, monkeypatch):
    import everserve.comm.slotkernel as sk

    monkeypatch.setattr(
        sk,
        "decide_variant",
        lambda devices, cache=None: sk.VariantChoice("host", "nvml", f"devices {devices}"),
    )
    spec = load(str(POLICY)).engine("rig")
    path = str(tmp_path / "slot")
    row, choice = prepare_region(spec, "node-c", 1, path, 0.03)
    assert row == 0 and choice.variant == "host" and "devices ['0']" in choice.detail
    r = Region(path)
    assert r.order(0) == [0] and r.bound(0) == 0.03
    r.close()


def test_on_an_engine_node_the_spare_waits_for_the_executor_s_region(tmp_path):
    spec = load(str(POLICY)).engine("rig")
    path = str(tmp_path / "slot")
    Region.create(path, "host", replace=True).close()  # the executor's, already there
    row, choice = prepare_region(spec, "node-a", 1, path, 0.03)
    assert row == spec.tensor_parallel and choice is None


class Link:
    """A taken spare's link: the notices it sends, and the placement and policy it has heard."""

    def __init__(self, placement, notice_min_ms=2000):
        from everserve.policy import Loading, Policy

        self.placement, self.notices = placement, []
        self.policy = Policy("v1", (), 1, Loading.PRELOADED, 1, notice_min_ms)

    def notice(self, rank, deadline_ms):
        self.notices.append((rank, deadline_ms))

    def notice_lead_s(self, wanted_s):
        from everserve.contract.daemon import HearsEngine

        return HearsEngine.notice_lead_s(self, wanted_s)


def placed(epoch, node_of_rank_1):
    from everserve.contract.signals import EnginePlacement, RankPlacement

    return EnginePlacement(
        "proof", epoch, (RankPlacement(0, "node-a", 0, ""), RankPlacement(1, node_of_rank_1, 0, ""))
    )


def test_a_spare_that_took_its_rank_gives_it_notice_and_stops_once_a_change_moved_it():
    """A spare serving a rank, stopped by its pod's deletion with no notice, leaves the streams
    waiting on the loss (3.7 s). Its rank is noticed and moved first."""
    import threading
    import time

    from everserve_vllm.spare_main import leave_taken

    link = Link(placed(1, "node-c"))  # this spare, on node-c, holds rank 1 since epoch 1
    out = []
    t = threading.Thread(
        target=lambda: out.append(leave_taken(link, 1, "node-c", 5.0, 5.0, poll_s=0.01))
    )
    before = time.time()
    t.start()
    time.sleep(0.1)
    assert out == []  # still serving until the change
    ((rank, deadline_ms),) = link.notices
    assert rank == 1 and before + 4.9 <= deadline_ms / 1e3 <= time.time() + 5.1
    link.placement = placed(2, "node-b")  # the change at the deadline: rank 1 is node-b's spare's
    t.join(2)
    assert out == [True]


def test_a_taken_spare_whose_rank_never_moves_stops_at_its_bound():
    from everserve_vllm.spare_main import leave_taken

    link = Link(placed(1, "node-c"))
    assert leave_taken(link, 1, "node-c", 5.0, 0.1, poll_s=0.01) is False
    assert [r for r, _ in link.notices] == [1]


def test_a_taken_spare_s_notice_is_never_under_the_policy_s_minimum():
    """A notice under the policy's noticeMinMs is read as none, and the rank is lost when the
    process ends: the lead the spare was started with gives way to the policy it heard."""
    import time

    from everserve.policy import NOTICE_TRANSIT_S
    from everserve_vllm.spare_main import leave_taken

    link = Link(placed(1, "node-c"), notice_min_ms=8000)
    before = time.time()
    leave_taken(link, 1, "node-c", 5.0, 0.05, poll_s=0.01)
    ((_, deadline_ms),) = link.notices
    # whole milliseconds on the wire: the deadline may read up to 1 ms under its exact value
    assert deadline_ms / 1e3 >= before + 8.0 + NOTICE_TRANSIT_S - 0.001


def spare_leaving(close_only: bool) -> int:
    """A process standing as a spare on a leave pipe, its writer told to leave (or gone)."""
    import os
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    child = f"""
import faulthandler, time
from everserve_vllm import STACKS_SIGNAL
from everserve_vllm.spare_main import leave_on_pipe

leave_on_pipe({read_fd}, "node-c", 5.0, 1.0)  # standing, nothing taken: it stops on the leave
faulthandler.register(STACKS_SIGNAL, all_threads=True)  # the plugin, once vLLM loads it
print("standing", flush=True)
time.sleep(20)
"""
    p = subprocess.Popen([sys.executable, "-c", child], pass_fds=(read_fd,), stdout=subprocess.PIPE)
    os.close(read_fd)
    assert p.stdout is not None and p.stdout.readline() == b"standing\n"
    if not close_only:
        from everserve_vllm.node_main import LEAVE_BYTE

        os.write(write_fd, LEAVE_BYTE)
    os.close(write_fd)
    return p.wait(timeout=15)


def test_a_spare_leaves_on_its_pipe_whatever_the_plugin_registers_for_signals():
    """A leave sent as a signal is taken by the plugin's stack dump (faulthandler, registered
    once vLLM loads it in the spare), and the spare never leaves. The leave is a pipe the
    standing node hands the spare: nothing a library registers reaches it."""
    import signal

    assert spare_leaving(close_only=False) == -signal.SIGTERM


def test_a_spare_whose_node_process_is_gone_leaves():
    import signal

    assert spare_leaving(close_only=True) == -signal.SIGTERM
