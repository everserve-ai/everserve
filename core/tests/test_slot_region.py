# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The slot region's control path on the host: what the executor writes and every member reads."""

import pytest

from everserve.comm.slotkernel import MAXW, Region, choose_variant


def test_the_region_records_its_variant_and_carries_control_words_between_mappings(tmp_path):
    r = Region(str(tmp_path / "slot"), create=True, variant="p2p")
    r.set_members(range(4), range(4))
    r.set_members([4], [4])
    assert [r.order(i) for i in range(5)] == [[0, 1, 2, 3]] * 4 + [[4]]
    r.set_members([0, 1, 3, 4], [0, 1, 4, 3])  # rank 2 replaced by the spare's row, in its place
    assert r.order(4) == [0, 1, 4, 3] and r.order(2) == [0, 1, 2, 3]
    r.set_abort(range(MAXW), 1)
    r.publish(3, bytes(range(64)))
    other = Region(str(tmp_path / "slot"))  # another process's mapping of the same file
    assert other.variant == "p2p"
    assert other.order(4) == [0, 1, 4, 3] and int(other.ctrl[0, 1]) == 1
    assert other.handle(3, timeout_s=1) == bytes(range(64))
    with pytest.raises(TimeoutError):
        other.handle(2, timeout_s=0.05)  # never published
    other.close()
    r.close()


def test_an_order_with_a_row_twice_or_out_of_range_is_refused(tmp_path):
    r = Region(str(tmp_path / "slot"), create=True, variant="host")
    for bad in ([0, 1, 1], [0, MAXW]):
        with pytest.raises(ValueError):
            r.set_members([0], bad)
    r.close()


def test_a_region_from_before_the_member_order_is_refused(tmp_path):
    r = Region(str(tmp_path / "slot"), create=True, variant="host")
    r.header[0] = 0x45565253_4C4F5431  # "EVRSLOT1": the member mask layout
    with pytest.raises(ValueError):
        Region(str(tmp_path / "slot"))
    r.close()


def test_a_file_that_is_not_a_region_and_an_unknown_variant_are_refused(tmp_path):
    with pytest.raises(ValueError):
        Region(str(tmp_path / "slot"), create=True, variant="nvshmem")
    (tmp_path / "other").write_bytes(b"\0" * 8)
    with pytest.raises(ValueError):
        Region(str(tmp_path / "other"))


def test_without_nvml_the_variant_is_host_memory(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", None)  # an import of it raises ImportError
    assert choose_variant([0, 1]) == "host"


def test_the_variant_is_peer_to_peer_only_when_every_pair_reads_its_peer(monkeypatch):
    import sys
    import types

    def fake(blocked):
        m = types.ModuleType("pynvml")
        m.NVMLError = RuntimeError
        m.NVML_P2P_CAPS_INDEX_READ = 0
        m.NVML_P2P_STATUS_OK = 0
        m.nvmlInit = m.nvmlShutdown = lambda: None
        m.nvmlDeviceGetHandleByIndex = lambda i: i
        m.nvmlDeviceGetHandleByUUID = lambda u: int(u.removeprefix("GPU-"))
        m.nvmlDeviceGetP2PStatus = lambda a, b, cap: 1 if {a, b} == blocked else 0
        return m

    monkeypatch.setitem(sys.modules, "pynvml", fake(set()))
    assert choose_variant([0, 1, 2, 4], confirm=False) == "p2p"
    monkeypatch.setitem(sys.modules, "pynvml", fake({1, 4}))
    assert choose_variant([0, 1, 2, 4], confirm=False) == "host"
    # UUID entries name the device exactly and mix with indices, as CUDA_VISIBLE_DEVICES allows
    assert choose_variant(["GPU-0", "GPU-1", "2", 4], confirm=False) == "host"
    assert choose_variant(["GPU-0", "GPU-2", "GPU-3"], confirm=False) == "p2p"


def test_peer_reads_nvml_reports_must_also_work_when_the_collective_is_run(monkeypatch):
    import sys
    import types

    from everserve.comm import p2pprobe

    m = types.ModuleType("pynvml")
    m.NVMLError = RuntimeError
    m.NVML_P2P_STATUS_OK = 0
    m.nvmlInit = m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetHandleByUUID = lambda u: u
    m.nvmlDeviceGetP2PStatus = lambda a, b, cap: 0  # NVML: every pair reads its peer
    monkeypatch.setitem(sys.modules, "pynvml", m)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runs = []

    def fake_probe(devices, works):
        runs.append(list(devices))
        return p2pprobe.ProbeResult(works, "exact" if works else "no answer in 60s", 1.0)

    monkeypatch.setattr(p2pprobe, "probe", lambda d: fake_probe(d, True))
    assert choose_variant([0, 1, 2]) == "p2p"
    monkeypatch.setattr(p2pprobe, "probe", lambda d: fake_probe(d, False))
    assert choose_variant([0, 1, 2]) == "host", "NVML said yes; running it said no"
    assert runs == [[0, 1, 2], [0, 1, 2]]
    m.nvmlDeviceGetP2PStatus = lambda a, b, cap: 1  # NVML says no: nothing is run
    assert choose_variant([0, 1, 2]) == "host" and len(runs) == 2


def test_index_entries_warn_unless_cuda_orders_devices_by_bus(monkeypatch, caplog):
    import logging
    import sys
    import types

    m = types.ModuleType("pynvml")
    m.NVMLError = RuntimeError
    m.NVML_P2P_STATUS_OK = 0
    m.nvmlInit = m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetHandleByUUID = lambda u: u
    m.nvmlDeviceGetP2PStatus = lambda a, b, cap: 0
    monkeypatch.setitem(sys.modules, "pynvml", m)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    with caplog.at_level(logging.WARNING, logger="everserve.comm.slotkernel"):
        assert choose_variant([0, 1], confirm=False) == "p2p"
    assert "CUDA_DEVICE_ORDER" in caplog.text
    caplog.clear()
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    with caplog.at_level(logging.WARNING, logger="everserve.comm.slotkernel"):
        assert choose_variant([0, 1], confirm=False) == "p2p"
    assert caplog.text == ""
    assert choose_variant(["GPU-a", "not-a-number"]) == "host", "a bad entry is the safe variant"


def test_the_verdict_is_kept_for_the_same_gpus_and_driver_and_probed_again_after(
    monkeypatch, tmp_path
):
    import sys
    import types

    from everserve.comm import p2pprobe
    from everserve.comm.slotkernel import decide_variant

    m = types.ModuleType("pynvml")
    m.NVMLError = RuntimeError
    m.NVML_P2P_STATUS_OK = 0
    m.nvmlInit = m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: i
    m.nvmlDeviceGetHandleByUUID = lambda u: u
    m.nvmlDeviceGetP2PStatus = lambda a, b, cap: 0
    m.nvmlDeviceGetUUID = lambda h: f"GPU-{h}"
    m.nvmlSystemGetDriverVersion = lambda: "580.1"
    monkeypatch.setitem(sys.modules, "pynvml", m)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    probes = []

    def fake_probe(devices):
        probes.append(list(devices))
        return p2pprobe.ProbeResult(False, "no answer from devices [2] in 15.0s", 15.0)

    monkeypatch.setattr(p2pprobe, "probe", fake_probe)
    cache = tmp_path / "slot-variant.json"
    first = decide_variant([0, 1, 2], cache=cache)
    assert (first.variant, first.how) == ("host", "probe") and len(probes) == 1
    again = decide_variant([0, 1, 2], cache=cache)
    assert (again.variant, again.how, again.detail) == ("host", "cache", first.detail)
    assert len(probes) == 1, "the known box is not probed again"
    m.nvmlSystemGetDriverVersion = lambda: "590.2"  # a new driver: probed again
    assert decide_variant([0, 1, 2], cache=cache).how == "probe" and len(probes) == 2


def test_a_region_is_created_atomically_and_a_second_creator_opens_the_first_s(tmp_path):
    from everserve.comm.slotkernel import HEADER_WAIT_S  # noqa: F401 - the bound the loser waits

    path = str(tmp_path / "slot")
    first = Region.create(path, "p2p", replace=False)
    first.set_members([0, 1], [0, 1])
    second = Region.create(path, "host", replace=False)  # a second spare: the first one's stands
    assert second.variant == "p2p" and second.order(0) == [0, 1]
    replaced = Region.create(path, "host", replace=True)  # an executor's run: a fresh region
    assert replaced.variant == "host" and replaced.order(0) == []
    assert not list(tmp_path.glob("*.tmp"))
    for r in (first, second, replaced):
        r.close()


def test_an_extension_built_with_another_region_layout_is_refused():
    from types import SimpleNamespace

    from everserve.comm import slotkernel

    ours = SimpleNamespace(MAX_ROWS=8, CTRL_WORDS=16, ABORT=1, LATE=11, BOUND=12)
    slotkernel.check_layout(ours)
    with pytest.raises(RuntimeError, match="LATE"):
        slotkernel.check_layout(
            SimpleNamespace(MAX_ROWS=8, CTRL_WORDS=16, ABORT=1, LATE=10, BOUND=12)
        )
    with pytest.raises(RuntimeError, match="MAX_ROWS"):
        slotkernel.check_layout(SimpleNamespace())  # a build from before the export
