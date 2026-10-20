# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What a pod makes in the node's shared directories every process of the node's group can use,
whatever its umask: processes of other pods map the slot region and the store's staging files."""

import os
import stat

from everserve import podcontract
from everserve.comm.slotkernel import Region


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_a_shared_file_is_the_group_s_whatever_the_umask(tmp_path):
    old = os.umask(0o077)  # the strictest a process might run with
    try:
        podcontract.make_shared(str(tmp_path / "f"), 4096)
    finally:
        os.umask(old)
    assert mode(tmp_path / "f") == podcontract.SHARED_MODE == 0o660
    assert (tmp_path / "f").stat().st_size == 4096


def test_the_slot_region_is_made_for_the_node_s_group(tmp_path):
    old = os.umask(0o022)  # an image's default
    try:
        r = Region(str(tmp_path / "slot"), create=True)
        r.close()
    finally:
        os.umask(old)
    assert mode(tmp_path / "slot") == 0o660


def test_an_engine_pod_never_runs_as_root_and_is_in_the_node_s_group():
    assert podcontract.security_context() == {
        "runAsNonRoot": True,
        "supplementalGroups": [podcontract.NODE_GID],
    }


def test_every_host_directory_is_mounted_at_its_host_path_and_the_cache_is_vllm_s():
    host = {v["name"]: v["hostPath"]["path"] for v in podcontract.volumes()}
    mounted = {m["name"]: m["mountPath"] for m in podcontract.mounts()}
    assert host == mounted  # the same path inside and out, so the daemon's records name it
    assert set(host.values()) == {
        podcontract.DAEMON_DIR,
        podcontract.SHM_DIR,
        podcontract.COMPILE_CACHE_DIR,
    }
    assert all(v["hostPath"]["type"] == "Directory" for v in podcontract.volumes())
    env = {e["name"]: e.get("value") for e in podcontract.env()}
    # vLLM compiles into the node's directory and loads from it (decision 0043)
    assert env["VLLM_CACHE_ROOT"] == podcontract.COMPILE_CACHE_DIR
