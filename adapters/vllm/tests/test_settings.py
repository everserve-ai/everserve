# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""One typed object for how a process was launched, read from the environment once: what it
reads, that a child gets the same back, and that no other module of the adapter reads the
environment itself."""

import ast
from pathlib import Path

from everserve_vllm import settings


def test_the_environment_reads_into_one_typed_object_and_back():
    env = {
        "EVERSERVE_GRAPH_SAFE": "1",
        "EVERSERVE_ENGINE": "rig",
        "EVERSERVE_NODE": "node-c",
        "EVERSERVE_SPARE": "1",
        "EVERSERVE_SPARE_ANY": "0,1",
        "EVERSERVE_SPARE_RANK": "0",
        "EVERSERVE_LAYOUT": "1,2",
        "EVERSERVE_SLOT_ROW": "0",
        "EVERSERVE_DETECT_FLOOR_MS": "45",
        "CUDA_VISIBLE_DEVICES": "2, 3",
        "VLLM_HOST_IP": "10.0.0.10",
    }
    s = settings.read(env)
    assert s.graph_safe and s.engine == "rig" and s.node == "node-c" and s.spare
    assert s.spare_any == (0, 1) and s.spare_rank == 0 and s.layout == (1, 2) and s.row == 0
    assert s.detect_floor_ms == 45.0 and s.detect_steps == 3.0 and s.hang_s == 10.0
    assert s.visible_devices == ("2", "3") and s.host_ip == "10.0.0.10"
    back = settings.read(
        {**settings.environ_of(s), "CUDA_VISIBLE_DEVICES": "2,3", "VLLM_HOST_IP": "10.0.0.10"}
    )
    assert back == s
    empty = settings.read({})
    assert empty == settings.Settings() and empty.node == "local" and empty.spare_rank is None


def test_current_is_read_once_and_export_reaches_the_environment_children_inherit(monkeypatch):
    import os

    monkeypatch.setattr(settings, "_CURRENT", None)
    monkeypatch.setenv("EVERSERVE_ENGINE", "first")
    assert settings.current().engine == "first"
    monkeypatch.setenv("EVERSERVE_ENGINE", "later")
    assert settings.current().engine == "first"  # kept: read at the first use
    assert settings.update(row=3).row == 3 and "EVERSERVE_SLOT_ROW" not in os.environ
    monkeypatch.setenv("EVERSERVE_SLOT_REGION", "/dev/shm/x")
    s = settings.export(region="", layout=(2, 2))
    assert s.layout == (2, 2) and os.environ["EVERSERVE_LAYOUT"] == "2,2"
    assert "EVERSERVE_SLOT_REGION" not in os.environ and os.environ["EVERSERVE_SLOT_ROW"] == "3"
    assert os.environ["EVERSERVE_ENGINE"] == "first"  # export writes the whole object
    monkeypatch.setenv("EVERSERVE_ENGINE", "entry")
    assert settings.reload().engine == "entry"  # a process entry that set its environment late
    for key in [k for k in os.environ if k.startswith("EVERSERVE_")]:
        monkeypatch.delenv(key, raising=False)


def test_no_module_but_the_entry_points_touches_the_environment():
    """Library code reads no environment: the settings module reads it; the entry points and the
    plugin's register write process configuration; the spare launcher swaps the child's devices
    around a spawn. Nothing else names os.environ."""
    allowed = {"settings.py", "__init__.py", "node_main.py", "spare_main.py", "spare.py"}
    root = Path(settings.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
