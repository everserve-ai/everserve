# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A build made ahead of time, where there is no GPU, is loaded only by a process it was built for:
the same torch, CUDA and sources. Anything else is built here, never loaded."""

import json

import pytest

from everserve import kernels


def test_a_build_without_a_gpu_names_its_architectures(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST"):
        kernels.build_into(tmp_path / "k")


def test_the_stamp_is_what_the_build_runs_against(tmp_path, monkeypatch):
    s = kernels.stamp()
    assert set(s) == {"torch", "cuda", "sources"} and len(s["sources"]) == 64
    # a build from other sources is another stamp: one byte of one kernel changed
    copied = tmp_path / "csrc"
    copied.mkdir()
    for name in (*kernels.SOURCES, "kernels.h"):
        (copied / name).write_bytes((kernels.sources_dir() / name).read_bytes())
    monkeypatch.setattr(kernels, "sources_dir", lambda: copied)
    assert kernels.stamp() == s  # the same sources, the same stamp
    (copied / "slot.cu").write_bytes((copied / "slot.cu").read_bytes() + b"\n// changed\n")
    assert kernels.stamp()["sources"] != s["sources"]


def test_a_named_prebuilt_that_cannot_be_loaded_is_refused_not_built_around(tmp_path, monkeypatch):
    monkeypatch.setenv(kernels.PREBUILT_ENV, str(tmp_path))
    with pytest.raises(kernels.KernelsUnavailable, match="no usable stamp"):
        kernels._prebuilt()
    (tmp_path / kernels.STAMP).write_text(json.dumps({**kernels.stamp(), "torch": "0.0"}))
    with pytest.raises(kernels.KernelsUnavailable, match="built against"):
        kernels._prebuilt()
    (tmp_path / kernels.STAMP).write_text(json.dumps({**kernels.stamp(), "archs": "9.0"}))
    monkeypatch.setattr(kernels, "device_arch", lambda: "8.9")
    with pytest.raises(kernels.KernelsUnavailable, match="this GPU"):
        kernels._prebuilt()
    monkeypatch.setattr(kernels, "device_arch", lambda: "9.0")
    with pytest.raises(kernels.KernelsUnavailable, match="no everserve_kernels library"):
        kernels._prebuilt()


def test_the_torch_paths_are_taken_by_choice_alone(monkeypatch):
    kernels.available.cache_clear()
    monkeypatch.setenv(kernels.PATHS_ENV, "torch")
    assert kernels.available() is False
    kernels.available.cache_clear()


def test_a_build_names_the_gpus_it_runs_on():
    assert kernels.built_archs("8.0;8.9;9.0+PTX;10.0") == {"8.0", "8.9", "9.0", "10.0"}
    assert kernels.built_archs("9.0 10.0") == {"9.0", "10.0"}


def test_a_prebuilt_build_is_not_loaded_on_a_gpu_it_was_not_built_for(tmp_path, monkeypatch):
    monkeypatch.setenv(kernels.PREBUILT_ENV, str(tmp_path))
    (tmp_path / kernels.STAMP).write_text(json.dumps({**kernels.stamp(), "archs": "9.0;10.0"}))
    (tmp_path / f"{kernels.NAME}.so").write_bytes(b"not loaded")
    monkeypatch.setattr(kernels, "device_arch", lambda: "8.9")  # an L40S
    with pytest.raises(kernels.KernelsUnavailable, match="this GPU is sm89"):
        kernels._prebuilt()


def test_a_prebuilt_directory_is_refused_unless_its_stamp_is_ours(tmp_path, monkeypatch):
    monkeypatch.setenv(kernels.PREBUILT_ENV, str(tmp_path))
    with pytest.raises(kernels.KernelsUnavailable, match="no usable stamp"):
        kernels._prebuilt()
    (tmp_path / kernels.STAMP).write_text(json.dumps({**kernels.stamp(), "torch": "0.0.1"}))
    with pytest.raises(kernels.KernelsUnavailable, match="built against"):
        kernels._prebuilt()  # another torch: a different binary interface
    (tmp_path / kernels.STAMP).write_text(json.dumps({**kernels.stamp(), "archs": "9.0"}))
    monkeypatch.setattr(kernels, "device_arch", lambda: None)
    with pytest.raises(kernels.KernelsUnavailable, match="library"):
        kernels._prebuilt()  # ours, but no library in it
    monkeypatch.delenv(kernels.PREBUILT_ENV)
    assert kernels._prebuilt() is None  # none named: built here
