# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Triton's driver helper built in place in every staged inductor Triton cache, after a staging
and off the take's path: each directory found, each built in its own process with its cache
pointed there, and what it gained recorded; a driver that cannot initialise fails the step."""

import stat

import triton_helper


def stand_in(tmp_path, body):
    """An interpreter that ignores its program and runs ``body`` (a shell line)."""
    p = tmp_path / "python"
    p.write_text(f"#!/bin/sh\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def staged(tmp_path):
    root = tmp_path / "torch_compile_cache"
    for key in ("13aba33f7a", "57f7aebbdc"):
        d = root / "torch_aot_compile" / key / "inductor_cache" / "triton" / "0" / "AB12"
        d.mkdir(parents=True)
        (d / "kernel.cubin").write_bytes(b"k")
    (root / "torch_aot_compile" / "13aba33f7a" / "inductor_cache" / "fxgraph").mkdir()
    return root


def test_every_staged_triton_directory_is_found_and_nothing_else(tmp_path):
    dirs = triton_helper.triton_dirs(staged(tmp_path))
    assert [d.parts[-4] for d in dirs] == ["13aba33f7a", "57f7aebbdc"]
    assert all(d.name == "0" and d.parent.name == "triton" for d in dirs)


def test_the_helper_is_built_where_the_take_would_look_for_it(tmp_path):
    d = triton_helper.triton_dirs(staged(tmp_path))[0]
    py = stand_in(
        tmp_path,
        'mkdir -p "$TRITON_CACHE_DIR/MIH4X" && : > "$TRITON_CACHE_DIR/MIH4X/cuda_utils.so"',
    )
    r = triton_helper.build(d, python=py)
    assert r["ok"] and r["gained"] == ["MIH4X/cuda_utils.so"]
    assert triton_helper.build(d, python=py)["gained"] == []  # built once, found after


def test_a_driver_that_cannot_initialise_fails_the_step(tmp_path):
    d = triton_helper.triton_dirs(staged(tmp_path))[0]
    r = triton_helper.build(d, python=stand_in(tmp_path, "echo 'no CUDA driver' >&2; exit 1"))
    assert not r["ok"] and "no CUDA driver" in r["error"]
