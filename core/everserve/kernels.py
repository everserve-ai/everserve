# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The Everserve CUDA kernels: build, load, and say whether they are usable here.

The sources are ``core/csrc`` (shipped in the wheel as ``everserve/_csrc``). They are built as the
torch extension ``everserve_kernels`` against the torch and CUDA toolkit installed beside the
engine, because the engine's wheel pins its own torch build and an extension must match it
exactly. torch caches the build by source hash and toolchain, so:

* an install builds it once ahead of time with ``python -m everserve.kernels``;
* each worker process then loads the cached build without compiling.

An image is built where there is no GPU, and run where there is no compiler: ``python -m
everserve.kernels --into DIR`` builds it there with the CUDA toolkit alone, for the architectures
``TORCH_CUDA_ARCH_LIST`` names, and stamps it with what it was built against (torch, CUDA, the
sources). A process whose ``EVERSERVE_KERNELS_DIR`` names that directory loads the build only when
the stamp is its own; any other build would be a different binary interface, and is never loaded.

A process with ``EVERSERVE_KERNELS_DIR`` set loads that build or refuses: a stamp that is not
its own, a GPU the build is not for, or a directory without the library is a deployment error
(:class:`KernelsUnavailable`), never a reason to build or to serve on torch paths, since one rank
on other kernels than its peers' is an engine whose detection and determinism claims
(decisions 0006, 0012, 0021) no longer hold. The torch paths are taken only by choice:
``EVERSERVE_KERNELS=torch``, which :func:`available` reads, or on a host without CUDA. This
module is for loading the extension, not for choosing between kernel variants; that is each
caller's decision.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path
from types import ModuleType

log = logging.getLogger(__name__)

NAME = "everserve_kernels"
SOURCES = ("kernels.cpp", "gather.cu", "slot.cu")
PREBUILT_ENV = "EVERSERVE_KERNELS_DIR"
"""A directory holding a build made ahead of time (:func:`build_into`), loaded when its stamp is
this process's and refused otherwise."""
PATHS_ENV = "EVERSERVE_KERNELS"
"""``torch`` to serve on the torch paths without the kernels, by choice; anything else, or unset,
loads them."""
STAMP = "stamp.json"


class KernelsUnavailable(RuntimeError):
    """The kernels cannot be loaded here, and why: a wrong prebuilt, no CUDA, a failed build."""


def sources_dir() -> Path:
    """Where the kernel sources are: inside the installed package, else the repository."""
    packaged = Path(__file__).parent / "_csrc"
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[1] / "csrc"


def stamp() -> dict[str, str]:
    """What a build is built against: torch, its CUDA, and the sources, by content."""
    import torch

    digest = hashlib.sha256()
    src = sources_dir()
    for name in (*SOURCES, "kernels.h"):
        digest.update(name.encode())
        digest.update((src / name).read_bytes())
    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "sources": digest.hexdigest(),
    }


def _compile(build_directory: str | None, verbose: bool) -> ModuleType:
    from torch.utils import cpp_extension

    src = sources_dir()
    return cpp_extension.load(
        name=NAME,
        sources=[str(src / s) for s in SOURCES],
        extra_include_paths=[str(src)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=build_directory,
        verbose=verbose,
    )


def build_into(directory: Path, verbose: bool = False) -> Path:
    """Build the extension into ``directory`` with the CUDA toolkit alone, and stamp it.

    No GPU is needed: the architectures are ``TORCH_CUDA_ARCH_LIST``'s, which must be set.

    Raises:
        RuntimeError: without ``TORCH_CUDA_ARCH_LIST``, or when the build fails.
    """
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if not archs:
        raise RuntimeError("a build without a GPU names its architectures: TORCH_CUDA_ARCH_LIST")
    directory.mkdir(parents=True, exist_ok=True)
    mod = _compile(str(directory), verbose)
    (directory / STAMP).write_text(json.dumps({**stamp(), "archs": archs}, indent=2) + "\n")
    return Path(mod.__file__)


def built_archs(archs: str) -> set[str]:
    """The compute capabilities a ``TORCH_CUDA_ARCH_LIST`` builds for (``"8.9;9.0+PTX"``: both)."""
    return {a.strip().removesuffix("+PTX") for a in archs.replace(" ", ";").split(";") if a.strip()}


def device_arch() -> str | None:
    """This process's GPU's compute capability (``"8.9"``), or None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _prebuilt() -> ModuleType | None:
    """The build made ahead of time, when ``EVERSERVE_KERNELS_DIR`` names one; None if unset.

    Raises:
        KernelsUnavailable: the directory holds no usable stamp, a stamp that is not this
            process's, a build for other GPUs than this one, or no library.
    """
    import importlib.machinery
    import importlib.util

    directory = os.environ.get(PREBUILT_ENV, "")
    if not directory:
        return None
    built = Path(directory)
    try:
        theirs = json.loads((built / STAMP).read_text())
    except (OSError, ValueError) as e:
        raise KernelsUnavailable(f"{PREBUILT_ENV}={built}: no usable stamp ({e})") from e
    ours = stamp()
    if any(theirs.get(k) != v for k, v in ours.items()):
        raise KernelsUnavailable(
            f"{PREBUILT_ENV}={built}: built against {theirs}, this process is {ours}"
        )
    here = device_arch()
    if here is not None and here not in built_archs(theirs.get("archs", "")):
        raise KernelsUnavailable(
            f"{PREBUILT_ENV}={built}: built for {theirs.get('archs')}, "
            f"this GPU is sm{here.replace('.', '')}"
        )
    so = next(iter(sorted(built.glob(f"{NAME}*.so"))), None)
    if so is None:
        raise KernelsUnavailable(f"{PREBUILT_ENV}={built}: stamped, but no {NAME} library in it")
    loader = importlib.machinery.ExtensionFileLoader(NAME, str(so))
    spec = importlib.util.spec_from_file_location(NAME, so, loader=loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


_SOURCE: str = ""
"""How this process got its kernels, once loaded: ``prebuilt`` or ``built``."""


@functools.lru_cache(maxsize=1)
def load(verbose: bool = False) -> ModuleType:
    """Import the extension: the build made ahead of time where one is named, else built here.

    A build here is made once and cached by torch. The result is cached for the process: the
    extension is a process-wide resource, like the CUDA context it runs in.

    Raises:
        KernelsUnavailable: without CUDA, when a named prebuilt cannot be loaded, or when the
            build here fails.
    """
    global _SOURCE
    import torch

    if not torch.cuda.is_available():
        raise KernelsUnavailable("the Everserve kernels need CUDA")
    prebuilt = _prebuilt()
    if prebuilt is not None:
        _SOURCE = "prebuilt"
        return prebuilt
    try:
        built = _compile(None, verbose)
    except Exception as e:  # noqa: BLE001 - torch's build raises many kinds; one word for all
        raise KernelsUnavailable(f"the Everserve kernels did not build here: {e}") from e
    _SOURCE = "built"
    return built


def source() -> str:
    """How this process got its kernels: ``prebuilt``, ``built``, or ``""`` before they loaded."""
    return _SOURCE


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """Whether this process serves on the kernels: False on a host without CUDA, or by choice.

    ``EVERSERVE_KERNELS=torch`` takes the torch paths without loading anything, said once in
    the log. Any other reason the kernels cannot load raises, so a rank never serves on other
    paths than its peers' without an operator having said so.

    Raises:
        KernelsUnavailable: a named prebuilt that cannot be loaded, or a build here that failed.
    """
    import torch

    if os.environ.get(PATHS_ENV, "") == "torch":
        log.info("%s=torch: the torch paths serve, the kernels are not loaded", PATHS_ENV)
        return False
    if not torch.cuda.is_available():
        return False
    load()
    return True


def main(argv: list[str] | None = None) -> None:
    """Build the extension now, so worker processes only load it."""
    import argparse

    ap = argparse.ArgumentParser(prog="everserve.kernels", description=main.__doc__)
    ap.add_argument(
        "--into",
        type=Path,
        help="build ahead of time into this directory, without a GPU (TORCH_CUDA_ARCH_LIST)",
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.into is not None:
        log.info("built %s at %s", NAME, build_into(a.into, verbose=True))
        return
    mod = load(verbose=True)
    log.info("built %s at %s", NAME, mod.__file__)


if __name__ == "__main__":
    main()
