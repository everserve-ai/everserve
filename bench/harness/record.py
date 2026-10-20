# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""What of a run's output is committed under ``bench/runs``, and what is scrubbed from it.

A run writes everything it sees into its output directory: the driver's log, every pod's and
process's stdout, the rows it backed up, the API reads it acted on. A committed record keeps what
a reader needs to check the result and drops what only a debugger of that run needed, so the
tree a clone pays for holds the evidence and not the dumps. The rule is one function,
:func:`kept`, and the check in ``bench/harness/tests/test_record.py`` fails when a tracked file
breaks it, so the rule in ``bench/runs/README.md`` is the rule the tree obeys.

A record is also read by people outside the account that ran it: :func:`scrub` replaces the
paths of the machine that drove the run, its temporary directories, the registry the images were
pushed to, the cloud project, and public addresses, each with a placeholder that says what stood
there.

Usage: ``python -m record RUN_DIR...`` curates and scrubs each directory in place and prints what
it removed; ``--check RUN_DIR...`` only reports what breaks the rule, exit 1 if anything does.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

SMALL_BYTES = 256 * 1024
"""A data file (JSON, text) is kept when it is at most this large: a read, a summary, a verdict."""

LOG_BYTES = 2 * 1024 * 1024
"""The driver's own log is kept when it is at most this large."""

KEPT_NAMES = frozenset({"results.json", "predictions.json", "summary.json", "verdict.json"})
"""Kept whatever their size: the predictions, and what the run reported against them."""

KEPT_LOGS = frozenset({"run.log", "deploy.log"})
"""The driver's and the deploy's own logs: what the harness did and when, not what a pod printed."""

KEPT_SUFFIXES = frozenset({".md", ".jsonl", ".csv", ".sha256"})
"""Prose, the harness's action log, measurements and digests are kept whatever their size."""

SMALL_SUFFIXES = frozenset({".json", ".txt", ".yaml", ".yml"})
"""Kept when small and not a dump: an API read, a captured list, a policy, a load record."""

DUMPS = re.compile(r"^(result|control|backup_.*|rank\d+_(holder|replica)|.*_rows)\.json$")
"""Per-session outputs and backed-up KV rows: the summary beside them is what the result reads."""

DROPPED_DIRS = re.compile(r"^(out|data\d*|captures|logs)$")
"""Pod and process stdout, raw data and captures: a debugger's, never the result's."""

DROPPED_SUFFIXES = frozenset({".bak", ".kill", ".armed", ".start", ".err", ".log", ".pid"})
"""Working files of a run in progress, and every log but the driver's own."""


def kept(relative: str, size: int) -> bool:
    """Whether a file at ``relative`` (under the run's directory) of ``size`` bytes is committed."""
    parts = relative.split("/")
    name = parts[-1]
    suffix = os.path.splitext(name)[1]
    if any(DROPPED_DIRS.match(p) for p in parts[:-1]):
        return False
    if name in KEPT_NAMES:
        return True
    if name in KEPT_LOGS:
        return size <= LOG_BYTES
    if suffix in DROPPED_SUFFIXES:
        return False
    if suffix in KEPT_SUFFIXES:
        return True
    if suffix in SMALL_SUFFIXES:
        return size <= SMALL_BYTES and not DUMPS.match(name)
    return False


_SCRUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    # the checkout that drove the run, and any worktree of it, before the home directory itself;
    # a node's own users (/home/everserve, /home/dynamo) are the deployment's, not the account's
    (
        re.compile(r"/Users/[^/\s\"'`]+/workspace/everserve(?:/\.claude/worktrees/[^/\s\"'`]+)?"),
        "<tree>",
    ),
    (re.compile(r"/Users/[^/\s\"'`]+"), "$HOME"),
    # macOS temporary directories
    (re.compile(r"(?:/private)?/var/folders/[^\s\"'`]*?/T/+"), "$TMPDIR/"),
    # the container registry the images were pushed to, and the cloud project
    (
        re.compile(r"cr\.eu-north1\.nebius\.cloud/e00[a-z0-9]+"),
        "cr.eu-north1.nebius.cloud/<registry>",
    ),
    (re.compile(r"\bproject-e0[0-9a-z]{16,}"), "project-<id>"),
    (re.compile(r"\btenant-e0[0-9a-z]{16,}"), "tenant-<id>"),
    # a public address where the record names one: a VM's public IP, an ssh target, a URL
    (
        re.compile(
            r'(?P<lead>"public_ip": "|@|://|\bssh |--master-addr[ =])'
            r"(?!10\.|127\.|0\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)"
            r"(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
        ),
        r"\g<lead><public-ip>",
    ),
)
"""Each pattern and its placeholder, applied in order."""


def scrub(text: str) -> str:
    """``text`` with every path, id and address of the account that ran it replaced."""
    for pattern, placeholder in _SCRUBS:
        text = pattern.sub(placeholder, text)
    return text


def unscrubbed(text: str) -> list[str]:
    """What :func:`scrub` would replace in ``text``, one match each."""
    return [m.group(0) for pattern, _ in _SCRUBS for m in pattern.finditer(text)]


def files_under(root: Path) -> Iterator[tuple[str, Path]]:
    """Every file under ``root``, as its path relative to ``root`` and its absolute path."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path.relative_to(root).as_posix(), path


def check(root: Path, relatives: Iterable[str] | None = None) -> list[str]:
    """What breaks the rule under ``root``: a file the rule drops, or text the scrub replaces.

    ``relatives`` limits the check to those files (the tracked ones, for the test).
    """
    broken: list[str] = []
    pairs = files_under(root) if relatives is None else ((r, root / r) for r in sorted(relatives))
    for relative, path in pairs:
        if not kept(relative, path.stat().st_size):
            broken.append(f"{relative}: not a kept kind or too large")
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            broken.append(f"{relative}: unreadable: {e}")
            continue
        for found in sorted(set(unscrubbed(text)))[:3]:
            broken.append(f"{relative}: holds {found!r}")
    return broken


def curate(root: Path) -> list[str]:
    """Remove every file the rule drops under ``root`` and scrub the rest; the removed paths."""
    removed: list[str] = []
    for relative, path in list(files_under(root)):
        if not kept(relative, path.stat().st_size):
            path.unlink()
            removed.append(relative)
            continue
        text = path.read_text(errors="replace")
        scrubbed = scrub(text)
        if scrubbed != text:
            path.write_text(scrubbed)
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return removed


def main(argv: list[str] | None = None) -> int:
    """The command line: curate (default) or ``--check`` each run directory given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories under bench/runs")
    parser.add_argument(
        "--check", action="store_true", help="report what breaks the rule; change nothing"
    )
    args = parser.parse_args(argv)
    broken_any = False
    for run in args.runs:
        if not run.is_dir():
            print(f"{run}: not a directory", file=sys.stderr)
            return 2
        if args.check:
            broken = check(run)
            broken_any |= bool(broken)
            for line in broken:
                print(f"{run}/{line}")
        else:
            removed = curate(run)
            print(f"{run}: removed {len(removed)} files")
    return 1 if broken_any else 0


if __name__ == "__main__":
    sys.exit(main())
