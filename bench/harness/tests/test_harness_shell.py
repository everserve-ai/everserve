# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The harness's shell scripts: an apostrophe inside a parameter expansion opens a string.

``${PREDICTIONS:?... the run's pre-registration}`` opens a single-quoted string at the apostrophe
that runs to the next quote, lines on, inside a function's Python: every assignment between is
never made, the battery dies at its first use of one with the VMs up, and ``bash -n`` passes, the
quotes balancing.
"""

import re
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1]).rglob("*.sh"))
MESSAGE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?[?\-=+]([^}]*)\}")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_parameter_expansion_opens_a_quote_it_does_not_close(script):
    bad = [
        f"{script.name}:{n}: {m.group(0)}"
        for n, line in enumerate(script.read_text().splitlines(), 1)
        for m in MESSAGE.finditer(line)
        if m.group(1).count("'") % 2
    ]
    assert not bad, bad


def test_the_check_finds_what_broke_run_2():
    line = "P=${PREDICTIONS:?the predictions.json beside the run's pre-registration}"
    assert any(m.group(1).count("'") % 2 for m in MESSAGE.finditer(line))


PLAIN_API = re.compile(r"http://[^ \"']*/v1/")


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_script_calls_the_operator_api_in_plain_http(script):
    """The operator API answers only over TLS, to an operator's certificate (decision 0039); a plain
    call is refused, and on a rental it would be found only when the call is first made."""
    bad = [
        f"{script.name}:{n}: {line.strip()[:100]}"
        for n, line in enumerate(script.read_text().splitlines(), 1)
        if PLAIN_API.search(line)
    ]
    assert not bad, bad
