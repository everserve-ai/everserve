# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process's probes: live while it runs, ready only once it is what it was started to be."""

import urllib.error
import urllib.request

import pytest

from everserve_vllm.health import Health


def status(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.fixture
def health():
    h = Health(0, host="127.0.0.1")
    yield h
    h.close()


def test_a_loading_process_is_live_and_not_ready_until_it_says_so(health):
    assert status(health.port, "/live") == 200
    assert status(health.port, "/ready") == 503  # loading: minutes, and not to be restarted
    health.set_ready()
    assert status(health.port, "/ready") == 200 and status(health.port, "/live") == 200


def test_an_unknown_path_is_not_found(health):
    assert status(health.port, "/metrics") == 404


def test_a_serve_loop_that_stops_beating_is_no_longer_live():
    h = Health(0, host="127.0.0.1", stale_s=0.2)
    try:
        assert status(h.port, "/live") == 200  # loading: no beat yet, the startup probe's to bound
        h.beat()
        assert status(h.port, "/live") == 200
        import time

        time.sleep(0.3)  # the loop is stuck: no beat past the bound
        assert status(h.port, "/live") == 503
        h.beat()
        assert status(h.port, "/live") == 200
    finally:
        h.close()
