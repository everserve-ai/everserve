# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""A process's liveness and readiness over HTTP, for an orchestrator's probes.

A pod that runs one of this adapter's processes (``node_main``, ``spare_main``) under Kubernetes
is probed by its kubelet; under NVIDIA Dynamo the operator injects probes against Dynamo's own
system port, which these processes do not serve, so the pod's template names this server's
instead (decision 0041). ``GET /live`` answers 200 while the process makes progress: once its
serve loop has started beating (:meth:`Health.beat`), a beat older than ``stale_s`` answers 503,
so a process stuck with its probe thread still running (a call into CUDA that holds no lock) is
restarted; before the first beat, while the process loads, it answers 200 and the pod's startup
probe bounds the load. ``GET /ready`` answers 200 once the process is
what it was started to be (a node's workers loaded and joined, a spare standing) and 503 before,
so a pod is not counted ready while it loads, which can take minutes.
"""

from __future__ import annotations

import http.server
import logging
import threading
import time

log = logging.getLogger(__name__)


class Health:
    """The probe server: ``/live`` while the process makes progress, ``/ready`` once ready."""

    def __init__(
        self,
        port: int,
        host: str = "0.0.0.0",
        stale_s: float = 30.0,
    ) -> None:
        """Bind ``host:port`` (port 0 picks one; every interface, for the pod's kubelet)."""
        self.ready = threading.Event()
        self.stale_s = stale_s
        self._beat: float | None = None
        health = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
                if self.path == "/live":
                    if health.live():
                        self._answer(200, b"live\n")
                    else:
                        self._answer(503, b"no progress\n")
                elif self.path == "/ready":
                    if health.ready.is_set():
                        self._answer(200, b"ready\n")
                    else:
                        self._answer(503, b"not ready\n")
                else:
                    self._answer(404, b"not found\n")

            def _answer(self, code: int, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:  # probes every few seconds: not logged
                return

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(
            target=self._server.serve_forever, daemon=True, name="EverserveHealth"
        ).start()
        log.info("health on port %d: /live, /ready", self.port)

    def beat(self) -> None:
        """The serve loop made progress: called on each of its turns."""
        self._beat = time.monotonic()

    def live(self) -> bool:
        """Whether the process makes progress: no beat yet (loading), or one within stale_s."""
        beat = self._beat
        return beat is None or time.monotonic() - beat < self.stale_s

    def set_ready(self) -> None:
        """The process is what it was started to be: ``/ready`` answers 200 from now."""
        self.ready.set()

    def close(self) -> None:
        """Stop serving."""
        self._server.shutdown()
        self._server.server_close()
