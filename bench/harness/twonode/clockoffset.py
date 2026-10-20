# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The clock offset between two nodes, so a cross-node span in a run record carries its error.

NTP's four timestamps, over one TCP connection (a rented VPC can pass no UDP between its VMs,
leaving every probe unanswered; TCP between the nodes is what the stores already use): the
client stamps its send (t1), the server its receive
(t2) and its reply (t3), the client the reply's arrival (t4). The offset of the server's clock
against the client's is ((t2 - t1) + (t3 - t4)) / 2 and the round trip (t4 - t1) - (t3 - t2);
the offset's error is at most half the round trip. The best of several exchanges (the shortest
round trip) is reported, as NTP does.

    clockoffset.py serve [--port 7410]
    clockoffset.py probe HOST [--port 7410] [--n 20]   -> one JSON line

A run records the probe from node-a to every other node at its start and at its end; the two
readings bound the drift over the run.
"""

import argparse
import json
import socket
import struct
import sys
import time

FMT = "!dd"  # two float64 seconds
PORT = 7410


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """``n`` bytes from ``sock``; empty when the peer closed first."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def serve(port: int) -> None:
    """Answer every 8-byte stamp on each connection with the receive and reply times here."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(8)
    while True:
        conn, _ = listener.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = _recv_exactly(conn, 8)
                t2 = time.time()
                if not data:
                    break
                conn.sendall(data + struct.pack(FMT, t2, time.time()))


def probe(host: str, port: int, n: int, timeout_s: float = 1.0) -> dict:
    """The offset of ``host``'s clock against this one, from the exchange with the shortest trip."""
    best: dict | None = None
    lost = 0
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as e:
        return {"host": host, "at": time.time(), "probes": n, "lost": n, "error": str(e)}
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for i in range(n):
        t1 = time.time()
        try:
            sock.sendall(struct.pack("!d", t1))
            data = _recv_exactly(sock, 24)
        except OSError:
            data = b""
        t4 = time.time()
        if len(data) < 24:
            lost = n - i  # the connection is gone: this exchange and the rest are lost
            break
        (sent,) = struct.unpack("!d", data[:8])
        if sent != t1:
            continue  # not this exchange's reply
        t2, t3 = struct.unpack(FMT, data[8:24])
        rtt = (t4 - t1) - (t3 - t2)
        offset = ((t2 - t1) + (t3 - t4)) / 2
        if best is None or rtt < best["rtt_s"]:
            best = {"offset_s": offset, "rtt_s": rtt}
        time.sleep(0.02)
    out = {"host": host, "at": time.time(), "probes": n, "lost": lost}
    if best is None:
        out["error"] = "no reply"
    else:
        out.update(
            offset_s=round(best["offset_s"], 6),
            rtt_s=round(best["rtt_s"], 6),
            error_s=round(best["rtt_s"] / 2, 6),
        )
    return out


def main(argv: list[str]) -> int:
    """The entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=PORT)
    p = sub.add_parser("probe")
    p.add_argument("host")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv[1:])
    if a.cmd == "serve":
        serve(a.port)
        return 0
    print(json.dumps(probe(a.host, a.port, a.n)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
