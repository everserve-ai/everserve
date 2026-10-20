# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
import socket
import threading

import pytest

from everserve.wire import MAX_FRAME, FrameError, read_frame, write_frame


def pair():
    a, b = socket.socketpair()
    return a, b


def test_frames_round_trip_including_empty():
    a, b = pair()
    for payload in (b"", b"x", bytes(range(256)) * 100):
        # the writer runs on its own thread: a frame larger than the socket buffer would
        # otherwise block before the reader starts
        t = threading.Thread(target=write_frame, args=(a, payload))
        t.start()
        assert read_frame(b) == payload
        t.join()


def test_partial_reads_are_reassembled():
    a, b = pair()
    payload = bytes(range(256)) * 1000

    def slow_writer():
        import struct

        data = struct.pack(">I", len(payload)) + payload
        for i in range(0, len(data), 777):
            a.sendall(data[i : i + 777])

    t = threading.Thread(target=slow_writer)
    t.start()
    assert read_frame(b) == payload
    t.join()


def test_closed_peer_is_eof():
    a, b = pair()
    a.close()
    with pytest.raises(EOFError):
        read_frame(b)


def test_oversized_frames_are_rejected_both_ways():
    a, b = pair()
    with pytest.raises(FrameError):
        write_frame(a, b"x" * (MAX_FRAME + 1))
    import struct

    a.sendall(struct.pack(">I", MAX_FRAME + 1))
    with pytest.raises(FrameError):
        read_frame(b)
