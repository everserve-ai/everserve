# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""Long streaming completions through an OpenAI-compatible front end, every token timed.

The Dynamo proof's load: ``--streams`` concurrent chat completions of ``--tokens`` tokens each,
streamed, each token's arrival recorded. While they run, the proof takes a node of the instance
away; a stream that survives it shows a pause where the replacement was and no error. Writes one
JSON record: per stream its tokens, its first and last token's times, its largest gap between
tokens and when it fell, its error if it ended in one, and the workers Dynamo says served it
(``nvext.worker_id``, which :mod:`served` names by pod). ``--worker-id`` sends every stream to one
worker instance (Dynamo's ``x-dynamo-worker-instance-id``), as a head that took over is shown
serving.

    stream_load.py --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct --streams 8 \
        --tokens 1500 --out load.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request


def workers_of(chunk: dict) -> list[int]:
    """The worker ids a chunk's ``nvext.worker_id`` names: its prefill and decode workers."""
    info = (chunk.get("nvext") or {}).get("worker_id") or {}
    ids = [info.get("prefill_worker_id"), info.get("decode_worker_id")]
    return [w for w in ids if isinstance(w, int)]


def stream(url: str, model: str, tokens: int, i: int, out: dict, worker: int | None = None) -> None:
    """One streamed completion, to ``worker`` alone when given; its record in ``out[i]``."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Stream {i}: write a long, detailed history of computing, "
                    "chapter by chapter, and do not stop early.",
                }
            ],
            "max_tokens": tokens,
            "min_tokens": tokens,  # the stream outlives the fault, whatever the model would say
            "temperature": 0.0,
            "stream": True,
            # the tokens generated, as the server counts them: a chunk can carry no text
            "stream_options": {"include_usage": True},
            # which workers served it, a migrated stream's each
            "nvext": {"extra_fields": ["worker_id"]},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if worker is not None:
        headers["x-dynamo-worker-instance-id"] = str(worker)
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    rec: dict = {
        "tokens": 0,
        "start": time.time(),
        "gap_s": 0.0,
        "gap_at": None,
        "error": None,
        "workers": [],
    }
    out[i] = rec
    last = None
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    rec["error"] = chunk["error"]
                    break
                usage = chunk.get("usage") or {}
                if "completion_tokens" in usage:
                    rec["usage_tokens"] = usage["completion_tokens"]
                for w in workers_of(chunk):
                    if w not in rec["workers"]:
                        rec["workers"].append(w)
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if not delta:
                    continue
                now = time.time()
                if last is not None and now - last > rec["gap_s"]:
                    rec["gap_s"], rec["gap_at"] = round(now - last, 3), round(last, 3)
                rec["first"] = rec.get("first", round(now, 3))
                rec["tokens"] += 1
                last = now
    except Exception as e:  # noqa: BLE001 - a broken stream is what the proof looks for
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["last"] = round(last, 3) if last is not None else None
    rec["end"] = round(time.time(), 3)
    if rec["error"] and last is not None and rec["end"] - last > rec["gap_s"]:
        # a stream that stalled and then failed: its last silence is its worst gap, though no
        # token closed it (28 s of silence before a 503 would otherwise read as an 82 ms gap)
        rec["gap_s"], rec["gap_at"] = round(rec["end"] - last, 3), round(last, 3)
    # the chunks that carried text, for the gaps; the server's count, when it gave one, for
    # whether the stream finished
    rec["text_chunks"] = rec["tokens"]
    rec["tokens"] = rec.get("usage_tokens", rec["tokens"])


def main() -> int:
    """Run the streams, write the record; exit 1 unless every stream finished its tokens.

    A stream that ended short of its tokens without an error (a front end that closed it with
    ``[DONE]`` and nothing else) is not a stream that survived: the record names it, and the
    exit says so.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-id", type=int, help="send every stream to this worker instance")
    a = ap.parse_args()
    out: dict = {}
    threads = [
        threading.Thread(target=stream, args=(a.url, a.model, a.tokens, i, out, a.worker_id))
        for i in range(a.streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record = {
        "streams": a.streams,
        "tokens": a.tokens,
        "completed": sum(1 for r in out.values() if not r["error"] and r["tokens"] >= a.tokens),
        "errors": [r["error"] for r in out.values() if r["error"]],
        # streams that ended with no error and fewer tokens than asked: not survivors
        "short": sorted(i for i, r in out.items() if not r["error"] and r["tokens"] < a.tokens),
        "max_gap_s": max((r["gap_s"] for r in out.values()), default=0.0),
        "per_stream": out,
    }
    with open(a.out, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k != "per_stream"}))
    return 0 if record["completed"] == a.streams else 1


if __name__ == "__main__":
    sys.exit(main())
