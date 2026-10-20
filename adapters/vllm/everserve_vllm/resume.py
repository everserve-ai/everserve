# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The resume after a change (decision 0013), apart from the executor so it imports without vLLM.

:func:`resume_running` keeps every running request running from its committed count, capped by
:func:`resume_point` so the last position is always recomputed, and returns the state every
worker re-adds the requests from. :class:`OneTokenReplay` holds the replay to one position per
step. The private vLLM lines they touch are listed in ``PRIVATE-INTERNALS.md`` (A11, A11a).
"""

from __future__ import annotations

from typing import Any


def resume_point(num_tokens: int, committed: int) -> int:
    """The computed length a running request resumes from: at most ``num_tokens - 1``.

    That is its committed count, capped. The engine's next token comes from computing the last
    token's position, so that position is always recomputed. A rank can have committed it
    already: on a notice the leaving rank drains every step it ran, and under async scheduling it
    ran the step for the last sampled token before the engine received that token's output.
    Resumed at the full count, the request would have nothing to compute and the engine would
    step empty batches forever. The recomputed position's rows are rewritten with the same
    values, by the same decode graph.
    """
    return min(committed, num_tokens - 1)


def resume_running(
    scheduler: Any, committed: dict[str, int], new_request_data: Any = None
) -> list[Any]:
    """Resume every running request from its committed count exactly (decision 0013).

    Each request stays running with its blocks: nothing is preempted, freed, evicted or looked
    up in the prefix cache, so no committed position is recomputed. Its computed length is set to
    ``committed[request]`` (capped by :func:`resume_point`) and the in-flight state of the steps
    that died is cleared; the next steps then recompute only the uncommitted tail, one position
    per step (see :class:`OneTokenReplay`), in decode shape on the captured graphs.

    A request absent from ``committed`` has no committed position: the replacement restored
    none of its rows (no store held them, as on a node with no replication target), so it
    resumes from zero and every position is recomputed. Resumed at its computed length
    instead, the spare serves its stage from empty rows and the outputs diverge from the
    control at the first token after the change.

    Returns, per request, the state every worker re-adds it from (``NewRequestData`` with its
    whole token list, blocks and the committed count), so the survivors drop what the aborted
    step left and the spare, which never saw the requests, holds the same state as they do.
    ``new_request_data`` builds that state (vLLM's ``NewRequestData.from_request`` by default).
    """
    if new_request_data is None:
        from vllm.v1.core.sched.output import NewRequestData

        new_request_data = NewRequestData.from_request
    km = scheduler.kv_cache_manager
    out = []
    for req in scheduler.running:
        req.num_computed_tokens = resume_point(
            req.num_tokens_with_spec,
            committed.get(req.request_id, 0),
        )
        req.num_output_placeholders = 0  # the steps they stood for died
        req.num_in_flight_tokens = 0
        req.num_stale_output_tokens = 0
        req.drop_stale_output = False
        req.spec_token_ids = []
        scheduler._inflight_prefills.discard(req)
        out.append(
            new_request_data(
                req, km.get_block_ids(req.request_id), prefill_token_ids=list(req.all_token_ids)
            )
        )
    return out


class OneTokenReplay:
    """Hold the replaying requests to one new token per step until each has caught up.

    With the one-step fence a request can be two positions past its committed count; scheduled
    as is, both would run as one two-token chunk, a shape no captured graph covers. While any
    replaying request is more than one position behind, the scheduler's per-request cap is one
    token and its step budget one token per request; both are restored once every replaying
    request has at most its next decode position left.
    """

    def __init__(self, scheduler: Any, ids: list[str]) -> None:
        """Wrap ``scheduler.schedule`` for the requests ``ids``."""
        self.scheduler, self.ids = scheduler, set(ids)
        self.schedule = scheduler.schedule
        scheduler.schedule = self

    def behind(self) -> bool:
        """Whether any replaying request still has more than one position to compute."""
        return any(
            r.num_tokens_with_spec - r.num_computed_tokens > 1
            for r in self.scheduler.running
            if r.request_id in self.ids
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """One scheduling step, capped while any replaying request is behind."""
        if not self.behind():
            self.scheduler.schedule = self.schedule
            return self.schedule(*args, **kwargs)
        s, cfg = self.scheduler, self.scheduler.scheduler_config
        saved = (cfg.long_prefill_token_threshold, s.adaptive_long_prefill_threshold)
        budget = s.max_num_scheduled_tokens
        cfg.long_prefill_token_threshold, s.adaptive_long_prefill_threshold = 1, False
        s.max_num_scheduled_tokens = max(1, len(self.ids))  # one request alone is not capped
        try:
            return self.schedule(*args, **kwargs)
        finally:
            cfg.long_prefill_token_threshold, s.adaptive_long_prefill_threshold = saved
            s.max_num_scheduled_tokens = budget
