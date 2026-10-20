# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The resume after a change: every running request computes at least its last position."""

from types import SimpleNamespace

from everserve_vllm.resume import resume_point, resume_running


class _Request(SimpleNamespace):
    __hash__ = object.__hash__  # the scheduler keeps requests in sets


def _req(rid, num_tokens):
    return _Request(
        request_id=rid,
        num_tokens_with_spec=num_tokens,
        all_token_ids=list(range(num_tokens)),
        num_computed_tokens=num_tokens + 1,  # what the steps in flight had claimed
        num_output_placeholders=1,
        num_in_flight_tokens=1,
        num_stale_output_tokens=0,
        drop_stale_output=False,
        spec_token_ids=[],
    )


def test_a_request_resumes_with_its_last_position_to_compute():
    # a dead rank's count lags the tokens: the tail from it is replayed
    assert resume_point(2486, 2484) == 2484
    # a leaving rank drained the step for the last sampled token: the count equals the tokens;
    # resumed there the request would have nothing to compute and the engine would step empty
    assert resume_point(2484, 2484) == 2483
    assert resume_point(2484, 2485) == 2483


def test_resume_running_sets_the_capped_count_and_clears_the_in_flight_state():
    reqs = [_req("a", 2484), _req("b", 2486)]
    scheduler = SimpleNamespace(
        running=reqs,
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda rid: ([1, 2, 3],)),
        _inflight_prefills=set(),
    )
    made = resume_running(
        scheduler,
        {"a": 2484, "b": 2484},
        new_request_data=lambda req, blocks, prefill_token_ids: (
            req.request_id,
            req.num_computed_tokens,
            len(prefill_token_ids),
        ),
    )
    assert made == [("a", 2483, 2484), ("b", 2484, 2486)]
    for r in reqs:
        assert r.num_tokens_with_spec - r.num_computed_tokens >= 1  # always a position to compute
        assert r.num_output_placeholders == 0 and r.num_in_flight_tokens == 0


def test_a_request_with_no_committed_count_resumes_from_zero():
    """Nothing restored for it means nothing committed: every position is recomputed."""
    reqs = [_req("a", 1241), _req("b", 1241)]
    scheduler = SimpleNamespace(
        running=reqs,
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda rid: ([1],)),
        _inflight_prefills=set(),
    )
    resume_running(
        scheduler,
        {"a": 900},
        new_request_data=lambda req, blocks, prefill_token_ids: req.request_id,
    )
    assert reqs[0].num_computed_tokens == 900
    assert reqs[1].num_computed_tokens == 0


def test_the_step_after_the_resume_is_a_decode_for_every_request_past_its_prompt():
    """The runner's prefill length becomes the committed count and the committed position's
    token sits in the last-sampled buffer; the token list and the count stay the control's."""
    import numpy as np
    import torch

    from everserve_vllm.worker import decode_shaped

    tokens = list(range(100, 112))  # 4 prompt tokens, 8 generated so far
    reqs = [
        SimpleNamespace(req_id="a", num_computed_tokens=11, prefill_token_ids=tokens, prompt_len=4),
        SimpleNamespace(
            req_id="b", num_computed_tokens=3, prefill_token_ids=tokens[:4], prompt_len=4
        ),
        SimpleNamespace(req_id="c", num_computed_tokens=12, prefill_token_ids=tokens, prompt_len=4),
    ]

    class PrefillLen:  # v2's UVA-backed tensor: the host array, and the device-visible copy
        def __init__(self):
            self.np = np.array([0] * 8, dtype=np.int32)
            self.gpu = self.np.copy()

        def copy_to_uva(self):
            self.gpu = self.np.copy()

    states = SimpleNamespace(
        req_id_to_index={"a": 5, "b": 2, "c": 7},
        prefill_len=PrefillLen(),
        last_sampled_tokens=torch.zeros(8, 1, dtype=torch.int64),
    )
    states.prefill_len.np[5] = len(tokens)  # what add_request set: one past the committed count
    states.prefill_len.copy_to_uva()  # and copied to the device, as add_requests ends
    runner = SimpleNamespace(req_states=states)
    assert decode_shaped(runner, reqs) == ["a"]
    # a: nothing left to prefill (prefill length = computed, on the device side the step reads
    # too), the next input is the committed token
    assert states.prefill_len.np[5] == 11 and states.prefill_len.gpu[5] == 11
    assert int(states.last_sampled_tokens[5, 0]) == tokens[11]
    assert reqs[0].prefill_token_ids == tokens and reqs[0].num_computed_tokens == 11  # untouched
    # b is inside its prompt (a real prefill), c has no position left: vLLM's own shape for both
    assert states.prefill_len.np[2] == 0 and states.prefill_len.np[7] == 0
    assert int(states.last_sampled_tokens[2, 0]) == 0 and int(states.last_sampled_tokens[7, 0]) == 0


def test_a_runner_without_v2_request_state_keeps_vllm_s_shape():
    from everserve_vllm.worker import decode_shaped

    req = SimpleNamespace(
        req_id="a", num_computed_tokens=5, prefill_token_ids=[1] * 6, prompt_len=2
    )
    assert decode_shaped(SimpleNamespace(), [req]) == []
