# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joining a streamed Gemini response must survive text-less chunks.

Regression cover for the original behaviour: ``get_result_text`` joined
``res.text`` blindly, so one chunk carrying only an image/thought part
(``.text is None``) raised ``TypeError`` and discarded the whole stage's
work; a blocked or token-truncated response was indistinguishable from a
normal one because finish_reason/prompt_feedback were never inspected.

These tests exercise the helpers directly so they need none of the heavy
model dependencies (torch/diffusers) that importing the provider classes
would pull in.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_response_helpers():
    """Load the Gemini response helpers out of vlm.py without its heavy deps."""
    source = (
        Path(__file__).resolve().parents[1] / "simfoundry" / "models" / "vlm.py"
    ).read_text(encoding="utf-8")
    start = source.index("class GeminiResponseIncomplete")
    end = source.index("class VLM_API:")
    namespace: dict = {}
    exec(source[start:end], namespace)
    return namespace


HELPERS = _load_response_helpers()


def _chunk(text, finish_reason=None, block_reason=None):
    candidate = SimpleNamespace(finish_reason=finish_reason)
    prompt_feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    return SimpleNamespace(text=text, candidates=[candidate], prompt_feedback=prompt_feedback)


def test_text_less_chunks_are_skipped_not_fatal():
    chunks = [_chunk("foo"), _chunk(None), _chunk("bar", finish_reason="STOP")]
    assert HELPERS["join_gemini_result_text"](chunks) == "foobar"


def test_normal_stream_joins_all_text():
    chunks = [_chunk("a"), _chunk("b"), _chunk("c", finish_reason="STOP")]
    assert HELPERS["join_gemini_result_text"](chunks) == "abc"


def test_safety_stop_raises_named_error():
    chunks = [_chunk("partial"), _chunk(None, finish_reason="SAFETY")]
    with pytest.raises(HELPERS["GeminiResponseIncomplete"], match="SAFETY"):
        HELPERS["join_gemini_result_text"](chunks, model="gemini-test")


def test_token_truncation_raises_named_error():
    chunks = [_chunk("partial", finish_reason="MAX_TOKENS")]
    with pytest.raises(HELPERS["GeminiResponseIncomplete"], match="MAX_TOKENS"):
        HELPERS["join_gemini_result_text"](chunks)


def test_blocked_prompt_raises_named_error():
    chunks = [_chunk(None, block_reason="PROHIBITED_CONTENT")]
    with pytest.raises(HELPERS["GeminiResponseIncomplete"], match="PROHIBITED_CONTENT"):
        HELPERS["join_gemini_result_text"](chunks)


def test_enum_like_finish_reasons_use_their_name():
    finish = SimpleNamespace(name="RECITATION")
    with pytest.raises(HELPERS["GeminiResponseIncomplete"], match="RECITATION"):
        HELPERS["join_gemini_result_text"]([_chunk("x", finish_reason=finish)])


def test_cache_replayed_chunks_have_no_metadata_and_pass():
    # _CachedGeminiChunk exposes .text and bare candidates without finish_reason
    # or prompt_feedback; replayed results must never be rejected.
    cached = SimpleNamespace(text="hello", candidates=[SimpleNamespace()])
    assert HELPERS["join_gemini_result_text"]([cached]) == "hello"


def test_all_text_less_but_normal_finish_yields_empty_string():
    chunks = [_chunk(None), _chunk(None, finish_reason="STOP")]
    assert HELPERS["join_gemini_result_text"](chunks) == ""


def test_problem_reporter_returns_none_when_clean():
    assert HELPERS["gemini_response_problem"]([_chunk("ok", finish_reason="STOP")]) is None
    assert HELPERS["gemini_response_problem"]([]) is None
    assert HELPERS["gemini_response_problem"](None) is None
