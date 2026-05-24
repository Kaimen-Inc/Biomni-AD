"""Tests for ``biomni.llm_resilience``: cache-control transform + usage tracker.

These tests don't require any LLM provider package — the transform is pure
data manipulation, and the usage tracker reads from response objects that we
fabricate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from biomni.llm_resilience import (
    LLMUsage,
    LLMUsageTracker,
    extract_usage,
    prepare_messages_for_cache,
    supports_prompt_caching,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# supports_prompt_caching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Anthropic", True),
        ("OpenAI", False),
        ("AzureOpenAI", False),
        ("Bedrock", False),
        (None, False),
        ("", False),
    ],
)
def test_supports_prompt_caching(source, expected) -> None:
    assert supports_prompt_caching(source) is expected


# ---------------------------------------------------------------------------
# prepare_messages_for_cache
# ---------------------------------------------------------------------------


def test_cache_transform_anthropic_marks_system_prompt() -> None:
    messages = [
        SystemMessage(content="You are a biomedical agent. " * 200),
        HumanMessage(content="What is APOE?"),
    ]

    out = prepare_messages_for_cache(messages, "Anthropic")

    assert out is not messages, "must return a new list, not mutate input"
    assert len(out) == 2
    sys_msg = out[0]
    assert isinstance(sys_msg, SystemMessage)
    assert isinstance(sys_msg.content, list)
    assert len(sys_msg.content) == 1
    block = sys_msg.content[0]
    assert block["type"] == "text"
    assert block["text"].startswith("You are a biomedical agent.")
    assert block["cache_control"] == {"type": "ephemeral"}
    # downstream messages must pass through untouched
    assert out[1] is messages[1]


def test_cache_transform_noop_for_openai() -> None:
    messages = [
        SystemMessage(content="long prompt"),
        HumanMessage(content="hi"),
    ]
    assert prepare_messages_for_cache(messages, "OpenAI") is messages


def test_cache_transform_disabled_flag() -> None:
    messages = [SystemMessage(content="x"), HumanMessage(content="y")]
    assert prepare_messages_for_cache(messages, "Anthropic", enabled=False) is messages


def test_cache_transform_handles_empty_list() -> None:
    assert prepare_messages_for_cache([], "Anthropic") == []


def test_cache_transform_no_system_message_passthrough() -> None:
    messages = [HumanMessage(content="just a user msg"), AIMessage(content="ack")]
    assert prepare_messages_for_cache(messages, "Anthropic") is messages


def test_cache_transform_skips_empty_system_content() -> None:
    """Empty system content shouldn't be wrapped — Anthropic rejects 0-byte cache blocks."""
    messages = [SystemMessage(content=""), HumanMessage(content="hi")]
    out = prepare_messages_for_cache(messages, "Anthropic")
    # Empty content → unchanged (passthrough), no cache markers attempted.
    assert out is messages


def test_cache_transform_respects_existing_cache_control() -> None:
    """If the caller has already structured the content with cache_control, don't clobber."""
    pre_marked = [
        {"type": "text", "text": "preamble"},
        {"type": "text", "text": "cached part", "cache_control": {"type": "ephemeral"}},
    ]
    messages = [SystemMessage(content=pre_marked), HumanMessage(content="q")]
    out = prepare_messages_for_cache(messages, "Anthropic")
    assert out[0].content == pre_marked


def test_cache_transform_marks_last_text_block_in_list_content() -> None:
    """When content is already a list but unmarked, mark the LAST text block."""
    blocks = [
        {"type": "text", "text": "preamble (don't cache)"},
        {"type": "text", "text": "main prompt"},
    ]
    messages = [SystemMessage(content=blocks), HumanMessage(content="q")]
    out = prepare_messages_for_cache(messages, "Anthropic")
    new_blocks = out[0].content
    assert "cache_control" not in new_blocks[0]
    assert new_blocks[1]["cache_control"] == {"type": "ephemeral"}


def test_cache_transform_preserves_additional_kwargs() -> None:
    """The new SystemMessage should carry over additional_kwargs from the original."""
    sys = SystemMessage(content="big prompt", additional_kwargs={"trace_id": "abc"})
    out = prepare_messages_for_cache([sys, HumanMessage(content="q")], "Anthropic")
    assert out[0].additional_kwargs.get("trace_id") == "abc"


# ---------------------------------------------------------------------------
# extract_usage / LLMUsageTracker
# ---------------------------------------------------------------------------


def _fake_response(usage_metadata=None, response_metadata=None):
    return SimpleNamespace(
        usage_metadata=usage_metadata,
        response_metadata=response_metadata or {},
    )


def test_extract_usage_from_langchain_usage_metadata() -> None:
    resp = _fake_response(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 80, "cache_creation": 20},
        }
    )
    usage = extract_usage(resp)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cache_read_tokens == 80
    assert usage.cache_creation_tokens == 20
    assert usage.total_tokens == 150


def test_extract_usage_anthropic_raw_shape() -> None:
    resp = _fake_response(
        response_metadata={
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 3,
            }
        }
    )
    usage = extract_usage(resp)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.cache_read_tokens == 7
    assert usage.cache_creation_tokens == 3


def test_extract_usage_openai_raw_shape() -> None:
    resp = _fake_response(response_metadata={"token_usage": {"prompt_tokens": 25, "completion_tokens": 10}})
    usage = extract_usage(resp)
    assert usage.input_tokens == 25
    assert usage.output_tokens == 10


def test_extract_usage_unknown_shape_returns_zero() -> None:
    usage = extract_usage(_fake_response())
    assert usage == LLMUsage()
    # None is also tolerated
    assert extract_usage(None) == LLMUsage()


def test_usage_tracker_accumulates() -> None:
    tracker = LLMUsageTracker()
    tracker.record(_fake_response(usage_metadata={"input_tokens": 10, "output_tokens": 5}))
    tracker.record(
        _fake_response(
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 3,
                "input_token_details": {"cache_read": 40},
            }
        )
    )

    assert tracker.calls == 2
    assert tracker.total.input_tokens == 10
    assert tracker.total.output_tokens == 8
    assert tracker.total.cache_read_tokens == 40

    summary = tracker.summary()
    assert summary["calls"] == 2
    assert summary["total_tokens"] == 18
    # cache hit ratio: 40 cached / (10 input + 40 cached) = 0.8
    assert summary["cache_hit_ratio"] == pytest.approx(0.8)


def test_usage_tracker_cache_hit_ratio_handles_zero_input() -> None:
    tracker = LLMUsageTracker()
    assert tracker.cache_hit_ratio == 0.0
