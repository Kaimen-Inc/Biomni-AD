"""LLM resilience and observability helpers.

Two cross-cutting concerns live here so they stay out of the agent core:

1. **Prompt caching.** Anthropic charges much less for cache-hit input tokens.
   For Biomni the system prompt is hundreds of lines and is re-sent on every
   ReAct turn, so flagging it with ``cache_control`` is the single highest-ROI
   token-cost optimization in the codebase. ``prepare_messages_for_cache``
   rewrites the system message into Anthropic's content-blocks format with
   one ephemeral cache breakpoint.

2. **Per-run usage telemetry.** ``LLMUsageTracker.record`` extracts the
   ``usage_metadata`` LangChain attaches to every chat-model response (input
   tokens, output tokens, and — when present — cache-read / cache-creation
   tokens) and accumulates totals so the agent can surface cost insight
   without depending on LangSmith.

Both helpers are no-ops for non-Anthropic providers / messages that don't
carry usage_metadata, so they are safe to call unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

# Sources for which Anthropic-style ``cache_control`` markers are honored.
# Azure-hosted Anthropic deployments route through the same SDK and accept
# the same content-blocks format.
_CACHE_SUPPORTED_SOURCES = frozenset({"Anthropic"})


def supports_prompt_caching(source: str | None) -> bool:
    """Return True if ``source`` is a provider known to honor cache_control."""
    return source in _CACHE_SUPPORTED_SOURCES


def prepare_messages_for_cache(
    messages: list[BaseMessage],
    source: str | None,
    *,
    enabled: bool = True,
) -> list[BaseMessage]:
    """Annotate the system message with a cache breakpoint for Anthropic.

    Returns a new list; the input is not mutated. Non-Anthropic providers and
    messages without a leading ``SystemMessage`` pass through unchanged.

    The transform converts::

        SystemMessage(content="...long prompt...")

    into::

        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "...long prompt...",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )

    which is the LangChain-Anthropic documented form for marking a cache
    breakpoint. Subsequent turns with the same system prompt hit the cache
    and are billed at the reduced cache-read input rate.
    """
    if not enabled or not supports_prompt_caching(source) or not messages:
        return messages

    head, *rest = messages
    if not isinstance(head, SystemMessage):
        return messages

    cached_content = _to_cached_text_blocks(head.content)
    if cached_content is None:
        return messages

    return [SystemMessage(content=cached_content, additional_kwargs=head.additional_kwargs), *rest]


def _to_cached_text_blocks(content: Any) -> list[dict[str, Any]] | None:
    """Coerce SystemMessage content into a single cache-marked text block.

    Returns ``None`` if the content is empty (caching nothing is wasteful and
    a 0-byte cache block triggers provider errors).
    """
    if isinstance(content, str):
        if not content:
            return None
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]

    if isinstance(content, list):
        # Already structured. Avoid clobbering pre-existing cache_control —
        # if the caller has already marked breakpoints, respect them.
        if any(isinstance(b, dict) and b.get("cache_control") for b in content):
            return content  # type: ignore[return-value]

        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if not text_blocks:
            return None

        # Mark the last text block — this lets earlier blocks (if any) act
        # as non-cached preamble, which matches Anthropic's recommended
        # "cache the prefix" usage when blocks are layered.
        marked: list[dict[str, Any]] = []
        last_text_idx = max(i for i, b in enumerate(content) if isinstance(b, dict) and b.get("type") == "text")
        for i, block in enumerate(content):
            if i == last_text_idx and isinstance(block, dict):
                marked.append({**block, "cache_control": {"type": "ephemeral"}})
            else:
                marked.append(block)
        return marked

    return None


@dataclass
class LLMUsage:
    """Token counts for a single LLM response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )


@dataclass
class LLMUsageTracker:
    """Aggregates token usage across an agent run.

    Designed to be passed a LangChain chat-model response object via
    :meth:`record`; pulls ``usage_metadata`` (LC ≥ 0.3 standard attribute)
    and accumulates totals. Falls back to ``response_metadata.usage`` for
    older provider integrations.
    """

    total: LLMUsage = field(default_factory=LLMUsage)
    calls: int = 0

    def record(self, response: Any) -> LLMUsage:
        """Record one response. Returns the parsed per-call usage."""
        usage = extract_usage(response)
        self.total = self.total + usage
        self.calls += 1
        return usage

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache. 0.0 when no input seen."""
        if self.total.input_tokens + self.total.cache_read_tokens == 0:
            return 0.0
        return self.total.cache_read_tokens / (self.total.input_tokens + self.total.cache_read_tokens)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.total.input_tokens,
            "output_tokens": self.total.output_tokens,
            "cache_read_tokens": self.total.cache_read_tokens,
            "cache_creation_tokens": self.total.cache_creation_tokens,
            "total_tokens": self.total.total_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
        }


def extract_usage(response: Any) -> LLMUsage:
    """Pull token counts from a LangChain chat-model response.

    Looks at ``usage_metadata`` first (LangChain standard since 0.3) and
    falls back to provider-specific ``response_metadata.usage`` /
    ``response_metadata.token_usage``. Unknown shapes return a zero usage
    object rather than raising — usage tracking is best-effort.
    """
    if response is None:
        return LLMUsage()

    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        input_tokens = int(usage_meta.get("input_tokens", 0) or 0)
        output_tokens = int(usage_meta.get("output_tokens", 0) or 0)
        details = usage_meta.get("input_token_details") or {}
        cache_read = int(details.get("cache_read", 0) or 0)
        cache_creation = int(details.get("cache_creation", 0) or 0)
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )

    response_meta = getattr(response, "response_metadata", None) or {}
    raw = response_meta.get("usage") or response_meta.get("token_usage") or {}
    if isinstance(raw, dict):
        # Anthropic raw shape: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens.
        # OpenAI raw shape: prompt_tokens, completion_tokens.
        return LLMUsage(
            input_tokens=int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0),
            output_tokens=int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0),
            cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
        )

    return LLMUsage()


__all__ = [
    "LLMUsage",
    "LLMUsageTracker",
    "extract_usage",
    "prepare_messages_for_cache",
    "supports_prompt_caching",
]
