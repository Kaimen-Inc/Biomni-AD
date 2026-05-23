"""Shared helpers for run-artifact bookkeeping.

These helpers are used by both `biomni.agent.a1.A1` and the Chainlit UI to
build run-directory IDs and to snapshot the filesystem before/after an
agent run. Centralising them prevents the two call sites from drifting on
the directory exclude list, which previously caused mismatched
initial/final file sets when the agent was driven from the UI.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable
from datetime import datetime

logger = logging.getLogger(__name__)

# Directories that should never appear in a run's file snapshot. Hidden
# dotted directories are excluded separately via the `startswith(".")` check.
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "runs",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
    }
)

_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "into", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
        "using", "use", "please", "can", "could", "would", "should", "do", "does",
        "analyze", "analysis", "show", "find", "run", "task", "generate", "get",
    }
)


def get_all_files(
    directory: str,
    *,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> set[str]:
    """Recursively collect non-hidden file paths under `directory`.

    Hidden files and directories (leading dot) are always skipped. Any
    directory name in `exclude_dirs` is skipped as well.
    """
    if not directory or not os.path.isdir(directory):
        return set()

    excluded = set(exclude_dirs)
    result: set[str] = set()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in excluded]
        for fname in files:
            if not fname.startswith("."):
                result.add(os.path.join(root, fname))
    return result


def summarize_topic_for_run_id(topic: str) -> str:
    """Extract a compact 1-3 word filesystem-safe summary from a prompt."""
    raw_tokens = re.findall(r"[A-Za-z0-9]+", topic or "")
    if not raw_tokens:
        return ""

    selected: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        if lower in _TOPIC_STOPWORDS:
            continue
        if len(lower) <= 2 and not lower.isdigit():
            continue
        selected.append(lower)
        if len(selected) == 3:
            break

    if not selected:
        selected = [t.lower() for t in raw_tokens[:3]]

    summary = "_".join(selected)
    summary = re.sub(r"[^0-9a-z_]+", "", summary)
    summary = re.sub(r"_+", "_", summary).strip("_")
    return summary[:40]


def build_run_id(
    topic: str | None = None,
    *,
    llm_summarizer: Callable[[str], str] | None = None,
    now: Callable[[], datetime] = datetime.now,
) -> str:
    """Build a run directory ID of the form `run_YYYYMMDD_HHMMSS[_topic_slug]`.

    If `llm_summarizer` is provided it is called first to produce a slug;
    a falsy or empty return falls back to the deterministic
    `summarize_topic_for_run_id` heuristic.
    """
    timestamp = now().strftime("%Y%m%d_%H%M%S")
    if not topic:
        return f"run_{timestamp}"

    topic_slug = ""
    if llm_summarizer is not None:
        try:
            topic_slug = llm_summarizer(topic) or ""
        except Exception:
            logger.debug("LLM run-id summarizer failed; falling back to heuristic", exc_info=True)
            topic_slug = ""
    if not topic_slug:
        topic_slug = summarize_topic_for_run_id(topic)
    if not topic_slug:
        return f"run_{timestamp}"

    return f"run_{timestamp}_{topic_slug}"
