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
import secrets
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "using",
        "use",
        "please",
        "can",
        "could",
        "would",
        "should",
        "do",
        "does",
        "analyze",
        "analysis",
        "show",
        "find",
        "run",
        "task",
        "generate",
        "get",
    }
)


class SnapshotCancelled(Exception):
    """A walk was stopped because nobody is waiting for its result any more."""


def get_all_files(
    directory: str,
    *,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    cancelled: Callable[[], bool] | None = None,
) -> set[str]:
    """Recursively collect non-hidden file paths under `directory`.

    Hidden files and directories (leading dot) are always skipped. Any
    directory name in `exclude_dirs` is skipped as well.

    ``cancelled`` is asked before each directory is read. Once it says yes the
    walk raises :class:`SnapshotCancelled` instead of returning: half a listing
    would pass every file it had not reached yet off as new.
    """
    if not directory or not os.path.isdir(directory):
        return set()

    excluded = set(exclude_dirs)
    result: set[str] = set()
    for root, dirs, files in os.walk(directory):
        if cancelled is not None and cancelled():
            raise SnapshotCancelled(directory)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in excluded]
        for fname in files:
            if not fname.startswith("."):
                result.add(os.path.join(root, fname))
    return result


def _walk_reaches(ancestor: str, root: str) -> bool:
    """Whether :func:`get_all_files` on ``ancestor`` also lists everything it would list for ``root``.

    True when ``root`` is below ``ancestor`` and the walk can get there: it does
    not enter hidden or excluded directories, and does not follow symlinks, so a
    root reached through one of those has to be walked on its own.
    """
    try:
        if os.path.commonpath([ancestor, root]) != ancestor:
            return False
    except ValueError:  # different drives on Windows
        return False
    here = ancestor
    for part in os.path.relpath(root, ancestor).split(os.sep):
        if part == os.curdir:
            continue
        if part.startswith(".") or part in DEFAULT_EXCLUDED_DIRS:
            return False
        here = os.path.join(here, part)
        if os.path.islink(here):
            return False
    return True


def _distinct_roots(directories: Iterable[str | None]) -> list[str]:
    """The directories worth walking: each once, and none that a walk of another already covers."""
    roots: list[str] = []
    for directory in directories:
        if directory and (root := os.path.abspath(directory)) not in roots:
            roots.append(root)
    return [root for root in roots if not any(other != root and _walk_reaches(other, root) for other in roots)]


def snapshot_files(*directories: str | None, cancelled: Callable[[], bool] | None = None) -> set[str]:
    """Every non-hidden file under any of ``directories``, as :func:`get_all_files` sees them.

    A ``None``, empty or missing directory contributes nothing. This is the walk
    behind a run's before/after comparison. It is unbounded, so on a network
    mount holding a large workspace it takes minutes: code running on an event
    loop must hand it to a worker thread instead of calling it, and can pass
    ``cancelled`` to have the thread stop when the caller gives up.

    A directory inside another one in the list is not walked again: where the
    app runs from ``/app`` and the workspace is mounted at ``/app/user-data``,
    walking both would read the whole workspace twice for the same listing.
    """
    files: set[str] = set()
    for directory in _distinct_roots(directories):
        files |= get_all_files(directory, cancelled=cancelled)
    return files


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
    token: str | None = None,
) -> str:
    """Build a run directory ID of the form `run_YYYYMMDD_HHMMSS_xxxxxx[_topic_slug]`.

    If `llm_summarizer` is provided it is called first to produce a slug;
    a falsy or empty return falls back to the deterministic
    `summarize_topic_for_run_id` heuristic.

    ``xxxxxx`` is random, and it is what makes the id unique. Timestamp plus
    slug is not: the app serves several chats at once, every caller creates the
    directory with ``exist_ok=True``, and two runs started in the same second
    from the same prompt therefore *shared* one directory and silently
    overwrote each other's ``report.md`` and ``trace.ipynb``. Identical prompts
    in the same second is not a contrived case - the UI's starter buttons send
    fixed text, so a room full of people clicking the same one is exactly it.

    The token sits after the timestamp rather than at the end so directory
    listings still sort chronologically and still read as the topic.
    ``token`` is injectable for tests; leave it unset in application code.
    """
    timestamp = now().strftime("%Y%m%d_%H%M%S")
    unique = token if token is not None else secrets.token_hex(3)
    stem = f"run_{timestamp}_{unique}" if unique else f"run_{timestamp}"
    if not topic:
        return stem

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
        return stem

    return f"{stem}_{topic_slug}"
