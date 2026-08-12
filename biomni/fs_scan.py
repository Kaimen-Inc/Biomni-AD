"""Bounded, cached filesystem scanning for workspace inventories.

Agent bootstrap and the Chainlit sidebar need a picture of the user's workspace
(file tree, counts, extensions). The naive approach - ``os.walk`` the whole tree
on every chat start - is fine for a handful of files but catastrophic for a
large workspace on a network-backed mount (Azure Files / blobfuse), where every
``readdir``/``stat`` is a round-trip. A multi-minute walk on the asyncio event
loop wedges the whole server: the ``/healthz`` liveness probe cannot be served,
so Kubernetes restarts the pod and the user sees "Agent could not start".

Every scan here is bounded by BOTH, whichever trips first:

* a hard **file-count cap** (``BIOMNI_WORKSPACE_MAX_FILES``, default 10000), and
* a wall-clock **deadline** (``BIOMNI_WORKSPACE_SCAN_TIMEOUT_S``, default 15s),

so a scan can never hang regardless of workspace size or mount latency. Results
are cached per (root, depth, exclusions, cap) for a short TTL
(``BIOMNI_WORKSPACE_SCAN_TTL_S``, default 60s) behind a per-key lock, so the
several inventory/sidebar builders that fire during one chat start traverse the
tree at most once instead of ~4 times.

When either bound trips, the scan sets ``truncated`` / ``timed_out`` and reports
counts as a *floor* (``file_count`` means "at least this many"). Callers should
render such totals with a trailing ``+`` and a note rather than as exact figures
- see :meth:`ScanResult.count_label`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILES = 10_000
_DEFAULT_TIMEOUT_S = 15.0
_DEFAULT_TTL_S = 60.0

# Directories never worth walking for a user-facing data inventory. Kept here so
# every scanner call (Chainlit sidebar + agent system prompt) prunes identically.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".chainlit",
        "node_modules",
        "site-packages",
        ".ipynb_checkpoints",
        ".cache",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        # Biomni's per-run output history (also pruned by
        # biomni.artifact.DEFAULT_EXCLUDED_DIRS). It can grow without bound and
        # is never user data, so keep the file cap / deadline for real datasets.
        "runs",
    }
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %d", name, raw, default)
        return default
    # <=0 is rejected on purpose: the bounds exist to keep scans finite, so
    # "0"/"unlimited" must not silently disable the guard - fall back to default.
    if value <= 0:
        logger.warning("%s=%r is <=0; ignoring (guard cannot be disabled), using default %d", name, raw, default)
        return default
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%r is <=0; ignoring (guard cannot be disabled), using default %s", name, raw, default)
        return default
    return value


def max_files() -> int:
    """Hard cap on files visited per scan (env ``BIOMNI_WORKSPACE_MAX_FILES``)."""
    return _int_env("BIOMNI_WORKSPACE_MAX_FILES", _DEFAULT_MAX_FILES)


def scan_timeout_s() -> float:
    """Wall-clock budget per scan (env ``BIOMNI_WORKSPACE_SCAN_TIMEOUT_S``)."""
    return _float_env("BIOMNI_WORKSPACE_SCAN_TIMEOUT_S", _DEFAULT_TIMEOUT_S)


def cache_ttl_s() -> float:
    """How long a scan result stays fresh (env ``BIOMNI_WORKSPACE_SCAN_TTL_S``)."""
    return _float_env("BIOMNI_WORKSPACE_SCAN_TTL_S", _DEFAULT_TTL_S)


@dataclass
class ScanResult:
    """Outcome of a bounded directory walk.

    ``file_count`` / ``dir_count`` are exact when neither bound tripped; when
    ``truncated`` or ``timed_out`` is set they are a *floor* (the real tree is at
    least this big). ``files`` holds relative POSIX paths, itself capped at the
    file cap so it is safe to embed in a prompt or render in a panel.

    READ-ONLY: instances are cached and handed to multiple threads by reference
    (see :func:`scan_directory`). Never mutate ``files`` / ``top_level_counts`` /
    ``extension_counts`` in place - copy first (``list(...)`` / ``dict(...)``).
    """

    root: str
    files: list[str] = field(default_factory=list)
    file_count: int = 0
    dir_count: int = 0
    truncated: bool = False  # hit the file-count cap
    timed_out: bool = False  # hit the wall-clock deadline
    top_level_counts: dict[str, int] = field(default_factory=dict)
    extension_counts: dict[str, int] = field(default_factory=dict)

    @property
    def bounded(self) -> bool:
        """True if the scan stopped early (counts are a lower bound)."""
        return self.truncated or self.timed_out

    def count_label(self) -> str:
        """Human file total, e.g. ``"328"`` or ``"10000+"`` when cut short."""
        return f"{self.file_count}+" if self.bounded else str(self.file_count)


# Deadline is re-checked inside a directory every this-many files so a single
# pathologically large directory cannot blow the wall-clock budget.
_DEADLINE_CHECK_EVERY = 2000


def _walk(
    path: str,
    max_depth: int,
    exclude_top: frozenset[str],
    cap: int,
    deadline: float,
    prune_subtrees: tuple[str, ...],
) -> ScanResult:
    """Single bounded ``os.walk`` pass. Stops at ``cap`` files or ``deadline``."""
    result = ScanResult(root=path)

    for root, dirs, files in os.walk(path):
        # os.walk is lazy: pruning ``dirs`` / breaking here actually stops
        # further readdir/stat traffic, which is what makes the deadline bite.
        if time.monotonic() >= deadline:
            result.timed_out = True
            break

        # Skip entire nested subtrees (e.g. a built-in data lake mounted under
        # the user root) - matched on absolute path prefix, so this covers dirs
        # deeper than the top level that ``exclude_top`` cannot reach.
        if prune_subtrees and any(root == p or root.startswith(p + os.sep) for p in prune_subtrees):
            dirs[:] = []
            continue

        rel_root = os.path.relpath(root, path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue

        visible = [d for d in dirs if not d.startswith(".") and d not in EXCLUDED_DIRS]
        if depth == 0 and exclude_top:
            visible = [d for d in visible if d not in exclude_top]
        dirs[:] = sorted(visible)
        result.dir_count += len(dirs)

        top = "[root]" if rel_root == "." else rel_root.split(os.sep, 1)[0]
        for file_name in sorted(files):
            if file_name.startswith("."):
                continue
            result.file_count += 1
            ext = Path(file_name).suffix.lower() or "[no_ext]"
            result.extension_counts[ext] = result.extension_counts.get(ext, 0) + 1
            result.top_level_counts[top] = result.top_level_counts.get(top, 0) + 1
            if len(result.files) < cap:
                rel = file_name if rel_root == "." else f"{rel_root}/{file_name}"
                result.files.append(rel.replace(os.sep, "/"))
            if result.file_count >= cap:
                result.truncated = True
                return result
            if result.file_count % _DEADLINE_CHECK_EVERY == 0 and time.monotonic() >= deadline:
                result.timed_out = True
                return result

    return result


# Cache maps (abspath, depth, excludes, cap) -> (monotonic_stamp, ScanResult).
# Cardinality is tiny (a few roots x a couple of depths), so no eviction is
# needed beyond TTL freshness; entries are simply overwritten on refresh.
_cache: dict[tuple, tuple[float, ScanResult]] = {}
_cache_lock = threading.Lock()  # guards _cache and _key_locks
_key_locks: dict[tuple, threading.Lock] = {}


def scan_directory(
    path: str | None,
    *,
    max_depth: int = 10,
    exclude_top_subdirs: set[str] | frozenset[str] | None = None,
    prune_subtrees: set[str] | frozenset[str] | list[str] | None = None,
    max_files_override: int | None = None,
) -> ScanResult:
    """Return a bounded, cached inventory of ``path``.

    Thread-safe: concurrent callers for the same key block on a per-key lock so
    exactly one thread walks a cold tree while the rest reuse its result. This
    collapses the several sidebar/inventory walks of one chat start into one.

    Args:
        path: Directory to scan. Non-existent / non-dir returns an empty result.
        max_depth: Maximum directory depth (top-level files are depth 0).
        exclude_top_subdirs: Top-level subdirectory names to skip entirely.
        prune_subtrees: Absolute paths whose subtrees are skipped entirely
            (e.g. a built-in data lake nested under the user root).
        max_files_override: Override the global file cap for this call (e.g. a
            retrieval index that only wants the first few hundred files).
    """
    if not path or not os.path.isdir(path):
        return ScanResult(root=path or "")

    exclude_top = frozenset(exclude_top_subdirs) if exclude_top_subdirs else frozenset()
    prune = tuple(sorted(os.path.abspath(p) for p in prune_subtrees)) if prune_subtrees else ()
    cap = max_files_override if (max_files_override and max_files_override > 0) else max_files()
    abspath = os.path.abspath(path)
    key = (abspath, int(max_depth), exclude_top, prune, int(cap))
    ttl = cache_ttl_s()

    # Fast path: fresh cache hit under the shared lock.
    with _cache_lock:
        hit = _cache.get(key)
        if hit and (time.monotonic() - hit[0]) < ttl:
            return hit[1]
        klock = _key_locks.get(key)
        if klock is None:
            klock = threading.Lock()
            _key_locks[key] = klock

    # Only one thread scans a given key at a time; others block then read the
    # value it just cached rather than re-walking the same (slow) tree.
    with klock:
        with _cache_lock:
            hit = _cache.get(key)
            if hit and (time.monotonic() - hit[0]) < ttl:
                return hit[1]

        started = time.monotonic()
        deadline = started + scan_timeout_s()
        result = _walk(abspath, int(max_depth), exclude_top, cap, deadline, prune)
        elapsed = time.monotonic() - started

        if result.bounded:
            logger.warning(
                "workspace scan of %s bounded after %.1fs (files>=%d, dirs>=%d, truncated=%s, timed_out=%s); "
                "listing is partial - set BIOMNI_WORKSPACE_MAX_FILES / BIOMNI_WORKSPACE_SCAN_TIMEOUT_S to tune",
                path,
                elapsed,
                result.file_count,
                result.dir_count,
                result.truncated,
                result.timed_out,
            )
        else:
            logger.info(
                "workspace scan of %s: %d files, %d dirs in %.2fs",
                path,
                result.file_count,
                result.dir_count,
                elapsed,
            )

        with _cache_lock:
            _cache[key] = (time.monotonic(), result)
        return result


def clear_cache() -> None:
    """Drop all cached scans (call after a known bulk change, or in tests)."""
    with _cache_lock:
        _cache.clear()
        _key_locks.clear()
