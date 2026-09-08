"""Per-user workspace preferences: which data to look at, where results go.

Two problems reported from the AD Workbench deployment share one root cause -
the app treats the mounted workspace as one undifferentiated blob and writes
its results wherever the process happens to be running:

* a workspace with thousands of files is scanned and described to the model in
  full, which is slow and drowns the agent in irrelevant paths;
* the output directory is not configurable, so artifacts land in the container
  filesystem and disappear with the pod.

This module is the answer to both: an explicit, persisted statement of *scope*
(the folders the user cares about) and *output directory*, resolved once per
session and reused by the UI, the agent's system prompt and the run artifacts.

Design notes:

* **Pure and Chainlit-free.** Everything here is testable without a browser or
  an event loop; the UI layer only renders what these functions decide.
* **Storage is pluggable** behind :class:`PrefsStore`. The default picked by
  :func:`build_prefs_store` prefers the user's *own workspace*
  (``<workspace>/.biomni/prefs.json``) when it is writable, because preferences
  stored there survive the application being deprovisioned without any extra
  infrastructure. It degrades to an operator-provided state directory, then to
  no persistence at all - never to an error.
* **A new user is not a special case.** No stored preferences simply means
  :func:`default_prefs`, which is why first-time GRIP users need no distinct
  code path.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Bumped when the on-disk shape changes incompatibly. Unknown/newer versions are
# discarded in favour of defaults rather than half-read (see from_dict).
PREFS_VERSION = 1

# Directory name used for both preferences and run records when they live inside
# the user's own workspace.
WORKSPACE_STATE_DIRNAME = ".biomni"

# Default folder name for run outputs under the resolved output root.
DEFAULT_OUTPUT_DIRNAME = "biomni-outputs"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var. Anything but a recognised true/false is ignored."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s=%r; using default %s", name, raw, default)
    return default


def _split_paths(raw: str) -> list[str]:
    """Split a delimited path list. Accepts commas and the OS path separator."""
    parts: list[str] = []
    for chunk in raw.replace(os.pathsep, ",").split(","):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #


@dataclass
class WorkspacePrefs:
    """One user's choices for one workspace.

    ``scope_paths`` are workspace-relative by convention (absolute paths are
    accepted and used as-is). An empty list means "the user has not chosen",
    which is materially different from "the user chose nothing" - the UI shows a
    shallow top-level listing in that case rather than hiding the workspace.
    """

    scope_paths: list[str] = field(default_factory=list)
    output_dir: str | None = None
    remember: bool = True
    version: int = PREFS_VERSION
    updated_at: str | None = None
    # Display-only identity echo, so a support engineer looking at a prefs file
    # can tell whose it is. Never used to build paths (see UserIdentity.storage_key).
    email: str | None = None

    @property
    def scope_is_default(self) -> bool:
        """True when the user has not made an explicit scope choice yet."""
        return not self.scope_paths

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> WorkspacePrefs | None:
        """Rebuild from stored JSON, or ``None`` if the payload is unusable.

        Tolerant by design: a corrupt or future-versioned preferences file must
        degrade to defaults, never break session start.
        """
        if not isinstance(data, dict):
            return None
        version = data.get("version")
        if not isinstance(version, int) or version > PREFS_VERSION:
            logger.warning("ignoring preferences with unsupported version %r", version)
            return None

        raw_scope = data.get("scope_paths") or []
        scope_paths = [str(p) for p in raw_scope if isinstance(p, str | os.PathLike) and str(p).strip()]

        output_dir = data.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            output_dir = None

        return cls(
            scope_paths=scope_paths,
            output_dir=output_dir or None,
            remember=bool(data.get("remember", True)),
            version=PREFS_VERSION,
            updated_at=data.get("updated_at") if isinstance(data.get("updated_at"), str) else None,
            email=data.get("email") if isinstance(data.get("email"), str) else None,
        )


def default_prefs() -> WorkspacePrefs:
    """Preferences for a user who has never chosen anything.

    Operators can seed a sensible starting scope per deployment with
    ``BIOMNI_DEFAULT_SCOPE_PATHS`` (comma or ``os.pathsep`` separated); with it
    unset, the scope stays empty and the UI shows a cheap top-level listing.
    """
    scope = _split_paths(os.getenv("BIOMNI_DEFAULT_SCOPE_PATHS", ""))
    return WorkspacePrefs(scope_paths=scope, output_dir=None, remember=True)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


class PrefsStore(Protocol):
    """Where a user's preferences live. Implementations must never raise."""

    def load(self, key: str) -> WorkspacePrefs | None: ...

    def save(self, key: str, prefs: WorkspacePrefs) -> bool: ...

    def delete(self, key: str) -> bool: ...

    @property
    def describe(self) -> str: ...


class NullPrefsStore:
    """No persistence. Settings apply to the current session only.

    ``reason`` is surfaced in the UI, because "your settings will not be
    remembered" is only actionable if the user is told why (no writable volume
    is an operator problem; no signed-in user is not).
    """

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "no writable preferences location configured"

    def load(self, key: str) -> WorkspacePrefs | None:
        return None

    def save(self, key: str, prefs: WorkspacePrefs) -> bool:
        return False

    def delete(self, key: str) -> bool:
        return False

    @property
    def describe(self) -> str:
        return f"this session only ({self.reason})"


def atomic_write_json(path: str, payload: dict[str, Any]) -> bool:
    """Write ``payload`` to ``path`` atomically. Returns False on failure.

    The temp file is created in the destination directory so ``os.replace`` is
    a same-filesystem rename: a pod evicted mid-write can leave a stray
    ``.tmp-`` file but never a truncated record that would wipe good state.
    Shared by the preferences store and the run registry.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        logger.warning("could not write %s", path, exc_info=True)
        return False
    return True


def read_json(path: str) -> Any | None:
    """Read a JSON file, returning ``None`` when absent or unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("could not read %s", path, exc_info=True)
        return None


class JsonFilePrefsStore:
    """One JSON file per user under ``root``, written atomically."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def _path(self, key: str) -> str:
        # `key` comes from UserIdentity.storage_key(), which is already
        # slugged/hashed; basename() is belt-and-braces against traversal.
        return os.path.join(self.root, f"{os.path.basename(key)}.json")

    def load(self, key: str) -> WorkspacePrefs | None:
        return WorkspacePrefs.from_dict(read_json(self._path(key)))

    def save(self, key: str, prefs: WorkspacePrefs) -> bool:
        prefs.touch()
        return atomic_write_json(self._path(key), prefs.to_dict())

    def delete(self, key: str) -> bool:
        """Forget a user's stored preferences. Missing is success, not failure."""
        try:
            os.unlink(self._path(key))
        except FileNotFoundError:
            return True
        except OSError:
            logger.warning("could not delete preferences for %s", key, exc_info=True)
            return False
        return True

    @property
    def describe(self) -> str:
        return self.root


def _nearest_existing(path: str) -> str:
    """Deepest existing ancestor of ``path`` (including itself)."""
    current = os.path.abspath(path)
    while current and not os.path.isdir(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


# Filesystem types that carry no durability even though they are a distinct
# mount: a RAM disk survives nothing, and the container's own writable layer is
# exactly what a restart discards.
_RAM_FSTYPES = frozenset({"tmpfs", "ramfs", "devtmpfs"})
_CONTAINER_LAYER_FSTYPES = frozenset({"overlay", "overlayfs", "aufs"})


def _mount_fstype(path: str) -> str | None:
    """Filesystem type backing ``path``, per ``/proc/self/mounts``.

    ``None`` when the mount table is unavailable (macOS, a container with no
    ``/proc``), which callers treat as "cannot tell" rather than as an answer.
    The longest matching mount point wins, since mount points nest.
    """
    try:
        with open("/proc/self/mounts", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return None

    target = os.path.abspath(path)
    best_mountpoint = ""
    fstype: str | None = None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        # Mount points are escaped octal-style in /proc (a space is \040).
        mountpoint = fields[1].replace("\\040", " ")
        if target != mountpoint and not target.startswith(mountpoint.rstrip("/") + "/"):
            continue
        if len(mountpoint) >= len(best_mountpoint):
            best_mountpoint, fstype = mountpoint, fields[2]
    return fstype


def on_mounted_volume(path: str) -> bool:
    """Whether ``path`` sits on durable storage an operator mounted here.

    This is what separates "results are lost on restart" from "results are on a
    volume" - a distinction the app used to infer from *which candidate won*,
    and so got wrong in the GRIP deployment, where a disk is mounted over the
    very directory the last-resort fallback uses.

    Deliberately requires positive evidence: a Linux mount table showing a
    distinct, non-RAM, non-container-layer filesystem. Anything it cannot verify
    reads as "not a volume", so the UI keeps warning rather than promising
    durability nobody checked.
    """
    target = _nearest_existing(path)
    if not target:
        return False

    fstype = _mount_fstype(target)
    if fstype is None or fstype in _RAM_FSTYPES or fstype in _CONTAINER_LAYER_FSTYPES:
        return False

    # A separate mount from the root filesystem is the thing that makes it a
    # volume; sharing the root's device just means "somewhere under /".
    try:
        return os.stat(target).st_dev != os.stat("/").st_dev
    except OSError:
        return False


def is_writable_dir(path: str | None) -> bool:
    """Whether ``path`` is (or could be created as) a writable directory.

    A non-existent path counts as writable when its nearest existing ancestor
    is, since the caller creates it on first use. Checked with ``os.access`` and
    never by writing a probe file - this runs on a network mount where a probe
    write is both slow and visible to the user.
    """
    if not path:
        return False
    target = _nearest_existing(path)
    if not target or not os.path.isdir(target):
        return False
    return os.access(target, os.W_OK | os.X_OK)


def build_prefs_store(workspace_root: str | None = None) -> PrefsStore:
    """Pick the best available preferences location for this deployment.

    Order, first writable wins:

    1. ``BIOMNI_PREFS_DIR`` - explicit operator override.
    2. ``BIOMNI_STATE_DIR/prefs`` - the app's own persistent volume.
    3. ``<workspace_root>/.biomni/prefs`` - the user's own storage. Preferred
       over an app volume conceptually (it outlives the app entirely), but it
       comes last because it only exists when the workspace mount is writable,
       which varies by deployment.
    4. Nothing - preferences apply for the session only.
    """
    explicit = os.getenv("BIOMNI_PREFS_DIR", "").strip()
    if explicit:
        if is_writable_dir(explicit):
            return JsonFilePrefsStore(explicit)
        logger.warning("BIOMNI_PREFS_DIR=%s is not writable; preferences will not persist", explicit)

    state_dir = os.getenv("BIOMNI_STATE_DIR", "").strip()
    if state_dir:
        candidate = os.path.join(state_dir, "prefs")
        if is_writable_dir(candidate):
            return JsonFilePrefsStore(candidate)
        logger.warning("BIOMNI_STATE_DIR=%s is not writable; preferences will not persist there", state_dir)

    if workspace_root:
        candidate = os.path.join(workspace_root, WORKSPACE_STATE_DIRNAME, "prefs")
        if is_writable_dir(candidate):
            return JsonFilePrefsStore(candidate)

    logger.info("no writable preferences location; user settings apply to this session only")
    return NullPrefsStore()


def load_prefs(store: PrefsStore, key: str) -> WorkspacePrefs:
    """Stored preferences for ``key``, or defaults for a first-time user."""
    stored = store.load(key)
    if stored is None:
        return default_prefs()
    return stored


# --------------------------------------------------------------------------- #
# Output directory resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OutputTarget:
    """Where this session's run artifacts will be written, and why there."""

    path: str
    source: str  # "preference" | "env" | "workspace" | "cwd-fallback"
    writable: bool
    reason: str | None = None
    # Whether the resolved path is on a mounted volume. Determined once, at
    # resolution time, so :attr:`is_ephemeral` stays a pure property that the UI
    # can read on every render without touching the filesystem.
    mounted_volume: bool = False

    @property
    def is_ephemeral(self) -> bool:
        """True when outputs land in storage that a restart discards.

        The last-resort location is container-local *unless* an operator mounted
        a volume over it - which the GRIP deployment does, with a disk at
        ``/app/runs``. Warning there was a false alarm that sent people looking
        for a configuration bug instead of the real one, so the claim is now
        checked against the mount table (see :func:`on_mounted_volume`).
        """
        return self.source == "cwd-fallback" and not self.mounted_volume


# How each candidate is named to a human. The env entry deliberately spells the
# variable, so a warning about it is directly actionable by an operator.
_SOURCE_LABELS = {
    "preference": "the output directory you configured",
    "env": "BIOMNI_OUTPUT_ROOT",
    "workspace": "the workspace default",
    "cwd-fallback": "the container-local fallback",
}


def _output_candidates(prefs: WorkspacePrefs, workspace_root: str | None) -> list[tuple[str, str]]:
    """(path, source) candidates in priority order."""
    candidates: list[tuple[str, str]] = []

    if prefs.output_dir:
        chosen = prefs.output_dir
        if not os.path.isabs(chosen) and workspace_root:
            chosen = os.path.join(workspace_root, chosen)
        candidates.append((os.path.abspath(chosen), "preference"))

    env_root = os.getenv("BIOMNI_OUTPUT_ROOT", "").strip()
    if env_root:
        candidates.append((os.path.abspath(env_root), "env"))

    if workspace_root:
        candidates.append((os.path.join(os.path.abspath(workspace_root), DEFAULT_OUTPUT_DIRNAME), "workspace"))

    # Historical behaviour, kept last so nothing breaks where neither a volume
    # nor a writable workspace exists - but flagged as ephemeral so the UI can
    # warn instead of silently losing the user's results.
    candidates.append((os.path.abspath(os.path.join(os.getcwd(), "runs")), "cwd-fallback"))
    return candidates


def _describe_skipped(entries: list[tuple[str, str]]) -> str:
    """Human explanation of which configured locations were passed over, and why.

    Names the setting and the path, because "fell back from env" told a reader
    that something was rejected without telling them what to go and fix.
    """
    uid = getattr(os, "getuid", lambda: None)()
    by_whom = f" by uid {uid}" if uid is not None else ""
    parts = [f"{_SOURCE_LABELS.get(source, source)} ({path}) is not writable{by_whom}" for path, source in entries]
    return "; ".join(parts)


def resolve_output_dir(prefs: WorkspacePrefs, *, workspace_root: str | None = None) -> OutputTarget:
    """Decide where run outputs go, preferring what the user asked for.

    Falls through to the next candidate when one is not writable, so a stale
    preference pointing at a deleted folder degrades instead of failing every
    run. The returned target always has a path; ``writable`` says whether it can
    actually be used.

    Every skipped candidate is logged at WARNING. Silence here cost a round-trip
    with the GRIP team: an unwritable ``BIOMNI_OUTPUT_ROOT`` looked exactly like
    an ignored ``BIOMNI_OUTPUT_ROOT``, and nothing in the logs or the UI could
    tell the two apart.
    """
    candidates = _output_candidates(prefs, workspace_root)
    skipped: list[tuple[str, str]] = []

    for path, source in candidates:
        if is_writable_dir(path):
            reason = None
            if skipped:
                reason = f"fell back from {_describe_skipped(skipped)}"
            return OutputTarget(
                path=path,
                source=source,
                writable=True,
                reason=reason,
                mounted_volume=on_mounted_volume(path),
            )
        if source != "cwd-fallback":
            logger.warning(
                "output location %s (%s) is not writable; falling back to the next candidate",
                _SOURCE_LABELS.get(source, source),
                path,
            )
        skipped.append((path, source))

    path, source = candidates[-1]
    logger.error("no writable output location found; tried %s", _describe_skipped(skipped))
    return OutputTarget(
        path=path,
        source=source,
        writable=False,
        reason="no writable output location found; set BIOMNI_OUTPUT_ROOT to a mounted volume",
    )


def ensure_output_dir(target: OutputTarget) -> bool:
    """Create the resolved output directory. Returns False if it cannot be made."""
    if not target.writable:
        return False
    try:
        os.makedirs(target.path, exist_ok=True)
    except OSError:
        logger.warning("could not create output directory %s", target.path, exc_info=True)
        return False
    return True


# --------------------------------------------------------------------------- #
# Scope resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScopeResolution:
    """The concrete directories a session may read, derived from preferences."""

    roots: list[str]
    missing: list[str]
    is_default: bool

    @property
    def has_selection(self) -> bool:
        return bool(self.roots)


def resolve_scope(prefs: WorkspacePrefs, workspace_root: str | None) -> ScopeResolution:
    """Turn stored scope entries into absolute directories that exist today.

    Entries that no longer exist are reported in ``missing`` rather than
    dropped silently - a remembered folder that was deleted or renamed is
    something the user needs to see, not a mystery empty scope.
    """
    if prefs.scope_is_default or not workspace_root:
        return ScopeResolution(roots=[], missing=[], is_default=True)

    root = os.path.abspath(workspace_root)
    seen: set[str] = set()
    roots: list[str] = []
    missing: list[str] = []

    for entry in prefs.scope_paths:
        candidate = entry if os.path.isabs(entry) else os.path.join(root, entry)
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            roots.append(candidate)
        else:
            missing.append(entry)

    return ScopeResolution(roots=roots, missing=missing, is_default=False)


def relative_label(path: str, workspace_root: str | None) -> str:
    """Display label for a scope entry: workspace-relative when possible."""
    if not workspace_root:
        return path
    try:
        rel = os.path.relpath(path, os.path.abspath(workspace_root))
    except ValueError:  # different drive on Windows
        return path
    return path if rel.startswith("..") else rel


def normalize_scope_entries(entries: Iterable[str], workspace_root: str | None) -> list[str]:
    """Normalise UI input into stored scope entries (relative, de-duplicated).

    Entries inside the workspace are stored relative to it so a preferences file
    stays valid when the mount path changes between deployments.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in entries:
        entry = str(raw).strip().strip("/")
        if not entry:
            continue
        if os.path.isabs(raw if isinstance(raw, str) else str(raw)):
            entry = relative_label(os.path.abspath(str(raw)), workspace_root)
        if entry in seen:
            continue
        seen.add(entry)
        normalized.append(entry)
    return normalized
