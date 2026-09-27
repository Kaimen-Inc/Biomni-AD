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
import re
import stat
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
# Network filesystems whose ownership the kubelet cannot change: fsGroup is
# silently ignored on them, so they need their own remedy.
_SMB_FSTYPES = frozenset({"cifs", "smb3", "smbfs"})
_NFS_FSTYPES = frozenset({"nfs", "nfs4"})
# SMB options under which the file server's ACLs, not the mount options,
# decide a file's owner and mode.
_SMB_ACL_OPTIONS = ("cifsacl", "modefromsid")

_MOUNTS_FILE = "/proc/self/mounts"
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{3})")


@dataclass(frozen=True)
class MountInfo:
    """One line of the mount table: what is mounted where, and how."""

    mountpoint: str
    source: str
    fstype: str
    options: tuple[str, ...]

    def option(self, name: str) -> str | None:
        """The value of ``name=value`` among the options, or ``None``."""
        prefix = f"{name}="
        for option in self.options:
            if option.startswith(prefix):
                return option[len(prefix) :]
        return None

    @property
    def read_only(self) -> bool:
        return "ro" in self.options


def _unescape_mount_field(field: str) -> str:
    # /proc escapes whitespace and backslashes octal-style (a space is \040).
    return _OCTAL_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 8)), field)


def mount_info(path: str) -> MountInfo | None:
    """The mount backing ``path``, per ``/proc/self/mounts``.

    ``None`` when the mount table is unavailable (macOS, a container with no
    ``/proc``), which callers treat as "cannot tell" rather than as an answer.
    The longest matching mount point wins, since mount points nest.

    Symlinks are resolved first: a configured path that links onto a share is
    stored on the share, not on whatever holds the link. And the table is read
    the way the OS reads file names, since the kernel escapes only whitespace
    and backslashes and passes any other byte of a mount point through as is.
    """
    try:
        with open(_MOUNTS_FILE, encoding="utf-8", errors="surrogateescape") as handle:
            lines = handle.readlines()
    except OSError:
        return None

    target = os.path.realpath(path)
    best: MountInfo | None = None
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        mountpoint = _unescape_mount_field(fields[1])
        if target != mountpoint and not target.startswith(mountpoint.rstrip("/") + "/"):
            continue
        if best is None or len(mountpoint) >= len(best.mountpoint):
            best = MountInfo(
                mountpoint=mountpoint,
                source=_unescape_mount_field(fields[0]),
                fstype=fields[2],
                options=tuple(fields[3].split(",")),
            )
    return best


def _mount_fstype(path: str) -> str | None:
    """Filesystem type backing ``path``, or ``None`` when it cannot be told."""
    info = mount_info(path)
    return info.fstype if info else None


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


def _process_ids() -> tuple[int | None, int | None, frozenset[int]]:
    """This process's uid, primary gid and every group it belongs to."""
    if not hasattr(os, "getuid"):
        return None, None, frozenset()
    return os.getuid(), os.getgid(), frozenset({os.getgid(), *os.getgroups()})


def _mode_allows(st: os.stat_result, uid: int, groups: frozenset[int], *, write: bool) -> bool:
    """Whether owner, group and mode bits alone let ``uid`` list (or add to) a directory."""
    if uid == 0:
        return True
    need = stat.S_IWUSR if write else stat.S_IRUSR
    mode = st.st_mode
    if st.st_uid == uid:
        return bool(mode & need and mode & stat.S_IXUSR)
    if st.st_gid in groups:
        return bool(mode & (need >> 3) and mode & stat.S_IXGRP)
    return bool(mode & (need >> 6) and mode & stat.S_IXOTH)


def _is_read_only(target: str, info: MountInfo | None) -> bool:
    try:
        return bool(os.statvfs(target).f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):
        return bool(info and info.read_only)


def _explain_access(target: str, *, write: bool, wanted: str | None = None) -> str:
    """Why this process cannot read or write the existing directory ``target``.

    Written for whoever has to fix it: who owns the directory, what storage it
    is on, and the remedy for that kind of storage. "Not writable by uid 57439"
    was true and still cost the GRIP team a round trip, because a share mounted
    without ownership options, one mounted read-only, and one whose new options
    had not reached the node yet all read exactly the same.

    ``wanted`` is the directory the caller actually needs, when ``target`` is
    only its nearest existing parent.
    """
    if not os.path.isdir(target):
        return f"{target} is not a directory" if os.path.exists(target) else f"{target} does not exist"
    info = mount_info(target)
    share = _without_server(info.source) if info else ""
    on = f" ({info.fstype} mount of {share})" if info else ""
    in_image = bool(info and info.fstype in _CONTAINER_LAYER_FSTYPES)
    if write and _is_read_only(target, info):
        if in_image:
            # readOnlyRootFilesystem: there is no volume here to remount.
            return (
                f"{target} is inside the container image, which is mounted read-only; "
                f"mount a writable volume at {wanted or target}"
            )
        return f"{target} is on a read-only mount{on}; mount the volume read-write"
    try:
        st = os.stat(target)
    except OSError as exc:
        return f"{target} cannot be inspected ({exc.strerror or exc})"

    uid, gid, groups = _process_ids()
    owner = f"owned by uid {st.st_uid}, gid {st.st_gid} with mode {stat.S_IMODE(st.st_mode):04o}"
    if uid is None:
        return f"{target} is {owner}"
    access = "writing" if write else "reading"
    if _mode_allows(st, uid, groups, write=write):
        return (
            f"{target} is {owner}, which permits {access}, so something else refuses it: an ACL, "
            "SELinux or AppArmor, or the file server itself"
        )
    if info and info.fstype in _SMB_FSTYPES:
        acl = next((name for name in _SMB_ACL_OPTIONS if name in info.options), None)
        if acl:
            return (
                f"{target} is {owner} on an SMB share ({share}) mounted with {acl}, so its owner and mode come "
                f"from the file server's ACLs; grant uid {uid} {'write' if write else 'read'} access in the "
                f"share's ACLs, or mount it without {acl} and with uid={uid},gid={gid},dir_mode=0770,"
                "file_mode=0770"
            )
        # SMB has no Unix ownership of its own: the client presents whatever the
        # mount options say, so the live options are the diagnosis.
        live = ",".join(f"{name}={info.option(name)}" for name in ("uid", "gid", "dir_mode") if info.option(name))
        return (
            f"{target} is {owner} on an SMB share ({share}). Its owner and mode come only from the mount "
            f"options, which are {live or 'unset (uid=0,gid=0)'} on the mount this container sees. Mount it with "
            f"uid={uid},gid={gid},dir_mode=0770,file_mode=0770; new options only apply once the share is "
            "mounted afresh on the node"
        )
    if info and info.fstype in _NFS_FSTYPES:
        return (
            f"{target} is {owner} on an NFS export ({share}), which ignores fsGroup; "
            f"grant uid {uid} {'write' if write else 'read'} access on the NFS server"
        )
    if info and info.fstype.startswith("fuse"):
        # blobfuse2, s3fs, gcsfuse...: the driver presents ownership from its own
        # options, and the kubelet cannot change it. The driver is named in the
        # type ("fuse.s3fs") or, for plain "fuse", as the source ("blobfuse2").
        driver = info.fstype.partition(".")[2] or share
        return (
            f"{target} is {owner} on a FUSE mount ({driver}), which ignores fsGroup; mount it "
            f"with the driver's options that let uid {uid} {'write' if write else 'read'} (allow_other, and "
            "its uid, gid or umask options)"
        )
    if in_image:
        # Usually a volume that is not mounted where the setting says. Opening
        # up the image would not help: whatever is written there is lost when
        # the pod restarts.
        return (
            f"{target} is {owner}, inside the container image rather than on a mounted volume; "
            f"mount {'a writable' if write else 'the'} volume at {wanted or target}"
        )
    if st.st_uid == uid:
        # Already ours: ownership advice would send someone chowning a
        # directory to the uid that owns it.
        return (
            f"{target} is {owner}{on}, which denies its own owner (uid {uid}); give the owner access with chmod u+rwx"
        )
    if st.st_gid in groups:
        return f"{target} is {owner}{on}, which denies its group; give the group access with chmod g+rwx"
    return (
        f"{target} is {owner}{on}; set securityContext.fsGroup: {gid} on the pod (it does not apply to hostPath "
        f"volumes), or chown it to uid {uid}"
    )


def _without_server(source: str) -> str:
    """A mount source minus the server it names.

    ``//acct.file.core.windows.net/share`` becomes ``/share`` and
    ``nfs.example:/export`` becomes ``/export``. The server of an Azure Files
    share is the storage account, which stays out of what users are shown and
    what is logged - the same reason :func:`storage_facts` drops the
    ``username=`` option. Devices and pseudo filesystems pass through.
    """
    if source.startswith("//"):
        return "/" + source[2:].partition("/")[2]
    server, separator, export = source.partition(":/")
    if separator and server and "/" not in server:
        return "/" + export
    return source


def explain_unwritable(path: str) -> str:
    """Why ``path`` cannot be written (or created), as specifically as the system can tell.

    About the directory actually in the way: ``path`` itself, or for a path
    that does not exist yet, the nearest parent it would be created in.
    """
    return _explain_access(_nearest_existing(path), write=True, wanted=os.path.abspath(path))


def is_readable_dir(path: str | None) -> bool:
    """Whether ``path`` is a directory this process can list and enter."""
    if not path:
        return False
    return os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK)


def explain_unreadable(path: str) -> str:
    """Why ``path`` cannot be listed, as specifically as the system can tell."""
    return _explain_access(os.path.abspath(path), write=False)


# The mount options worth a log line - the ones that decide who may write. The
# rest (an SMB mount's username, cache settings) are noise at best.
_LOGGED_FLAGS = frozenset({"ro", "rw", "noperm", "forceuid", "forcegid", "cifsacl", "noexec"})
_LOGGED_KEYS = frozenset({"uid", "gid", "file_mode", "dir_mode", "vers", "sec"})


def storage_facts(path: str) -> dict[str, Any]:
    """What a log line needs to diagnose ``path`` without shell access to the pod."""
    target = _nearest_existing(path)
    facts: dict[str, Any] = {
        "path": os.path.abspath(path),
        "checked": target,
        "readable": is_readable_dir(target),
        "writable": is_writable_dir(path),
    }
    try:
        st = os.stat(target)
        facts.update(owner_uid=st.st_uid, owner_gid=st.st_gid, mode=f"{stat.S_IMODE(st.st_mode):04o}")
    except OSError:
        pass
    info = mount_info(target)
    if info is not None:
        facts.update(
            mountpoint=info.mountpoint,
            fstype=info.fstype,
            source=_without_server(info.source),
            options=",".join(o for o in info.options if o in _LOGGED_FLAGS or o.split("=", 1)[0] in _LOGGED_KEYS),
        )
    if not facts["readable"]:
        facts["read_reason"] = explain_unreadable(target)
    if not facts["writable"]:
        facts["write_reason"] = explain_unwritable(path)
    return facts


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

    # One entry per directory, under the name that put it first. Setting
    # BIOMNI_OUTPUT_ROOT to <workspace>/biomni-outputs - the GRIP setup - makes it
    # the workspace default too, and a directory that cannot be used would
    # otherwise be reported twice, under two names.
    unique: dict[str, str] = {}
    for path, source in candidates:
        unique.setdefault(path, source)
    return list(unique.items())


def _describe_skipped(entries: list[tuple[str, str, str]]) -> str:
    """Which configured locations were passed over, and exactly why.

    ``entries`` are ``(path, source, reason)``. Names the setting and the path,
    because "fell back from env" told a reader that something was rejected
    without telling them what to go and fix. Grouped by the directory actually
    in the way: ``BIOMNI_OUTPUT_ROOT`` and the workspace default usually fail
    together, on the same mount, and explaining that twice buried the one fix
    both needed.
    """
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for entry in entries:
        groups.setdefault(_nearest_existing(entry[0]), []).append(entry)
    uid = getattr(os, "getuid", lambda: None)()
    by_whom = f" by uid {uid}" if uid is not None else ""
    sentences = []
    for members in groups.values():
        names = _join_names([f"{_SOURCE_LABELS.get(source, source)} ({path})" for path, source, _ in members])
        verb = "is" if len(members) == 1 else "are"
        sentence = f"{names} {verb} not writable{by_whom}: {members[0][2]}"
        # Most labels start "the ...", and each one starts a sentence.
        sentences.append(sentence[0].upper() + sentence[1:])
    return ". ".join(sentences)


def _join_names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


# (source, path, reason) combinations already warned about. Resolution runs
# several times a session, and the same unwritable mount repeated on every one
# buried everything else in the log; a changed reason is new, and is logged.
_warned_unwritable: set[tuple[str, str, str]] = set()


def _warn_unwritable_once(source: str, path: str, reason: str) -> None:
    key = (source, path, reason)
    if key in _warned_unwritable:
        return
    _warned_unwritable.add(key)
    logger.warning(
        "output location %s (%s) is not writable, falling back to the next candidate: %s",
        _SOURCE_LABELS.get(source, source),
        path,
        reason,
    )


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
    skipped: list[tuple[str, str, str]] = []

    for path, source in candidates:
        if is_writable_dir(path):
            return OutputTarget(
                path=path,
                source=source,
                writable=True,
                reason=_describe_skipped(skipped) if skipped else None,
                mounted_volume=on_mounted_volume(path),
            )
        # Worked out once: it reads the mount table and asks the file system,
        # which on an SMB share is a round trip to the server.
        reason = explain_unwritable(path)
        if source != "cwd-fallback":
            _warn_unwritable_once(source, path, reason)
        skipped.append((path, source, reason))

    path, source = candidates[-1]
    tried = _describe_skipped(skipped)
    logger.error("no writable output location found: %s", tried)
    return OutputTarget(path=path, source=source, writable=False, reason=tried)


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
