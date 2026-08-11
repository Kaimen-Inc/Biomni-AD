"""Rendering for the workspace scope panel, inventory and run history.

Everything here is a pure string/data builder: no Chainlit import, no event
loop, no I/O beyond the bounded scanner. The Chainlit layer decides *where* to
show these strings; this module decides *what* they say, which is the part worth
testing (and CI installs no chainlit extra, so it has to be importable without
it).

The central behaviour is the difference between a chosen scope and no choice
at all:

* **Scope chosen** - only those folders are scanned and described to the agent.
  A workspace with thousands of files costs the same as one with ten, because
  the files nobody selected are never visited.
* **No choice yet** - a new user must not pay for a full traversal just to ask
  a question, so the workspace is advertised by top-level folder *names* only,
  read with a single directory listing, and the agent is told to enumerate on
  demand. This is what a first-time GRIP user gets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from biomni.fs_scan import EXCLUDED_DIRS, scan_directory

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from biomni.run_registry import RunRecord
    from biomni.workspace_prefs import OutputTarget, ScopeResolution

logger = logging.getLogger(__name__)

# How many files of a selected folder to show in the agent-facing inventory.
_INVENTORY_PREVIEW_FILES = 500
_INVENTORY_TREE_LINES = 300
# ... and in the human-facing sidebar, which needs to stay skimmable.
_PANEL_PREVIEW_FILES = 200
_PANEL_TREE_LINES = 60

_STATUS_ICONS = {
    "running": "🔄",
    "completed": "✅",
    "failed": "⚠️",
    "interrupted": "⏸",
    "cancelled": "🚫",
}


# --------------------------------------------------------------------------- #
# Cheap workspace introspection
# --------------------------------------------------------------------------- #


def list_top_level_dirs(workspace_root: str | None, *, limit: int = 200) -> list[str]:
    """Names of the immediate subdirectories of the workspace.

    One ``scandir`` call, no recursion: this runs on every session start and
    must stay cheap on a network-backed mount where each stat is a round trip.
    Deliberately returns names without file counts - counting means walking, and
    walking every folder just to label a picker is the cost this whole change
    exists to remove.
    """
    if not workspace_root or not os.path.isdir(workspace_root):
        return []
    names: list[str] = []
    try:
        with os.scandir(workspace_root) as entries:
            for entry in entries:
                if entry.name.startswith(".") or entry.name in EXCLUDED_DIRS:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        names.append(entry.name)
                except OSError:  # pragma: no cover - transient mount error
                    continue
    except OSError:
        logger.warning("could not list workspace root %s", workspace_root, exc_info=True)
        return []
    return sorted(names)[:limit]


@dataclass(frozen=True)
class ScopeEntrySummary:
    """One selected folder, with the cost of looking at it."""

    label: str
    path: str
    file_count: int
    bounded: bool

    def count_label(self) -> str:
        return f"{self.file_count}+" if self.bounded else str(self.file_count)


def summarize_scope(scope: ScopeResolution, workspace_root: str | None) -> list[ScopeEntrySummary]:
    """Bounded scan of each selected root, for display and for the inventory."""
    from biomni.workspace_prefs import relative_label

    summaries: list[ScopeEntrySummary] = []
    for root in scope.roots:
        result = scan_directory(root, max_depth=10)
        summaries.append(
            ScopeEntrySummary(
                label=relative_label(root, workspace_root),
                path=root,
                file_count=result.file_count,
                bounded=result.bounded,
            )
        )
    return summaries


# --------------------------------------------------------------------------- #
# Tree preview
# --------------------------------------------------------------------------- #


def build_tree_preview_lines(paths: list[str], max_lines: int = 60, max_depth: int = 3) -> list[str]:
    """Render relative file paths as a compact folder tree preview.

    Directory counts are computed from the provided path sample.
    """
    # Tree node shape: {"dirs": {name: node}, "files": [name, ...]}
    tree: dict[str, object] = {"dirs": {}, "files": []}

    for rel_path in sorted(paths):
        parts = [p for p in rel_path.split("/") if p]
        if not parts:
            continue

        node = tree
        for idx, part in enumerate(parts):
            is_file = idx == len(parts) - 1
            if is_file:
                files = node.setdefault("files", [])
                if isinstance(files, list):
                    files.append(part)
            else:
                dirs = node.setdefault("dirs", {})
                if not isinstance(dirs, dict):
                    break
                if part not in dirs:
                    dirs[part] = {"dirs": {}, "files": []}
                child = dirs.get(part)
                if not isinstance(child, dict):
                    break
                node = child

    lines: list[str] = []

    def _count_files(node: dict[str, object]) -> int:
        count = 0
        files = node.get("files", [])
        dirs = node.get("dirs", {})

        if isinstance(files, list):
            count += len(files)
        if isinstance(dirs, dict):
            for child in dirs.values():
                if isinstance(child, dict):
                    count += _count_files(child)
        return count

    def _render(node: dict[str, object], prefix: str, depth: int) -> bool:
        if len(lines) >= max_lines:
            return False
        dirs = node.get("dirs", {})
        files = node.get("files", [])

        dir_names = sorted(dirs.keys()) if isinstance(dirs, dict) else []
        file_names = sorted(str(f) for f in files) if isinstance(files, list) else []
        entries: list[tuple[str, str, object | None]] = []
        for dirname in dir_names:
            child = dirs.get(dirname) if isinstance(dirs, dict) else None
            entries.append(("dir", dirname, child))
        for filename in file_names:
            entries.append(("file", filename, None))

        for idx, (kind, name, child) in enumerate(entries):
            is_last = idx == len(entries) - 1
            branch = "└─ " if is_last else "├─ "
            next_prefix = prefix + ("   " if is_last else "│  ")

            if kind == "dir":
                child_count = _count_files(child) if isinstance(child, dict) else 0
                lines.append(f"{prefix}{branch}📁 {name}/ ({child_count})")
                if len(lines) >= max_lines:
                    return False
                if isinstance(child, dict):
                    if depth + 1 < max_depth:
                        if not _render(child, next_prefix, depth + 1):
                            return False
                    elif child_count > 0:
                        lines.append(f"{next_prefix}…")
                        if len(lines) >= max_lines:
                            return False
            else:
                lines.append(f"{prefix}{branch}📄 {name}")
                if len(lines) >= max_lines:
                    return False

        return True

    _render(tree, prefix="", depth=0)
    return lines


# --------------------------------------------------------------------------- #
# Agent-facing inventory
# --------------------------------------------------------------------------- #


def build_scope_inventory(
    scope: ScopeResolution,
    workspace_root: str | None,
    *,
    top_level_dirs: Sequence[str] | None = None,
) -> str:
    """The workspace description injected into the agent's system prompt.

    With a selection, this is a real file tree of the selected folders. Without
    one, it is a list of folder names plus an explicit instruction to enumerate
    on demand - the agent must never conclude the workspace is empty just
    because nothing has been indexed yet.
    """
    if not workspace_root:
        return ""

    if scope.is_default or not scope.roots:
        names = list(top_level_dirs if top_level_dirs is not None else list_top_level_dirs(workspace_root))
        lines = [f"User workspace ({workspace_root})", ""]
        if scope.missing:
            # Every selected folder has disappeared. Saying "nothing selected"
            # here would hide a real problem from the agent, which would then
            # happily plan around data the user believes is in scope.
            lines.append(
                "The folders previously selected as the data scope are GONE (deleted or renamed): "
                + ", ".join(scope.missing)
                + ". Tell the user rather than silently substituting other data."
            )
            lines.append("")
        lines.append("No data scope is currently available, so the workspace has NOT been indexed.")
        if names:
            lines.append("Top-level folders available:")
            lines.extend(f"  📁 {name}/" for name in names)
        else:
            lines.append("No top-level folders were found directly under the workspace root.")
        lines.append(
            "Use os.listdir()/glob on the paths above to discover files when the task needs them. "
            "Do not assume the workspace is empty."
        )
        return "\n".join(lines)

    sections: list[str] = []
    for summary in summarize_scope(scope, workspace_root):
        result = scan_directory(summary.path, max_depth=10)
        preview = result.files[:_INVENTORY_PREVIEW_FILES]
        tree_lines = build_tree_preview_lines(preview, max_lines=_INVENTORY_TREE_LINES, max_depth=6)
        section = f"{summary.label} ({summary.path}) - {summary.count_label()} files:\n" + "\n".join(tree_lines)
        if result.bounded:
            section += (
                f"\n  ... only the first {len(preview)} files are listed (scan bounded for responsiveness; "
                "use os.listdir()/glob on the path for the rest)"
            )
        elif result.file_count > len(preview):
            section += f"\n  ... and {result.file_count - len(preview)} more files"
        sections.append(section)

    header = (
        f"The user selected these folders of their workspace ({workspace_root}) as the data scope "
        "for this session. Prefer them over anything else on disk."
    )
    body = "\n\n".join(sections) if sections else "(the selected folders contain no files)"

    if scope.missing:
        body += "\n\nSelected but no longer present (do not attempt to read): " + ", ".join(scope.missing)

    return f"{header}\n\n{body}"


# --------------------------------------------------------------------------- #
# Human-facing panels
# --------------------------------------------------------------------------- #


def build_scope_panel(
    scope: ScopeResolution,
    workspace_root: str | None,
    output: OutputTarget | None,
    *,
    summaries: Sequence[ScopeEntrySummary] | None = None,
    top_level_dirs: Sequence[str] | None = None,
    persistence: str | None = None,
) -> str:
    """The sidebar page: what the agent can read, and where results will go."""
    lines: list[str] = []

    if not workspace_root:
        lines.append("No user workspace is configured for this deployment.")
    else:
        lines.append(f"**Workspace**\n`{workspace_root}`")
        lines.append("")
        lines.append("**Active data scope**")
        if scope.roots:
            entries = list(summaries if summaries is not None else summarize_scope(scope, workspace_root))
            lines.append("")
            for entry in entries:
                suffix = "file" if (entry.file_count == 1 and not entry.bounded) else "files"
                lines.append(f"- 📁 `{entry.label}/` - {entry.count_label()} {suffix}")
        else:
            names = list(top_level_dirs if top_level_dirs is not None else list_top_level_dirs(workspace_root))
            lines.append("")
            lines.append("_Nothing selected yet - the agent will look only where a task points it._")
            if names:
                lines.append("")
                lines.append(f"Top-level folders ({len(names)}):")
                lines.extend(f"- 📁 `{name}/`" for name in names[:40])
                if len(names) > 40:
                    lines.append(f"- _... and {len(names) - 40} more_")
            lines.append("")
            lines.append("Use ⚙️ **Settings** to pick the folders you are working with.")

        # Reported whether or not anything is still selected: a scope that
        # resolved to nothing because its folders were deleted must not read as
        # "you never chose".
        if scope.missing:
            lines.append("")
            lines.append("**Selected but missing**")
            lines.extend(f"- ⚠️ `{name}` (deleted or renamed)" for name in scope.missing)

    if output is not None:
        lines.append("")
        lines.append("**Outputs**")
        lines.append(f"`{output.path}`")
        if not output.writable:
            lines.append("")
            lines.append(f"⚠️ Not writable. {output.reason or ''}".rstrip())
        elif output.is_ephemeral:
            lines.append("")
            lines.append(
                "⚠️ This is container-local storage: results are lost when the application restarts. "
                "Set an output directory in ⚙️ Settings, or ask an operator to configure "
                "`BIOMNI_OUTPUT_ROOT`."
            )

    if persistence:
        lines.append("")
        lines.append(f"_Settings storage: {persistence}_")

    return "\n".join(lines)


def build_runs_panel(records: Sequence[RunRecord], *, limit: int = 10) -> str:
    """The sidebar page listing this user's recent runs across sessions."""
    if not records:
        return "No runs recorded yet.\n\nCompleted runs will be listed here, including ones from earlier sessions."

    lines: list[str] = []
    for record in list(records)[:limit]:
        icon = _STATUS_ICONS.get(record.status, "•")
        lines.append(f"{icon} **{record.label}**")
        lines.append(f"   {record.status} · {record.created_at}")
        if record.output_dir:
            lines.append(f"   `{record.output_dir}`")
        if record.error:
            lines.append(f"   _{record.error}_")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_previous_runs_notice(records: Iterable[RunRecord]) -> str | None:
    """A chat message about runs that ended badly while the user was away.

    Only unfinished work is worth interrupting someone with on arrival;
    completed runs are in the sidebar for whenever they want them.
    """
    unfinished = [r for r in records if r.status in {"interrupted", "failed"}]
    if not unfinished:
        return None

    lines = [
        f"**{len(unfinished)} earlier run{'s' if len(unfinished) > 1 else ''} did not finish.**",
        "",
    ]
    for record in unfinished[:5]:
        icon = _STATUS_ICONS.get(record.status, "•")
        lines.append(f"- {icon} {record.label} ({record.status})")
        if record.output_dir:
            lines.append(f"  Partial results: `{record.output_dir}`")
    if len(unfinished) > 5:
        lines.append(f"- _... and {len(unfinished) - 5} more_")
    lines.append("")
    lines.append("Ask again to re-run any of them.")
    return "\n".join(lines)


def scope_choice_items(names: Iterable[str]) -> dict[str, str]:
    """``{label: value}`` for the settings multi-select."""
    return {f"📁 {name}": name for name in names}
