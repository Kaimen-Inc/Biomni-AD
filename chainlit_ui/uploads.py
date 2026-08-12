"""Put attached files where the user can find them again.

Chainlit stores an upload as ``<app root>/.files/<session id>/<uuid>.<ext>`` and
treats that directory as scratch: it deletes the session's folder when the
session ends, and the whole tree on shutdown. Two things follow, and both are
wrong for a research tool:

* the file the user attached is gone by their next visit, along with any
  analysis that points at it;
* the agent is handed ``ef4edb5c-f251-465d-8bd0-9d2531dc68e2.csv``, so the
  filename the user chose - which is usually what the file *means* - never
  reaches the model or the transcript.

So attachments are copied into the output directory the user configured, under
``uploads/``, keeping their original names. Copied rather than moved, and
Chainlit's own copy is left alone: it owns that lifecycle, it serves the file
back to the browser from there, and pointing its scratch path at the user's
directory would have it delete their results on session end.

Pure path/IO helpers, no Chainlit import, so the naming rules are testable.
"""

from __future__ import annotations

import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

UPLOADS_DIRNAME = "uploads"

# Anything outside this set is replaced. Deliberately strict: the name comes
# from a filesystem this process does not control, it is echoed into prompts and
# log lines, and it becomes a path.
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FALLBACK_NAME = "upload"
_MAX_NAME_LEN = 120


def uploads_dir(output_dir: str) -> str:
    """Where attachments for this session are kept."""
    return os.path.join(output_dir, UPLOADS_DIRNAME)


def safe_filename(name: str | None) -> str:
    """A filename that cannot escape its directory or surprise a shell.

    Keeps the extension, since that is what tells the agent (and pandas) how to
    read the file.
    """
    base = os.path.basename(name or "").strip()
    # A Windows-style path arrives as one segment on POSIX; basename would keep
    # the whole thing, so split on the other separator too.
    base = base.rsplit("\\", 1)[-1]
    cleaned = _UNSAFE_CHARS_RE.sub("_", base).strip("._-")
    if not cleaned:
        return _FALLBACK_NAME
    if len(cleaned) <= _MAX_NAME_LEN:
        return cleaned
    stem, ext = os.path.splitext(cleaned)
    return stem[: max(1, _MAX_NAME_LEN - len(ext))] + ext


def unique_destination(directory: str, filename: str) -> str:
    """A path in ``directory`` that does not overwrite an existing file.

    Second and later copies of ``results.csv`` become ``results (2).csv``, the
    convention every desktop uses. Silently overwriting would be worse than a
    cluttered folder: two runs would disagree about what "the file I uploaded"
    contains.
    """
    stem, ext = os.path.splitext(safe_filename(filename))
    candidate = os.path.join(directory, f"{stem}{ext}")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem} ({counter}){ext}")
        counter += 1
    return candidate


def store_upload(source_path: str, directory: str, filename: str | None = None) -> str | None:
    """Copy one attachment into ``directory``. ``None`` if it could not be.

    Never raises. A failed copy means the caller falls back to Chainlit's
    temporary path, which still works for the current message - losing the
    durable copy is not a reason to lose the attachment.
    """
    if not source_path or not os.path.isfile(source_path):
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        destination = unique_destination(directory, filename or os.path.basename(source_path))
        shutil.copy2(source_path, destination)
        return destination
    except OSError:
        logger.warning("could not copy upload into %s", directory, exc_info=True)
        return None


def store_uploads(sources: list[tuple[str, str | None]], directory: str) -> list[str]:
    """Copy several attachments, returning the path each one ended up at.

    Falls back to the original path per file, so a partial failure still gives
    the agent something usable for every attachment.
    """
    stored: list[str] = []
    for source_path, name in sources:
        stored.append(store_upload(source_path, directory, name) or source_path)
    return stored
