"""Provider credentials, and keeping them out of reach of generated code.

The agent executes LLM-generated Python in-process (:func:`biomni.tool.support_tools.run_python_repl`
calls ``exec``), so every environment variable the process holds is readable by
whoever writes the prompt - including ``ANTHROPIC_API_KEY`` and friends. A single
``print(os.environ)`` exfiltrates them, and because ``subprocess`` children
inherit the parent environment, so does a single ``printenv``.

This module closes that hole with a scrub window: :func:`scrubbed_environ`
removes credential-shaped variables from the real ``os.environ`` for the
duration of a block and puts them back afterwards. Mutating the real mapping is
deliberate - handing a filtered copy into the ``exec`` globals would be defeated
by a bare ``import os`` inside the snippet, and would not cover what a
subprocess inherits.

Two consequences of that choice are handled here rather than left to callers:

* **Concurrent runs must not fight over the window.** Sessions run in parallel
  (one per chat), so the window is reference-counted: the first entrant strips
  the variables, the last one out restores them. Nested and overlapping windows
  are both safe, and no run ever waits on another - the lock is held only for
  the bookkeeping, never for the duration of the block.
* **Legitimate readers still need the values.** Anything that genuinely needs a
  credential (the LLM factory) reads it through :func:`getenv`, which returns
  the true value whether or not a scrub is in flight. That is what makes the
  window race-free: a chat starting while another chat runs code still builds a
  working client, and never has to reach into ``os.environ`` to find out.

Restoration overwrites rather than merges, so a snippet that assigns
``os.environ["ANTHROPIC_API_KEY"] = "junk"`` inside the window cannot poison the
real credential either.

The classifier is name-based (:func:`is_credential`) and deliberately matches
secret-*shaped* names only. Blanket-blocking ``AWS_*`` / ``AZURE_*`` would hide
``AWS_REGION`` and ``ENDPOINT_URL``, which generated analysis code has a
legitimate reason to read; those are endpoints, not secrets.

**What this is not.** It is not a sandbox. Any secret this process can recover,
code running *in* this process can recover the same way - including through this
module. It closes the accidental and casual paths, which is the realistic
failure mode: a model that prints its configuration, a library that dumps the
environment into a traceback, a snippet that shells out. A determined attacker
who knows the codebase is out of scope, and defending against one needs the
executor moved to a separate process or sandbox (see ARCHITECTURE.md §9.4).
Deployments exposed to untrusted users should pair this with a provider-side
spend cap and a key scoped to that deployment.

**Known gap.** boto3 (and therefore ``ChatBedrock``) resolves AWS credentials
from the environment when its client is constructed, so a Bedrock client built
inside a scrub window would not find env-supplied keys. It falls back to
``~/.aws`` and instance metadata, which are unaffected. Nothing in this repo
uses Bedrock by default; wire the keys through :func:`getenv` explicitly if that
changes.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Substrings that make a variable name credential-shaped, matched
# case-insensitively against the whole name.
#
# Bare "AUTH" is deliberately absent: it would match configuration such as
# BIOMNI_AUTH_USER_ID_HEADER (a header *name*, not a secret), while the actual
# auth secrets in use - SYNAPSE_AUTH_TOKEN, CHAINLIT_AUTH_SECRET - are already
# caught by TOKEN and SECRET.
_CREDENTIAL_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "APIKEY",
    "CREDENTIAL",
    "PASSWD",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_KEY",
    "TOKEN",
)

# Credentials whose names carry no marker of their own.
_CREDENTIAL_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
)

# Secrets the server itself re-reads at request time. Hiding these does not
# protect them - CHAINLIT_AUTH_SECRET is persisted to
# ``$BIOMNI_STATE_DIR/threads/auth_secret`` by chainlit_ui/persistence.py, so
# generated code can read it off disk whatever the environment says - but it
# does break the process: Chainlit calls ``get_jwt_secret()`` on every JWT
# operation, so while one chat ran a code step, every other user reconnecting or
# resuming a thread hit ``assert secret`` and got a 500.
_NEVER_SCRUBBED = frozenset({"CHAINLIT_AUTH_SECRET"})

_EXTRA_ENV = "BIOMNI_SCRUB_ENV_EXTRA"
_ALLOW_ENV = "BIOMNI_SCRUB_ENV_ALLOW"


@functools.lru_cache(maxsize=4)
def _name_set(env_name: str) -> frozenset[str]:
    """Parse a comma-separated env override into a set of upper-cased names.

    Cached on first read, and deliberately so. Neither override name is itself
    credential-shaped, so without this a snippet could set
    ``BIOMNI_SCRUB_ENV_ALLOW=ANTHROPIC_API_KEY`` inside one window and read the
    key in the next - process-wide, disabling the scrub for every other chat.
    Configuration is a deployment decision, not something a prompt gets to make.
    """
    raw = os.getenv(env_name, "")
    return frozenset(part.strip().upper() for part in raw.split(",") if part.strip())


def is_credential(name: str) -> bool:
    """Whether ``name`` looks like a secret that generated code must not read.

    Deployments can extend the set with ``BIOMNI_SCRUB_ENV_EXTRA`` or exempt a
    variable with ``BIOMNI_SCRUB_ENV_ALLOW`` (both comma-separated). The
    allowlist wins, so a deployment whose analysis code genuinely needs a
    token-shaped variable has an escape hatch that does not require disabling
    the scrub wholesale.
    """
    upper = name.upper()
    if upper in _NEVER_SCRUBBED or upper in _name_set(_ALLOW_ENV):
        return False
    if upper in _CREDENTIAL_NAMES or upper in _name_set(_EXTRA_ENV):
        return True
    return any(marker in upper for marker in _CREDENTIAL_MARKERS)


# Bookkeeping for the scrub window. ``_lock`` guards ``_depth``/``_stashed`` and
# is never held across the caller's block - only across the strip and restore.
_lock = threading.Lock()
_depth = 0
# Windows currently open per thread. Reference counting alone is correct while
# every window is closed by the thread that opened it, but ``run_with_timeout``
# abandons threads it cannot kill, and an abandoned increment would leave the
# process permanently stripped. :func:`release_thread` lets the survivor drop
# exactly the abandoned thread's share - resetting the global count instead
# would clobber a window another chat still has open.
_thread_depth: dict[int, int] = {}
_stashed: dict[str, str] = {}


def _restore_locked() -> None:
    """Put the credentials back. Caller holds the lock and has seen depth reach 0."""
    for name, value in _stashed.items():
        os.environ[name] = value
    _stashed.clear()


@contextlib.contextmanager
def scrubbed_environ() -> Iterator[frozenset[str]]:
    """Hide credential-shaped variables from ``os.environ`` for the block.

    Yields the names that were hidden (empty when a window was already open,
    since the outermost entrant owns the strip). Restoration is unconditional:
    values assigned to a scrubbed name inside the window are discarded.
    """
    global _depth

    ident = threading.get_ident()
    with _lock:
        hidden: frozenset[str] = frozenset()
        if _depth == 0:
            hidden = frozenset(name for name in list(os.environ) if is_credential(name))
            for name in hidden:
                _stashed[name] = os.environ.pop(name)
            if hidden:
                logger.debug("scrubbed %d credential variable(s) for code execution", len(hidden))
        _depth += 1
        _thread_depth[ident] = _thread_depth.get(ident, 0) + 1

    try:
        yield hidden
    finally:
        with _lock:
            _depth -= 1
            remaining = _thread_depth.get(ident, 1) - 1
            if remaining > 0:
                _thread_depth[ident] = remaining
            else:
                _thread_depth.pop(ident, None)
            if _depth <= 0:
                _depth = 0
                _restore_locked()


def getenv(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, seeing through an open scrub window.

    Use this - not ``os.getenv`` - wherever the process itself needs a
    credential, so that building an LLM client while another session is
    executing code still finds the key.
    """
    with _lock:
        if _depth > 0 and name in _stashed:
            return _stashed[name]
    return os.environ.get(name, default)


def release_thread(ident: int | None) -> int:
    """Drop any windows left open by thread ``ident``. Returns how many.

    Called by :func:`biomni.utils.run_with_timeout` after a timeout, because the
    thread it gave up on may have been inside a window and will never run its
    own ``finally``. Only that thread's share is released, so a window another
    chat is still inside is untouched.
    """
    global _depth
    if ident is None:
        return 0
    with _lock:
        orphaned = _thread_depth.pop(ident, 0)
        if not orphaned:
            return 0
        _depth = max(0, _depth - orphaned)
        if _depth == 0:
            _restore_locked()
    logger.warning("released %d credential scrub window(s) left open by an abandoned thread", orphaned)
    return orphaned


def scrub_active() -> bool:
    """Whether a scrub window is currently open. Diagnostics and tests only."""
    with _lock:
        return _depth > 0


__all__ = ["getenv", "is_credential", "release_thread", "scrub_active", "scrubbed_environ"]
