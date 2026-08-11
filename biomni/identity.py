"""Caller identity, as supplied by the deployment's authentication gateway.

On the GRIP platform Biomni sits behind an authentication gateway that validates
access and forwards the caller's details as HTTP headers (user id, e-mail and
workspace id). Nothing in this module talks to an identity provider itself - it
only *reads* what the gateway asserts, which is the whole point: the trust
boundary is the gateway, and the app must never accept these headers from a
client that reached it directly.

Three things make this safe to land before the gateway spec is final:

* **Header names are configurable.** Each field is looked up through a list of
  candidate names, overridable per-deployment via ``BIOMNI_AUTH_*_HEADER``
  (comma-separated). When GRIP publishes its final header names, that is a
  values change in the manifest, not a code change.
* **Header shape is normalised.** ``extract_headers`` accepts a plain mapping,
  a WSGI/socket.io ``environ`` (``HTTP_X_FOO`` keys) or a raw ASGI scope
  (list of byte pairs), because Chainlit hands a different shape depending on
  where in the request lifecycle you ask.
* **Absence is not an error.** With no gateway in front (local dev, the current
  deployment), :func:`resolve_identity` returns an anonymous identity keyed by
  the Chainlit session. Per the GRIP team, anonymous sessions are not a GRIP
  use case, but a *new* user is: they authenticate normally and simply have no
  stored preferences yet, which is handled one layer up by falling back to
  defaults rather than by anything here.

Storage keys are deliberately not raw identifiers - see :meth:`UserIdentity.storage_key`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

# Candidate header names per field, tried in order. The defaults cover the
# conventions of the common gateways (oauth2-proxy, Envoy/ext_authz, generic
# reverse proxies) so a deployment usually needs no override at all.
_DEFAULT_USER_ID_HEADERS = (
    "x-auth-request-user-id",
    "x-auth-request-user",
    "x-forwarded-user",
    "x-user-id",
)
_DEFAULT_EMAIL_HEADERS = (
    "x-auth-request-email",
    "x-forwarded-email",
    "x-user-email",
)
_DEFAULT_WORKSPACE_HEADERS = (
    "x-workspace-id",
    "x-auth-request-workspace-id",
    "x-grip-workspace-id",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SLUG_UNSAFE_RE = re.compile(r"[^a-z0-9._-]+")


def _header_candidates(env_name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    """Header names to try for one field: the env override first, then defaults.

    The override is additive rather than exclusive so that setting one field's
    header name cannot accidentally break the others' discovery.
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return defaults
    overrides = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not overrides:
        return defaults
    return overrides + tuple(d for d in defaults if d not in overrides)


def _decode(value: Any) -> str:
    """Coerce a header value (``str`` or ``bytes``) to a stripped ``str``."""
    if isinstance(value, bytes):
        try:
            return value.decode("latin-1").strip()
        except (UnicodeDecodeError, AttributeError):
            return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def extract_headers(source: Any) -> dict[str, str]:
    """Normalise any of Chainlit's header carriers to a lowercase-keyed dict.

    Accepts:

    * a mapping of header name -> value (``starlette.datastructures.Headers``,
      a plain dict, ...),
    * a WSGI / socket.io ``environ`` where headers appear as ``HTTP_X_FOO``,
    * an ASGI scope containing ``headers`` as a list of ``(bytes, bytes)``.

    Unknown shapes yield an empty dict rather than raising - identity is
    best-effort and must never be able to break session start.
    """
    if not source:
        return {}

    headers: dict[str, str] = {}
    try:
        # ASGI scope (or anything else carrying a raw header pair list).
        raw_pairs = source.get("headers") if isinstance(source, dict) else None
        if isinstance(raw_pairs, list | tuple):
            for pair in raw_pairs:
                if not isinstance(pair, list | tuple) or len(pair) != 2:
                    continue
                name = _decode(pair[0]).lower()
                if name:
                    headers[name] = _decode(pair[1])

        if isinstance(source, dict):
            for key, value in source.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("HTTP_"):
                    # WSGI/socket.io environ: HTTP_X_AUTH_REQUEST_EMAIL -> x-auth-request-email
                    headers[key[5:].replace("_", "-").lower()] = _decode(value)
                elif key == "headers":
                    continue  # already handled above
                elif "-" in key or key.islower():
                    # Already header-shaped (plain mapping of real header names).
                    headers.setdefault(key.lower(), _decode(value))
            # An ASGI scope is often nested under a socket.io environ.
            nested = source.get("asgi.scope")
            if isinstance(nested, dict):
                for name, value in extract_headers(nested).items():
                    headers.setdefault(name, value)
        else:
            items = getattr(source, "items", None)
            if callable(items):
                for key, value in items():
                    if isinstance(key, str):
                        headers[key.lower()] = _decode(value)
    except Exception:  # pragma: no cover - defensive: never break session start
        logger.debug("could not extract headers from %r", type(source), exc_info=True)

    return {name: value for name, value in headers.items() if value}


def _first_present(headers: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = headers.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class UserIdentity:
    """Who the gateway says is calling.

    ``user_id`` is the stable subject to key preferences by. ``email`` is for
    display only - never build storage paths from it directly (see
    :meth:`storage_key`). ``workspace_id`` scopes a user's preferences to one
    workspace, so the same person working in two GRIP workspaces gets the data
    scope and output directory appropriate to each.
    """

    user_id: str | None = None
    email: str | None = None
    workspace_id: str | None = None
    source: str = "anonymous"  # "headers" | "chainlit-user" | "anonymous"

    @property
    def is_authenticated(self) -> bool:
        return self.source != "anonymous" and bool(self.user_id or self.email)

    @property
    def display_name(self) -> str:
        """Human label for the UI. Prefers e-mail; never returns an empty string."""
        return self.email or self.user_id or "anonymous"

    def storage_key(self) -> str:
        """Filesystem-safe, stable key for this user's stored preferences.

        Two properties matter more than readability:

        * **No path traversal.** The key becomes a filename, and the raw value
          is attacker-influenced in the sense that we do not control what the
          gateway sends. Everything outside ``[a-z0-9._-]`` is replaced, and a
          hash suffix keeps distinct ids from colliding after slugging.
        * **No e-mail addresses on disk.** When the only identifier we have is
          an e-mail, the key is hash-only. The address is still stored *inside*
          the preferences payload where it is useful for support, but it does
          not leak into directory listings or log lines that echo the key.
          This matches the deployment's ``BIOMNI_LOG_REDACT_EMAILS`` posture.
        """
        raw = (self.user_id or self.email or "").strip()
        if not raw:
            return "anonymous"

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        if _EMAIL_RE.match(raw):
            return f"u-{digest}"

        slug = _SLUG_UNSAFE_RE.sub("-", raw.lower()).strip("-._")[:40]
        return f"{slug}-{digest}" if slug else f"u-{digest}"

    def scoped_key(self) -> str:
        """Storage key including the workspace, when the gateway supplies one."""
        base = self.storage_key()
        if not self.workspace_id:
            return base
        ws = _SLUG_UNSAFE_RE.sub("-", self.workspace_id.lower()).strip("-._")[:32]
        return f"{base}@{ws}" if ws else base


def identity_from_headers(source: Any) -> UserIdentity | None:
    """Build an identity from gateway headers, or ``None`` when absent.

    Returning ``None`` (rather than an anonymous identity) lets callers tell
    "no gateway in front of us" apart from "gateway said anonymous", which are
    different situations operationally.
    """
    headers = extract_headers(source)
    if not headers:
        return None

    user_id = _first_present(headers, _header_candidates("BIOMNI_AUTH_USER_ID_HEADER", _DEFAULT_USER_ID_HEADERS))
    email = _first_present(headers, _header_candidates("BIOMNI_AUTH_EMAIL_HEADER", _DEFAULT_EMAIL_HEADERS))
    workspace_id = _first_present(
        headers, _header_candidates("BIOMNI_AUTH_WORKSPACE_HEADER", _DEFAULT_WORKSPACE_HEADERS)
    )

    if not (user_id or email):
        return None

    return UserIdentity(user_id=user_id, email=email, workspace_id=workspace_id, source="headers")


def trust_auth_headers() -> bool:
    """Whether identity headers may be believed (env ``BIOMNI_TRUST_AUTH_HEADERS``).

    Fails closed, and that is the whole point. These headers are an assertion
    that only an authentication gateway is entitled to make: if the app is
    reachable without one in front - which is true of the shipped manifest, of
    local development, and of any misrouted ingress - then anyone can send
    ``x-user-id: <someone else>`` and read or overwrite that person's stored
    preferences and run history.

    So a deployment must state explicitly that it sits behind a gateway which
    strips client-supplied copies of these headers. Until it does, headers are
    ignored and every session is anonymous.
    """
    return os.getenv("BIOMNI_TRUST_AUTH_HEADERS", "").strip().lower() in {"1", "true", "yes", "on"}


def anonymous_identity(session_id: str | None = None) -> UserIdentity:
    """Identity for a session with no gateway in front of it.

    Keyed by the Chainlit session so preferences still persist for the life of
    that session (and no further) - enough for local development, and never
    reached on GRIP where the gateway always authenticates.
    """
    return UserIdentity(user_id=f"session-{session_id}" if session_id else None, source="anonymous")


def resolve_identity(header_source: Any = None, *, session_id: str | None = None) -> UserIdentity:
    """Identity for a session: gateway headers when trusted, else anonymous.

    Headers are only consulted when :func:`trust_auth_headers` is on, so an
    unprotected deployment cannot be impersonated by a hand-crafted request.
    """
    if not trust_auth_headers():
        if identity_from_headers(header_source) is not None:
            # Someone is sending identity headers at an app that has not been
            # told it sits behind a gateway. That is either a misconfiguration
            # or an impersonation attempt; both are worth a log line.
            logger.warning(
                "ignoring identity headers: BIOMNI_TRUST_AUTH_HEADERS is not enabled. "
                "Set it only when an authentication gateway strips client-supplied copies of these headers."
            )
        return anonymous_identity(session_id)

    identity = identity_from_headers(header_source)
    if identity is not None:
        return identity
    return anonymous_identity(session_id)
