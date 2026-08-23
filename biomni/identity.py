"""Caller identity, as supplied by the deployment's authentication gateway.

On the GRIP platform Biomni sits behind an authentication gateway that validates
access and forwards the caller's details as HTTP headers (subject id, e-mail,
name and workspace id). Nothing in this module talks to an identity provider
itself - it only *reads* what the gateway asserts, which is the whole point: the
trust boundary is the gateway, and the app must never accept these headers from a
client that reached it directly.

Four things keep this robust across deployments:

* **GRIP's context headers are the primary source.** The gateway sends two
  composite headers whose value is a comma-separated list of ``key=value``
  pairs::

      Ai-App-User-Context: email=jane@grip.org,given_name=Jane,family_name=Doe,sub=abc-123,iss=https://keycloak/realms/grip
      Ai-App-Workspace-Context: uuid=6f1c0e9e-...

  ``sub`` is the stable subject id, and is what preferences, run records and
  chat threads are keyed by; ``uuid`` scopes that user to one workspace.
  ``iss`` is optional and, when sent, is folded into the storage key: a subject
  id is unique within an issuer's realm rather than globally, so the pair is
  what survives an IdP change. It must be sent from the first deployment or not
  at all - introducing it later re-keys every existing user (see
  :meth:`UserIdentity.storage_key`).

  Values are read as CSV: a value containing a comma may be quoted, and a
  literal quote inside a quoted value is doubled per RFC 4180.
* **Other gateways still work, and header names stay configurable.** Every
  field also has a list of single-value header names (oauth2-proxy,
  Envoy/ext_authz, generic reverse proxies), consulted for whatever the context
  headers did not supply and overridable per-deployment via
  ``BIOMNI_AUTH_*_HEADER`` (comma-separated). A renamed header is therefore a
  values change in the manifest, never a code change.
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

# GRIP's composite context headers. Each carries several fields as a
# comma-separated list of key=value pairs, so they are parsed rather than read.
_DEFAULT_USER_CONTEXT_HEADERS = ("ai-app-user-context",)
_DEFAULT_WORKSPACE_CONTEXT_HEADERS = ("ai-app-workspace-context",)

# Keys inside those headers, per the GRIP specification.
_CONTEXT_SUBJECT_KEY = "sub"
_CONTEXT_ISSUER_KEY = "iss"
_CONTEXT_EMAIL_KEY = "email"
_CONTEXT_GIVEN_NAME_KEY = "given_name"
_CONTEXT_FAMILY_NAME_KEY = "family_name"
_CONTEXT_WORKSPACE_KEY = "uuid"

# Candidate header names per field, tried in order, for gateways that send one
# value per header instead. The defaults cover the conventions of the common
# gateways (oauth2-proxy, Envoy/ext_authz, generic reverse proxies) so a
# deployment usually needs no override at all.
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
# The OIDC issuer. ``sub`` is unique only within one issuer's realm, so a
# deployment that may ever change IdP needs both to key storage safely.
_DEFAULT_ISSUER_HEADERS = (
    "x-auth-request-issuer",
    "x-auth-issuer",
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


def _split_context_pairs(value: str) -> list[str]:
    """Split a context header on commas, ignoring commas inside double quotes.

    A family name legitimately contains a comma ("Smith, Jr."), and a gateway
    that quotes such a value would otherwise have it silently truncated.
    Unquoted values behave exactly like a plain ``split(",")``, so this costs
    nothing in the ordinary case.

    The gateway states it emits valid CSV, so a literal quote inside a quoted
    value arrives doubled per RFC 4180 (``"Smith ""Bud"", Jr."``). A doubled
    quote is consumed as one character and does not end the quoting; the pair is
    collapsed in :func:`parse_context_header` once the surrounding quotes come
    off. If the gateway settles on backslash escaping instead, this is the one
    function that changes.
    """
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    index = 0
    while index < len(value):
        char = value[index]
        if char == '"':
            if in_quotes and value[index + 1 : index + 2] == '"':
                current.append('""')
                index += 2
                continue
            in_quotes = not in_quotes
            current.append(char)
        elif char == "," and not in_quotes:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def parse_context_header(value: str | None) -> dict[str, str]:
    """Parse ``key=value,key=value`` into a lowercase-keyed mapping.

    Tolerant by design - this is untrusted input from another team's service,
    and one malformed field must not cost us the rest of the identity:

    * whitespace around keys, values and separators is stripped (the published
      example has a space after one of the commas);
    * a value may itself contain ``=`` (base64-ish subject ids do), so only the
      first one separates key from value;
    * optional surrounding double quotes are removed, and a doubled quote inside
      them collapses to one (RFC 4180 - see :func:`_split_context_pairs`);
    * fragments with no ``=``, or with an empty key or value, are dropped.
    """
    if not value:
        return {}

    fields: dict[str, str] = {}
    for fragment in _split_context_pairs(value):
        key, separator, raw = fragment.partition("=")
        if not separator:
            continue
        name = key.strip().lower()
        item = raw.strip()
        if len(item) >= 2 and item[0] == '"' and item[-1] == '"':
            item = item[1:-1].replace('""', '"').strip()
        if name and item:
            fields[name] = item
    return fields


def _context_fields(headers: Mapping[str, str], env_name: str, defaults: tuple[str, ...]) -> dict[str, str]:
    """Parsed contents of the first context header present, or an empty dict."""
    return parse_context_header(_first_present(headers, _header_candidates(env_name, defaults)))


@dataclass(frozen=True)
class UserIdentity:
    """Who the gateway says is calling.

    ``user_id`` is the stable subject to key preferences by - GRIP's ``sub``.
    ``issuer`` is the OIDC ``iss`` that minted it; ``sub`` is unique only within
    one issuer's realm, so the pair is what actually identifies a person across
    an IdP change (see :meth:`storage_key`).
    ``email``, ``given_name`` and ``family_name`` are for display only; never
    build storage paths from them directly (see :meth:`storage_key`).
    ``workspace_id`` - GRIP's workspace ``uuid`` - scopes a user's preferences to
    one workspace, so the same person working in two GRIP workspaces gets the
    data scope and output directory appropriate to each.
    """

    user_id: str | None = None
    email: str | None = None
    workspace_id: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    issuer: str | None = None
    source: str = "anonymous"  # "headers" | "local" | "anonymous"

    @property
    def is_authenticated(self) -> bool:
        """Whether somebody actually proved who they are.

        The ``local`` source is deliberately excluded: it is a stable key for an
        un-gated deployment, not a claim about identity, and callers use this
        property to decide how much to trust the session.
        """
        return self.source not in {"anonymous", "local"} and bool(self.user_id or self.email)

    @property
    def full_name(self) -> str:
        """The caller's name as the gateway spells it, or an empty string."""
        return " ".join(part for part in (self.given_name, self.family_name) if part).strip()

    @property
    def display_name(self) -> str:
        """Human label for the UI; never returns an empty string.

        A name when the gateway supplies one, because that is what a person
        recognises as themselves in the corner of a page, then e-mail, then the
        opaque subject id as a last resort.
        """
        return self.full_name or self.email or self.user_id or "anonymous"

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

        # Scoped to the issuer when the gateway names one: a subject id is
        # unique within an IdP realm, not globally, so two realms could
        # otherwise mint the same ``sub`` for two different people and hand one
        # the other's preferences and run history.
        #
        # Note this changes the key. A gateway that starts sending ``iss`` after
        # go-live re-keys every existing user, who then silently finds an empty
        # history - so it must be sent from the first deployment or not at all.
        keyed = f"{self.issuer}|{raw}" if self.issuer else raw
        digest = hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:10]
        if _EMAIL_RE.match(raw):
            return f"u-{digest}"

        # The readable half stays derived from the subject alone: it is there to
        # make a directory listing navigable, and issuer URLs slug into noise.
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

    # GRIP's composite headers first; a gateway that sends them is authoritative
    # about every field it carries.
    user_context = _context_fields(headers, "BIOMNI_AUTH_USER_CONTEXT_HEADER", _DEFAULT_USER_CONTEXT_HEADERS)
    workspace_context = _context_fields(
        headers, "BIOMNI_AUTH_WORKSPACE_CONTEXT_HEADER", _DEFAULT_WORKSPACE_CONTEXT_HEADERS
    )

    # Single-value headers fill in whatever the context headers did not carry,
    # which is how non-GRIP gateways (and a partially populated context) keep
    # working unchanged.
    user_id = user_context.get(_CONTEXT_SUBJECT_KEY) or _first_present(
        headers, _header_candidates("BIOMNI_AUTH_USER_ID_HEADER", _DEFAULT_USER_ID_HEADERS)
    )
    email = user_context.get(_CONTEXT_EMAIL_KEY) or _first_present(
        headers, _header_candidates("BIOMNI_AUTH_EMAIL_HEADER", _DEFAULT_EMAIL_HEADERS)
    )
    workspace_id = workspace_context.get(_CONTEXT_WORKSPACE_KEY) or _first_present(
        headers, _header_candidates("BIOMNI_AUTH_WORKSPACE_HEADER", _DEFAULT_WORKSPACE_HEADERS)
    )
    issuer = user_context.get(_CONTEXT_ISSUER_KEY) or _first_present(
        headers, _header_candidates("BIOMNI_AUTH_ISSUER_HEADER", _DEFAULT_ISSUER_HEADERS)
    )

    if not (user_id or email):
        return None

    return UserIdentity(
        user_id=user_id,
        email=email,
        workspace_id=workspace_id,
        given_name=user_context.get(_CONTEXT_GIVEN_NAME_KEY),
        family_name=user_context.get(_CONTEXT_FAMILY_NAME_KEY),
        issuer=issuer,
        source="headers",
    )


def trust_auth_headers() -> bool:
    """Whether identity headers may be believed (env ``BIOMNI_TRUST_AUTH_HEADERS``).

    Fails closed, and that is the whole point. These headers are an assertion
    that only an authentication gateway is entitled to make: if the app is
    reachable without one in front - which is true of the shipped manifest, of
    local development, and of any misrouted ingress - then anyone can send
    ``Ai-App-User-Context: sub=<someone else>`` and read or overwrite that
    person's stored preferences and run history.

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


def password_identity(username: str) -> UserIdentity:
    """Identity for a caller who signed in with the deployment password.

    ``source="password"`` makes :attr:`UserIdentity.is_authenticated` true, which
    is what turns on durable preferences, run records and the conversation list -
    so each person who picks a distinct username gets their own history, without
    the deployment having to opt into ``BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE``.

    The name is normalised because it becomes a storage key: trimmed, lowercased
    and stripped of anything outside ``[a-z0-9._-]``. An empty result yields an
    unauthenticated identity, which the caller must reject rather than serve.
    """
    slug = _SLUG_UNSAFE_RE.sub("-", (username or "").strip().lower()).strip("-._")[:40]
    if not slug:
        return UserIdentity(source="anonymous")
    return UserIdentity(user_id=slug, source="password")


LOCAL_USER_ID = "local"


def single_user_mode() -> bool:
    """Whether an un-gated deployment should behave as one persistent user.

    With no gateway, :func:`anonymous_identity` keys everything by connection,
    so nothing written can ever be read back - settings reset on reload and no
    chat history accumulates. That is the safe default for a deployment reachable
    by more than one person, but it makes the app untestable locally and hides
    the persistence features entirely.

    ``BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE`` opts into a single shared local user.
    Everyone who reaches the app then shares one set of preferences, one run
    history and one list of chat threads, so it is for development and
    single-user deployments only - which is why it is off by default.
    """
    return os.getenv("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE", "").strip().lower() in {"1", "true", "yes", "on"}


def local_identity() -> UserIdentity:
    """The single shared identity used in :func:`single_user_mode`.

    Not ``is_authenticated``: nobody proved anything. It is merely *stable*,
    which is all persistence needs.
    """
    return UserIdentity(user_id=LOCAL_USER_ID, source="local")


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
        return local_identity() if single_user_mode() else anonymous_identity(session_id)

    identity = identity_from_headers(header_source)
    if identity is not None:
        return identity
    return anonymous_identity(session_id)
