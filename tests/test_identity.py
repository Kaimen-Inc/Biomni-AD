"""Tests for biomni.identity - caller identity from the auth gateway's headers.

Two properties matter most and are locked in here: header extraction works
across every shape Chainlit hands us (plain mapping, WSGI environ, ASGI scope),
and ``storage_key`` can never produce a path that escapes its directory or
writes a user's e-mail address into a filename.
"""

from __future__ import annotations

import pytest
from biomni import identity

_ENV_TO_CLEAR = (
    "BIOMNI_AUTH_USER_ID_HEADER",
    "BIOMNI_AUTH_EMAIL_HEADER",
    "BIOMNI_AUTH_WORKSPACE_HEADER",
    "BIOMNI_TRUST_AUTH_HEADERS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Header extraction
# --------------------------------------------------------------------------- #


def test_extract_headers_from_plain_mapping():
    headers = identity.extract_headers({"X-Auth-Request-Email": "a@b.org", "X-Workspace-Id": "ws1"})
    assert headers["x-auth-request-email"] == "a@b.org"
    assert headers["x-workspace-id"] == "ws1"


def test_extract_headers_from_wsgi_environ():
    environ = {
        "HTTP_X_AUTH_REQUEST_USER_ID": "user-42",
        "HTTP_X_WORKSPACE_ID": "ws-9",
        "REQUEST_METHOD": "GET",
    }
    headers = identity.extract_headers(environ)
    assert headers["x-auth-request-user-id"] == "user-42"
    assert headers["x-workspace-id"] == "ws-9"


def test_extract_headers_from_asgi_scope_bytes():
    scope = {"headers": [(b"x-auth-request-user-id", b"user-7"), (b"x-user-email", b"c@d.org")]}
    headers = identity.extract_headers(scope)
    assert headers["x-auth-request-user-id"] == "user-7"
    assert headers["x-user-email"] == "c@d.org"


def test_extract_headers_from_environ_with_nested_asgi_scope():
    environ = {"asgi.scope": {"headers": [(b"x-user-id", b"nested-1")]}}
    assert identity.extract_headers(environ)["x-user-id"] == "nested-1"


def test_extract_headers_tolerates_junk():
    assert identity.extract_headers(None) == {}
    assert identity.extract_headers(object()) == {}
    assert identity.extract_headers({"headers": "not-a-list"}) == {}


def test_extract_headers_drops_empty_values():
    assert "x-user-id" not in identity.extract_headers({"X-User-Id": "   "})


# --------------------------------------------------------------------------- #
# Identity resolution
# --------------------------------------------------------------------------- #


def test_identity_from_headers_reads_all_three_fields():
    who = identity.identity_from_headers(
        {"x-auth-request-user-id": "u1", "x-auth-request-email": "a@b.org", "x-workspace-id": "ws1"}
    )
    assert who is not None
    assert (who.user_id, who.email, who.workspace_id) == ("u1", "a@b.org", "ws1")
    assert who.source == "headers"
    assert who.is_authenticated


def test_identity_from_headers_returns_none_without_a_subject():
    # A workspace header alone does not identify anyone.
    assert identity.identity_from_headers({"x-workspace-id": "ws1"}) is None
    assert identity.identity_from_headers({}) is None


def test_env_override_takes_precedence_over_default_header_names(monkeypatch):
    monkeypatch.setenv("BIOMNI_AUTH_USER_ID_HEADER", "x-grip-subject")
    who = identity.identity_from_headers({"x-grip-subject": "grip-1", "x-user-id": "ignored"})
    assert who is not None
    assert who.user_id == "grip-1"


def test_env_override_is_additive_so_defaults_still_resolve(monkeypatch):
    monkeypatch.setenv("BIOMNI_AUTH_USER_ID_HEADER", "x-grip-subject")
    who = identity.identity_from_headers({"x-user-id": "fallback-1"})
    assert who is not None
    assert who.user_id == "fallback-1"


def test_email_only_identity_is_still_authenticated():
    who = identity.identity_from_headers({"x-forwarded-email": "only@mail.org"})
    assert who is not None
    assert who.user_id is None
    assert who.is_authenticated
    assert who.display_name == "only@mail.org"


def test_anonymous_identity_is_keyed_by_session():
    who = identity.anonymous_identity("sess-1")
    assert not who.is_authenticated
    assert who.source == "anonymous"
    assert who.storage_key() != "anonymous"  # distinct per session


def test_resolve_identity_prefers_headers_then_falls_back(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    assert identity.resolve_identity({"x-user-id": "u9"}, session_id="s1").user_id == "u9"
    assert identity.resolve_identity({}, session_id="s1").source == "anonymous"


# --------------------------------------------------------------------------- #
# Storage keys - safety properties
# --------------------------------------------------------------------------- #


def test_storage_key_never_contains_path_separators():
    who = identity.UserIdentity(user_id="../../etc/passwd", source="headers")
    key = who.storage_key()
    assert "/" not in key and "\\" not in key
    assert ".." not in key


def test_storage_key_omits_the_email_address():
    who = identity.UserIdentity(email="jane.doe@example.org", source="headers")
    key = who.storage_key()
    assert "jane" not in key
    assert "example" not in key
    assert key.startswith("u-")


def test_storage_key_is_stable_and_distinct():
    a = identity.UserIdentity(user_id="user-1", source="headers")
    b = identity.UserIdentity(user_id="user-2", source="headers")
    assert a.storage_key() == identity.UserIdentity(user_id="user-1", source="headers").storage_key()
    assert a.storage_key() != b.storage_key()


def test_storage_key_distinguishes_ids_that_slug_identically():
    # Both slug to "a-b"; the hash suffix keeps their files apart.
    a = identity.UserIdentity(user_id="a b", source="headers").storage_key()
    b = identity.UserIdentity(user_id="a/b", source="headers").storage_key()
    assert a != b


def test_storage_key_without_any_identifier():
    assert identity.UserIdentity().storage_key() == "anonymous"


def test_scoped_key_separates_workspaces():
    base = identity.UserIdentity(user_id="u1", source="headers")
    scoped = identity.UserIdentity(user_id="u1", workspace_id="ws-A", source="headers")
    other = identity.UserIdentity(user_id="u1", workspace_id="ws-B", source="headers")
    assert scoped.scoped_key() != other.scoped_key()
    assert scoped.scoped_key().startswith(base.storage_key())
    assert base.scoped_key() == base.storage_key()


def test_scoped_key_sanitizes_workspace_id():
    who = identity.UserIdentity(user_id="u1", workspace_id="../evil", source="headers")
    key = who.scoped_key()
    assert "/" not in key and ".." not in key


# --------------------------------------------------------------------------- #
# Header trust gate
# --------------------------------------------------------------------------- #


def test_headers_are_ignored_unless_the_deployment_opts_in(monkeypatch):
    """Fail closed: without a declared gateway, headers are an impersonation vector."""
    monkeypatch.delenv("BIOMNI_TRUST_AUTH_HEADERS", raising=False)
    who = identity.resolve_identity({"x-user-id": "victim"}, session_id="s1")
    assert who.source == "anonymous"
    assert who.user_id != "victim"


def test_headers_are_used_once_trusted(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    who = identity.resolve_identity({"x-user-id": "u1"}, session_id="s1")
    assert who.source == "headers"
    assert who.user_id == "u1"


def test_trust_flag_spellings(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", raw)
        assert identity.trust_auth_headers() is True
    for raw in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", raw)
        assert identity.trust_auth_headers() is False


def test_trusted_but_headerless_session_is_still_anonymous(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    assert identity.resolve_identity({}, session_id="s1").source == "anonymous"
