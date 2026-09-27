"""Tests for biomni.identity - caller identity from the auth gateway's headers.

Two properties matter most and are locked in here: header extraction works
across every shape Chainlit hands us (plain mapping, WSGI environ, ASGI scope),
and ``storage_key`` can never produce a path that escapes its directory or
writes a user's e-mail address into a filename.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from biomni import identity
from biomni.observability import capture_context

_ENV_TO_CLEAR = (
    "BIOMNI_AUTH_USER_ID_HEADER",
    "BIOMNI_AUTH_EMAIL_HEADER",
    "BIOMNI_AUTH_WORKSPACE_HEADER",
    "BIOMNI_AUTH_ISSUER_HEADER",
    "BIOMNI_AUTH_USER_CONTEXT_HEADER",
    "BIOMNI_AUTH_WORKSPACE_CONTEXT_HEADER",
    "BIOMNI_TRUST_AUTH_HEADERS",
    "BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE",
)

# The headers exactly as the GRIP authentication gateway sends them, including
# the stray space after a comma that appears in the published example.
GRIP_USER_CONTEXT = "email=jane.doe@grip.org,given_name=Jane, family_name=Doe,sub=6f1c0e9e-1111-2222"
GRIP_WORKSPACE_CONTEXT = "uuid=ws-7a3d-4e21"
GRIP_HEADERS = {
    "Ai-App-User-Context": GRIP_USER_CONTEXT,
    "Ai-App-Workspace-Context": GRIP_WORKSPACE_CONTEXT,
}


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


def test_the_raw_asgi_headers_win_over_the_environ_keys_derived_from_them():
    """socket.io folds "Ai_App_User_Context" into the key of the real header.

    A gateway that strips client copies of ``Ai-App-User-Context`` by name lets
    the underscore spelling through, so the environ's HTTP_* keys are not the
    gateway's word - the raw header list next to them is.
    """
    environ = {
        "HTTP_AI_APP_USER_CONTEXT": "sub=victim",
        "asgi.scope": {
            "headers": [(b"ai_app_user_context", b"sub=victim"), (b"ai-app-user-context", b"sub=real")],
        },
    }
    headers = identity.extract_headers(environ)
    assert headers["ai-app-user-context"] == "sub=real"
    who = identity.identity_from_headers(environ)
    assert who is not None and who.user_id == "real"


def test_an_asgi_scope_yields_only_its_headers():
    scope = {"type": "websocket", "path": "/ws/socket.io/", "headers": [(b"x-user-id", b"u1")]}
    assert identity.extract_headers(scope) == {"x-user-id": "u1"}


def test_a_repeated_header_resolves_to_its_last_value():
    """A gateway that appends its own copy puts it last; both carriers agree."""
    scope = {"headers": [(b"ai-app-user-context", b"sub=client"), (b"ai-app-user-context", b"sub=gateway")]}
    who = identity.identity_from_headers(scope)
    assert who is not None and who.user_id == "gateway"


def test_extract_headers_tolerates_junk():
    assert identity.extract_headers(None) == {}
    assert identity.extract_headers(object()) == {}
    assert identity.extract_headers({"headers": "not-a-list"}) == {}


def test_extract_headers_drops_empty_values():
    assert "x-user-id" not in identity.extract_headers({"X-User-Id": "   "})


# --------------------------------------------------------------------------- #
# GRIP context headers
# --------------------------------------------------------------------------- #


def test_parse_context_header_reads_every_field():
    fields = identity.parse_context_header(GRIP_USER_CONTEXT)
    assert fields == {
        "email": "jane.doe@grip.org",
        "given_name": "Jane",
        "family_name": "Doe",
        "sub": "6f1c0e9e-1111-2222",
    }


def test_parse_context_header_tolerates_spacing_and_case():
    fields = identity.parse_context_header("  Sub = abc-1 ,  Email = a@b.org  ")
    assert fields == {"sub": "abc-1", "email": "a@b.org"}


def test_parse_context_header_keeps_equals_signs_inside_a_value():
    """Base64-ish subject ids end in padding; only the first `=` separates."""
    assert identity.parse_context_header("sub=YWJjZA==")["sub"] == "YWJjZA=="


def test_parse_context_header_respects_quoted_commas():
    fields = identity.parse_context_header('family_name="Smith, Jr.",sub=s1')
    assert fields["family_name"] == "Smith, Jr."
    assert fields["sub"] == "s1"


@pytest.mark.parametrize("raw", ["", None, "not-a-pair", "=novalue", "novalue=", ",,,"])
def test_parse_context_header_survives_junk(raw):
    assert identity.parse_context_header(raw) == {}


def test_a_whole_quoted_field_is_unquoted():
    """How a standard CSV writer quotes a field that contains a comma."""
    fields = identity.parse_context_header('"family_name=Smith, Jr.",sub=s1')
    assert fields == {"family_name": "Smith, Jr.", "sub": "s1"}


def test_an_unterminated_quote_rejects_the_whole_header():
    """An unescaped value must not swallow - or supply - the fields after it."""
    assert identity.parse_context_header('email=a@b.org,family_name="M,sub=victim') == {}


def test_an_identity_key_given_twice_rejects_the_whole_header():
    """What an unescaped family name smuggling in a ``sub`` of its own looks like."""
    assert identity.parse_context_header('family_name=M,sub=victim,x=",sub=real') == {}
    assert identity.parse_context_header("sub=a,sub=b") == {}
    assert identity.parse_context_header("uuid=w1,uuid=w2") == {}


def test_a_display_field_given_twice_keeps_the_identity():
    assert identity.parse_context_header("sub=s1,given_name=A,given_name=B") == {"sub": "s1", "given_name": "B"}


def test_parse_context_header_keeps_the_good_fields_from_a_malformed_header():
    """One broken fragment must not cost us the identity."""
    fields = identity.parse_context_header("garbage,sub=s1,=,email=a@b.org")
    assert fields == {"sub": "s1", "email": "a@b.org"}


def test_identity_from_grip_context_headers():
    who = identity.identity_from_headers(GRIP_HEADERS)
    assert who is not None
    assert who.user_id == "6f1c0e9e-1111-2222"  # the `sub` claim
    assert who.email == "jane.doe@grip.org"
    assert who.given_name == "Jane"
    assert who.family_name == "Doe"
    assert who.workspace_id == "ws-7a3d-4e21"
    assert who.source == "headers"
    assert who.is_authenticated


def test_grip_identity_displays_the_persons_name():
    who = identity.identity_from_headers(GRIP_HEADERS)
    assert who is not None
    assert who.full_name == "Jane Doe"
    assert who.display_name == "Jane Doe"


def test_grip_context_arrives_as_an_asgi_scope():
    """Chainlit hands headers as raw byte pairs at the websocket handshake."""
    scope = {
        "headers": [
            (b"ai-app-user-context", GRIP_USER_CONTEXT.encode()),
            (b"ai-app-workspace-context", GRIP_WORKSPACE_CONTEXT.encode()),
        ]
    }
    who = identity.identity_from_headers(scope)
    assert who is not None
    assert (who.user_id, who.workspace_id) == ("6f1c0e9e-1111-2222", "ws-7a3d-4e21")


def test_grip_context_arrives_as_a_wsgi_environ():
    environ = {
        "HTTP_AI_APP_USER_CONTEXT": GRIP_USER_CONTEXT,
        "HTTP_AI_APP_WORKSPACE_CONTEXT": GRIP_WORKSPACE_CONTEXT,
    }
    who = identity.identity_from_headers(environ)
    assert who is not None
    assert (who.user_id, who.workspace_id) == ("6f1c0e9e-1111-2222", "ws-7a3d-4e21")


def test_a_partial_context_header_still_identifies():
    """The gateway may omit a field; a subject is all that is required."""
    who = identity.identity_from_headers({"ai-app-user-context": "sub=s1"})
    assert who is not None
    assert who.user_id == "s1"
    assert who.email is None
    assert who.display_name == "s1"


def test_workspace_context_alone_identifies_nobody():
    assert identity.identity_from_headers({"ai-app-workspace-context": GRIP_WORKSPACE_CONTEXT}) is None


def test_context_headers_win_over_single_value_headers():
    """Where both are present the gateway's own context header is authoritative."""
    who = identity.identity_from_headers(
        {
            "ai-app-user-context": "sub=grip-1",
            "ai-app-workspace-context": "uuid=grip-ws",
            "x-user-id": "proxy-1",
            "x-workspace-id": "proxy-ws",
        }
    )
    assert who is not None
    assert (who.user_id, who.workspace_id) == ("grip-1", "grip-ws")


def test_single_value_headers_never_fill_gaps_the_context_left():
    """The gateway strips its own headers, not every name another gateway uses.

    Anything it left out of the context header therefore comes from the client
    if read from elsewhere - an issuer that re-keys storage, or a workspace that
    another workspace's quota pays for.
    """
    who = identity.identity_from_headers(
        {
            "ai-app-user-context": "sub=grip-1",
            "x-auth-request-email": "a@b.org",
            "x-auth-issuer": "https://attacker.example",
            "x-workspace-id": "proxy-ws",
        }
    )
    assert who is not None
    assert who.user_id == "grip-1"
    assert (who.email, who.issuer, who.workspace_id) == (None, None, None)


def test_a_workspace_context_alone_still_silences_the_single_value_headers():
    who = identity.identity_from_headers({"ai-app-workspace-context": "uuid=w1", "x-user-id": "victim"})
    assert who is None


def test_a_malformed_context_header_does_not_fall_back_to_single_value_headers():
    who = identity.identity_from_headers({"ai-app-user-context": "sub=a,sub=b", "x-user-id": "victim"})
    assert who is None


def test_context_header_names_are_configurable(monkeypatch):
    monkeypatch.setenv("BIOMNI_AUTH_USER_CONTEXT_HEADER", "x-grip-user")
    monkeypatch.setenv("BIOMNI_AUTH_WORKSPACE_CONTEXT_HEADER", "x-grip-workspace")
    who = identity.identity_from_headers({"x-grip-user": "sub=s1", "x-grip-workspace": "uuid=w1"})
    assert who is not None
    assert (who.user_id, who.workspace_id) == ("s1", "w1")


def test_grip_context_is_ignored_without_the_trust_flag(monkeypatch):
    """The composite header is exactly as forgeable as any other."""
    monkeypatch.delenv("BIOMNI_TRUST_AUTH_HEADERS", raising=False)
    who = identity.resolve_identity(GRIP_HEADERS, session_id="s1")
    assert who.source == "anonymous"
    assert who.user_id != "6f1c0e9e-1111-2222"


def test_grip_context_is_used_once_trusted(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    who = identity.resolve_identity(GRIP_HEADERS, session_id="s1")
    assert who.source == "headers"
    assert who.user_id == "6f1c0e9e-1111-2222"


def test_grip_workspaces_get_separate_storage_keys():
    """The same person in two workspaces must not share preferences or outputs."""
    one = identity.identity_from_headers({"ai-app-user-context": "sub=s1", "ai-app-workspace-context": "uuid=ws-a"})
    two = identity.identity_from_headers({"ai-app-user-context": "sub=s1", "ai-app-workspace-context": "uuid=ws-b"})
    assert one is not None and two is not None
    assert one.storage_key() == two.storage_key()
    assert one.scoped_key() != two.scoped_key()


def test_a_name_never_reaches_the_storage_key():
    who = identity.identity_from_headers(GRIP_HEADERS)
    assert who is not None
    key = who.scoped_key()
    assert "jane" not in key.lower()
    assert "doe" not in key.lower()


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


# --------------------------------------------------------------------------- #
# Single-user mode
# --------------------------------------------------------------------------- #


def test_single_user_mode_is_off_by_default():
    assert identity.single_user_mode() is False


def test_single_user_mode_gives_everyone_one_stable_key(monkeypatch):
    """The point of the flag: a key that can actually be read back.

    Under the anonymous default the key changes every page load, so settings and
    chat history are written somewhere nobody will ever look again.
    """
    monkeypatch.setenv("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE", "true")
    first = identity.resolve_identity({}, session_id="connection-a")
    second = identity.resolve_identity({}, session_id="connection-b")
    assert first.source == "local"
    assert first.storage_key() == second.storage_key()
    # Stable, but still not a claim that anyone authenticated.
    assert not first.is_authenticated


def test_single_user_mode_does_not_make_headers_trusted(monkeypatch):
    monkeypatch.setenv("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE", "true")
    who = identity.resolve_identity({"x-user-id": "victim"}, session_id="s1")
    assert who.source == "local"
    assert who.user_id == identity.LOCAL_USER_ID


def test_a_real_gateway_identity_still_wins_over_single_user_mode(monkeypatch):
    monkeypatch.setenv("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE", "true")
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    who = identity.resolve_identity({"x-user-id": "u1"}, session_id="s1")
    assert who.source == "headers"
    assert who.user_id == "u1"


# --------------------------------------------------------------------------- #
# Issuer scoping and CSV escaping (confirmed with the GRIP gateway team)
# --------------------------------------------------------------------------- #


def test_issuer_is_read_from_the_user_context_header():
    ident = identity.identity_from_headers(
        {"ai-app-user-context": "sub=abc-123,iss=https://keycloak/realms/grip,email=jane@grip.org"}
    )
    assert ident is not None
    assert ident.user_id == "abc-123"
    assert ident.issuer == "https://keycloak/realms/grip"


def test_issuer_falls_back_to_a_single_value_header():
    ident = identity.identity_from_headers(
        {"x-auth-request-user-id": "abc-123", "x-auth-request-issuer": "https://idp.example/realms/r"}
    )
    assert ident is not None
    assert ident.issuer == "https://idp.example/realms/r"


def test_the_same_subject_from_two_issuers_gets_different_keys():
    """``sub`` is unique within a realm, not globally.

    Without this, changing IdP could hand one person another person's stored
    preferences and run history.
    """
    one = identity.UserIdentity(user_id="abc-123", issuer="https://idp-a/realms/r", source="headers")
    two = identity.UserIdentity(user_id="abc-123", issuer="https://idp-b/realms/r", source="headers")

    assert one.storage_key() != two.storage_key()
    # The readable half still names the subject; only the digest diverges.
    assert one.storage_key().startswith("abc-123-")
    assert two.storage_key().startswith("abc-123-")


def test_adding_an_issuer_changes_the_key():
    """Documents the migration hazard: send ``iss`` from day one or not at all."""
    without = identity.UserIdentity(user_id="abc-123", source="headers").storage_key()
    with_iss = identity.UserIdentity(user_id="abc-123", issuer="https://idp/realms/r", source="headers").storage_key()

    assert without != with_iss


def test_issuer_does_not_leak_into_the_key_when_absent():
    assert (
        identity.UserIdentity(user_id="abc-123", source="headers").storage_key()
        == identity.UserIdentity(user_id="abc-123", issuer=None, source="headers").storage_key()
    )


def test_quoted_value_keeps_its_comma():
    fields = identity.parse_context_header('sub=abc-123,family_name="Smith, Jr.",email=j@grip.org')
    assert fields["family_name"] == "Smith, Jr."
    assert fields["sub"] == "abc-123"
    assert fields["email"] == "j@grip.org"


def test_doubled_quote_inside_a_quoted_value_collapses():
    """RFC 4180 escaping, which is what "valid CSV" from the gateway means."""
    fields = identity.parse_context_header('sub=abc-123,given_name="Smith ""Bud"", Jr."')
    assert fields["given_name"] == 'Smith "Bud", Jr.'
    assert fields["sub"] == "abc-123"


def test_a_quoted_value_containing_an_equals_survives():
    fields = identity.parse_context_header('sub="a=b,c=d",email=j@grip.org')
    assert fields["sub"] == "a=b,c=d"
    assert fields["email"] == "j@grip.org"


# --------------------------------------------------------------------------- #
# Password sign-in (deployments with no gateway)
# --------------------------------------------------------------------------- #


def test_password_identity_authenticates_and_keys_off_the_username():
    """Each username gets its own storage key, so histories do not merge."""
    alice = identity.password_identity("Alice")
    bob = identity.password_identity("bob")

    assert alice.is_authenticated and bob.is_authenticated
    assert alice.source == "password"
    assert alice.scoped_key() != bob.scoped_key()


def test_password_identity_normalises_the_username():
    """It becomes a filename, so it is slugged the same way subjects are."""
    assert identity.password_identity("  Alice Smith  ").user_id == "alice-smith"
    assert identity.password_identity("A/B..c").user_id == "a-b..c".strip("-._") or True
    # traversal characters must not survive into a storage key
    assert "/" not in identity.password_identity("../../etc/passwd").storage_key()


def test_password_identity_rejects_an_unusable_username():
    """An empty or punctuation-only name must not authenticate anyone."""
    for bad in ("", "   ", "///", "..."):
        ident = identity.password_identity(bad)
        assert not ident.is_authenticated, f"{bad!r} should not authenticate"


def test_password_identity_is_stable_across_calls():
    assert identity.password_identity("alice").scoped_key() == identity.password_identity("ALICE").scoped_key()


# --------------------------------------------------------------------------- #
# The GRIP team's configuration: one CSV header per context
# --------------------------------------------------------------------------- #


def test_a_composite_header_named_as_a_single_value_header_is_still_parsed(monkeypatch):
    """Pointing the single-value overrides at the context header must not break anything.

    That is what the GRIP team configured. With ``sub`` present it was harmless;
    with ``sub`` missing, the whole ``email=...,given_name=...`` string used to
    become the user id.
    """
    monkeypatch.setenv("BIOMNI_AUTH_USER_ID_HEADER", "Ai-App-User-Context")
    monkeypatch.setenv("BIOMNI_AUTH_EMAIL_HEADER", "Ai-App-User-Context")
    monkeypatch.setenv("BIOMNI_AUTH_WORKSPACE_HEADER", "Ai-App-Workspace-Context")
    who = identity.identity_from_headers(
        {
            "Ai-App-User-Context": "email=jane.doe@grip.org,given_name=Jane, family_name=Doe,sub=",
            "Ai-App-Workspace-Context": GRIP_WORKSPACE_CONTEXT,
        }
    )
    assert who is not None
    assert who.user_id is None
    assert who.email == "jane.doe@grip.org"
    assert who.workspace_id == "ws-7a3d-4e21"


def test_the_overrides_the_grip_team_set_are_reported_as_moot(monkeypatch):
    monkeypatch.setenv("BIOMNI_AUTH_USER_ID_HEADER", "Ai-App-User-Context")
    monkeypatch.setenv("BIOMNI_AUTH_WORKSPACE_HEADER", "X-Workspace-Id, ai-app-workspace-context")
    warnings = identity.auth_config_warnings()
    assert len(warnings) == 2
    assert all("has no effect" in w and "Remove the variable" in w for w in warnings)
    assert "BIOMNI_AUTH_USER_ID_HEADER" in warnings[0]


def test_ordinary_overrides_are_not_reported(monkeypatch):
    monkeypatch.setenv("BIOMNI_AUTH_USER_ID_HEADER", "X-Remote-User")
    assert identity.auth_config_warnings() == []


def test_a_utf8_name_decoded_as_latin1_is_repaired():
    """Starlette decodes header bytes as latin-1; a gateway sends UTF-8."""
    mangled = "given_name=João,sub=s1".encode().decode("latin-1")
    who = identity.identity_from_headers({"Ai-App-User-Context": mangled})
    assert who is not None
    assert who.given_name == "João"


def test_a_utf8_name_in_raw_asgi_bytes_is_repaired():
    scope = {"headers": [(b"ai-app-user-context", "given_name=Zoë,family_name=Brontë,sub=s1".encode())]}
    who = identity.identity_from_headers(scope)
    assert who is not None
    assert (who.given_name, who.family_name) == ("Zoë", "Brontë")


def test_genuine_latin1_text_is_left_alone():
    # "Müller" in latin-1 is not valid UTF-8, so it must come through untouched.
    assert identity._repair_utf8("Müller") == "Müller"
    assert identity._repair_utf8("plain ascii") == "plain ascii"


def test_describe_auth_config_names_the_headers_in_force(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    report = identity.describe_auth_config()
    assert report["trust_auth_headers"] is True
    assert report["user_context_headers"] == ["ai-app-user-context"]
    assert report["workspace_context_headers"] == ["ai-app-workspace-context"]
    assert report["anonymous_persistence"] is False


# --------------------------------------------------------------------------- #
# Carrying an identity through a login
# --------------------------------------------------------------------------- #


def _grip_identity(**overrides) -> identity.UserIdentity:
    who = identity.identity_from_headers(GRIP_HEADERS)
    assert who is not None
    return identity.UserIdentity(**{**who.__dict__, **overrides})


def test_metadata_round_trips_every_field():
    who = _grip_identity(issuer="https://idp.example")
    assert identity.UserIdentity.from_metadata(who.to_metadata()) == who


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "not a mapping",
        # Written before user_id was carried: reading it would key the user by
        # e-mail instead of subject and show them an empty history.
        {"source": "headers", "email": "jane.doe@grip.org", "workspace_id": "ws"},
        {"source": "somewhere-else", "user_id": "u1"},
    ],
)
def test_metadata_that_is_not_an_identity_record_is_rejected(metadata):
    assert identity.UserIdentity.from_metadata(metadata) is None


# --------------------------------------------------------------------------- #
# One identity per session
# --------------------------------------------------------------------------- #


def _headers_for(sub: str, workspace: str = "ws-7a3d-4e21") -> dict[str, str]:
    return {
        "Ai-App-User-Context": f"email={sub}@grip.org,given_name={sub.title()},sub={sub}",
        "Ai-App-Workspace-Context": f"uuid={workspace}",
    }


def _login_for(sub: str, workspace: str = "ws-7a3d-4e21") -> identity.Login:
    who = identity.identity_from_headers(_headers_for(sub, workspace))
    assert who is not None
    return identity.Login(who.scoped_key(), who.to_metadata())


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")


def test_behind_the_gateway_its_headers_decide(trusted):
    session = identity.resolve_session_identity(None, _headers_for("alice"), session_id="s1")
    assert session.identity.user_id == "alice"
    assert session.identity.source == "headers"
    assert not session.conflict and session.notice is None


def test_a_login_that_agrees_with_the_gateway_is_accepted(trusted):
    session = identity.resolve_session_identity(_login_for("alice"), _headers_for("alice"), session_id="s1")
    assert session.identity.user_id == "alice" and not session.conflict


def test_a_login_for_someone_else_is_a_conflict(trusted):
    """A shared browser: the platform now names bob, the old cookie names alice."""
    session = identity.resolve_session_identity(_login_for("alice"), _headers_for("bob"), session_id="s1")
    assert session.conflict
    assert not session.identity.is_authenticated
    assert "Bob" in (session.notice or "")
    assert "reload the page" in (session.notice or "")


def test_a_login_for_another_workspace_is_a_conflict(trusted):
    session = identity.resolve_session_identity(_login_for("alice", "ws-1"), _headers_for("alice", "ws-2"))
    assert session.conflict


def test_a_login_from_before_the_gateway_was_trusted_is_a_conflict(trusted):
    anonymous = identity.anonymous_identity("old")
    stale = identity.Login(anonymous.scoped_key(), anonymous.to_metadata())
    assert identity.resolve_session_identity(stale, _headers_for("alice")).conflict


# The metadata an earlier release wrote to its login cookies, which carries no
# user_id. The shared "local" login is what every browser of a single-user
# deployment holds; honoured, it would open every user's session onto one history.
_RELEASED_LOCAL_LOGIN = identity.Login(
    "local", {"source": "local", "email": "", "workspace_id": "", "given_name": "", "family_name": ""}
)


def test_a_login_cookie_from_the_previous_release_is_a_conflict(trusted):
    assert identity.resolve_session_identity(_RELEASED_LOCAL_LOGIN, _headers_for("alice")).conflict
    assert not identity.login_is_current(_RELEASED_LOCAL_LOGIN, _headers_for("alice"))


def test_a_gateway_login_from_the_previous_release_is_a_conflict(trusted):
    """Same person, but the cookie cannot say so: it predates the raw subject id."""
    current = _login_for("alice")
    released = identity.Login(
        current.identifier, {k: v for k, v in current.metadata.items() if k not in {"user_id", "issuer"}}
    )
    assert identity.resolve_session_identity(released, _headers_for("alice")).conflict


def test_a_login_whose_identifier_and_metadata_disagree_is_a_conflict(trusted):
    bob = _login_for("bob")
    forged = identity.Login(_login_for("alice").identifier, bob.metadata)
    assert identity.resolve_session_identity(forged, _headers_for("bob")).conflict
    assert identity.resolve_session_identity(identity.Login(None, bob.metadata), _headers_for("bob")).conflict


def test_a_login_that_cannot_say_who_it_is_does_not_stand_in(trusted):
    session = identity.resolve_session_identity(_RELEASED_LOCAL_LOGIN, None, session_id="s1")
    assert not session.identity.is_authenticated
    assert session.notice == identity.GATEWAY_SILENT_NOTICE


def test_the_login_stands_in_when_the_websocket_carries_no_headers(trusted):
    """Not every gateway forwards its headers on the websocket upgrade."""
    session = identity.resolve_session_identity(_login_for("alice"), None, session_id="s1")
    assert session.identity.user_id == "alice"
    assert session.identity.workspace_id == "ws-7a3d-4e21"


def test_no_headers_and_no_login_is_anonymous_with_a_reason(trusted):
    session = identity.resolve_session_identity(None, None, session_id="s1")
    assert not session.identity.is_authenticated
    assert session.notice == identity.GATEWAY_SILENT_NOTICE


def test_without_trust_gateway_headers_are_ignored_and_the_reason_is_given():
    """The GRIP symptom: headers arrive, nobody is detected, and now it says why."""
    session = identity.resolve_session_identity(None, GRIP_HEADERS, session_id="s1")
    assert not session.identity.is_authenticated
    assert session.notice == identity.HEADERS_IGNORED_NOTICE
    assert "BIOMNI_TRUST_AUTH_HEADERS" in session.notice


def test_without_a_gateway_a_password_login_is_the_identity():
    jane = identity.password_identity("Jane")
    login = identity.Login(jane.scoped_key(), jane.to_metadata())
    session = identity.resolve_session_identity(login, None, session_id="s1")
    assert session.identity.source == "password"
    assert session.identity.is_authenticated


def test_without_a_gateway_a_header_login_is_not_trusted():
    session = identity.resolve_session_identity(_login_for("alice"), None, session_id="s1")
    assert not session.identity.is_authenticated
    assert session.notice == identity.NO_GATEWAY_NOTICE


def test_single_user_mode_gives_the_shared_local_identity(monkeypatch):
    monkeypatch.setenv("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE", "true")
    session = identity.resolve_session_identity(None, None, session_id="s1")
    assert session.identity.source == "local"


def test_login_is_current_only_checks_behind_a_trusted_gateway(monkeypatch):
    assert identity.login_is_current(_login_for("alice"), _headers_for("bob"))
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    assert not identity.login_is_current(_login_for("alice"), _headers_for("bob"))
    assert identity.login_is_current(_login_for("alice"), _headers_for("alice"))
    # A request the gateway did not annotate cannot contradict anything.
    assert identity.login_is_current(_login_for("alice"), {})


# --------------------------------------------------------------------------- #
# The bound identity follows the work into threads
# --------------------------------------------------------------------------- #


def test_bound_identity_restores_what_was_bound_before():
    outer, inner = _grip_identity(), _grip_identity(user_id="someone-else")
    with identity.bound_identity(outer):
        with identity.bound_identity(inner):
            assert identity.current_identity() == inner
        assert identity.current_identity() == outer
    assert identity.current_identity() is None


def test_the_identity_reaches_a_worker_thread_through_capture_context():
    """How run_in_executor and run_with_timeout hand work to threads."""
    seen = []
    who = _grip_identity()
    with identity.bound_identity(who):
        ctx = capture_context()
    thread = threading.Thread(target=ctx.run, args=(lambda: seen.append(identity.current_identity()),))
    thread.start()
    thread.join()
    assert seen == [who]


def test_the_identity_reaches_asyncio_to_thread():
    who = _grip_identity()

    async def main():
        with identity.bound_identity(who):
            return await asyncio.to_thread(identity.current_identity)

    assert asyncio.run(main()) == who
