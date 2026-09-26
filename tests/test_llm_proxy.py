"""Tests for the platform LLM proxy: settings, the identity rule, and the wire.

The first half is pure Python (``biomni.llm_proxy``). The second half runs the
real chat-model classes against a local stub of the proxy and asserts on what
actually crosses the wire - the headers the proxy refuses to go without, the
bearer token, and the absence of any direct provider credential - because that
is the contract the platform states, and mocks of the SDKs would only restate
our assumptions about them.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from biomni import llm_proxy
from biomni.identity import UserIdentity, bound_identity, password_identity
from biomni.llm_proxy import LLMProxyIdentityError

_PROXY_ENV = ("BIOMNI_LLM_PROXY_URL", "BIOMNI_LLM_PROXY_SCHEMA", "BIOMNI_LLM_PROXY_API_KEY")

GRIP_USER = UserIdentity(
    user_id="6f1c0e9e-1111-2222",
    email="jane.doe@grip.org",
    workspace_id="ws-7a3d-4e21",
    given_name="Jane",
    family_name="Doe",
    source="headers",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*_PROXY_ENV, "LLM_SOURCE", "BIOMNI_SOURCE", "BIOMNI_TRUST_AUTH_HEADERS"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_no_proxy_url_means_no_proxy():
    assert llm_proxy.llm_proxy_settings() is None
    assert llm_proxy.describe_llm_proxy() == {"enabled": False}


@pytest.mark.parametrize(
    "url",
    ["https://proxy.example", "https://proxy.example/", "https://proxy.example/v1", "https://proxy.example/v1/"],
)
def test_the_root_and_the_v1_base_mean_the_same_proxy(monkeypatch, url):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", url)
    settings = llm_proxy.llm_proxy_settings()
    assert settings is not None
    assert settings.root_url == "https://proxy.example"
    assert settings.openai_base_url == "https://proxy.example/v1"
    assert settings.anthropic_base_url == "https://proxy.example"


def test_the_schema_defaults_to_openai_and_accepts_anthropic(monkeypatch):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example")
    assert llm_proxy.llm_proxy_settings().schema == "openai"
    monkeypatch.setenv("BIOMNI_LLM_PROXY_SCHEMA", " Anthropic ")
    assert llm_proxy.llm_proxy_settings().schema == "anthropic"


def test_an_unknown_schema_fails_loudly(monkeypatch):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_SCHEMA", "azure")
    with pytest.raises(ValueError, match="BIOMNI_LLM_PROXY_SCHEMA"):
        llm_proxy.llm_proxy_settings()
    assert "error" in llm_proxy.describe_llm_proxy()


def test_describe_reports_the_token_without_revealing_it(monkeypatch):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_API_KEY", "sk-secret-value")
    report = llm_proxy.describe_llm_proxy()
    assert report == {"enabled": True, "url": "https://proxy.example", "schema": "openai", "token_configured": True}
    assert "sk-secret-value" not in json.dumps(report)


# --------------------------------------------------------------------------- #
# Who a call is for
# --------------------------------------------------------------------------- #


def test_a_gateway_identity_names_the_user_and_workspace():
    with bound_identity(GRIP_USER):
        assert llm_proxy.proxy_identity_headers() == {
            "X-User-Id": "6f1c0e9e-1111-2222",
            "X-Workspace-Id": "ws-7a3d-4e21",
        }


def test_a_call_outside_any_session_is_refused():
    with pytest.raises(LLMProxyIdentityError, match="not being made for any chat session") as info:
        llm_proxy.proxy_identity_headers()
    # Generated code reads this. It must be told to stay in its session, not
    # handed the means to make up an identity of its own.
    assert "bound_identity" not in str(info.value)
    assert "thread pool" in str(info.value)


def test_behind_a_trusted_gateway_an_unidentified_session_is_told_to_sign_in_again(monkeypatch):
    """Telling it to set the flag that is already set sends the operator in circles."""
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    problem = llm_proxy.proxy_identity_problem(UserIdentity(user_id="session-1", source="anonymous"))
    assert problem is not None
    assert "did not identify this session" in problem and "Reload the page" in problem
    assert "BIOMNI_TRUST_AUTH_HEADERS" not in problem


@pytest.mark.parametrize(
    "who,expected",
    [
        (password_identity("jane"), "not signed in through the platform's authentication gateway"),
        (UserIdentity(source="anonymous"), "identity source: anonymous"),
        (UserIdentity(user_id="u1", source="headers"), "the workspace id (uuid in Ai-App-Workspace-Context)"),
        (
            UserIdentity(email="a@b.org", workspace_id="ws", source="headers"),
            "the user id (sub in Ai-App-User-Context)",
        ),
        (UserIdentity(user_id="Jöhn", workspace_id="ws", source="headers"), "cannot be sent as HTTP headers"),
        (UserIdentity(user_id="u1\r\nX-Evil: 1", workspace_id="ws", source="headers"), "cannot be sent"),
    ],
)
def test_identities_the_proxy_cannot_bill_are_refused_with_the_reason(who, expected):
    problem = llm_proxy.proxy_identity_problem(who)
    assert problem is not None and expected in problem
    with bound_identity(who), pytest.raises(LLMProxyIdentityError):
        llm_proxy.proxy_identity_headers()


def test_a_complete_gateway_identity_has_no_problem():
    assert llm_proxy.proxy_identity_problem(GRIP_USER) is None


# --------------------------------------------------------------------------- #
# On the wire
# --------------------------------------------------------------------------- #

_OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "stub",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "stub",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class _StubProxy:
    """A local HTTP server that answers like the proxy and records each request."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        recorded = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                recorded.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                )
                reply = _ANTHROPIC_REPLY if self.path.endswith("/messages") else _OPENAI_REPLY
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _StubProxy:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def proxy(monkeypatch):
    """The stub proxy, configured the way GRIP configures the real one.

    Direct provider keys are set too, to prove none of them ever leaves the pod.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct-must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-direct-must-not-leak")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_API_KEY", "proxy-token")
    with _StubProxy() as stub:
        monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", f"{stub.url}/v1")
        yield stub


def _assert_no_direct_credential(request: dict) -> None:
    sent = json.dumps(request["headers"])
    assert "must-not-leak" not in sent
    assert "x-api-key" not in request["headers"]


def test_openai_schema_calls_chat_completions_with_both_headers(proxy):
    pytest.importorskip("langchain_openai")
    from biomni.llm import get_llm

    llm = get_llm("gpt-4.1", max_retries=0)
    with bound_identity(GRIP_USER):
        assert llm.invoke("hi").content == "hello"

    (request,) = proxy.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer proxy-token"
    assert request["headers"]["x-user-id"] == "6f1c0e9e-1111-2222"
    assert request["headers"]["x-workspace-id"] == "ws-7a3d-4e21"
    assert request["body"]["model"] == "gpt-4.1"
    _assert_no_direct_credential(request)


def test_openai_schema_async_calls_carry_the_headers_too(proxy):
    pytest.importorskip("langchain_openai")
    import asyncio

    from biomni.llm import get_llm

    llm = get_llm("gpt-4.1", max_retries=0)

    async def call():
        with bound_identity(GRIP_USER):
            return await llm.ainvoke("hi")

    assert asyncio.run(call()).content == "hello"
    assert proxy.requests[0]["headers"]["x-user-id"] == "6f1c0e9e-1111-2222"


def test_one_model_instance_names_whichever_session_is_calling(proxy):
    """Instances outlive requests; the ids must come from the call, never the instance."""
    pytest.importorskip("langchain_openai")
    from biomni.llm import get_llm

    llm = get_llm("gpt-4.1", max_retries=0)
    other = UserIdentity(user_id="bob", workspace_id="ws-2", source="headers")
    for who in (GRIP_USER, other):
        with bound_identity(who):
            llm.invoke("hi")
    assert [r["headers"]["x-user-id"] for r in proxy.requests] == ["6f1c0e9e-1111-2222", "bob"]
    assert [r["headers"]["x-workspace-id"] for r in proxy.requests] == ["ws-7a3d-4e21", "ws-2"]


def test_a_reasoning_model_is_sent_no_stop_or_temperature(proxy):
    pytest.importorskip("langchain_openai")
    from biomni.llm import get_llm

    llm = get_llm("gpt-5-mini", stop_sequences=["</execute>"], temperature=0.2, max_retries=0)
    with bound_identity(GRIP_USER):
        llm.invoke("hi")
    body = proxy.requests[0]["body"]
    assert "stop" not in body and "temperature" not in body


def test_a_call_with_no_session_is_refused_before_anything_is_sent(proxy):
    pytest.importorskip("langchain_openai")
    from biomni.llm import get_llm

    llm = get_llm("gpt-4.1", max_retries=0)
    with pytest.raises(LLMProxyIdentityError):
        llm.invoke("hi")
    with bound_identity(password_identity("jane")), pytest.raises(LLMProxyIdentityError):
        llm.invoke("hi")
    assert proxy.requests == []


def test_anthropic_schema_calls_messages_with_a_bearer_token_and_both_headers(proxy, monkeypatch):
    pytest.importorskip("langchain_anthropic")
    from biomni.llm import get_llm

    monkeypatch.setenv("BIOMNI_LLM_PROXY_SCHEMA", "anthropic")
    llm = get_llm("claude-sonnet-4-5", stop_sequences=["</execute>"], max_retries=0)
    with bound_identity(GRIP_USER):
        assert llm.invoke("hi").content == "hello"

    (request,) = proxy.requests
    assert request["path"] == "/v1/messages"
    assert request["headers"]["authorization"] == "Bearer proxy-token"
    assert request["headers"]["x-user-id"] == "6f1c0e9e-1111-2222"
    assert request["headers"]["x-workspace-id"] == "ws-7a3d-4e21"
    assert request["body"]["model"] == "claude-sonnet-4-5"
    assert request["body"]["stop_sequences"] == ["</execute>"]
    _assert_no_direct_credential(request)


def test_the_proxy_outranks_any_provider_a_call_site_asks_for(proxy):
    from biomni.llm import resolve_source

    assert resolve_source("claude-sonnet-4-5", "Anthropic", None) == "LLMProxy"
    assert resolve_source("gpt-4o", None, "http://elsewhere/v1") == "LLMProxy"


def test_a_proxy_without_its_token_is_refused_at_construction(proxy, monkeypatch):
    pytest.importorskip("langchain_openai")
    from biomni.llm import get_llm

    monkeypatch.delenv("BIOMNI_LLM_PROXY_API_KEY")
    with pytest.raises(ValueError, match="BIOMNI_LLM_PROXY_API_KEY"):
        get_llm("gpt-4.1")


def test_llmproxy_as_a_source_needs_the_proxy_url():
    from biomni.llm import resolve_source

    with pytest.raises(ValueError, match="BIOMNI_LLM_PROXY_URL"):
        resolve_source("gpt-4o", "LLMProxy", None)
