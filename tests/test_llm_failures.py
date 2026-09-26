"""Tests for chainlit_ui.llm_failures - what a user reads when a model call fails.

Most exceptions are stand-ins shaped like the provider SDKs' (a ``status_code``
and a JSON ``body``), since the module is duck-typed on exactly that. The last
section raises the real thing, through each SDK's own client, for the bodies
proxies actually send - skipped for an SDK that is not installed.
"""

from __future__ import annotations

import pytest
from biomni.llm_proxy import LLMProxyIdentityError
from chainlit_ui.llm_failures import describe_llm_failure, is_access_failure


class APIStatusError(Exception):
    def __init__(self, status_code: int, message: str = "", body: object = None) -> None:
        super().__init__(f"Error code: {status_code} - {body!r}")
        self.status_code = status_code
        self.message = message
        self.body = body


class APIConnectionError(Exception):
    pass


class ConnectTimeout(APIConnectionError):
    pass


@pytest.fixture
def behind_proxy(monkeypatch):
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")


@pytest.fixture(autouse=True)
def _direct(monkeypatch):
    monkeypatch.delenv("BIOMNI_LLM_PROXY_URL", raising=False)
    monkeypatch.delenv("BIOMNI_LLM_PROXY_SCHEMA", raising=False)


def test_a_missing_identity_is_explained_in_its_own_words():
    exc = LLMProxyIdentityError("the LLM proxy needs the user and workspace every model call is made for, and x")
    assert (
        describe_llm_failure(exc) == "The LLM proxy needs the user and workspace every model call is made for, and x."
    )
    assert is_access_failure(exc)


def test_the_servers_message_is_shown_rather_than_the_raw_body(behind_proxy):
    exc = APIStatusError(403, body={"error": {"message": "workspace ws-1 has no access to model gpt-4.1"}})
    text = describe_llm_failure(exc)
    assert (
        text
        == "The platform's LLM proxy refused the request (HTTP 403): workspace ws-1 has no access to model gpt-4.1."
    )
    assert "{" not in text


def test_a_rejected_token_names_the_setting_behind_the_proxy(behind_proxy):
    assert "BIOMNI_LLM_PROXY_API_KEY" in describe_llm_failure(APIStatusError(401))


def test_a_rejected_key_without_the_proxy_names_the_api_key():
    text = describe_llm_failure(APIStatusError(401))
    assert text.startswith("The language model service rejected")
    assert "API key" in text


def test_a_usage_limit_explains_the_platforms_per_user_limits(behind_proxy):
    text = describe_llm_failure(APIStatusError(429))
    assert "per user and per workspace" in text and "Try again later" in text


def test_an_unknown_model_points_at_the_model_setting(behind_proxy):
    assert "BIOMNI_LLM" in describe_llm_failure(APIStatusError(404, body={"message": "no such deployment"}))


def test_a_server_error_suggests_trying_again():
    assert describe_llm_failure(APIStatusError(502)).endswith("Try again shortly.")


def test_a_connection_failure_is_recognised_through_its_base_class(behind_proxy):
    exc = ConnectTimeout("timed out")
    assert describe_llm_failure(exc) == "Could not reach the platform's LLM proxy. Try again shortly."
    assert is_access_failure(exc)


def test_anything_else_still_says_what_was_raised():
    assert describe_llm_failure(KeyError("messages")) == "KeyError: 'messages'."
    assert describe_llm_failure(RuntimeError()) == "RuntimeError."


def test_a_very_long_server_message_is_truncated():
    exc = APIStatusError(400, body={"error": {"message": "x" * 5000}})
    assert len(describe_llm_failure(exc)) < 400


@pytest.mark.parametrize(
    "status,unreachable", [(400, False), (401, True), (403, True), (404, True), (429, True), (500, True)]
)
def test_which_failures_end_the_run(status, unreachable):
    assert is_access_failure(APIStatusError(status)) is unreachable


def test_an_ordinary_bug_is_not_an_access_failure():
    assert not is_access_failure(ValueError("bad plan"))


# --------------------------------------------------------------------------- #
# The real SDKs' exceptions
# --------------------------------------------------------------------------- #


def _raised_by(sdk_name: str, status: int, content: bytes, content_type: str) -> Exception:
    """What the SDK raises when the proxy answers with this response."""
    sdk = pytest.importorskip(sdk_name)
    httpx = pytest.importorskip("httpx")

    def respond(request):
        return httpx.Response(status, content=content, headers={"content-type": content_type})

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(sdk.APIStatusError) as info:
        if sdk_name == "openai":
            client = sdk.OpenAI(
                api_key="t", base_url="https://proxy.example/v1", http_client=http_client, max_retries=0
            )
            client.chat.completions.create(model="m", messages=messages)
        else:
            client = sdk.Anthropic(
                api_key="t", base_url="https://proxy.example", http_client=http_client, max_retries=0
            )
            client.messages.create(model="m", max_tokens=1, messages=messages)
    return info.value


_SDKS = ["openai", "anthropic"]


@pytest.mark.parametrize("sdk_name", _SDKS)
def test_a_detail_body_is_read(behind_proxy, sdk_name):
    exc = _raised_by(sdk_name, 403, b'{"detail": "workspace ws-1 is over its limit"}', "application/json")
    assert describe_llm_failure(exc) == (
        "The platform's LLM proxy refused the request (HTTP 403): workspace ws-1 is over its limit."
    )


@pytest.mark.parametrize("sdk_name", _SDKS)
def test_a_plain_string_error_body_is_read(behind_proxy, sdk_name):
    exc = _raised_by(sdk_name, 400, b'{"error": "model m is not enabled"}', "application/json")
    assert (
        describe_llm_failure(exc) == "The platform's LLM proxy rejected the request (HTTP 400): model m is not enabled."
    )


@pytest.mark.parametrize("sdk_name", _SDKS)
def test_an_html_error_page_is_not_pasted_into_the_chat(behind_proxy, sdk_name):
    page = b"<html><head><title>502 Bad Gateway</title></head><body><center>nginx</center></body></html>"
    exc = _raised_by(sdk_name, 502, page, "text/html")
    assert describe_llm_failure(exc) == "The platform's LLM proxy failed to answer (HTTP 502). Try again shortly."


@pytest.mark.parametrize("sdk_name", _SDKS)
def test_the_providers_own_error_shape_is_read(behind_proxy, sdk_name):
    body = b'{"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}'
    exc = _raised_by(sdk_name, 400, body, "application/json")
    assert describe_llm_failure(exc) == "The platform's LLM proxy rejected the request (HTTP 400): prompt is too long."
