"""Tests for ``biomni.llm`` - source resolution and retry/timeout plumbing.

We mock the optional provider packages (``langchain_anthropic``,
``langchain_openai``, etc.) so these tests run without those extras
installed. The goal is to verify that ``get_llm`` passes the right
resilience kwargs to whichever provider class it picks - not to test
the provider implementations themselves.
"""

from __future__ import annotations

import sys
import types

import pytest
from biomni.config import BiomniConfig
from biomni.llm import get_llm, resolve_source


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "LLM_SOURCE",
        "BIOMNI_SOURCE",
        "AZURE_ANTHROPIC_API_KEY",
        "ENDPOINT_URL",
        "DEPLOYMENT_NAME",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# resolve_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-5-sonnet-20241022", "Anthropic"),
        ("claude-sonnet-4-5", "Anthropic"),
        ("gpt-4o", "OpenAI"),
        ("gpt-5-mini", "OpenAI"),
        ("gpt-oss-20b", "Ollama"),
        ("azure-foo", "AzureOpenAI"),
        ("gemini-1.5-pro", "Gemini"),
        ("some-groq-mixtral", "Groq"),
        # Bedrock prefixes that don't collide with Ollama substring heuristics:
        ("anthropic.claude-3-haiku", "Bedrock"),
        ("amazon.titan-text-express", "Bedrock"),
        ("cohere.command-text", "Bedrock"),
        ("ai21.j2-ultra", "Bedrock"),
        # Local model heuristics (substring match): wins over Bedrock for
        # ambiguous names like ``meta.llama-3-70b`` - Bedrock users for those
        # should pass ``source="Bedrock"`` explicitly.
        ("llama-3-8b", "Ollama"),
        ("mistral-7b-instruct", "Ollama"),
        ("vendor/some-model", "Ollama"),
    ],
)
def test_resolve_source_autodetect(model: str, expected: str) -> None:
    assert resolve_source(model) == expected


@pytest.mark.parametrize("model", ["meta.llama-3-70b", "us.meta.llama-3-70b"])
def test_bedrock_llama_requires_explicit_source(model: str) -> None:
    """Bedrock model IDs containing 'llama' get caught by the Ollama heuristic.
    Explicit source overrides resolve correctly."""
    assert resolve_source(model) == "Ollama"  # documents existing behavior
    assert resolve_source(model, source="Bedrock") == "Bedrock"


def test_resolve_source_explicit_overrides_heuristic() -> None:
    assert resolve_source("claude-3-5-sonnet", source="OpenAI") == "OpenAI"


def test_resolve_source_invalid_raises() -> None:
    with pytest.raises(ValueError):
        resolve_source("claude-3-5-sonnet", source="NotARealProvider")  # type: ignore[arg-type]


def test_resolve_source_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_SOURCE", "OpenAI")
    assert resolve_source("claude-3-5-sonnet") == "OpenAI"


def test_resolve_source_base_url_routes_to_custom() -> None:
    assert resolve_source("totally-custom-model", base_url="http://localhost:8000/v1") == "Custom"


def test_resolve_source_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unable to determine model source"):
        resolve_source("zzz-mystery-model-9000")


# ---------------------------------------------------------------------------
# get_llm - retry/timeout pass-through (provider modules stubbed)
# ---------------------------------------------------------------------------


def _install_fake_provider_module(monkeypatch, module_name, class_name) -> list[dict]:
    """Install a synthetic langchain-* module exporting one chat-model class
    whose __init__ records every kwarg it was constructed with.

    Returns the list that captures construction calls (one dict per call).
    """
    captured: list[dict] = []

    class FakeChat:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.kwargs = kwargs

    fake_mod = types.ModuleType(module_name)
    setattr(fake_mod, class_name, FakeChat)
    monkeypatch.setitem(sys.modules, module_name, fake_mod)
    return captured


def test_anthropic_get_llm_forwards_retry_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_provider_module(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # avoid bash_profile lookup

    llm = get_llm("claude-3-5-sonnet", max_retries=5, request_timeout=42.0)

    assert captured, "ChatAnthropic was not constructed"
    kw = captured[-1]
    assert kw["max_retries"] == 5
    assert kw["default_request_timeout"] == 42.0
    assert kw["max_tokens"] == 8192
    assert llm.kwargs["model"] == "claude-3-5-sonnet"


def test_openai_get_llm_forwards_retry_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_provider_module(monkeypatch, "langchain_openai", "ChatOpenAI")
    get_llm("gpt-4o", max_retries=4, request_timeout=10.0)
    kw = captured[-1]
    assert kw["max_retries"] == 4
    assert kw["timeout"] == 10.0


def test_get_llm_inherits_config_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_provider_module(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    cfg = BiomniConfig(llm_max_retries=9, llm_request_timeout=7.5)
    get_llm("claude-sonnet-4-5", config=cfg)

    kw = captured[-1]
    assert kw["max_retries"] == 9
    assert kw["default_request_timeout"] == 7.5


def test_get_llm_config_none_timeout_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit None on config means disable per-call timeout (vs. default-fill)."""
    captured = _install_fake_provider_module(monkeypatch, "langchain_anthropic", "ChatAnthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    cfg = BiomniConfig(llm_request_timeout=None)
    get_llm("claude-sonnet-4-5", config=cfg)

    assert captured[-1]["default_request_timeout"] is None
