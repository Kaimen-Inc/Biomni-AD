"""Tests for biomni.config — env-driven default LLM resolution."""

from __future__ import annotations

import pytest
from biomni.config import resolve_default_llm


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "BIOMNI_LLM",
        "DEPLOYMENT_NAME",
        "ENDPOINT_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def test_resolve_falls_back_to_default() -> None:
    assert resolve_default_llm() == "claude-sonnet-4-5"


def test_resolve_respects_custom_fallback() -> None:
    assert resolve_default_llm(fallback="my-model") == "my-model"


def test_biomni_llm_overrides_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_LLM", "gpt-4o")
    monkeypatch.setenv("DEPLOYMENT_NAME", "dep1")
    monkeypatch.setenv("ENDPOINT_URL", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert resolve_default_llm() == "gpt-4o"


def test_azure_openai_auto_when_all_three_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_NAME", "mydep")
    monkeypatch.setenv("ENDPOINT_URL", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert resolve_default_llm() == "azure-mydep"


def test_azure_anthropic_auto_when_endpoint_has_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_NAME", "mydep")
    monkeypatch.setenv("ENDPOINT_URL", "https://x.anthropic.azure.com")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "k")
    assert resolve_default_llm() == "mydep"


def test_azure_openai_wins_when_both_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """If endpoint contains 'anthropic' but AZURE_OPENAI_API_KEY is also set,
    Azure OpenAI takes precedence (matches the legacy chainlit behavior)."""
    monkeypatch.setenv("DEPLOYMENT_NAME", "mydep")
    monkeypatch.setenv("ENDPOINT_URL", "https://x.anthropic.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k1")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_KEY", "k2")
    assert resolve_default_llm() == "azure-mydep"


def test_partial_azure_config_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployment without endpoint, or vice-versa, should not auto-select."""
    monkeypatch.setenv("DEPLOYMENT_NAME", "mydep")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert resolve_default_llm() == "claude-sonnet-4-5"
