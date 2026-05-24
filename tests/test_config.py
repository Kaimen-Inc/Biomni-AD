"""Tests for biomni.config — env-driven default LLM resolution + resilience knobs."""

from __future__ import annotations

import pytest
from biomni.config import BiomniConfig, resolve_default_llm


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "BIOMNI_LLM",
        "DEPLOYMENT_NAME",
        "ENDPOINT_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_ANTHROPIC_API_KEY",
        "BIOMNI_LLM_MAX_RETRIES",
        "BIOMNI_LLM_REQUEST_TIMEOUT",
        "BIOMNI_ENABLE_PROMPT_CACHING",
        "BIOMNI_ENABLE_LLM_TELEMETRY",
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


# ---------------------------------------------------------------------------
# Resilience knobs: retry / timeout / caching / telemetry
# ---------------------------------------------------------------------------


def test_resilience_defaults() -> None:
    cfg = BiomniConfig()
    assert cfg.llm_max_retries == 3
    assert cfg.llm_request_timeout == 120.0
    assert cfg.enable_prompt_caching is True
    assert cfg.enable_llm_telemetry is False
    snap = cfg.to_dict()
    for key in (
        "llm_max_retries",
        "llm_request_timeout",
        "enable_prompt_caching",
        "enable_llm_telemetry",
    ):
        assert key in snap


def test_env_overrides_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMNI_LLM_MAX_RETRIES", "7")
    monkeypatch.setenv("BIOMNI_LLM_REQUEST_TIMEOUT", "45.5")
    monkeypatch.setenv("BIOMNI_ENABLE_PROMPT_CACHING", "false")
    monkeypatch.setenv("BIOMNI_ENABLE_LLM_TELEMETRY", "true")
    cfg = BiomniConfig()
    assert cfg.llm_max_retries == 7
    assert cfg.llm_request_timeout == 45.5
    assert cfg.enable_prompt_caching is False
    assert cfg.enable_llm_telemetry is True


@pytest.mark.parametrize("raw", ["none", "None", "0"])
def test_request_timeout_none_sentinels(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Explicit sentinels disable the per-call timeout. Empty string ≠ disable
    (matches shell convention — an unset/blank var falls back to the default).
    """
    monkeypatch.setenv("BIOMNI_LLM_REQUEST_TIMEOUT", raw)
    cfg = BiomniConfig()
    assert cfg.llm_request_timeout is None
