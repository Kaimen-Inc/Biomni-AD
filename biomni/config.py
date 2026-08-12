"""
Biomni Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def resolve_data_lake_root() -> str:
    """Where the built-in data lake (including the AD-specific ``biomniAD`` subtree) lives on disk.

    Defaults to the ``data/biomni_data/data_lake`` folder shipped inside the repo, so a
    fresh checkout works with no configuration. Override with ``BIOMNI_DATA_LAKE_PATH``
    when the data lake is mounted somewhere else on the server (a dedicated volume, a
    different disk) — every caller resolves the location through this one function, so
    moving it later means setting one env var rather than hunting down hardcoded paths.
    """
    override = os.getenv("BIOMNI_DATA_LAKE_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    repo_root = Path(__file__).resolve().parent.parent
    return str((repo_root / "data" / "biomni_data" / "data_lake").resolve())


def resolve_default_llm(fallback: str = "claude-sonnet-4-5") -> str:
    """Pick the default LLM model name from environment.

    Precedence: BIOMNI_LLM > Azure OpenAI deployment > Azure Anthropic deployment > fallback.

    Azure OpenAI is selected only when DEPLOYMENT_NAME + ENDPOINT_URL + AZURE_OPENAI_API_KEY
    are all set. Azure Anthropic requires DEPLOYMENT_NAME + ENDPOINT_URL containing
    "anthropic" + AZURE_ANTHROPIC_API_KEY. This avoids misrouting users who configure
    both Azure providers with overlapping env vars.
    """
    override = os.getenv("BIOMNI_LLM")
    if override:
        return override

    deployment = os.getenv("DEPLOYMENT_NAME")
    endpoint = os.getenv("ENDPOINT_URL")
    if deployment and endpoint:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            return f"azure-{deployment}"
        if "anthropic" in endpoint and os.getenv("AZURE_ANTHROPIC_API_KEY"):
            return deployment

    return fallback


@dataclass
class BiomniConfig:
    """Central configuration for Biomni agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = BiomniConfig()

        # Override specific settings
        config = BiomniConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "/custom/data/path"
    """

    # Data and execution settings
    path: str = os.path.join(os.path.expanduser("~"), ".biomni", "data")
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    llm: str = "claude-sonnet-4-5"
    temperature: float = 0.7

    # LLM resilience settings
    # Provider SDKs (anthropic, openai, etc.) implement their own exponential
    # backoff on 429 / 5xx — we forward these knobs to the SDK constructor.
    llm_max_retries: int = 3
    # Per-call request timeout (seconds). None disables. Distinct from
    # `timeout_seconds`, which gates code/tool execution, not LLM HTTP calls.
    llm_request_timeout: float | None = 120.0

    # Total wall-clock budget for one agent run (seconds). None disables (default,
    # preserving existing behavior). When set, the agent stops cleanly between
    # ReAct turns once the budget is exceeded — bounding the *number* of turns,
    # complementing `timeout_seconds` (which bounds a single code/tool step) and
    # the recursion limit. Recommended for interactive/demo deployments so a
    # long-running query fails fast and visibly instead of spinning.
    run_timeout_seconds: int | None = None

    # Prompt caching (currently honored for Anthropic models). When True the
    # agent annotates the large system prompt with cache_control so the
    # provider can charge cached-input rates on subsequent turns.
    enable_prompt_caching: bool = True

    # Per-run LLM usage / cost telemetry. Cheap; off by default to avoid
    # changing existing log output for users not opted in.
    enable_llm_telemetry: bool = False

    # Tool settings
    use_tool_retriever: bool = True
    auto_network_limited_mode: bool = True

    # Data licensing settings
    commercial_mode: bool = False  # If True, excludes non-commercial datasets

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = None

    # Third-party integrations
    protocols_io_access_token: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist."""
        # Check for environment variable overrides (optional)
        # Support all known path env names for backwards compatibility.
        # Priority keeps BIOMNI_USER_DATA_PATH as the explicit user data root when set.
        # BIOMNI_DATA_PATH is preferred over BIOMNI_PATH because BIOMNI_PATH is often
        # reserved for built-in app data in container deployments.
        if os.getenv("BIOMNI_USER_DATA_PATH") or os.getenv("BIOMNI_DATA_PATH") or os.getenv("BIOMNI_PATH"):
            self.path = os.getenv("BIOMNI_USER_DATA_PATH") or os.getenv("BIOMNI_DATA_PATH") or os.getenv("BIOMNI_PATH")
        if os.getenv("BIOMNI_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS"))
        if os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL"):
            self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BIOMNI_LLM_MODEL")
        if os.getenv("BIOMNI_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("BIOMNI_USE_TOOL_RETRIEVER").lower() == "true"
        if os.getenv("BIOMNI_AUTO_NETWORK_LIMITED_MODE"):
            self.auto_network_limited_mode = os.getenv("BIOMNI_AUTO_NETWORK_LIMITED_MODE").lower() == "true"
        if os.getenv("BIOMNI_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("BIOMNI_COMMERCIAL_MODE").lower() == "true"
        if os.getenv("BIOMNI_TEMPERATURE"):
            self.temperature = float(os.getenv("BIOMNI_TEMPERATURE"))
        if os.getenv("BIOMNI_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("BIOMNI_CUSTOM_BASE_URL")
        if os.getenv("BIOMNI_CUSTOM_API_KEY"):
            self.api_key = os.getenv("BIOMNI_CUSTOM_API_KEY")
        if os.getenv("BIOMNI_SOURCE"):
            self.source = os.getenv("BIOMNI_SOURCE")

        # LLM resilience env-var overrides.
        if os.getenv("BIOMNI_LLM_MAX_RETRIES"):
            self.llm_max_retries = int(os.getenv("BIOMNI_LLM_MAX_RETRIES"))
        if os.getenv("BIOMNI_LLM_REQUEST_TIMEOUT"):
            raw = os.getenv("BIOMNI_LLM_REQUEST_TIMEOUT").strip().lower()
            self.llm_request_timeout = None if raw in ("", "none", "0") else float(raw)
        if os.getenv("BIOMNI_RUN_TIMEOUT_SECONDS"):
            raw = os.getenv("BIOMNI_RUN_TIMEOUT_SECONDS").strip().lower()
            # Non-positive (incl. a negative typo, which would arm a deadline in
            # the past and abort every run on turn 1) disables the budget.
            self.run_timeout_seconds = None if raw in ("", "none") else (int(raw) if int(raw) > 0 else None)
        if os.getenv("BIOMNI_ENABLE_PROMPT_CACHING"):
            self.enable_prompt_caching = os.getenv("BIOMNI_ENABLE_PROMPT_CACHING").lower() == "true"
        if os.getenv("BIOMNI_ENABLE_LLM_TELEMETRY"):
            self.enable_llm_telemetry = os.getenv("BIOMNI_ENABLE_LLM_TELEMETRY").lower() == "true"

        # Protocols.io access token (prefer specific env vars)
        env_token = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN")
        if env_token:
            self.protocols_io_access_token = env_token

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "llm_max_retries": self.llm_max_retries,
            "llm_request_timeout": self.llm_request_timeout,
            "run_timeout_seconds": self.run_timeout_seconds,
            "enable_prompt_caching": self.enable_prompt_caching,
            "enable_llm_telemetry": self.enable_llm_telemetry,
            "use_tool_retriever": self.use_tool_retriever,
            "auto_network_limited_mode": self.auto_network_limited_mode,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


# Global default config instance (optional, for convenience)
default_config = BiomniConfig()
