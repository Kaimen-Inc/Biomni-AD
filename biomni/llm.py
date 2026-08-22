import logging
import os
from typing import TYPE_CHECKING, Literal, Optional, cast, get_args

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from biomni import credentials

load_dotenv(override=True)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biomni.config import BiomniConfig

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: set[str] = set(get_args(SourceType))


def _openai_key_kwargs() -> dict[str, object]:
    """``api_key`` for the OpenAI client, sourced through the credential vault.

    The SDK's own fallback reads ``OPENAI_API_KEY`` from ``os.environ``, which is
    stripped while generated code runs (see :mod:`biomni.credentials`), so a chat
    opened during another chat's code step would build a keyless client. Passing
    it explicitly removes that dependency. The key is omitted rather than passed
    as ``None`` when unset, so the SDK's own resolution still applies to
    deployments that populate it by some other means.
    """
    key = credentials.getenv("OPENAI_API_KEY")
    return {"api_key": key} if key else {}


def resolve_source(
    model: str,
    source: SourceType | None = None,
    base_url: str | None = None,
) -> SourceType:
    """Resolve the provider source for a model string.

    Mirrors the auto-detection logic in :func:`get_llm` but is callable
    independently so callers (e.g. the A1 agent) can know the resolved
    source without re-implementing the heuristic.

    Precedence: explicit ``source`` arg -> ``LLM_SOURCE`` / ``BIOMNI_SOURCE``
    env vars -> model-name prefix heuristics -> ``base_url`` presence ->
    Azure-Anthropic env detection. Raises ``ValueError`` if the source can't
    be determined.
    """
    if source is not None:
        if source not in ALLOWED_SOURCES:
            raise ValueError(f"Unknown source: {source!r}. Valid: {sorted(ALLOWED_SOURCES)}")
        return source

    env_source = os.getenv("LLM_SOURCE") or os.getenv("BIOMNI_SOURCE")
    if env_source in ALLOWED_SOURCES:
        return cast("SourceType", env_source)

    if model.startswith("claude-"):
        return "Anthropic"
    if model.startswith("gpt-oss"):
        return "Ollama"
    if model.startswith("gpt-"):
        return "OpenAI"
    if model.startswith("azure-"):
        return "AzureOpenAI"
    if model.startswith("gemini-"):
        return "Gemini"
    if "groq" in model.lower():
        return "Groq"
    if base_url is not None:
        return "Custom"
    if "/" in model or any(
        name in model.lower()
        for name in ("llama", "mistral", "qwen", "gemma", "phi", "dolphin", "orca", "vicuna", "deepseek")
    ):
        return "Ollama"
    if model.startswith(("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")):
        return "Bedrock"
    if (
        credentials.getenv("AZURE_ANTHROPIC_API_KEY")
        and os.getenv("ENDPOINT_URL")
        and "anthropic" in os.getenv("ENDPOINT_URL", "")
    ):
        return "Anthropic"

    raise ValueError("Unable to determine model source. Please specify 'source' parameter.")


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["BiomniConfig"] = None,
    *,
    max_retries: int | None = None,
    request_timeout: float | None = None,
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (BiomniConfig): Optional configuration object. If provided, unspecified parameters will use config values
        max_retries: Provider-SDK retry attempts on 429/5xx. Falls back to ``config.llm_max_retries`` then ``3``.
        request_timeout: Per-call HTTP timeout in seconds. Falls back to ``config.llm_request_timeout`` then ``120``.
    """
    # Use config values for any unspecified parameters
    if config is not None:
        if model is None:
            model = config.llm
        if temperature is None:
            temperature = config.temperature
        if source is None:
            if config.source in ALLOWED_SOURCES:
                source = cast("SourceType", config.source)
        if base_url is None:
            base_url = config.base_url
        if api_key is None:
            api_key = config.api_key or "EMPTY"
        if max_retries is None:
            max_retries = config.llm_max_retries
        if request_timeout is None:
            request_timeout = config.llm_request_timeout

    # Use defaults if still not specified
    if model is None:
        model = "claude-3-5-sonnet-20241022"
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = "EMPTY"
    if max_retries is None:
        max_retries = 3
    if request_timeout is None and config is None:
        # Only default-fill when no config was passed; explicit None from a
        # configured caller means "disable per-call timeout".
        request_timeout = 120.0
    # Resolve source via shared helper (auto-detection + env precedence).
    source = resolve_source(model, source, base_url)

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        # Newer OpenAI models (e.g., gpt-5-*) require the Responses API and may reject
        # legacy Chat Completions parameters like `stop`. Force Responses API when
        # using gpt-5 models to avoid 400 errors such as: "Unsupported parameter: 'stop'".
        use_responses = model.startswith("gpt-5")

        if use_responses:
            # Define a minimal subclass that drops the `stop` field when using the
            # Responses API, since certain models (gpt-5-*) reject it entirely.
            class _ChatOpenAIResponsesNoStop(ChatOpenAI):
                def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
                    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                    try:
                        # If this call will use the Responses API, drop `stop` to avoid 400s.
                        if hasattr(self, "_use_responses_api") and self._use_responses_api(payload):  # type: ignore[attr-defined]
                            payload.pop("stop", None)
                            # Also drop temperature for gpt-5 models as they only support default value
                            payload.pop("temperature", None)
                    except Exception:
                        # Be conservative: if anything goes wrong, still remove `stop` and `temperature`.
                        payload.pop("stop", None)
                        payload.pop("temperature", None)
                    return payload

            return _ChatOpenAIResponsesNoStop(
                model=model,
                temperature=1,  # Set to default value for gpt-5, will be removed in payload
                stop_sequences=stop_sequences,
                base_url=os.getenv("OPENAI_BASE_URL"),
                **_openai_key_kwargs(),
                use_responses_api=True,
                output_version="v0",
                max_retries=max_retries,
                timeout=request_timeout,
            )
        else:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                stop_sequences=stop_sequences,
                base_url=os.getenv("OPENAI_BASE_URL"),
                **_openai_key_kwargs(),
                max_retries=max_retries,
                timeout=request_timeout,
            )

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        API_VERSION = "2024-12-01-preview"
        # Derive deployment name: strip "azure-" prefix if present, else fall back to DEPLOYMENT_NAME env var
        deployment = (
            model.replace("azure-", "") if model.startswith("azure-") else (os.getenv("DEPLOYMENT_NAME") or model)
        )

        # Some Azure-hosted models (e.g. gpt-5.*) reject any temperature value other
        # than the default. Use a subclass that silently drops the parameter.
        class _AzureChatOpenAINoTemp(AzureChatOpenAI):
            def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                payload.pop("temperature", None)
                return payload

        return _AzureChatOpenAINoTemp(
            openai_api_key=credentials.getenv("AZURE_OPENAI_API_KEY") or credentials.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT_URL") or os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=deployment,
            openai_api_version=API_VERSION,
            temperature=1,  # default; will be stripped from payload by subclass
            max_retries=max_retries,
            timeout=request_timeout,
        )

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        azure_endpoint = os.getenv("ENDPOINT_URL")
        uses_azure_anthropic = bool(azure_endpoint and "azure.com" in azure_endpoint and "anthropic" in azure_endpoint)
        azure_deployment = os.getenv("DEPLOYMENT_NAME")

        # Azure Anthropic routes by deployment name rather than canonical Claude model IDs.
        if uses_azure_anthropic and azure_deployment and model.startswith("claude-"):
            model = azure_deployment

        # Allow Azure Anthropic credentials while keeping backwards compatibility.
        # Resolved into locals and handed to the client below rather than written
        # back into os.environ: an env write landing inside another session's
        # code step would hand the key straight to model-written code.
        azure_anthropic_key = credentials.getenv("AZURE_ANTHROPIC_API_KEY")
        anthropic_key = azure_anthropic_key if uses_azure_anthropic else None
        anthropic_base_url = azure_endpoint if uses_azure_anthropic else None
        if anthropic_key is None:
            anthropic_key = credentials.getenv("ANTHROPIC_API_KEY")
        if anthropic_base_url is None:
            anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")

        # Ensure ANTHROPIC_API_KEY is loaded from bash_profile if not in environment
        if not anthropic_key:
            try:
                import subprocess

                result = subprocess.run(
                    ["bash", "-c", "source ~/.bash_profile 2>/dev/null && echo $ANTHROPIC_API_KEY"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    anthropic_key = result.stdout.strip()
                    logger.info("Loaded ANTHROPIC_API_KEY from ~/.bash_profile")
            except Exception:
                logger.warning("Could not load ANTHROPIC_API_KEY from bash_profile", exc_info=True)

        # Passed explicitly rather than left to the SDK's os.environ lookup:
        # credentials are stripped from the environment while generated code
        # runs, so a chat starting during another chat's code step would
        # otherwise build a client with no key.
        anthropic_kwargs: dict[str, object] = {}
        if anthropic_key:
            anthropic_kwargs["api_key"] = anthropic_key
        if anthropic_base_url:
            anthropic_kwargs["base_url"] = anthropic_base_url

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            max_retries=max_retries,
            default_request_timeout=request_timeout,
            **anthropic_kwargs,
        )

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=credentials.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
            max_retries=max_retries,
            timeout=request_timeout,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=credentials.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
            max_retries=max_retries,
            timeout=request_timeout,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        # ChatOllama exposes a transport-level timeout, not max_retries; pass
        # what's supported and let local Ollama retries stay manual.
        ollama_kwargs: dict[str, object] = {"model": model, "temperature": temperature}
        if request_timeout is not None:
            ollama_kwargs["timeout"] = request_timeout
        return ChatOllama(**ollama_kwargs)

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        # Bedrock retry config lives on the boto3 client; pass through via
        # ``config`` kwarg when available. Older langchain-aws versions don't
        # accept ``config`` directly, so build defensively.
        bedrock_kwargs: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
        }
        return ChatBedrock(**bedrock_kwargs)

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=request_timeout,
        )
        return llm

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', or 'Ollama'"
        )
