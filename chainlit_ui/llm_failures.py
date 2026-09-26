"""Say why a model call failed, in words a user and an operator can act on.

The provider SDKs raise exceptions whose text is a status code and a raw JSON
body - ``Error code: 400 - {'error': {'message': ...}}`` - and that is what the
UI used to show. Behind the platform's LLM proxy the causes that matter are few
and specific (a missing identity, a revoked token, a usage limit), so they get
sentences of their own; everything else still says what was raised.

Pure Python: duck-typed on the SDKs' exception shape (``status_code``, ``body``)
rather than importing them, so it is testable without either installed.
"""

from __future__ import annotations

from biomni.llm_proxy import LLMProxyIdentityError, llm_proxy_settings

# Failures after which trying again in the same run is pointless: the next call
# would meet the same refusal. The pipeline stops on these instead of carrying
# on "without a plan" into an execution that can only fail the same way.
_ACCESS_STATUSES = frozenset({401, 403, 404, 429})
_CONNECTION_ERRORS = frozenset({"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout"})

_MAX_DETAIL = 300


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_connection_error(exc: BaseException) -> bool:
    return any(cls.__name__ in _CONNECTION_ERRORS for cls in type(exc).__mro__)


def _error_body(exc: BaseException) -> object:
    """The error response's JSON body, however the SDK chose to keep it.

    Read from the response itself where there is one: the OpenAI SDK keeps only
    the ``error`` member as ``body`` - a bare string when the proxy sent
    ``{"error": "..."}`` - and a response that was not JSON at all as the raw
    text, and the two cannot be told apart from ``body`` alone.
    """
    read_json = getattr(getattr(exc, "response", None), "json", None)
    if callable(read_json):
        try:
            return read_json()
        except Exception:
            return None  # not JSON: a gateway's HTML error page, say
    return getattr(exc, "body", None)


def _server_message(exc: BaseException) -> str:
    """The message inside the error body, or an empty string.

    Reads the shapes providers and the proxies in front of them send:
    ``{"error": {"message": ...}}`` (OpenAI, Anthropic), ``{"error": "..."}``,
    ``{"message": ...}`` and FastAPI's ``{"detail": "..."}``. Anything else says
    nothing a user can act on and is left out: a body that is not JSON, and the
    SDK's own ``message``, which is only the status code followed by the raw
    body.
    """
    body = _error_body(exc)
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    candidates = (error.get("message") if isinstance(error, dict) else error, body.get("message"), body.get("detail"))
    text = next((c.strip() for c in candidates if isinstance(c, str) and c.strip()), "")
    return text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 1] + "…"


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _behind_proxy() -> bool:
    try:
        return llm_proxy_settings() is not None
    except ValueError:
        return True


def is_access_failure(exc: BaseException) -> bool:
    """Whether the model is unreachable for this session, not just this call."""
    if isinstance(exc, LLMProxyIdentityError):
        return True
    status = _status(exc)
    if status is not None:
        return status in _ACCESS_STATUSES or status >= 500
    return _is_connection_error(exc)


def describe_llm_failure(exc: BaseException) -> str:
    """One or two sentences explaining ``exc``, naming the fix when there is one."""
    if isinstance(exc, LLMProxyIdentityError):
        return _sentence(str(exc))

    proxy = _behind_proxy()
    service = "The platform's LLM proxy" if proxy else "The language model service"
    status = _status(exc)
    detail = _server_message(exc)
    said = f": {detail}" if detail else ""

    if status == 401:
        fix = "Check BIOMNI_LLM_PROXY_API_KEY." if proxy else "Check the deployment's API key."
        return f"{service} rejected this deployment's credentials (HTTP 401). {fix}"
    if status == 403:
        return _sentence(f"{service} refused the request (HTTP 403){said}")
    if status == 404:
        return _sentence(f"{service} does not offer the configured model (HTTP 404{said}). Check BIOMNI_LLM")
    if status == 429:
        scope = " The platform limits model usage per user and per workspace." if proxy else ""
        return f"The language model usage limit has been reached (HTTP 429).{scope} Try again later."
    if status is not None and status >= 500:
        return _sentence(f"{service} failed to answer (HTTP {status}){said}. Try again shortly")
    if status is not None:
        return _sentence(f"{service} rejected the request (HTTP {status}){said}")
    if _is_connection_error(exc):
        return _sentence(f"Could not reach {service[0].lower()}{service[1:]}. Try again shortly")

    text = str(exc).strip() or type(exc).__name__
    if len(text) > _MAX_DETAIL:
        text = text[: _MAX_DETAIL - 1] + "…"
    return _sentence(f"{type(exc).__name__}: {text}" if text != type(exc).__name__ else text)


__all__ = ["describe_llm_failure", "is_access_failure"]
