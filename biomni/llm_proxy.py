"""The platform LLM proxy: where it is, and who each request is for.

On GRIP every model call goes through the platform's LLM proxy in front of
Azure AI Foundry ("AI Apps Onboarding", section Azure Foundry). It speaks the
provider's own API, authenticates the app with its own bearer token, and meters
usage per user and per workspace - so each request must name both, in two
headers it refuses to go without::

    X-User-Id:      <sub from Ai-App-User-Context>
    X-Workspace-Id: <uuid from Ai-App-Workspace-Context>

Those are the ids the authentication gateway asserted for the session a call is
made for (see :mod:`biomni.identity`). They are read when the request is built,
from the identity bound to the running session, rather than baked into a client
- which is what makes it impossible for a model instance to speak for anyone
but the session it is running in.

This module is deliberately import-light (no LangChain), so the readiness probe
and the session-start check can use it. The chat-model classes that attach the
headers live in :mod:`biomni.llm`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from biomni import credentials
from biomni.identity import current_identity, trust_auth_headers

if TYPE_CHECKING:
    from biomni.identity import UserIdentity

PROXY_USER_HEADER = "X-User-Id"
PROXY_WORKSPACE_HEADER = "X-Workspace-Id"
ProxySchema = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class LLMProxySettings:
    """Where the proxy is and how to talk to it, from the environment."""

    root_url: str
    api_key: str | None
    schema: ProxySchema

    @property
    def openai_base_url(self) -> str:
        """Base URL for the OpenAI SDK, which appends ``/chat/completions``."""
        return f"{self.root_url}/v1"

    @property
    def anthropic_base_url(self) -> str:
        """Base URL for the Anthropic SDK, which appends ``/v1/messages``."""
        return self.root_url


def llm_proxy_settings() -> LLMProxySettings | None:
    """The configured LLM proxy, or ``None`` when model calls go direct.

    ``BIOMNI_LLM_PROXY_URL`` switches it on, and accepts either the root or the
    ``/v1`` base the platform documents - ``https://proxy.example`` and
    ``https://proxy.example/v1`` mean the same thing, since which of the two an
    operator is handed is a coin toss. ``BIOMNI_LLM_PROXY_SCHEMA`` picks the API
    to speak: ``openai`` (the default, and what the platform documents) or
    ``anthropic``, for a Claude deployment the proxy serves in its native schema.

    Raises ``ValueError`` for a schema it does not know, so a typo fails loudly
    instead of quietly speaking the wrong API.
    """
    raw = os.getenv("BIOMNI_LLM_PROXY_URL", "").strip().rstrip("/")
    if not raw:
        return None
    root = raw[: -len("/v1")] if raw.endswith("/v1") else raw
    schema = (os.getenv("BIOMNI_LLM_PROXY_SCHEMA") or "openai").strip().lower()
    if schema not in get_args(ProxySchema):
        raise ValueError(f"BIOMNI_LLM_PROXY_SCHEMA must be 'openai' or 'anthropic', not {schema!r}")
    return LLMProxySettings(
        root_url=root.rstrip("/"),
        # Through the vault: os.environ loses credentials while generated code
        # runs, and a chat can start during another chat's code step.
        api_key=credentials.getenv("BIOMNI_LLM_PROXY_API_KEY") or None,
        schema=cast("ProxySchema", schema),
    )


def describe_llm_proxy() -> dict[str, Any]:
    """The proxy settings in force, for one structured line at startup."""
    try:
        settings = llm_proxy_settings()
    except ValueError as exc:
        return {"enabled": True, "error": str(exc)}
    if settings is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "url": settings.root_url,
        "schema": settings.schema,
        "token_configured": bool(settings.api_key),
    }


class LLMProxyIdentityError(RuntimeError):
    """A model call through the LLM proxy has no user or workspace to name.

    Raised before anything is sent: the proxy would refuse the request anyway,
    and its refusal would not say which of our sessions lacked what.

    Code calling a model from Python outside the app binds the identity itself,
    with :func:`biomni.identity.bound_identity`. The message does not say so,
    because generated code reads it too, and the fix it must reach for is to
    stay inside the session it runs for, not to make up an identity.
    """


_NEEDS = "the LLM proxy needs the user and workspace every model call is made for"


def _header_safe(value: str) -> bool:
    return value.isascii() and value.isprintable()


def proxy_identity_problem(identity: UserIdentity | None) -> str | None:
    """Why ``identity`` cannot be named to the proxy, or ``None`` if it can.

    Only a gateway identity will do. The proxy meters and limits usage per
    user, and every other kind of session - a demo password login, the shared
    local user, an anonymous visitor - has an id the platform has never heard
    of, so sending it would charge the usage to nobody, or to a stranger.
    """
    if identity is None:
        return (
            f"{_NEEDS}, and this call is not being made for any chat session: it runs in a thread started "
            "without one. Make model calls from the code step itself or from a concurrent.futures thread "
            "pool, which carry the session with them."
        )
    if identity.source != "headers":
        if trust_auth_headers():
            return (
                f"{_NEEDS}, and the platform's authentication gateway did not identify this session. "
                "Reload the page to sign in again."
            )
        return (
            f"{_NEEDS}, and this session was not signed in through the platform's authentication gateway "
            f"(identity source: {identity.source}). Set BIOMNI_TRUST_AUTH_HEADERS=true on a deployment "
            "behind the gateway."
        )
    missing = []
    if not identity.user_id:
        missing.append("the user id (sub in Ai-App-User-Context)")
    if not identity.workspace_id:
        missing.append("the workspace id (uuid in Ai-App-Workspace-Context)")
    if missing:
        return f"{_NEEDS}, and the authentication gateway did not send {' or '.join(missing)}."
    if not (_header_safe(identity.user_id or "") and _header_safe(identity.workspace_id or "")):
        return f"{_NEEDS}, and the ids the authentication gateway sent cannot be sent as HTTP headers."
    return None


def proxy_identity_headers() -> dict[str, str]:
    """``X-User-Id`` and ``X-Workspace-Id`` for the session this call is for.

    Raises :class:`LLMProxyIdentityError` when there is no such session, or it
    is not one the proxy can bill (see :func:`proxy_identity_problem`).
    """
    identity = current_identity()
    problem = proxy_identity_problem(identity)
    if problem is not None or identity is None:
        raise LLMProxyIdentityError(problem)
    return {PROXY_USER_HEADER: identity.user_id or "", PROXY_WORKSPACE_HEADER: identity.workspace_id or ""}


__all__ = [
    "PROXY_USER_HEADER",
    "PROXY_WORKSPACE_HEADER",
    "LLMProxyIdentityError",
    "LLMProxySettings",
    "describe_llm_proxy",
    "llm_proxy_settings",
    "proxy_identity_headers",
    "proxy_identity_problem",
]
