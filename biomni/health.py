"""Liveness / readiness probes for the Chainlit app under Kubernetes (AKS).

Kubernetes distinguishes two probe kinds and they must mean different things:

* **liveness** (``/healthz``) - "is the process wedged?". Must be cheap and must
  not depend on external systems, or a transient upstream blip triggers a
  pod restart. So ``/healthz`` returns 200 as long as the event loop can serve
  a request.
* **readiness** (``/readyz``) - "should this pod receive traffic *right now*?".
  Here we verify the agent could actually function: the user's workspace is
  mounted, when one is configured, and a model can be reached. A misconfigured
  pod reports 503 and is pulled from the Service endpoints instead of failing
  user requests.

Both endpoints are registered at the *front* of the router. Chainlit mounts a
catch-all ``GET /{full_path:path}`` (for the SPA) as its last route; a route
appended after it would be shadowed, so we insert ahead of everything.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from biomni import credentials
from biomni.identity import trust_auth_headers
from biomni.llm_proxy import llm_proxy_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request

# Provider credentials the agent can use. Readiness needs at least one (or a
# custom base URL pointing at a self-hosted / proxy model).
_PROVIDER_KEY_ENVS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_BEARER_TOKEN_BEDROCK",
)


def build_info() -> dict[str, str]:
    """Image provenance, surfaced in probe payloads to confirm which build is live."""
    info: dict[str, str] = {}
    sha = os.getenv("BIOMNI_GIT_SHA") or os.getenv("GIT_SHA")
    ref = os.getenv("BIOMNI_GIT_REF") or os.getenv("GIT_REF")
    if sha:
        info["git_sha"] = sha
    if ref:
        info["git_ref"] = ref
    return info


def _workspace_check() -> dict[str, Any]:
    """Whether the user's workspace is mounted, when one is configured.

    Worth holding traffic back for: without it every question about the user's
    own data is answered as if there were none. With no workspace configured
    there is nothing to wait for. ``BIOMNI_PATH`` is deliberately not consulted:
    it is the app's own data directory, which the image does not contain and the
    agent creates the first time it is built - and that takes a chat, which an
    unready pod never receives.
    """
    # Env read directly rather than through biomni.config: readiness must reflect
    # the *current* environment and stay import-light for the probe path.
    for env in ("BIOMNI_USER_DATA_PATH", "BIOMNI_DATA_PATH"):
        path = os.getenv(env, "").strip()
        if path:
            return {"ok": os.path.isdir(path), "path": path}
    return {"ok": True, "path": None}


def _llm_credential_check() -> dict[str, Any]:
    """Whether some usable LLM credential / endpoint is configured.

    With an LLM proxy configured every model call goes through it, so its token
    is the only credential that counts: a provider key left over in the
    environment must not make a pod that cannot reach any model look ready. Nor
    must a proxy that will refuse every call, because nothing tells it whom the
    call is for: its user and workspace ids come only from a trusted gateway.
    """
    try:
        proxy = llm_proxy_settings()
    except ValueError as exc:
        return {"ok": False, "source": "BIOMNI_LLM_PROXY_URL", "detail": str(exc)}
    if proxy is not None:
        if not proxy.api_key:
            return {
                "ok": False,
                "source": "BIOMNI_LLM_PROXY_URL",
                "detail": "BIOMNI_LLM_PROXY_API_KEY is not set; the proxy needs its token",
            }
        if not trust_auth_headers():
            return {
                "ok": False,
                "source": "BIOMNI_LLM_PROXY_URL",
                "detail": "BIOMNI_TRUST_AUTH_HEADERS is not enabled; the proxy needs the user and workspace ids "
                "only the authentication gateway supplies",
            }
        return {"ok": True, "source": "BIOMNI_LLM_PROXY_URL"}

    for env in _PROVIDER_KEY_ENVS:
        # Via the vault, not os.getenv: credentials are hidden from os.environ
        # while generated code runs, and a probe landing in that window must
        # not report the pod unready.
        if credentials.getenv(env):
            return {"ok": True, "source": env}
    if os.getenv("BIOMNI_CUSTOM_BASE_URL"):
        return {"ok": True, "source": "BIOMNI_CUSTOM_BASE_URL"}
    return {"ok": False, "source": None}


def readiness_report() -> tuple[bool, dict[str, Any]]:
    """Evaluate readiness checks.

    Returns ``(ok, report)`` where ``report`` carries per-check detail suitable
    for the probe body and for debugging a pod stuck in ``NotReady``.
    """
    checks: dict[str, dict[str, Any]] = {
        "data_path": _workspace_check(),
        "llm_credential": _llm_credential_check(),
    }

    ok = all(c["ok"] for c in checks.values())
    report: dict[str, Any] = {"status": "ready" if ok else "not_ready", "checks": checks}
    build = build_info()
    if build:
        report["build"] = build
    return ok, report


def register_routes_first(app: Any, handlers: list[tuple[str, Callable]]) -> None:
    """Put ``handlers`` (``[(path, endpoint), ...]``) ahead of every other route.

    Shared by every operational endpoint this app exposes, because they all face
    the same two problems: Chainlit's catch-all ``GET /{full_path:path}`` would
    otherwise answer them with the SPA, and the entry module is re-imported on
    dev reload, which would append a second copy of each route.

    Registration is therefore idempotent - any existing route on one of these
    paths is dropped first - and order is preserved, so the caller's first
    handler ends up first in the router.
    """
    from starlette.routing import Route

    paths = {path for path, _ in handlers}
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in paths]

    # insert(0, ...) per route, reversed so the final order matches ``handlers``.
    for path, handler in reversed(handlers):
        app.router.routes.insert(0, Route(path, handler, methods=["GET"]))


def register_health_routes(
    app: Any,
    *,
    liveness_path: str = "/healthz",
    readiness_path: str = "/readyz",
) -> None:
    """Register liveness/readiness routes at the front of ``app``'s router.

    ``app`` is the Starlette/FastAPI application (``chainlit.server.app``).
    """
    from starlette.responses import JSONResponse

    async def healthz(_request: Request) -> JSONResponse:
        payload: dict[str, Any] = {"status": "ok"}
        build = build_info()
        if build:
            payload["build"] = build
        return JSONResponse(payload, status_code=200)

    async def readyz(_request: Request) -> JSONResponse:
        ok, report = readiness_report()
        return JSONResponse(report, status_code=200 if ok else 503)

    register_routes_first(app, [(liveness_path, healthz), (readiness_path, readyz)])


__all__ = ["build_info", "readiness_report", "register_health_routes", "register_routes_first"]
