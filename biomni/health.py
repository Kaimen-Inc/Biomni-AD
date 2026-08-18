"""Liveness / readiness probes for the Chainlit app under Kubernetes (AKS).

Kubernetes distinguishes two probe kinds and they must mean different things:

* **liveness** (``/healthz``) - "is the process wedged?". Must be cheap and must
  not depend on external systems, or a transient upstream blip triggers a
  pod restart. So ``/healthz`` returns 200 as long as the event loop can serve
  a request.
* **readiness** (``/readyz``) - "should this pod receive traffic *right now*?".
  Here we verify the agent could actually function: its data directory is
  mounted and an LLM credential is configured. A misconfigured pod reports 503
  and is pulled from the Service endpoints instead of failing user requests.

Both endpoints are registered at the *front* of the router. Chainlit mounts a
catch-all ``GET /{full_path:path}`` (for the SPA) as its last route; a route
appended after it would be shadowed, so we insert ahead of everything.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

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


def _data_path() -> str:
    """The data directory the app expects to be mounted (honors env overrides)."""
    # Read env directly rather than importing biomni.config: readiness must
    # reflect the *current* environment and stay import-light for the probe path.
    return (
        os.getenv("BIOMNI_USER_DATA_PATH")
        or os.getenv("BIOMNI_DATA_PATH")
        or os.getenv("BIOMNI_PATH")
        or os.path.join(os.path.expanduser("~"), ".biomni", "data")
    )


def _has_llm_credential() -> tuple[bool, str]:
    """Whether some usable LLM credential / endpoint is configured."""
    for env in _PROVIDER_KEY_ENVS:
        if os.getenv(env):
            return True, env
    if os.getenv("BIOMNI_CUSTOM_BASE_URL"):
        return True, "BIOMNI_CUSTOM_BASE_URL"
    return False, ""


def readiness_report() -> tuple[bool, dict[str, Any]]:
    """Evaluate readiness checks.

    Returns ``(ok, report)`` where ``report`` carries per-check detail suitable
    for the probe body and for debugging a pod stuck in ``NotReady``.
    """
    checks: dict[str, dict[str, Any]] = {}

    data_path = _data_path()
    data_ok = os.path.isdir(data_path)
    checks["data_path"] = {"ok": data_ok, "path": data_path}

    cred_ok, cred_source = _has_llm_credential()
    checks["llm_credential"] = {"ok": cred_ok, "source": cred_source or None}

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
