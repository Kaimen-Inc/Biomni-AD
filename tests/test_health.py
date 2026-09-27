"""Tests for biomni.health - readiness logic and probe-route registration.

The route-precedence tests matter most: Chainlit registers a catch-all
``GET /{full_path:path}`` last, so health routes must be inserted ahead of it
or they get shadowed by the SPA fallback.
"""

from __future__ import annotations

import pytest
from biomni import health

# readiness_report() / build_info() are pure (os.getenv only) and run everywhere.
# The route-registration tests need fastapi/starlette, which are OPTIONAL deps
# (the `chainlit` extra) - absent in the minimal CI test job. Guard those tests
# so they skip there instead of erroring at collection, while the
# dependency-free readiness tests still run.
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

requires_fastapi = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/starlette not installed (chainlit extra)")

# Every credential / path env that readiness inspects - cleared per test so the
# host's real keys don't leak into assertions.
_ENV_TO_CLEAR = (
    *health._PROVIDER_KEY_ENVS,
    "BIOMNI_CUSTOM_BASE_URL",
    "BIOMNI_PATH",
    "BIOMNI_DATA_PATH",
    "BIOMNI_USER_DATA_PATH",
    "BIOMNI_GIT_SHA",
    "GIT_SHA",
    "BIOMNI_GIT_REF",
    "GIT_REF",
    "BIOMNI_LLM_PROXY_URL",
    "BIOMNI_LLM_PROXY_SCHEMA",
    "BIOMNI_LLM_PROXY_API_KEY",
    "BIOMNI_TRUST_AUTH_HEADERS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# build_info
# ---------------------------------------------------------------------------


def test_build_info_empty_without_env():
    assert health.build_info() == {}


def test_build_info_reads_git_env(monkeypatch):
    monkeypatch.setenv("BIOMNI_GIT_SHA", "abc1234")
    monkeypatch.setenv("BIOMNI_GIT_REF", "main")
    assert health.build_info() == {"git_sha": "abc1234", "git_ref": "main"}


# ---------------------------------------------------------------------------
# readiness_report
# ---------------------------------------------------------------------------


def test_behind_the_llm_proxy_its_token_is_the_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_API_KEY", "proxy-token")
    monkeypatch.setenv("BIOMNI_TRUST_AUTH_HEADERS", "true")
    ok, report = health.readiness_report()
    assert ok is True
    assert report["checks"]["llm_credential"] == {"ok": True, "source": "BIOMNI_LLM_PROXY_URL"}


def test_a_proxy_that_will_refuse_every_call_is_not_ready(monkeypatch, tmp_path):
    """Without the gateway's ids the proxy refuses every request, so the pod cannot serve."""
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_API_KEY", "proxy-token")
    ok, report = health.readiness_report()
    assert ok is False
    assert "BIOMNI_TRUST_AUTH_HEADERS" in report["checks"]["llm_credential"]["detail"]


def test_a_leftover_provider_key_does_not_make_a_tokenless_proxy_ready(monkeypatch, tmp_path):
    """Every model call goes through the proxy, so without its token none can succeed."""
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    ok, report = health.readiness_report()
    assert ok is False
    check = report["checks"]["llm_credential"]
    assert check["ok"] is False
    assert "BIOMNI_LLM_PROXY_API_KEY" in check["detail"]


def test_a_misspelt_proxy_schema_makes_the_pod_unready(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("BIOMNI_LLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_API_KEY", "proxy-token")
    monkeypatch.setenv("BIOMNI_LLM_PROXY_SCHEMA", "openia")
    ok, report = health.readiness_report()
    assert ok is False
    assert "BIOMNI_LLM_PROXY_SCHEMA" in report["checks"]["llm_credential"]["detail"]


def test_readiness_ok_when_data_and_credential_present(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    ok, report = health.readiness_report()
    assert ok is True
    assert report["status"] == "ready"
    assert report["checks"]["data_path"]["ok"] is True
    assert report["checks"]["llm_credential"] == {"ok": True, "source": "ANTHROPIC_API_KEY"}


def test_readiness_fails_without_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    ok, report = health.readiness_report()
    assert ok is False
    assert report["status"] == "not_ready"
    assert report["checks"]["llm_credential"]["ok"] is False


@pytest.mark.parametrize("env", ["BIOMNI_USER_DATA_PATH", "BIOMNI_DATA_PATH"])
def test_readiness_waits_for_a_configured_workspace(monkeypatch, tmp_path, env):
    monkeypatch.setenv(env, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-whatever")
    ok, report = health.readiness_report()
    assert ok is False
    assert report["checks"]["data_path"] == {"ok": False, "path": str(tmp_path / "does-not-exist")}


def test_readiness_does_not_wait_for_the_apps_own_data_directory(monkeypatch, tmp_path):
    """The image sets BIOMNI_PATH to a directory only a chat creates - and an unready pod gets no chats."""
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path / "created-on-first-use"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-whatever")
    ok, report = health.readiness_report()
    assert ok is True
    assert report["checks"]["data_path"] == {"ok": True, "path": None}


def test_readiness_accepts_custom_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("BIOMNI_CUSTOM_BASE_URL", "http://vllm:8000/v1")
    ok, report = health.readiness_report()
    assert ok is True
    assert report["checks"]["llm_credential"]["source"] == "BIOMNI_CUSTOM_BASE_URL"


# ---------------------------------------------------------------------------
# register_health_routes
# ---------------------------------------------------------------------------


def _app_with_catchall() -> FastAPI:
    """A FastAPI app whose last route is a catch-all, like Chainlit's."""
    app = FastAPI()

    async def catchall(_request):
        return PlainTextResponse("SPA-FALLBACK")

    app.router.routes.append(Route("/{full_path:path}", catchall, methods=["GET"]))
    return app


@requires_fastapi
def test_healthz_returns_200():
    app = FastAPI()
    health.register_health_routes(app)
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@requires_fastapi
def test_readyz_200_when_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMNI_PATH", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    app = FastAPI()
    health.register_health_routes(app)
    resp = TestClient(app).get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@requires_fastapi
def test_readyz_503_when_not_ready():
    app = FastAPI()
    health.register_health_routes(app)
    resp = TestClient(app).get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


@requires_fastapi
def test_health_routes_take_precedence_over_catchall():
    app = _app_with_catchall()
    health.register_health_routes(app)
    client = TestClient(app)
    # Without front-insertion these would return "SPA-FALLBACK".
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] in ("ready", "not_ready")
    # A genuinely unknown path still falls through to the catch-all.
    assert client.get("/some/spa/route").text == "SPA-FALLBACK"


@requires_fastapi
def test_register_is_idempotent():
    app = _app_with_catchall()
    health.register_health_routes(app)
    health.register_health_routes(app)
    health.register_health_routes(app)
    healthz_routes = [r for r in app.router.routes if getattr(r, "path", None) == "/healthz"]
    assert len(healthz_routes) == 1
    assert TestClient(app).get("/healthz").status_code == 200


@requires_fastapi
def test_healthz_includes_build_info(monkeypatch):
    monkeypatch.setenv("BIOMNI_GIT_SHA", "deadbeef")
    app = FastAPI()
    health.register_health_routes(app)
    assert TestClient(app).get("/healthz").json()["build"] == {"git_sha": "deadbeef"}
