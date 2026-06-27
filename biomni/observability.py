"""Structured logging and observability for container / Kubernetes deployments.

This module is the single place that decides *how* Biomni emits operational
telemetry. It exists because the default Python logging setup (plain text to
stderr, no correlation, no redaction) is nearly unusable once the app runs as
many pods behind Azure Kubernetes Service:

* **Azure Container Insights ingests stdout/stderr line-by-line.** Emitting one
  JSON object per line turns each field into a queryable column in the Log
  Analytics workspace (KQL), instead of an opaque text blob.
* **Interleaved pods need correlation.** ``session_id`` / ``run_id`` are carried
  in :mod:`contextvars` and auto-injected into every record, so one user's agent
  trajectory can be reconstructed across pods with a single filter.
* **The log sink retains and indexes everything.** A redaction chokepoint scrubs
  credentials (and, opt-in, e-mail-shaped PII) out of every line before it is
  written — important for an app that handles API keys and biomedical data.

Nothing here touches the root logger on import. Call :func:`setup_logging` once
at process start (the Chainlit entrypoint does this). Library/test imports stay
side-effect free.

Public surface:

* :func:`setup_logging` — install the JSON/text handler on the root logger.
* :func:`emit_event` — log a structured event with arbitrary fields.
* :func:`log_llm_usage` — convenience wrapper for per-run token/cost telemetry.
* :func:`bind_run`, :func:`set_session_id`, :func:`set_run_id`, :func:`get_context`
  — manage correlation IDs.
* :func:`capture_context` — snapshot the caller's context for replay in a worker
  thread (executors do not copy contextvars otherwise).
* :class:`Redactor` — the secret/PII scrubber (exposed for testing).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Mapping  # runtime use (isinstance in RunHeartbeat)
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# ---------------------------------------------------------------------------
# Correlation context
# ---------------------------------------------------------------------------

# Set at request/run boundaries (e.g. the Chainlit message handler) and read by
# the formatter so every line — including ones emitted deep in worker threads —
# carries the same ids. ``None`` means "not in a run"; such keys are omitted
# from output rather than logged as null.
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("biomni_session_id", default=None)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("biomni_run_id", default=None)


def set_session_id(value: str | None) -> contextvars.Token:
    """Bind the current session id. Returns a token for :meth:`ContextVar.reset`."""
    return session_id_var.set(value)


def set_run_id(value: str | None) -> contextvars.Token:
    """Bind the current run id. Returns a token for :meth:`ContextVar.reset`."""
    return run_id_var.set(value)


def get_context() -> dict[str, str]:
    """Return the currently-bound correlation ids (omitting unset ones)."""
    ctx: dict[str, str] = {}
    sid = session_id_var.get()
    rid = run_id_var.get()
    if sid is not None:
        ctx["session_id"] = sid
    if rid is not None:
        ctx["run_id"] = rid
    return ctx


@contextmanager
def bind_run(*, session_id: str | None = None, run_id: str | None = None):
    """Scope a session/run id to a block, resetting on exit.

    Only the ids that are passed are bound; ``None`` leaves the existing value
    untouched so a per-message ``run_id`` can be set without clobbering the
    chat-wide ``session_id``.
    """
    tokens: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
    if session_id is not None:
        tokens.append((session_id_var, session_id_var.set(session_id)))
    if run_id is not None:
        tokens.append((run_id_var, run_id_var.set(run_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def capture_context() -> contextvars.Context:
    """Snapshot the *current* thread's context for replay in a worker thread.

    ``ThreadPoolExecutor`` / ``loop.run_in_executor`` do not propagate
    contextvars into the worker thread, so correlation ids set in an async
    handler would be lost in the agent graph that runs there.

    This MUST be called in the originating thread, then the returned context's
    ``run`` is what crosses the boundary::

        ctx = capture_context()  # caller thread
        executor.submit(ctx.run, fn, *args)  # worker replays caller's ids

    Calling ``copy_context()`` inside the worker instead would capture the
    worker's empty context and propagate nothing. Each call returns a fresh
    copy, so distinct executor tasks never share (and contend over) one context.
    """
    return contextvars.copy_context()


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Names whose *values* are treated as secrets and scrubbed verbatim wherever
# they appear. Substring match (case-insensitive) so ``ANTHROPIC_API_KEY`` and
# ``MY_SERVICE_TOKEN`` both qualify. The minimum-length guard in ``Redactor``
# protects against a short, non-secret value (e.g. an empty placeholder key)
# triggering collateral redaction of unrelated log text.
_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")

REDACTED = "[REDACTED]"

# High-precision provider-key shapes. Ordered longest/most-specific first.
_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),  # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{16,}"),  # OpenAI project keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / generic sk- keys
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}"),  # Authorization: Bearer ...
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{6,}"),  # JWTs
)

# base64 data URIs (matplotlib plots are embedded as these). Collapse to a short
# marker — keeps multi-MB image blobs out of the log pipeline.
_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+")

# e-mail-shaped strings; redaction is opt-in (PII, but also matches legitimate
# log content), gated by BIOMNI_LOG_REDACT_EMAILS.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


class Redactor:
    """Scrubs secrets (and optionally e-mails) from log strings.

    Two layers: (1) verbatim removal of known secret *values* harvested from the
    environment, which catches custom credentials no regex would, and (2)
    pattern matching for well-known key/token shapes. Both run on every line, so
    a credential that leaks via an unexpected code path is still caught.
    """

    def __init__(self, secret_values: Iterable[str] = (), *, redact_emails: bool = False) -> None:
        # Longest first: redacting a longer secret before a shorter substring of
        # it avoids leaving a recognizable tail behind.
        self._secret_values = sorted({s for s in secret_values if s and len(s) >= 6}, key=len, reverse=True)
        self._redact_emails = redact_emails

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None, *, redact_emails: bool | None = None) -> Redactor:
        env = os.environ if environ is None else environ
        secrets: list[str] = []
        for name, value in env.items():
            if not value:
                continue
            upper = name.upper()
            if any(hint in upper for hint in _SECRET_ENV_HINTS):
                secrets.append(value)
        if redact_emails is None:
            redact_emails = _env_flag(env.get("BIOMNI_LOG_REDACT_EMAILS"), default=False)
        return cls(secrets, redact_emails=redact_emails)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for secret in self._secret_values:
            if secret in text:
                text = text.replace(secret, REDACTED)
        text = _DATA_URI.sub("data:[REDACTED base64]", text)
        for pat in _KEY_PATTERNS:
            text = pat.sub(REDACTED, text)
        if self._redact_emails:
            text = _EMAIL.sub("[REDACTED-EMAIL]", text)
        return text


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

# LogRecord attributes that are framework-internal; everything else attached to
# a record (i.e. caller-supplied ``extra=``) is treated as a structured field.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


# Static service identity (service/version/env/host/pid) stamped on every
# record. Populated once by setup_logging; empty until then so pre-setup and
# library/test imports stay side-effect free. See _resolve_service_fields.
_SERVICE_FIELDS: dict[str, Any] = {}


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Assemble the base + context + extra fields for a record (pre-redaction)."""
    try:
        message = record.getMessage()
    except Exception:  # pragma: no cover - malformed %-args should never crash logging
        message = str(record.msg)

    fields: dict[str, Any] = {
        "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "msg": message,
    }
    # Static identity first, so correlation ids and caller-supplied fields below
    # win on any key collision (they never collide in practice).
    fields.update(_SERVICE_FIELDS)
    # Correlation ids (read live from contextvars — works in worker threads when
    # the caller propagated context via capture_context).
    fields.update(get_context())

    # Caller-supplied structured fields.
    for key, value in record.__dict__.items():
        if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
            fields[key] = value

    if record.exc_info:
        fields["exc"] = logging.Formatter().formatException(record.exc_info)
    return fields


class JsonFormatter(logging.Formatter):
    """Render each record as a single redacted JSON line."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or Redactor()

    def format(self, record: logging.LogRecord) -> str:
        fields = _record_fields(record)
        try:
            text = json.dumps(fields, default=str, ensure_ascii=False)
        except Exception:
            # A field with a hostile __str__ slipped past default=str (its
            # exception propagates out of json.dumps). The logging chokepoint
            # must never raise, so fall back to a minimal record (ts/level/
            # logger/msg are always plain strings) flagged for triage rather
            # than dropping the line entirely.
            text = json.dumps(
                {
                    "ts": fields.get("ts"),
                    "level": fields.get("level"),
                    "logger": fields.get("logger"),
                    "msg": fields.get("msg"),
                    "log_format_error": True,
                },
                ensure_ascii=False,
            )
        return self._redactor.redact(text)


class HumanFormatter(logging.Formatter):
    """Compact single-line text format for local dev (still redacted)."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        super().__init__()
        self._redactor = redactor or Redactor()

    def format(self, record: logging.LogRecord) -> str:
        fields = _record_fields(record)
        ts = fields.pop("ts")
        level = fields.pop("level")
        logger_name = fields.pop("logger")
        msg = fields.pop("msg")
        exc = fields.pop("exc", None)
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{ts} {level:<7} {logger_name} {msg}"
        if extras:
            line += f" | {extras}"
        if exc:
            line += f"\n{exc}"
        return self._redactor.redact(line)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_MANAGED_ATTR = "_biomni_managed"
# Libraries whose INFO/DEBUG chatter would drown the app's own logs. Pinned to
# WARNING unless the operator raises the level explicitly.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "watchfiles", "asyncio")


def _env_flag(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    name = (level or os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def _detect_version() -> str | None:
    """Installed biomni package version, or None if it can't be determined."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("biomni")
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - importlib.metadata should always exist
        return None


def _resolve_service_fields() -> dict[str, Any]:
    """Static identity stamped on every log line so records are attributable.

    Lets you filter a shared Log Analytics workspace by service/version/env and
    tell pods (and workers within a pod) apart. All env-overridable; ``service``
    follows the OpenTelemetry ``OTEL_SERVICE_NAME`` convention. ``host`` is the
    pod name under Kubernetes (the pod's hostname). ``version``/``env`` are
    omitted when unknown rather than logged as null.
    """
    fields: dict[str, Any] = {
        "service": os.getenv("OTEL_SERVICE_NAME") or os.getenv("BIOMNI_SERVICE_NAME") or "biomni",
        "pid": os.getpid(),
    }
    host = os.getenv("HOSTNAME") or socket.gethostname()
    if host:
        fields["host"] = host
    version = os.getenv("BIOMNI_VERSION") or _detect_version()
    if version:
        fields["version"] = version
    env = os.getenv("BIOMNI_ENV") or os.getenv("DEPLOY_ENV") or os.getenv("ENVIRONMENT")
    if env:
        fields["env"] = env
    return fields


def setup_logging(
    level: str | int | None = None,
    *,
    fmt: str | None = None,
    stream: Any | None = None,
    force: bool = False,
    quiet_noisy_loggers: bool = True,
) -> logging.Handler:
    """Configure root logging for structured stdout output. Idempotent.

    Parameters
    ----------
    level:
        Log level (name or int). Falls back to ``$LOG_LEVEL`` then ``INFO``.
    fmt:
        ``"json"`` (default, for AKS/Container Insights) or ``"text"`` (dev).
        Falls back to ``$BIOMNI_LOG_FORMAT`` then ``"json"``.
    stream:
        Output stream (default ``sys.stdout`` — container log collectors read
        stdout; stderr is reserved for genuine process errors).
    force:
        Re-configure even if already configured (replaces our handler).

    Returns the installed handler.
    """
    root = logging.getLogger()
    managed = [h for h in root.handlers if getattr(h, _MANAGED_ATTR, False)]
    if managed and not force:
        # Already configured. Just keep level in sync (e.g. a later call with an
        # explicit level) and return the existing handler.
        root.setLevel(_resolve_level(level))
        return managed[0]

    # Claim the root logger: drop *every* existing handler, not just a prior
    # managed one. Libraries imported before us install their own root handler
    # — chainlit's CLI calls logging.basicConfig() at import time, adding a
    # plain-text StreamHandler. Left in place it double-emits every record
    # (once plain, once JSON), corrupting the structured stream Container
    # Insights parses. Safe because this function is the single source of
    # truth for biomni's stdout logging.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = (fmt or os.getenv("BIOMNI_LOG_FORMAT") or "json").strip().lower()
    redactor = Redactor.from_environ()
    formatter: logging.Formatter = JsonFormatter(redactor) if fmt != "text" else HumanFormatter(redactor)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    setattr(handler, _MANAGED_ATTR, True)

    root.addHandler(handler)
    root.setLevel(_resolve_level(level))

    # Resolve service identity once, here, so it reflects the process env at
    # configure time and is stamped on every record by _record_fields.
    global _SERVICE_FIELDS
    _SERVICE_FIELDS = _resolve_service_fields()

    if quiet_noisy_loggers:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    return handler


# ---------------------------------------------------------------------------
# Structured events
# ---------------------------------------------------------------------------


def _sanitize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Rename keys that collide with LogRecord internals (logging would raise)."""
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
            safe[f"{key}_"] = value
        else:
            safe[key] = value
    return safe


def emit_event(
    event: str,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured ``event`` with arbitrary fields.

    The message text is the event name (greppable) and ``event`` plus every
    field becomes a top-level key in the JSON output. Never raises — telemetry
    must not break the path it instruments.
    """
    log = logger or logging.getLogger("biomni.events")
    try:
        extra = _sanitize_fields(fields)
        extra["event"] = event
        log.log(level, event, extra=extra)
    except Exception:  # pragma: no cover - defensive: logging must never break callers
        log.debug("emit_event failed for %s", event, exc_info=True)


def log_llm_usage(
    summary: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> None:
    """Emit an ``llm_usage`` event from an :class:`LLMUsageTracker` summary."""
    payload = dict(summary)
    payload.update(fields)
    emit_event("llm_usage", logger=logger, **payload)


def diff_usage_summary(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Per-run delta between two :meth:`LLMUsageTracker.summary` snapshots.

    Token/`calls` counts are subtracted; ``cache_hit_ratio`` is recomputed from
    the delta (a straight subtraction of ratios would be meaningless).
    """
    counters = ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "total_tokens")
    delta = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in counters}
    cached = delta["cache_read_tokens"]
    fresh = delta["input_tokens"]
    delta["cache_hit_ratio"] = round(cached / (cached + fresh), 4) if (cached + fresh) > 0 else 0.0
    return delta


# ---------------------------------------------------------------------------
# Liveness heartbeat
# ---------------------------------------------------------------------------


class RunHeartbeat:
    """Emit a periodic ``run_heartbeat`` event while a long operation is in flight.

    A multi-step agent run can spend minutes inside a single LLM call or code
    execution with no log output, making a healthy-but-slow run indistinguishable
    from a wedged one. This context manager spawns a daemon thread that logs an
    elapsed-time event every ``interval`` seconds until the block exits, so the
    run's liveness is continuously visible in the log pipeline.

    The caller's correlation context is captured at ``__enter__`` (in the calling
    thread) and replayed for each emit, so heartbeats carry the run's session_id /
    run_id even though they fire from a separate thread.

    Usage::

        with RunHeartbeat(interval=15, status=lambda: {"step": agent.react_step}):
            run_the_agent()
    """

    def __init__(
        self,
        *,
        interval: float = 15.0,
        event: str = "run_heartbeat",
        logger: logging.Logger | None = None,
        status: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        # Small positive floor guards against a 0/negative interval busy-looping
        # the thread; callers pick a sane cadence (Chainlit defaults to ~15s).
        self._interval = max(0.05, float(interval))
        self._event = event
        self._logger = logger
        self._status = status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ctx: contextvars.Context | None = None
        self._start = 0.0

    def __enter__(self) -> RunHeartbeat:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("RunHeartbeat is single-use and not reentrant")
        self._ctx = capture_context()
        self._start = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="biomni-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        return False  # never suppress exceptions

    def _loop(self) -> None:
        # Event.wait returns True once stopped; the loop ends without a final emit.
        # Guard the whole body: a heartbeat that dies on an unexpected error must
        # not leave an unjoinable thread or a stack trace on stderr.
        while not self._stop.wait(self._interval):
            try:
                self._ctx.run(self._emit) if self._ctx is not None else self._emit()
            except Exception:  # pragma: no cover - defensive: keep the heartbeat alive
                logging.getLogger("biomni.events").debug("heartbeat emit failed", exc_info=True)

    def _emit(self) -> None:
        fields: dict[str, Any] = {"elapsed_ms": round((time.monotonic() - self._start) * 1000, 1)}
        if self._status is not None:
            try:
                extra = self._status()
                if isinstance(extra, Mapping):
                    fields.update(extra)
            except Exception:  # status callback must never break the heartbeat
                logging.getLogger("biomni.events").debug("heartbeat status callback failed", exc_info=True)
        emit_event(self._event, logger=self._logger, **fields)


__all__ = [
    "HumanFormatter",
    "JsonFormatter",
    "REDACTED",
    "Redactor",
    "RunHeartbeat",
    "bind_run",
    "capture_context",
    "diff_usage_summary",
    "emit_event",
    "get_context",
    "log_llm_usage",
    "run_id_var",
    "session_id_var",
    "set_run_id",
    "set_session_id",
    "setup_logging",
]
