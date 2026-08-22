# syntax=docker/dockerfile:1.7

# Multi-stage build:
#   1. ``builder``  - provisions the conda env, installs Chainlit, installs
#                     the biomni package editable. Uses BuildKit cache mounts
#                     for the conda pkg cache and pip cache so neither lands
#                     in a published layer.
#   2. ``runtime``  - copies just the prepared env (``/opt/conda/envs/biomni_e1``)
#                     and the application code. Activates the env via ``PATH``
#                     so HEALTHCHECK and ENTRYPOINT can run python directly
#                     without per-invocation ``micromamba run`` overhead.
#
# Base image is pinned by multi-arch manifest digest in both stages. The
# digest is inlined on each FROM (not stored in an ARG) so Dependabot's
# Docker updater can recognise and bump it - it only rewrites the literal
# FROM line, not ARG defaults.

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.10@sha256:e3797091302382ea841498bc93a7b0a50f7c1448333d5e946d2d1608d0c5f43d AS builder

# The AD Workbench environment is the default because it is the one that
# matches what the agent is *told* it has: biomni/env_desc.py advertises ~113
# libraries to the model and instructs it to prefer locally installed ones.
# The minimal environment.yml supplies only ten of them, so the agent would
# routinely write scanpy / gseapy / biopython code that fails at the import.
# This file costs image size and buys correctness; override it for a small
# deployment that only needs the chat and pandas-level analysis:
#
#   docker build --build-arg BIOMNI_ENV_FILE=biomni_env/environment.yml .
ARG BIOMNI_ENV_FILE=biomni_env/adworkbench_env.yml

# Run install steps as root inside the builder so cache mounts at
# system paths work without uid plumbing; the runtime stage drops back
# to the base image's default non-root user automatically.
USER root
WORKDIR /app

COPY ${BIOMNI_ENV_FILE} /tmp/biomni_env.yml

# BuildKit cache mounts: pkg downloads and pip's HTTP cache persist across
# builds for speed, but never enter any image layer. ``sharing=locked`` so
# parallel builds (e.g. arm64 + amd64) don't corrupt the cache.
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba create -y -n biomni_e1 -f /tmp/biomni_env.yml && \
    micromamba run -n biomni_e1 pip install --upgrade pip && \
    micromamba run -n biomni_e1 pip install "chainlit[custom-data]>=2.8" "aiosqlite>=0.20" --index-url https://pypi.org/simple/

# Editable install of the biomni package. Stage 2 copies the same source
# tree to the same /app path so the .pth pointer resolves at runtime.
COPY pyproject.toml README.md MANIFEST.in /app/
COPY biomni /app/biomni
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba run -n biomni_e1 pip install -e /app

# Drop the compiler toolchain now that everything needing it has been built.
# The runtime stage copies this environment wholesale, so anything left here
# ships. Must be the last builder step for that reason. The script verifies its
# own work - every module still on disk has to still import - so a bad prune
# fails the build rather than producing an image that dies on first use.
COPY docker/prune-build-tools.sh /tmp/prune-build-tools.sh
RUN bash /tmp/prune-build-tools.sh /opt/conda/envs/biomni_e1

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ──────────────────────────────────────────────────────────────────────────
# Same digest as stage 1 - keep both lines in sync when bumping. The
# duplication is intentional: Dependabot's Docker updater rewrites the
# literal FROM line and won't follow an ARG.
FROM mambaorg/micromamba:1.5.10@sha256:e3797091302382ea841498bc93a7b0a50f7c1448333d5e946d2d1608d0c5f43d AS runtime

# Build-arg metadata for provenance. The CI workflow passes GIT_SHA / GIT_REF
# so a pulled image can be traced back to a commit without registry tag soup.
ARG GIT_SHA=""
ARG GIT_REF=""

# OCI image labels.
#
# These are the default labels baked into the image. The CI publish path
# (.github/workflows/docker.yml) layers additional / overriding labels via
# docker-metadata-action - specifically ``image.revision`` (= github.sha)
# and ``image.version`` (= the primary tag, e.g. ``sha-abc1234``). For
# images built outside CI (``docker build .``) these fields fall back to
# whatever GIT_SHA / GIT_REF build-args the operator passes, or empty.
# That's intentional: in-CI metadata wins, local builds get whatever the
# builder feeds in.
LABEL org.opencontainers.image.title="Biomni-AD" \
      org.opencontainers.image.description="Alzheimer's-specialized biomedical AI agent (Chainlit UI)" \
      org.opencontainers.image.source="https://github.com/Kaimen-Inc/Biomni-AD" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Kaimen Inc." \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${GIT_REF}"

WORKDIR /app

# Copy the prepared env. Includes python, chainlit, biomni's editable
# install pointer, and all bio-Python deps. Ownership matches the base
# image's mambauser (UID 57439).
COPY --from=builder --chown=57439:57439 /opt/conda/envs/biomni_e1 /opt/conda/envs/biomni_e1

# Copy application sources. The biomni dir path must match the editable
# install location used in stage 1 (/app/biomni); the others are runtime
# assets. ``--chown=57439:57439`` matches the base image's mambauser
# (MAMBA_USER_ID; stable across the 1.5.x series - if upstream bumps it,
# both the FROM digest and these uids must be updated together) so the
# tree is owned correctly whether the image runs as root (compose default
# for first-time setup) or 57439:57439 (production-hardened mode).
# Chainlit's init writes translation files into /app/.chainlit/, which
# would fail if the tree were root-owned and the container ran as
# mambauser.
#
# chainlit_ui/ is shipped source-only on purpose (pyproject excludes it
# from the wheel) - the chainlit_app.py entrypoint imports it relative
# to cwd, so the directory must be present at the chainlit working dir.
#
# pyproject.toml, README.md, MANIFEST.in are NOT copied here: the
# editable install in stage 1 baked the package metadata into the env's
# site-packages (.pth + .dist-info), so the runtime stage doesn't need
# the originals.
COPY --chown=57439:57439 biomni /app/biomni
# chainlit.md.template (not chainlit.md) is the tracked source - chainlit_app.py
# rewrites chainlit.md on every launch with the local data inventory, so
# the file itself is gitignored.
COPY --chown=57439:57439 chainlit_app.py chainlit.md.template /app/
COPY --chown=57439:57439 chainlit_ui /app/chainlit_ui
COPY --chown=57439:57439 .chainlit /app/.chainlit
COPY --chown=57439:57439 public /app/public
COPY --chmod=755 --chown=57439:57439 docker/entrypoint.sh /app/docker/entrypoint.sh

# Activate the env by prepending its bin to PATH. ENTRYPOINT and
# HEALTHCHECK then call ``python`` directly, avoiding the ~200ms
# per-invocation cost of ``micromamba run -n biomni_e1``.
#
# Logging defaults target Azure Container Insights / Log Analytics:
#   LOG_LEVEL=INFO            structured-log verbosity (DEBUG for triage)
#   BIOMNI_LOG_FORMAT=json    one JSON object per stdout line (KQL-queryable)
#   BIOMNI_ENABLE_LLM_TELEMETRY=true  emit per-run token/cost telemetry events
# GIT_SHA/GIT_REF are surfaced as env too so the /healthz + /readyz probes can
# report which image build is live (build-args are otherwise label-only).
ENV PATH=/opt/conda/envs/biomni_e1/bin:$PATH \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000 \
    BIOMNI_PATH=/app/data \
    LOG_LEVEL=INFO \
    BIOMNI_LOG_FORMAT=json \
    BIOMNI_ENABLE_LLM_TELEMETRY=true \
    BIOMNI_GIT_SHA=${GIT_SHA} \
    BIOMNI_GIT_REF=${GIT_REF}

EXPOSE 8000

# HTTP liveness probe against /healthz - confirms the app is actually serving
# (not just that the port is open). start-period covers the ~30–60s of import
# time for the full agent stack on cold start. Kubernetes uses its own probes
# (see deploy/k8s/); this HEALTHCHECK is for Docker/Compose deployments.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c \
      "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)" \
      || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
