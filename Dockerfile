# syntax=docker/dockerfile:1.7

# Multi-stage build:
#   1. ``builder``  — provisions the conda env, installs Chainlit, installs
#                     the biomni package editable. Uses BuildKit cache mounts
#                     for the conda pkg cache and pip cache so neither lands
#                     in a published layer.
#   2. ``runtime``  — copies just the prepared env (``/opt/conda/envs/biomni_e1``)
#                     and the application code. Activates the env via ``PATH``
#                     so HEALTHCHECK and ENTRYPOINT can run python directly
#                     without per-invocation ``micromamba run`` overhead.
#
# Base image is pinned by multi-arch manifest digest in both stages. The
# digest is inlined on each FROM (not stored in an ARG) so Dependabot's
# Docker updater can recognise and bump it — it only rewrites the literal
# FROM line, not ARG defaults.

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.10@sha256:e3797091302382ea841498bc93a7b0a50f7c1448333d5e946d2d1608d0c5f43d AS builder

ARG BIOMNI_ENV_FILE=biomni_env/environment.yml

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
    micromamba run -n biomni_e1 pip install "chainlit>=1.0" --index-url https://pypi.org/simple/

# Editable install of the biomni package. Stage 2 copies the same source
# tree to the same /app path so the .pth pointer resolves at runtime.
COPY pyproject.toml README.md MANIFEST.in /app/
COPY biomni /app/biomni
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    micromamba run -n biomni_e1 pip install -e /app

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ──────────────────────────────────────────────────────────────────────────
# Same digest as stage 1 — keep both lines in sync when bumping. The
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
# docker-metadata-action — specifically ``image.revision`` (= github.sha)
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
# (MAMBA_USER_ID; stable across the 1.5.x series — if upstream bumps it,
# both the FROM digest and these uids must be updated together) so the
# tree is owned correctly whether the image runs as root (compose default
# for first-time setup) or 57439:57439 (production-hardened mode).
# Chainlit's init writes translation files into /app/.chainlit/, which
# would fail if the tree were root-owned and the container ran as
# mambauser.
#
# chainlit_ui/ is shipped source-only on purpose (pyproject excludes it
# from the wheel) — the chainlit_app.py entrypoint imports it relative
# to cwd, so the directory must be present at the chainlit working dir.
#
# pyproject.toml, README.md, MANIFEST.in are NOT copied here: the
# editable install in stage 1 baked the package metadata into the env's
# site-packages (.pth + .dist-info), so the runtime stage doesn't need
# the originals.
COPY --chown=57439:57439 biomni /app/biomni
# chainlit.md.template (not chainlit.md) is the tracked source — chainlit_app.py
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
ENV PATH=/opt/conda/envs/biomni_e1/bin:$PATH \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000 \
    BIOMNI_PATH=/app/data

EXPOSE 8000

# TCP probe: Chainlit binds 8000 only after startup completes, so a
# successful connect implies the app is serving. start-period covers the
# ~30–60s of import time for the full agent stack on cold start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c \
      "import socket; s=socket.create_connection(('127.0.0.1', 8000), timeout=3); s.close()" \
      || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
