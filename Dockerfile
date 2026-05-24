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
# Base image is pinned by multi-arch manifest digest in both stages.
# Update the digest deliberately when bumping the base version.

ARG MICROMAMBA_BASE=mambaorg/micromamba:1.5.10@sha256:e3797091302382ea841498bc93a7b0a50f7c1448333d5e946d2d1608d0c5f43d

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────────────────────────────────
FROM ${MICROMAMBA_BASE} AS builder

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
FROM ${MICROMAMBA_BASE} AS runtime

# Build-arg metadata for provenance. The CI workflow passes GIT_SHA / GIT_REF
# so a pulled image can be traced back to a commit without registry tag soup.
ARG GIT_SHA=""
ARG GIT_REF=""

# OCI image labels — surfaced by GHCR, used by image scanners and policy
# engines. Keep these in lockstep with the metadata-action labels in
# .github/workflows/docker.yml so registry-side and image-embedded metadata
# agree.
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
# assets. ``--chown`` makes the tree owned by the base image's non-root
# mambauser so the image can run as either root (default for first-time
# compose runs) or 57439:57439 (production-hardened mode) — chainlit's
# init writes translation files into /app/.chainlit, which would fail
# read-only.
#
# chainlit_ui/ is shipped source-only on purpose (pyproject excludes it
# from the wheel) — the chainlit_app.py entrypoint imports it from cwd,
# so the directory must be present at the chainlit working directory.
COPY --chown=57439:57439 pyproject.toml README.md MANIFEST.in /app/
COPY --chown=57439:57439 biomni /app/biomni
COPY --chown=57439:57439 chainlit_app.py chainlit.md /app/
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
