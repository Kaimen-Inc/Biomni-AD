#!/usr/bin/env bash
#
# Container entrypoint for the Biomni-AD Chainlit UI.
#
# The Dockerfile prepends ``/opt/conda/envs/biomni_e1/bin`` to PATH so the
# correct python interpreter is the first ``python`` on PATH. We exec it
# directly (no ``micromamba run`` wrapper) — saves the per-startup
# activation cost and shrinks the runtime dependency surface.

set -euo pipefail

HOST="${CHAINLIT_HOST:-0.0.0.0}"
PORT="${CHAINLIT_PORT:-8000}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: Invalid CHAINLIT_PORT '$PORT'. Use an integer between 1 and 65535." >&2
    exit 1
fi

exec python -m chainlit run /app/chainlit_app.py --host "$HOST" --port "$PORT"
