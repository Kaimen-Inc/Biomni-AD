#!/usr/bin/env bash

set -euo pipefail

HOST="${CHAINLIT_HOST:-0.0.0.0}"
PORT="${CHAINLIT_PORT:-8000}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: Invalid CHAINLIT_PORT '$PORT'. Use an integer between 1 and 65535."
    exit 1
fi

exec micromamba run -n biomni_e1 \
    python -m chainlit run /app/chainlit_app.py --host "$HOST" --port "$PORT"