#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_chainlit.sh - launch the Biomni-AD Chainlit UI from the project root.
#
# The whole point of this wrapper is to invoke the conda env's python even
# when a virtualenv (.venv) is active in the same shell. Without it, a
# half-set-up .venv silently wins on PATH and Chainlit imports break.
#
# Usage:
#   bash run_chainlit.sh                       # default port 8000
#   bash run_chainlit.sh --port 8080           # any chainlit flag is passed through
#   bash run_chainlit.sh --headless --debug    # multiple flags fine
#
# Environment variables:
#   BIOMNI_CONDA_ENV   conda env name              (default: biomni_e1)
#   BIOMNI_LLM         LLM model name              (forwarded to chainlit_app.py)
#   BIOMNI_PATH        data directory              (forwarded)
#   ANTHROPIC_API_KEY / OPENAI_API_KEY             as required by your LLM
#
# For container deployments use the Dockerfile entrypoint instead; this
# script is the local-dev launcher only.
# ---------------------------------------------------------------------------

set -euo pipefail

REQUIRED_ENV="${BIOMNI_CONDA_ENV:-biomni_e1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/chainlit_app.py"

# Prefer conda when both conda and micromamba are present - conda's
# activation scripts handle a wider range of env edge cases.
if command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif command -v micromamba &>/dev/null; then
    CONDA_CMD="micromamba"
else
    echo "ERROR: neither conda nor micromamba is on PATH." >&2
    echo "       Install Miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi

# A bare `conda run` against a missing env produces a long stack trace; the
# explicit check gives a one-line actionable error instead.
#
# awk with exact-string comparison (no regex) so env names containing
# dots/plus/brackets (technically allowed by conda) don't confuse us.
if ! "$CONDA_CMD" env list | awk -v name="$REQUIRED_ENV" 'NR > 1 && $1 == name { found=1 } END { exit !found }'; then
    echo "ERROR: conda environment '${REQUIRED_ENV}' not found." >&2
    echo "       Create it with:  cd biomni_env && bash setup.sh" >&2
    exit 1
fi

# Strip any active .venv from the env so it can't shadow conda's python.
# MPLBACKEND=Agg keeps matplotlib usable on headless boxes.
# --no-capture-output lets chainlit's startup banner stream live.
exec env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME MPLBACKEND=Agg \
    "$CONDA_CMD" run --no-capture-output -n "$REQUIRED_ENV" \
    python -m chainlit run "$APP" "$@"
