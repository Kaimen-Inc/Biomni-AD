#!/bin/bash
# ---------------------------------------------------------------------------
# run_chainlit.sh — Launch the Biomni Chainlit UI
#
# Always uses the biomni_e1 conda environment's Python directly, so it works
# correctly even when a virtualenv (.venv) is also active in the shell.
#
# Usage:
#   bash run_chainlit.sh                   # default port 8000
#   bash run_chainlit.sh --port 8080       # custom port (auto-fallback if busy)
#   bash run_chainlit.sh --headless        # no browser auto-open (CI/servers)
#
# Environment variables (optional):
#   BIOMNI_LLM    LLM model name           (default: claude-sonnet-4-5)
#   BIOMNI_PATH   Data directory           (default: ./data)
#   ANTHROPIC_API_KEY / OPENAI_API_KEY     as required by your chosen LLM
# ---------------------------------------------------------------------------

set -e

REQUIRED_ENV="biomni_e1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/chainlit_app.py"
DEFAULT_PORT=8000

# ---- Parse args and resolve requested port --------------------------------
REQUESTED_PORT="$DEFAULT_PORT"
CHAINLIT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "ERROR: --port requires a numeric value."
                exit 1
            fi
            REQUESTED_PORT="$2"
            shift 2
            ;;
        --port=*)
            REQUESTED_PORT="${1#*=}"
            shift
            ;;
        *)
            CHAINLIT_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$REQUESTED_PORT" =~ ^[0-9]+$ ]] || (( REQUESTED_PORT < 1 || REQUESTED_PORT > 65535 )); then
    echo "ERROR: Invalid port '$REQUESTED_PORT'. Use an integer between 1 and 65535."
    exit 1
fi

port_in_use() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

PORT="$REQUESTED_PORT"
if port_in_use "$PORT"; then
    START_PORT="$PORT"
    while (( PORT <= 65535 )) && port_in_use "$PORT"; do
        ((PORT++))
    done

    if (( PORT > 65535 )); then
        echo "ERROR: Could not find a free TCP port starting from $START_PORT."
        exit 1
    fi

    echo ""
    echo "WARNING: Port $START_PORT is already in use. Falling back to port $PORT."
fi

# ---- Resolve the conda env's Python explicitly ----------------------------
# We do NOT rely on PATH (a .venv activated in the same shell would win).
# `conda run` always uses the right interpreter regardless of PATH ordering.

if ! command -v conda &>/dev/null && ! command -v micromamba &>/dev/null; then
    echo ""
    echo "ERROR: conda (or micromamba) is not available."
    echo "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_CMD="conda"
command -v conda &>/dev/null || CONDA_CMD="micromamba"

# Verify the target env exists
if ! "$CONDA_CMD" env list | grep -q "^${REQUIRED_ENV}[[:space:]]"; then
    echo ""
    echo "ERROR: conda environment '$REQUIRED_ENV' not found."
    echo ""
    echo "  Set it up first:"
    echo "    cd biomni_env && bash setup.sh"
    echo ""
    exit 1
fi

# ---- Locate the app file ---------------------------------------------------
if [[ ! -f "$APP" ]]; then
    echo "ERROR: chainlit_app.py not found at $APP"
    exit 1
fi

# ---- Ensure chainlit is installed in the target env -----------------------
if ! "$CONDA_CMD" run -n "$REQUIRED_ENV" python -c "import chainlit" &>/dev/null 2>&1; then
    echo "chainlit not found in '$REQUIRED_ENV'. Installing now..."
    "$CONDA_CMD" run -n "$REQUIRED_ENV" pip install "chainlit>=1.0"
fi

# ---- Launch ----------------------------------------------------------------
echo ""
echo "Starting Biomni Chainlit UI..."
echo "  Conda env   : $REQUIRED_ENV"
echo "  LLM         : ${BIOMNI_LLM:-claude-sonnet-4-5}"
echo "  Data path   : ${BIOMNI_PATH:-./data}"
echo "  Port        : $PORT"
echo ""

# Use `conda run` to guarantee we use biomni_e1's Python, not any .venv.
# Force headless matplotlib backend to avoid macOS GUI/thread crashes.
export MPLBACKEND=Agg

exec "$CONDA_CMD" run --no-capture-output -n "$REQUIRED_ENV" \
    python -m chainlit run "$APP" --port "$PORT" "${CHAINLIT_ARGS[@]}"
