#!/usr/bin/env bash
# ============================================================================
#  Research AI Intelligence Platform v3.1 — Start Script
#  For LOCAL development. On Hugging Face Spaces, app.py is the entrypoint.
# ============================================================================
set -e

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
if [ -f .env ]; then
  echo "Loading .env..."
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
python3 --version >/dev/null 2>&1 || { echo "Python 3 is required"; exit 1; }

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt --quiet
fi

# ---------------------------------------------------------------------------
# PYTHONPATH — must include src/ so 'import research_ai' resolves correctly
# ---------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
: "${LLM_BACKEND:=cloud}"
: "${CLOUD_LLM_PROVIDER:=gemini}"
: "${PORT:=8000}"
: "${HOST:=0.0.0.0}"
: "${ENABLE_PYTHON_EXECUTION:=false}"

echo ""
echo "=============================================="
echo "  Research AI Intelligence Platform v3.1"
echo "=============================================="
echo "  Backend  : ${LLM_BACKEND}"
echo "  Provider : ${CLOUD_LLM_PROVIDER}"
echo "  Model    : ${GEMINI_MODEL:-${GOOGLE_MODEL:-gemini-2.0-flash}}"
echo "  URL      : http://localhost:${PORT}"
echo "  Python X : ${ENABLE_PYTHON_EXECUTION}"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Mode: production vs development
# ---------------------------------------------------------------------------
if [ "${DEV_MODE:-false}" = "true" ]; then
  echo "Starting in DEVELOPMENT mode (--reload enabled)..."
  exec uvicorn research_ai.api.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --reload \
    --reload-dir src
else
  echo "Starting in PRODUCTION mode (no --reload)..."
  exec uvicorn research_ai.api.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
fi
