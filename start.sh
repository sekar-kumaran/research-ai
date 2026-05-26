#!/usr/bin/env bash
set -e

if [ -f .env ]; then
  echo "Loading .env..."
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

python3 --version >/dev/null 2>&1 || { echo "Python 3 is required"; exit 1; }

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt --quiet
fi

echo ""
echo "=========================================="
echo "  Research AI Intelligence Platform v3.1"
echo "=========================================="
echo "  Backend : ${LLM_BACKEND:-cloud}"
echo "  Provider: ${CLOUD_LLM_PROVIDER:-groq}"
echo "  URL     : http://localhost:${PORT:-8000}"
echo "=========================================="
echo ""

# PYTHONPATH must include src/ so 'import research_ai' resolves correctly
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"

uvicorn research_ai.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --reload \
  --reload-dir src
