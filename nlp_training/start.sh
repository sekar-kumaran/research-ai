#!/usr/bin/env bash
# ── Research Assistant Startup ────────────────────────────────────────────────
set -e

# Load .env if present
if [ -f .env ]; then
  echo "📦 Loading .env..."
  export $(grep -v '^#' .env | xargs)
fi

# Check Python
python3 --version >/dev/null 2>&1 || { echo "❌ Python 3 required"; exit 1; }

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "📦 Installing dependencies..."
  pip install -r requirements.txt --quiet
fi

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    Arxiv Research AI — Starting Server       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Backend : ${LLM_BACKEND:-cloud}"
echo "  Provider: ${CLOUD_LLM_PROVIDER:-groq}"
echo "  URL     : http://localhost:8000"
echo ""

# Start FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
