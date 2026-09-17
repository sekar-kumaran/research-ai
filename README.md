# ResearchAI

Local ML + GenAI research intelligence platform for arXiv-style paper discovery, retrieval, analysis, and persistent research chat.

## What Runs Locally

ResearchAI is now wired for a local Docker architecture:

- `backend`: FastAPI app, agent orchestrator, auth, database access, local ML services
- `postgres`: persistent users, sessions, conversations, messages, papers, analysis records, saved papers, graph records
- `ollama`: preferred local LLM provider
- `/artifacts`: canonical runtime ML artifact directory
- `/data`: runtime cache, uploads, and SQLite fallback data

The LLM is the synthesis layer. Paper discovery stays ML/retrieval-first:

```text
query -> classifier -> FAISS + BM25 -> ranking -> paper analysis -> Ollama synthesis -> database
```

## Install Docker

1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Start Docker Desktop.
3. Wait until Docker says the Linux engine is running.
4. Verify:

```bash
docker version
docker compose version
```

## Start The App

Copy the environment file and edit secrets if needed:

```bash
cp .env.example .env
```

Start services:

```bash
docker compose up --build
```

Open:

```text
http://localhost:7860
```

Create an account in the UI, then log in. Conversations are stored in the database and survive container restarts.

## Ollama

Docker Compose uses:

```env
CLOUD_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_MODEL=qwen3:8b
```

Recommended models:

- `qwen3:8b`
- `qwen3:4b`
- `llama3.1:8b`

Pull a model:

```bash
docker compose exec ollama ollama pull qwen3:8b
```

If Ollama is offline or the model is missing, retrieval and persistence still work; LLM synthesis reports the issue instead of pretending it ran.

## ML Artifacts

Canonical layout:

```text
artifacts/
  classification/
    classifier.joblib
    tfidf_vectorizer.joblib
  similarity/
    paper_index.faiss
    paper_metadata.parquet
    embedding_model_name.joblib
  clustering/
    kmeans.joblib
    cluster_assignments.parquet
```

Startup validates artifacts and exposes status at:

```text
GET /system/artifacts
GET /health
```

If large artifacts are hosted in a Hugging Face Dataset repo, configure:

```env
HF_ARTIFACTS_REPO=owner/research-ai-artifacts
HF_TOKEN=optional_private_repo_token
```

Then run:

```bash
python download_artifacts.py
```

The downloader stores files under `ARTIFACTS_ROOT`. It does not silently replace valid files.

## Current Artifact Status

In this workspace, FAISS and metadata validate successfully:

```text
paper_index.faiss: 8000 vectors, dim=384
paper_metadata.parquet: 8000 rows
```

Missing or unavailable artifacts are reported honestly:

```text
classifier.joblib: missing
kmeans.joblib: missing
cluster_assignments.parquet: missing
```

That means search works locally when the SentenceTransformer model is available, while classification and clustering remain unavailable until the real artifacts are restored.

## API Highlights

Auth:

```text
POST /api/auth/signup
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
PATCH /api/auth/profile
```

Research and chat:

```text
POST /chat/message
POST /chat/stream
GET  /conversations
GET  /conversations/{id}
PATCH /conversations/{id}
POST /conversations/{id}/clear
DELETE /conversations/{id}
POST /search
POST /classify
POST /summarize
POST /chat/upload
POST /chat/load-arxiv
POST /chat/ask
```

Diagnostics:

```text
GET /health
GET /stats
GET /system/artifacts
GET /models/list
```

## Local Development Without Docker

```powershell
$env:PYTHONPATH="src"
python -m uvicorn research_ai.api.main:app --host 127.0.0.1 --port 8000 --reload
```

The app uses `DATABASE_URL` if set. Without it, it falls back to:

```text
data/researchai.sqlite3
```

## Tests

Run:

```bash
pytest -q
```

Current local result:

```text
138 passed, 8 deselected
```

Docker image build was not completed in this session because Docker Desktop was not running:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

Start Docker Desktop and rerun:

```bash
docker compose config --quiet
docker compose build backend
docker compose up
```

## Interview Demo Flow

1. Start Docker Compose.
2. Pull the configured Ollama model.
3. Open `http://localhost:7860`.
4. Sign up and log in.
5. Ask: `Find recent transformer research for crop disease detection.`
6. Show `/system/artifacts` and `/health` to explain which ML features are ready.
7. Show FAISS + BM25 search results and source cards.
8. Open a conversation from the history panel after restart to demonstrate persistence.
9. If classifier/clustering artifacts are restored, show those pipeline steps too.

## Troubleshooting

- `classifier.joblib missing`: set `HF_ARTIFACTS_REPO` or copy the real file into `artifacts/classification/`.
- `Embedding model unavailable`: pre-download `all-MiniLM-L6-v2` into the model cache, or set `MODEL_LOCAL_FILES_ONLY=false` when network is available.
- `Ollama not reachable`: start the `ollama` service and pull the configured model.
- `Docker API not found`: start Docker Desktop and wait for the Linux engine.
- `Database connection failed`: check `DATABASE_URL` and Postgres container health.
