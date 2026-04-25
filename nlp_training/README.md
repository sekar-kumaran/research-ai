# Research AI Application (nlp_training)

Production-ready NLP application for research-paper search, classification, summarization, similarity scoring, and interactive paper chat.

This README is scoped to the application runtime inside this folder. For repository-level context, see the root README.

## Table of Contents

1. Overview
2. Features
3. Architecture
4. Prerequisites
5. Setup
6. Configuration
7. Run
8. API Surface
9. Artifacts and Data
10. Troubleshooting

## Overview

The application serves:

- A FastAPI backend with task-specific and mediated agent endpoints.
- A static web frontend mounted by the backend.
- Retrieval and inference workflows powered by prebuilt artifacts under `artifacts`.

## Features

- Semantic paper retrieval over FAISS index.
- Paper category classification with confidence output.
- Text summarization and arXiv-ID paper summarization.
- Text-to-text semantic similarity scoring.
- RAG-style question answering.
- Paper chat sessions from arXiv papers or uploaded documents.
- Multi-paper chat and bulk paper loading.

## Architecture

```text
nlp_training/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   └── research_assistant/
│       ├── agents.py
│       ├── cloud_llm.py
│       ├── paper_chat.py
│       ├── rag.py
│       ├── similarity.py
│       ├── summarization.py
│       └── config.py
├── pipeline_scripts/
├── artifacts/
├── requirements.txt
└── start.sh
```

## Prerequisites

- Python 3.10+
- pip
- Access to one cloud LLM provider key (recommended)

## Setup

Run all commands from this folder (`nlp_training`).

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create environment file

Use the repository template file and copy it into this folder:

Windows PowerShell:

```powershell
Copy-Item ..\.env.example .env
```

Linux/macOS:

```bash
cp ../.env.example .env
```

Set at least one valid provider key (for example `GROQ_API_KEY`) in `.env`.

### 4. Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Configuration

Important environment variables:

| Variable | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `cloud` | `cloud` or `local` |
| `CLOUD_LLM_PROVIDER` | `groq` | `groq`, `openrouter`, `google` |
| `GROQ_API_KEY` | empty | Required for Groq |
| `OPENROUTER_API_KEY` | empty | Required for OpenRouter |
| `GOOGLE_API_KEY` | empty | Required for Google |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Cloud model selection |
| `ARTIFACTS_ROOT` | `artifacts` | Model and index artifacts |
| `DATA_ROOT` | `data` | Dataset location |

## Run

### Option A: Direct uvicorn

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Startup script (Linux/macOS)

```bash
bash start.sh
```

Open:

- App: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## API Surface

Key endpoints:

- `GET /health`
- `GET /stats`
- `POST /search`
- `POST /classify`
- `POST /summarize`
- `POST /summarize-paper`
- `POST /similarity`
- `POST /ask`
- `POST /agent/run`
- `POST /agent/run/stream`
- `POST /chat/load-arxiv`
- `POST /chat/upload`
- `POST /chat/ask`
- `POST /chat/multi-ask`
- `POST /chat/bulk-load`
- `GET /chat/session/{id}`

Use Swagger for request and response contracts.

## Artifacts and Data

Runtime components rely on artifact availability:

- `artifacts/classification`: classifier and vectorizer files.
- `artifacts/similarity`: FAISS index and paper metadata.
- Other artifact folders support clustering, preprocessing outputs, and showcase assets.

If artifacts are missing, affected routes return unready or unavailable states.

## Troubleshooting

- Check `GET /health` for classifier, summarizer, and RAG readiness.
- Verify `.env` keys if cloud generation, routing, or summarization fails.
- Confirm `artifacts/similarity` exists if search and ask responses are empty.
- On Windows PowerShell activation errors, set process execution policy before activating the venv.
