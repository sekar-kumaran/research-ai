# Research AI

Research AI is an end-to-end NLP platform for research-paper discovery, classification, summarization, semantic retrieval, and conversational Q&A over paper content.

It combines a trained ML/NLP artifact pipeline with a FastAPI backend and a web frontend for interactive use.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Capabilities](#core-capabilities)
3. [Repository Structure](#repository-structure)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Running the Application](#running-the-application)
9. [API Overview](#api-overview)
10. [Model Artifacts](#model-artifacts)
11. [Development Notes](#development-notes)
12. [Troubleshooting](#troubleshooting)

## Project Overview

This repository contains two primary work areas:

- `dataset/`: Data sources, chunked datasets, and dataset notes.
- `nlp_training/`: Core application, training outputs, backend API, frontend UI, and research assistant modules.

The runtime application is launched from `nlp_training/` and serves both API endpoints and the static frontend.

## Core Capabilities

- Semantic paper search over FAISS index artifacts
- Paper category classification (e.g., cs, math, physics, q-bio)
- Text and paper summarization
- Similarity scoring between two text inputs
- Retrieval-augmented question answering over indexed papers
- Session-based paper chat (arXiv load, upload, and multi-paper query)

## Repository Structure

```text
.
├── dataset/
│   ├── dataset detail.txt
│   ├── arxiv_chunks/
│   └── data/
├── nlp_training/
│   ├── README.md
│   ├── requirements.txt
│   ├── start.sh
│   ├── artifacts/
│   ├── frontend/
│   ├── pipeline_scripts/
│   ├── resources/
│   └── src/
├── .env.example
└── .gitignore
```

## Tech Stack

- Backend: FastAPI, Uvicorn, Pydantic
- ML/NLP: scikit-learn, sentence-transformers, FAISS, transformers, NLTK, PyTorch
- Data: pandas, NumPy, pyarrow, polars
- Frontend: HTML/CSS/JavaScript (served by FastAPI)

## Prerequisites

- Python 3.10+
- `pip`
- Git
- Optional: API key for cloud LLM provider (recommended for full functionality)

## Quick Start

### 1. Navigate to the app directory

```powershell
cd nlp_training
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

From repository root:

```bash
cp ../.env.example .env
```

Then update `.env` with valid provider credentials (for example `GROQ_API_KEY`).

### 5. Download required NLTK assets

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Configuration

Key environment variables:

- `LLM_BACKEND`: `cloud` or `local`
- `CLOUD_LLM_PROVIDER`: `groq`, `openrouter`, or `google`
- `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`: provider credentials
- `ARTIFACTS_ROOT`: defaults to `artifacts`
- `DATA_ROOT`: defaults to `data`

The canonical template is in `.env.example` at repository root.

## Running the Application

From `nlp_training/`:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Alternative startup script:

```bash
bash start.sh
```

Open:

- App: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

## API Overview

Main endpoints include:

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

Refer to Swagger docs for request/response schemas.

## Model Artifacts

Application artifacts are stored under `nlp_training/artifacts/` and include:

- Classification models and vectorizers
- Clustering outputs
- Feature-engineering manifests
- Similarity index and metadata
- Processed dataset state and summaries

If artifacts are missing, related API features may return unavailable/503 responses.

## Development Notes

- Primary backend entry point: `nlp_training/src/api/main.py`
- Domain logic: `nlp_training/src/research_assistant/`
- Frontend assets: `nlp_training/frontend/`
- Pipeline scripts: `nlp_training/pipeline_scripts/`

## Troubleshooting

- If API starts but some features fail, check `GET /health` to verify loaded components.
- If summarization/routing quality is poor, verify cloud provider key and model settings.
- If FAISS search is unavailable, ensure similarity artifacts exist under `nlp_training/artifacts/similarity/`.
- If PowerShell blocks activation, run session-scoped execution policy command shown in Quick Start.
