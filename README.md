---
title: Research AI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---

# Research AI Intelligence Platform

**Version 3.1 · Agentic · Gemini-powered · Hugging Face Ready**

An autonomous AI research intelligence platform that orchestrates specialised local ML models,
hybrid retrieval (FAISS + BM25), citation intelligence, and sandboxed execution to analyse arXiv papers.
Deployed on [Hugging Face Spaces](https://huggingface.co/spaces) with **Google Gemini** as the cloud LLM.

---

## Architecture at a Glance

```
User Request
    │
    ▼
FastAPI (src/research_ai/api/main.py)
    │
    ▼
ResearchOrchestrator
    ├── PlannerAgent        ← Intent analysis, dynamic tool plan (Gemini)
    ├── RetrievalAgent      ← Strategy-aware retrieval specialist
    ├── MLExecutionAgent    ← Tool dispatch registry
    ├── EvaluatorAgent      ← Quality scoring + retry decision
    └── SynthesisAgent      ← Gemini synthesis over grounded outputs
         │
         ▼
    Tool Registry (13 tools)
         ├── classify_query         (sklearn classifier on arXiv data)
         ├── hybrid_search          (FAISS + BM25 + metadata reranking)
         ├── smart_retrieve         (strategy-aware RetrievalAgent)
         ├── summarize              (distilBART / cloud LLM)
         ├── methodology_extract    (regex NLP)
         ├── citation_signals       (category/year co-occurrence)
         ├── citation_proxy         (full proxy citation graph)
         ├── trend_analysis         (year/category statistics)
         ├── metadata_analyse       (author, quality, completeness)
         ├── paper_chat             (per-session FAISS paper Q&A)
         ├── metadata_rag           (retrieval-grounded Gemini answer)
         ├── run_pipeline           (named multi-step pipelines)
         ├── python_execute         (sandboxed scientific computation)
         └── conversation           (greetings / small talk)
```

---

## Quick Start — Local Development (Gemini)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY to your key from https://aistudio.google.com/

# 3. Run (development mode with hot-reload)
DEV_MODE=true ./start.sh

# On Windows:
# set PYTHONPATH=src
# uvicorn research_ai.api.main:app --reload --reload-dir src --port 8000

# API docs: http://localhost:8000/docs
```

---

## Hugging Face Spaces Deployment

### 1. Push the repository to a Gradio Space

Create a new Space at https://huggingface.co/new-space with:
- **SDK**: Gradio
- **Hardware**: ZeroGPU (or CPU Basic for free tier)
- **Visibility**: Public

Push (or link) this repository to that Space. The `app.py` at the root is
the HF entrypoint — it launches the real FastAPI server on port 7860.

### 2. Upload runtime artifacts

Commit these directly to the Space repo (small enough for standard git):

| File | Size | Required |
|---|---|---|
| `artifacts/similarity/paper_index.faiss` | ~12 MB | ✅ Yes |
| `artifacts/similarity/paper_metadata.parquet` | ~5 MB | ✅ Yes |
| `artifacts/similarity/metadata_parts/part_00000.parquet` | ~25 MB | ✅ Yes |
| `artifacts/similarity/embedding_model_name.joblib` | tiny | ✅ Yes |
| `artifacts/classification/tfidf_vectorizer.joblib` | tiny | ✅ Yes |
| `artifacts/classification/labels.joblib` | tiny | ✅ Yes |
| `artifacts/classification/classifier.joblib` | ~608 MB | ⚠️ Git LFS |
| `artifacts/clustering/kmeans.joblib` | ~80 MB | Optional |

For `classifier.joblib` (608 MB), use **Git LFS**:
```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add artifacts/classification/classifier.joblib
git commit -m "feat: add classifier artifact via LFS"
```

Alternatively host it on a HF Dataset repo and set `HF_ARTIFACTS_REPO` — the
`download_artifacts.py` script will fetch it at startup.

### 3. Set Hugging Face Secrets

**Space → Settings → Repository secrets → New secret**

| Secret Name | Value | Required |
|---|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from https://aistudio.google.com | ✅ Required |
| `LLM_BACKEND` | `cloud` | ✅ Required |
| `CLOUD_LLM_PROVIDER` | `gemini` | ✅ Required |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Optional |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Optional |
| `ENABLE_PYTHON_EXECUTION` | `false` | Optional (already default) |
| `HF_ARTIFACTS_REPO` | `your-name/research-ai-artifacts` | Only if using HF Dataset for large artifacts |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `cloud` | `cloud` or `local` |
| `CLOUD_LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `openrouter`, `google`, `ollama` |
| `GEMINI_API_KEY` | — | **Gemini API key** (primary for HF Spaces) |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model name |
| `GOOGLE_API_KEY` | — | Legacy alias for `GEMINI_API_KEY` |
| `GROQ_API_KEY` | — | Groq API key (if using Groq) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for FAISS |
| `DATA_ROOT` | `data` | Directory containing arXiv parquet shards |
| `ARTIFACTS_ROOT` | `artifacts` | Directory for trained model artifacts |
| `ENABLE_PYTHON_EXECUTION` | `false` | Enable sandboxed Python runner |
| `PORT` | `7860` | Server port (7860 = HF Spaces standard) |
| `HOST` | `0.0.0.0` | Server bind address |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `HF_ARTIFACTS_REPO` | — | HF Dataset repo ID for large artifact download |
| `HF_TOKEN` | — | HF token for private Dataset repos |
| `DEV_MODE` | `false` | Enable hot-reload in `start.sh` (local dev only) |

---

## Key API Endpoints

### Orchestration
| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Agentic research analysis (all tools) |
| POST | `/agent/run/stream` | Streaming SSE version of agent run |
| POST | `/ask` | Shorthand for auto-mode agent run |
| POST | `/chat/message` | Unified conversational AI endpoint |
| POST | `/chat/stream` | Streaming conversational AI |

### ML Models
| Method | Path | Description |
|---|---|---|
| POST | `/classify` | arXiv category classification |
| POST | `/search` | Hybrid FAISS+BM25 paper search |
| POST | `/summarize` | Scientific summarization |
| POST | `/similarity` | Semantic similarity between two texts |

### Research Intelligence
| Method | Path | Description |
|---|---|---|
| POST | `/metadata/analyse` | Author/category/quality analysis |
| POST | `/citation/proxy` | Proxy citation graph from metadata |
| POST | `/citation/clusters` | Co-citation topic clustering |
| POST | `/citation/timeline` | Chronological influence timeline |
| GET  | `/knowledge-graph` | Session knowledge graph summary |

### Paper Chat
| Method | Path | Description |
|---|---|---|
| POST | `/chat/upload` | Upload a PDF or text file |
| POST | `/chat/load-arxiv` | Load paper from arXiv by ID |
| POST | `/chat/ask` | Ask a question about a loaded paper |
| POST | `/chat/multi-ask` | Ask across multiple loaded papers |
| POST | `/chat/bulk-load` | Load multiple arXiv papers at once |

### Pipelines
| Method | Path | Description |
|---|---|---|
| POST | `/pipeline/run` | Run a named multi-step pipeline |
| GET  | `/pipeline/list` | List available pipelines |

### Execution
| Method | Path | Description |
|---|---|---|
| POST | `/execution/python` | Run sandboxed Python code (disabled by default) |

---

## Available Pipelines

| Pipeline Name | Steps | Use Case |
|---|---|---|
| `full_research_analysis` | classify → retrieve → methodology → citations → trends → RAG | Deep paper analysis |
| `quick_search_and_summarize` | retrieve → RAG | Fast answers |
| `classify_and_find_similar` | classify → retrieve | Category-aware search |
| `trend_report` | retrieve → trends → citations | Research trend overview |

---

## Directory Structure

```
├── app.py                     ← Hugging Face Space entrypoint
├── download_artifacts.py      ← Startup artifact downloader (HF Dataset)
├── start.sh                   ← Local development launcher
├── requirements.txt
├── .env.example
├── src/research_ai/
│   ├── agents/
│   │   ├── planner/           Intent analysis, dynamic tool plan
│   │   ├── orchestrator/      Plan→Execute→Evaluate→Synthesize loop
│   │   ├── retrieval_agent/   Strategy-aware retrieval specialist
│   │   ├── ml_execution_agent/ Tool dispatch
│   │   ├── evaluator_agent/   Quality scoring + retry logic
│   │   └── synthesis_agent/   Gemini-powered synthesis
│   ├── ml_models/
│   │   ├── classifier/        arXiv category classifier (sklearn)
│   │   ├── summarizer/        Scientific summarizer
│   │   ├── methodology_extractor/ Method/experiment signal extraction
│   │   ├── similarity/        Cosine similarity comparison
│   │   ├── ranking/           Retrieval score + metadata ranking
│   │   └── citation_graph/    Category/year co-occurrence signals
│   ├── retrieval/
│   │   ├── embeddings/        SentenceTransformer with LRU cache
│   │   ├── hybrid_search/     FAISS + BM25 + metadata reranking
│   │   ├── rerankers/         Keyword overlap reranker
│   │   ├── vector_store/      FAISS index + Parquet metadata
│   │   └── chunking.py        Sentence-aware contextual chunking
│   ├── research/
│   │   ├── paper_ingestion/   arXiv PDF fetch, session creation
│   │   ├── citation_engine/   Proxy citation graph from metadata
│   │   ├── metadata/          Author/quality/completeness analysis
│   │   └── trend_analysis/    Year/category trend statistics
│   ├── memory/
│   │   ├── knowledge_graph/   Cross-session concept tracker
│   │   └── session_memory/    Per-session chat history + FAISS index
│   ├── execution/
│   │   ├── python_runner/     Sandboxed subprocess execution
│   │   ├── sandbox/           AST-level static analysis validator
│   │   └── pipelines/         Composable multi-step pipelines
│   ├── api/                   FastAPI routes + Pydantic schemas
│   ├── configs/               Typed settings with env-var binding
│   ├── common/                Text cleaning, secret redaction
│   ├── llm.py                 Cloud LLM client (Gemini/Groq/OpenRouter/Ollama)
│   └── platform.py            Composition root / dependency injection
├── frontend/                  Existing HTML/CSS/JS UI (preserved)
└── artifacts/
    ├── similarity/            FAISS index + metadata (committed to repo)
    └── classification/        sklearn classifier artifacts (Git LFS for large files)
```

---

## Extending the Platform

### Add a new tool
1. Implement the function in the appropriate `ml_models/` or `research/` service.
2. Register it in `platform.py` → `_build_tool_registry()`.
3. Add its name and schema to the `PlannerAgent` `TOOL_CATALOG`.

### Add a new pipeline
Add an entry to `PIPELINES` in `src/research_ai/execution/pipelines/service.py`.

### Switch LLM provider
Change `CLOUD_LLM_PROVIDER` to `groq`, `openrouter`, or `ollama` and set the
corresponding API key. The `CloudLLMClient` in `llm.py` handles all providers
through the same `generate()` / `chat()` interface.

### Swap the embedding model
Set `EMBEDDING_MODEL=your-model-name` in `.env` and rebuild the FAISS index.
