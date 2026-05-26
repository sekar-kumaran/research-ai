# Research AI Intelligence Platform

**Version 3.1 · Agentic · Local-first · Scientific**

An autonomous AI research intelligence platform that orchestrates specialised local ML models, hybrid retrieval, citation intelligence, and sandboxed execution to analyse arXiv papers.

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
    ├── PlannerAgent        ← Intent analysis, dynamic tool plan
    ├── RetrievalAgent      ← Strategy-aware retrieval specialist
    ├── MLExecutionAgent    ← Tool dispatch registry
    ├── EvaluatorAgent      ← Quality scoring + retry decision
    └── SynthesisAgent      ← Cloud LLM synthesis over grounded outputs
         │
         ▼
    Tool Registry (13 tools)
         ├── classify_query         (sklearn classifier on arXiv data)
         ├── hybrid_search          (FAISS + BM25 + metadata reranking)
         ├── smart_retrieve         (strategy-aware RetrievalAgent)
         ├── summarize              (distilBART / cloud LLM)
         ├── methodology_extract    (regex NLP + extensible to fine-tuned tagger)
         ├── citation_signals       (category/year co-occurrence)
         ├── citation_proxy         (full proxy citation graph)
         ├── trend_analysis         (year/category statistics)
         ├── metadata_analyse       (author, quality, completeness)
         ├── paper_chat             (per-session FAISS paper Q&A)
         ├── metadata_rag           (retrieval-grounded LLM answer)
         ├── run_pipeline           (named multi-step pipelines)
         ├── python_execute         (sandboxed scientific computation)
         └── conversation           (greetings / small talk)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your LLM API key

# 3. Run the server
./start.sh
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `cloud` | `cloud` or `local` |
| `CLOUD_LLM_PROVIDER` | `groq` | `groq`, `openrouter`, or `google` |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `DATA_ROOT` | `data` | Directory containing arXiv parquet shards |
| `ARTIFACTS_ROOT` | `artifacts` | Directory for trained model artifacts |
| `ENABLE_PYTHON_EXECUTION` | `false` | Enable sandboxed Python runner |
| `PYTHON_EXEC_TIMEOUT` | `5` | Execution timeout in seconds |
| `PORT` | `8000` | Server port |

---

## Key API Endpoints

### Orchestration
| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Agentic research analysis (all tools) |
| POST | `/agent/run/stream` | Streaming SSE version of agent run |
| POST | `/ask` | Shorthand for auto-mode agent run |

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

### Pipelines
| Method | Path | Description |
|---|---|---|
| POST | `/pipeline/run` | Run a named multi-step pipeline |
| GET  | `/pipeline/list` | List available pipelines |

### Paper Chat
| Method | Path | Description |
|---|---|---|
| POST | `/chat/upload` | Upload a PDF or text file |
| POST | `/chat/load-arxiv` | Load paper from arXiv by ID |
| POST | `/chat/ask` | Ask a question about a loaded paper |
| POST | `/chat/multi-ask` | Ask across multiple loaded papers |
| POST | `/chat/bulk-load` | Load multiple arXiv papers at once |

### Execution
| Method | Path | Description |
|---|---|---|
| POST | `/execution/python` | Run sandboxed Python code |

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
├── src/research_ai/
│   ├── agents/
│   │   ├── planner/           Intent analysis, dynamic tool plan
│   │   ├── orchestrator/      Plan→Execute→Evaluate→Synthesize loop
│   │   ├── retrieval_agent/   Strategy-aware retrieval specialist
│   │   ├── ml_execution_agent/ Tool dispatch
│   │   ├── evaluator_agent/   Quality scoring + retry logic
│   │   └── synthesis_agent/   LLM-powered synthesis
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
│   ├── llm.py                 Cloud LLM client (Groq/OpenRouter/Google)
│   └── platform.py            Composition root / dependency injection
├── frontend/                  Static web UI
├── docs/                      Architecture + execution flow docs
├── requirements.txt
├── start.sh
└── .env.example
```

---

## Extending the Platform

### Add a new tool
1. Implement the function in the appropriate `ml_models/` or `research/` service.
2. Register it in `platform.py` → `_tools()`.
3. Add its name to the `PlannerAgent.SYSTEM` prompt so the LLM can select it.

### Add a new pipeline
Add an entry to `PIPELINES` in `src/research_ai/execution/pipelines/service.py`.

### Swap the embedding model
Set `EMBEDDING_MODEL=your-model-name` in `.env` and rebuild the FAISS index.

### Add a persistent knowledge graph
Replace `memory/knowledge_graph/service.py` with a Neo4j or NetworkX-backed
implementation. The `KnowledgeGraph` interface is unchanged.
