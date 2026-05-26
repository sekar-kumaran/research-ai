# Changelog — Research AI Intelligence Platform

## v3.1 — Agentic Orchestration Overhaul

### Critical Bug Fixes (this session)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `PlannerAgent` SYSTEM prompt had no tool descriptions, argument schemas, or examples | LLM guessed arguments and produced malformed JSON plans | Replaced with 7,200-char prompt: full tool catalog (14 tools × name + purpose + args + when-to-use) + 5 few-shot JSON examples |
| 2 | 4 new tools (`smart_retrieve`, `metadata_analyse`, `citation_proxy`, `run_pipeline`) not listed in SYSTEM prompt | LLM could never select any of the new tools | All 14 tools now enumerated in SYSTEM prompt |
| 3 | `MLExecutionAgent.execute()` had no argument type coercion | LLM-generated `top_k: "5"` (string) crashed tool calls | `_coerce_args()` normalises strings → int/bool for known param names |
| 4 | Data-flow injection (`from: search_results`) only tracked `hybrid_search`, ignored `smart_retrieve` | `methodology_extract`, `citation_signals`, etc. received empty paper lists when `smart_retrieve` was called | `execute_plan()` now tracks any tool in `_SEARCH_TOOLS` set |
| 5 | `CloudLLMClient.__init__()` raised `ValueError` if API key was missing | Any plan/synthesis step crashed mid-request if key was absent; server startup failed with LLM configured but key missing | `api_key` moved to a `@property` — raises only on the first real API call; startup always succeeds |
| 6 | `_is_conversation()` didn't strip punctuation | `"Hello!"` → `"hello!"` failed greeting match → routed to full research pipeline | `re.sub(r"[^\w\s]", "", q)` strips punctuation before matching |

### New Services (v3.1 architecture)

| Service | File | Description |
|---------|------|-------------|
| `RetrievalAgent` | `agents/retrieval_agent/service.py` | Strategy-aware search (hybrid/filtered/citation-aware + query expansion) |
| `KnowledgeGraph` | `memory/knowledge_graph/service.py` | In-memory concept co-occurrence tracker, fed on every search |
| `CitationEngine` | `research/citation_engine/service.py` | Proxy citation graph: category + keyword overlap + temporal proximity |
| `MetadataService` | `research/metadata/service.py` | Author network, abstract quality scoring, metadata completeness |
| `PipelineRunner` | `execution/pipelines/service.py` | Named composable multi-step analysis pipelines |
| `SandboxValidator` | `execution/sandbox/service.py` | AST-level pre-execution static analysis |

### Retrieval Improvements

- `HybridSearchService`: true 3-stage pipeline — FAISS (60%) + Okapi BM25 (25%) + metadata reranking (15%). BM25 implemented from scratch, zero new dependencies.
- `EmbeddingService`: LRU query cache (512 slots, MD5-keyed), `warm_up()` method.
- `contextual_chunks()`: sentence-boundary-aware splitting (regex, no NLP dependency), overlap uses whole sentences.

### Evaluator Improvements

- `EvaluatorAgent`: 4-dimension scoring (retrieval 0.4, answer completeness 0.3, evidence grounding 0.2, error absence 0.1). Previous version only checked if search returned 0 results.

### Dead Code Removed

| Removed | Reason |
|---------|--------|
| `research_ai/` (root ghost package) | `__path__` manipulation import shim — anti-pattern replaced by correct `PYTHONPATH` in `start.sh` |
| `src/api/main.py` + `src/api/schemas.py` | 2-line re-export wrapper with no functionality |
| `src/__init__.py` | Empty, unnecessary with PYTHONPATH set |
| `pipeline_scripts/__pycache__/` | Orphaned bytecode, source files gone |

### API New Endpoints (v3.1)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/metadata/analyse` | Author/year/quality analysis over a paper list |
| POST | `/citation/proxy` | Proxy citation graph from metadata |
| POST | `/citation/clusters` | Co-citation topic clustering |
| POST | `/citation/timeline` | Chronological influence ordering |
| GET  | `/knowledge-graph` | Session concept summary |
| GET  | `/knowledge-graph/concepts` | Top N tracked concepts |
| POST | `/pipeline/run` | Run a named multi-step pipeline |
| GET  | `/pipeline/list` | List available pipelines |
