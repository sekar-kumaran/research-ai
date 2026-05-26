# LLM and ML Execution Flow

This document describes the current Research AI runtime after the architecture refactor.

## Startup

1. `uvicorn src.api.main:app` imports the compatibility entrypoint.
2. `src.api.main` delegates to `research_ai.api.main`.
3. `ResearchAIPlatform` wires services:
   - planner, orchestrator, evaluator, executor, synthesizer
   - local classifier service
   - lazy embedding service
   - lazy FAISS vector store
   - hybrid retrieval and reranking
   - summarizer
   - paper ingestion/chat
   - similarity, ranking, citation signals, trend analysis
   - disabled-by-default Python runner
4. Heavy artifacts load lazily on first feature use.

## Agentic Request Flow

1. The user sends a request to `/ask`, `/agent/run`, or `/agent/run/stream`.
2. `PlannerAgent` creates a dynamic tool plan.
3. `ResearchOrchestrator` executes each tool through `MLExecutionAgent`.
4. Local tools produce evidence:
   - classification
   - hybrid retrieval
   - methodology signals
   - citation/category/year signals
   - trend analysis
   - metadata RAG
   - paper chat
   - optional Python execution
5. `EvaluatorAgent` checks whether evidence is sufficient and can trigger a retrieval retry.
6. `SynthesisAgent` produces the final answer from tool outputs.

## Local-First ML

Always local:

- trained classifier artifacts
- FAISS vector index
- paper metadata
- SentenceTransformer embeddings
- similarity computation
- methodology signal extraction
- ranking and trend/citation-signal computation

Conditional:

- cloud LLM planning and synthesis when `LLM_BACKEND=cloud` and provider credentials exist
- local transformer summarization/generation when `LLM_BACKEND=local`

## Retrieval Flow

1. Query is encoded locally.
2. FAISS returns semantic candidates.
3. Metadata filters are applied when provided.
4. Keyword reranker adds `keyword_score` and `hybrid_score`.
5. Ranking service adds category/recency adjustments.

## Execution Flow

`/execution/python` is disabled by default. When enabled, code runs in a subprocess with isolated mode, restricted builtins, forbidden operation checks, and timeout.

