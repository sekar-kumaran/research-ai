# API Exposure Audit

This document details the actual routes exposed by the FastAPI server located in `src/research_ai/api/main.py`.

## Core Health & System
- `GET /` (Hidden): Redirects or standard check.
- `GET /health`: Basic liveness check.
- `GET /ready`: Deeper readiness check (checks models/dependencies).
- `GET /stats`: Application statistics.
- `GET /system/artifacts`: Lists loaded artifacts.

## ML & Direct Inference Endpoints
These directly hit the underlying ML components (mostly bypassing LLM orchestration unless explicitly requested).
- `POST /classify`: Calls `ClassifierService.classify()`.
- `POST /search`: Calls `HybridSearchService.search()` (Local FAISS).
- `POST /similarity`: Calls `SimilarityService`.
- `POST /summarize`: Generic string summarization.
- `POST /summarize-paper`: Specific paper summarization.

## Research Intelligence
- `POST /metadata/analyse`: Analyzes query metadata using LLM.
- `POST /citation/proxy`: Looks up citations.
- `POST /citation/clusters`: Clusters papers.
- `POST /citation/timeline`: Citation timeline.
- `GET /knowledge-graph`: Returns current knowledge graph state.
- `GET /knowledge-graph/concepts`: Returns concepts.

## RAG Orchestration
- `POST /ask`: The primary RAG route. Uses `ResearchOrchestrator.run()`.
- `POST /agent/run`: Extended orchestration access.
- `POST /agent/run/stream`: Streaming orchestrator response.

## Conversational / Chat Routes
- `POST /chat/message`: Main multi-turn conversational chat. Uses `ConversationStore`.
- `POST /chat/stream`: Streaming multi-turn.
- `GET /conversations`: List all conversation history.
- `GET /conversations/{conversation_id}`: Get specific conversation history.
- `PATCH /conversations/{conversation_id}`: Update conversation metadata.
- `POST /conversations/{conversation_id}/clear`: Clear history for ID.
- `DELETE /conversations/{conversation_id}`: Delete conversation.

## Document / Paper Chat
- `POST /chat/upload`: Uploads a PDF (temporarily stores it, processes it, creates a `session_id`).
- `POST /chat/load-arxiv`: Loads paper from ArXiv to session.
- `POST /chat/ask`: Ask a question directly bounded by the loaded document session context.
- `POST /chat/multi-ask`: Ask multiple questions.
- `POST /chat/bulk-load`: Load multiple papers.
- `GET /chat/session/{session_id}`: Retrieve document session state.

## Execution & Auth Endpoints
- `POST /execution/python`: Sandboxed python execution (Disabled by default).
- `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me`: Auth routes (Currently placeholder/mock logic in v3.1).

## Dependencies
- **ML Dependencies:** `/classify`, `/search`, `/similarity`, and all Document/RAG queries implicitly rely on `ClassifierService`, `FaissVectorStore`, and `EmbeddingService`.
- **LLM Dependencies:** All Orchestrator, RAG, and Chat generation routes rely on the `cloud_factory` (Groq/OpenRouter).
- **Storage Dependencies:** 
  - `ConversationStore` keeps chat history in process RAM.
  - `/chat/upload` writes to local ephemeral `/tmp` filesystem for PDF processing before deletion.
