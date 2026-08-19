# Backend: API Schemas

## File Path: `src/research_ai/api/schemas.py`
## Status: Active / Stable

## Description
This file defines the Pydantic models (data schemas) for the FastAPI backend. It enforces strict type checking and validation on all incoming JSON payloads, preventing malformed data from reaching the internal components.

## File Breakdown

### 1. `src/research_ai/api/schemas.py`
- **Core Request Models**:
  - `ChatMessageRequest`: Sent by `app.js`. Contains `query` (string), `conversation_id` (string/null), `session_id` (string/null), `top_k` (int), and `debug` (bool). Used by the `POST /chat/stream` endpoint.
  - `BulkChatRequest`: Used for batch processing multiple messages.
- **Service Request Models**:
  - `SearchRequest`: Used by the raw `/search` endpoint to hit the HF space directly.
  - `ClassifyRequest`: Used by the raw `/classify` endpoint.
  - `SummarizeRequest`: Used by the raw `/summarize` endpoint.
- **Agent Models**:
  - `AgentRequest`: Base class for orchestrator requests.
  - `PipelineRequest`: Used for running predefined, rigid pipelines via the Executor agent.
- **Research Models**:
  - `CitationProxyRequest`: Used to fetch citation data from Semantic Scholar.
  - `MetadataAnalyseRequest`: Used to process and extract insights from raw metadata.
  - `PaperChatRequest`: Used specifically for RAG queries against an uploaded PDF/TXT document.
