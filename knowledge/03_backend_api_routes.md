# Backend: API Routes

## File Path: `src/research_ai/api/main.py`
## Status: Active / Stable

## Description
This is the entry point for the FastAPI backend. It mounts the frontend static files and exposes both low-level ML endpoints and the high-level orchestration endpoint (`/chat/stream`).

## Key Endpoints

### 1. `/health` (GET)
- Returns `{"status": "ok", "remote_connected": True/False}`
- Checked by the frontend to update the "Connecting..." status dot in the bottom left of the sidebar.
- Verifies connection to the Hugging Face ZeroGPU space.

### 2. `/chat/stream` (POST)
- **The Core Endpoint**. Receives a user's raw text query.
- Spawns `platform.chat()` in a background thread to prevent blocking the async event loop.
- Uses `StreamingResponse` to push Server-Sent Events (SSE) back to `app.js`.
- Yields `: keepalive` comments every 3 seconds to prevent connection drops while the orchestrator runs synchronously.
- Streams the final `{"delta": "..."}` chunks, followed by `{"event": "sources"}` and `{"event": "done"}`.

### 3. `/conversations/{conversation_id}` (GET)
- Retrieves a specific conversation history from the in-memory `ConversationStore`.
- If the server has restarted, it returns `404 Not Found`, prompting the frontend to start a new chat.

### 4. `/chat/upload` and `/chat/load-arxiv` (POST)
- Accepts PDF/TXT files or an arXiv ID.
- Offloads parsing and chunking to the `IngestionAgent` (or similar document loader) via the `ResearchAIPlatform`.
- Returns a `session_id` to scope subsequent queries to those uploaded documents.

### 5. Remote Model Proxies (`/search`, `/classify`, `/summarize`)
- Direct REST wrappers around the remote HF Space capabilities.
- Useful for testing the ML models without running the full planner pipeline.
