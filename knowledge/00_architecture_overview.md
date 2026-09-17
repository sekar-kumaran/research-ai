# Architecture Overview

## Status: Active / Stable
## Description
The Research AI Platform is a unified, agent-orchestrated intelligent assistant for academic research. The system is broken into three main tiers:

1. **Frontend (Browser)**
   - Pure HTML/CSS/Vanilla JS interface.
   - Stream-based rendering using Server-Sent Events (SSE) to receive real-time updates from the backend orchestrator.
   - Maintains UI state (current conversation ID, loaded papers, UI themes).

2. **Backend Application Server (Hugging Face Spaces - Docker / FastAPI)**
   - Single canonical deployment target via Docker on Hugging Face Spaces.
   - FastAPI server acting as the orchestration layer.
   - Uses `ResearchAIPlatform` to route user queries into the Agent System (`PlannerAgent`, `ExecutorAgent`, `SynthesisAgent`).
   - Maintains in-memory session history and routes heavy ML requests to the remote Hugging Face ML microservice space.

3. **Remote ML Engine (Hugging Face ZeroGPU Space)**
   - A separate microservice running on Hugging Face Spaces (using Gradio).
   - Responsible for computationally intensive tasks: generating embeddings, querying the FAISS index, classifying intents, and extracting methodologies.
   - Connected via `gradio_client` proxy services from the backend application server.

## Data Flow (User Query)
1. User types query in UI and hits "Send" (`app.js`).
2. `app.js` posts to `POST /chat/stream` on the FastAPI backend.
3. `main.py` invokes `platform.chat()` inside a background thread pool.
4. `platform.chat()` looks up the Conversation ID and passes history to `PlannerAgent`.
5. `PlannerAgent` decides which tool to run (e.g., `hybrid_search`, `classify_query`).
6. If `hybrid_search` is called, `proxy_services.py` uses `gradio_client` to hit the Hugging Face space.
7. HF Space runs FAISS search over 8k+ papers and returns JSON.
8. `SynthesisAgent` structures the JSON and (optionally) runs an LLM to answer the question using the retrieved context.
9. Backend streams the text back via SSE, then emits `sources` and `done` events.
10. `app.js` renders the message and updates the local history state.
