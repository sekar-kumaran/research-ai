# Backend: Remote Services & Hugging Face

## File Paths: `src/research_ai/remote_ml/proxy_services.py`, `hf_microservice/app.py`
## Status: Active / Stable

## Description
Because vector indexing (FAISS) and ML clustering require significant RAM and GPU compute, they are decoupled from the FastAPI orchestration server. They run on a Hugging Face Space (ZeroGPU). The FastAPI server communicates with this space via the `gradio_client` library.

## 1. The Remote Client Singleton (`proxy_services.py`)
- **`RemoteMLClient`**: A singleton class that maintains a persistent websocket/HTTP connection to the HF Space URL (`sekarkumaran461-research-ai.hf.space`).
- Uses lazy initialization: the connection is only opened on the first request.
- Contains a `reset()` method to destroy and recreate the client if a network timeout occurs, preventing the server from hanging indefinitely.

## 2. Proxy Services (`proxy_services.py`)
These classes wrap the ugly `gradio_client.predict()` calls into clean Python dictionaries expected by the Agents.
- **`RemoteHybridSearchService`**: Calls the `/hybrid_search_api` endpoint. Coerces raw string outputs into `{"results": [...], "count": N}` dictionaries. Contains `_safe_json()` fallback logic.
- **`RemoteClassifierService`**: Calls the `/classify_api` endpoint. Handles errors by returning safe fallbacks (e.g., `{"predicted_category": "cs.LG", "error": ...}`).
- **`RemoteMethodologyExtractor`** / **`RemoteClusteringService`**: Provide identical proxy wrappers for specific ML tasks.

## 3. Hugging Face Microservice (`hf_microservice/app.py`)
- Runs a standalone Gradio application.
- Loads a 8,000-paper FAISS index (`faiss_index.bin`) into RAM on startup.
- Exposes `gr.Interface` endpoints (e.g., `fn=api_hybrid_search`) configured with `api_name="hybrid_search_api"`.
- This code is deployed separately to Hugging Face and is NOT executed on the Render server.
