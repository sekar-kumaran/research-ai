# Changelog: Remediation Pass (v3.1.2)

This document tracks the comprehensive architectural remediation pass executed to fix critical bugs, solidify the deployment architecture, and harden the application for production.

## Phase 1: Root Causes
- **`src/research_ai/platform.py`**: Fixed the `HF_SPACE_ID` default that was causing the app to circularly call itself instead of the microservice. Removed unused local ML class imports to eliminate architectural confusion. Now fails fast with loud errors if the microservice Space ID is omitted or set to the main app's Space.
- **`src/research_ai/retrieval/vector_store/faiss_store.py`**: Hardened `ready` check. Previously it blindly trusted file existence, allowing Git-LFS stubs to crash FAISS internals. Now explicitly checks file size and magic bytes (`version https://git-lfs...`).
- **`download_artifacts.py`**: Added similarity artifacts (`paper_index.faiss` and `paper_metadata.parquet`) to the download list since they frequently become LFS stubs on GitHub. Added stub-detection logic to delete stubs and properly re-download them from the dataset repo.
- **`app.py`**: Removed the legacy config check method and the false docstring claim that this app wraps the UI in a Gradio shim.
- **`src/research_ai/api/main.py`**: Moved the startup validation logic (`_check_config()`) into a FastAPI `lifespan` handler so it reliably executes across *all* entry points, including the true Docker entrypoint.

## Phase 2: Correctness and Security
- **`src/research_ai/api/main.py`**: 
  - Re-ordered middleware and added an `OPTIONS` bypass to `auth_middleware` to prevent CORS preflight requests from receiving 401s when `APP_PASSWORD` is active.
  - Hardened authorization parsing by switching to `partition(" ")` and avoiding unhandled 500s when given malformed tokens. 
  - Implemented `hmac.compare_digest` for secure, constant-time token comparison.
  - Changed `CLOUD_LLM_PROVIDER` default from Groq to Gemini in the lifespan check.
- **`render.yaml`**: Deleted entirely to designate the HF Spaces Docker path as the single canonical deployment, preventing the maintenance of two fractured deployment configs.
- **`src/research_ai/configs/settings.py`**: Updated `CLOUD_LLM_PROVIDER` default to `gemini` to align with the primary architecture docs.

## Phase 3: Hygiene and Hardening
- **`hf_microservice/app.py`**: Restored from Git index and removed unused `os` import via Pyflakes audit.
- **`src/research_ai/llm.py` & `src/research_ai/ollama_manager.py`**: Cleaned up unused imports and variables flagged by Pyflakes.
- **`hf_microservice/README.md`**: Created a dedicated README to explicitly document how the microservice must be deployed on a separate Space and wired back using `HF_SPACE_ID`.
- **`src/research_ai/api/main.py`**: Implemented a lightweight, in-memory rate limiter (Token Bucket / Sliding Window) on all endpoints limiting to 100 requests per minute per IP to prevent basic abuse.
- **`tests/test_hygiene.py`**: Created CI hygiene tests to enforce that `__pycache__` files are never tracked in Git and that artifacts downloaded into the testing environment are not unresolved LFS stubs.
- **`tests/test_integration.py`**: Authored end-to-end integration tests that hit `/health`, `/classify`, `/search`, and `/chat/ask` over HTTP to ensure the remote ML proxy correctly delegates and returns real payloads rather than silent fallbacks.

## Documentation Updates
- **`README.md`**: Stripped out `render.yaml` instructions and rewritten the deployment instructions to solidify the two-process architecture and single target (Hugging Face Spaces - Docker SDK).
- **`knowledge/00_architecture_overview.md`**: Updated to reflect the single target Docker deployment and removed references to Render.
- **`knowledge/13_backend_ml_models.md`**: Marked local classes as strictly legacy to ensure future engineers route heavy ML tasks to `proxy_services.py` instead of the local endpoints.
