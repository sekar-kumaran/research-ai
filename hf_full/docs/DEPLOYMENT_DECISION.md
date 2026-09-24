# Deployment Decision Matrix

| Gate | Status | Evidence / Notes |
| :--- | :--- | :--- |
| **BUILD** | PASS | `hf_full` initializes locally without syntax errors or missing dependencies. |
| **RUNTIME** | PASS | `app.py` successfully intercepts FastAPI and routes it through Gradio mounts. |
| **API** | PASS | REST routes (health, search, classify) remain intact without UI-coupling. |
| **ML** | PASS | Classifier and FAISS load cleanly on CPU. SentenceTransformer correctly delegates to `@spaces.GPU` if available. |
| **RAG** | PASS | `/ask` orchestrates the tool chain (planner -> local retrieval -> LLM synthesis) successfully. |
| **LLM** | PASS | `LLM_BACKEND=cloud` ensures no local Ollama dependencies break the memory limits. |
| **MEMORY** | PASS WITH KNOWN LIMITATIONS | Space fits well within 16GB RAM limits of basic ZeroGPU spaces. Chat memory is not persistent across cold restarts, which is a known and accepted limitation for this phase. |
| **SECURITY** | PASS | Python execution disabled. File uploads handled strictly. CORS restricted. |
| **PARITY** | PASS | End-to-end classification and RAG functionality aligns with original codebase expectations (accounting for minor 1.6 vs 1.9 scikit-learn variation). |

## FINAL STATUS
**PASS WITH KNOWN LIMITATIONS**

The single-space architecture successfully consolidates the backend API and the ML runtime into a single deployable artifact without using microservices, satisfying all core requirements. The primary limitation to address in future phases is plugging in an external Postgres DB to persist `/chat/message` history beyond container lifespans.
