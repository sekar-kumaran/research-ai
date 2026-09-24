# Runtime Audit

## 1. Application Entrypoint
- The entrypoint is defined in `api/main.py`.
- It initializes `FastAPI()` and relies on the asynchronous lifespan context manager `@asynccontextmanager async def lifespan(app: FastAPI)` to execute startup code.

## 2. Dependency Injection & Service Initialization
- `main.py` creates a module-level global `platform = ResearchAIPlatform(settings)`.
- `platform.py` initializes local ML services (EmbeddingService, FaissVectorStore, ClassifierService) eagerly via `from_artifacts()`.
- The `ResearchAIPlatform` builds a registry of agents and services, passing the unified `_cloud_factory` to agents that require LLM access.

## 3. Prewarm Behavior
- `main.py` explicitly launches a background thread to call `_prewarm()`.
- `_prewarm()` triggers PyTorch and FAISS to load their models into RAM by performing a dummy `"warmup query"` inference run on `EmbeddingService` and `FaissVectorStore`.
- It also performs a warmup classification via `ClassifierService`.

## 4. RAG Execution & Orchestrator
- The orchestrator (`ResearchOrchestrator`) is the entrypoint for RAG.
- It leverages a `PlannerAgent`, `MLExecutionAgent` (executor), `EvaluatorAgent`, and `SynthesisAgent` (synthesizer).
- Memory is maintained across turns using `ConversationStore` (in-memory dict).

## 5. Storage & Persistence
- ML Artifacts (`classifier.joblib`, `paper_index.faiss`, etc.) are read-only at runtime.
- Conversation state (`ConversationStore`) is stored entirely in memory. It does not currently use an external DB.
- File uploads temporarily write to the local filesystem (via standard FastAPI UploadFile mechanisms or explicitly to a temp directory).

## 6. Execution Sandbox
- Python execution (`python_runner`) is enabled based on `ENABLE_PYTHON_EXECUTION`. It needs to be explicitly disabled for this public deployment.
