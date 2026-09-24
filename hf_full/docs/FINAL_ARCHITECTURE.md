# Final Architecture

## Visual Architecture Diagram

```mermaid
graph TD
    User((User / Browser)) -->|HTTPS| Vercel[React Frontend on Vercel]
    Vercel -->|HTTPS REST API| HFSpace[sekarkumaran461/ResearchAi-Full]
    
    subgraph HFSpace [Hugging Face Space (ZeroGPU)]
        Gradio[Gradio Server / FastAPI]
        API[Research AI Routes]
        
        Gradio --> API
        
        API -->|/ask, /chat| Orchestrator[RAG Orchestrator]
        API -->|/search, /similarity| Retriever[Local Retrieval Service]
        API -->|/classify| Classifier[Local Classifier]
        
        Orchestrator --> Agents[Agents & Memory]
        Agents --> Retriever
        
        subgraph Local ML Runtime
            Classifier -->|CPU| SGD[SGDClassifier]
            Retriever -->|CPU| FAISS[FAISS Index]
            Retriever -->|ZeroGPU| Embed[SentenceTransformer]
        end
    end
    
    Orchestrator -->|Cloud API Call| LLM((Cloud LLM Provider\nGroq/OpenRouter))
```

## 1. Components
- **Vercel / React**: The future frontend, decoupled entirely from the Space.
- **Hugging Face Space**: A single Gradio SDK Space acting as the unified backend container.
- **FastAPI**: Provides the REST API contract matching the original repository.
- **Local ML**: The Classifier and FAISS index are loaded into CPU memory. The SentenceTransformer embedding model uses `@spaces.GPU` for ZeroGPU execution.
- **Cloud LLM**: RAG reasoning, planning, and synthesis are strictly offloaded to external providers (Groq/Google) via API calls.

## 2. Failure Boundaries
- **Cloud LLM Failure**: If the LLM provider fails (429/5xx), the Orchestrator will catch the error and return a structured JSON response instead of crashing the Space.
- **ZeroGPU Unavailability**: If the Space is booted on CPU only, the `ZeroGPUEmbeddingService` gracefully detects the absence of `spaces` or CUDA and falls back to CPU encoding automatically.
- **Ephemeral Storage**: Uploaded files (`/chat/upload`) write to `/tmp` and must be deleted after parsing.

## 3. Authentication & Security
- CORS is configurable via `CORS_ORIGINS`.
- Sandboxed Python execution is disabled by default (`ENABLE_PYTHON_EXECUTION=false`).
- All API keys (Groq, OpenRouter) are handled securely via HF Space Secrets and are never committed to the repository.
