
import os
import time

print(f'\n[DIAGNOSTIC] IMPORTING app.py (PID: {os.getpid()}, PPID: {os.getppid()})')
import os
import sys
import logging
from pathlib import Path

# Prepend src to sys.path so internal imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import gradio as gr
from gradio import Server
import spaces
from fastapi import Request

# Import the existing backend platform and original FastAPI app
from research_ai.api.main import app as fastapi_app, platform, _primary_text
from research_ai.api.schemas import (
    SearchRequest, SimilarityRequest, AskRequest, ChatMessageRequest
)
from deployment.zero_gpu import ZeroGPUEmbeddingService

logger = logging.getLogger(__name__)

# --- 1. SETUP ZEROGPU SERVICE ---
print(f"ResearchAI startup PID={os.getpid()}, PPID={os.getppid()}")
gpu_embedding = ZeroGPUEmbeddingService(
    model_name=platform.settings.retrieval.embedding_model_name,
    device=os.environ.get("EMBEDDING_DEVICE", "auto")
)

# Patch the platform components to use the GPU-enabled embedding service
platform.embedding_service = gpu_embedding
platform.retriever.embedding_service = gpu_embedding
platform.similarity.embedding_service = gpu_embedding
platform.paper_chat.embedding_service = gpu_embedding

# --- 2. CREATE GRADIO SERVER ---
# This REPLACES gr.mount_gradio_app(fastapi_app, demo)
app = Server(
    title=fastapi_app.title,
    version=fastapi_app.version,
    description=fastapi_app.description,
    lifespan=fastapi_app.router.lifespan_context, 
)

# Preserve middlewares from original FastAPI app
app.user_middleware.extend(fastapi_app.user_middleware)
# Do NOT build the middleware stack early, because app.launch() will add more middleware and build it.


# --- 3. PRESERVE NORMAL FASTAPI ROUTES ---
# We copy all existing routes from the original app EXCEPT the ones we are 
# converting to Gradio @app.api() queued endpoints.
_gpu_endpoints = {"/search", "/similarity", "/ask", "/chat/message"}
for route in fastapi_app.router.routes:
    path = getattr(route, "path", None)
    if path not in _gpu_endpoints and path not in ["/openapi.json", "/docs", "/redoc"]:
        app.router.routes.append(route)


# --- 4. MOVE MODEL-COMPUTE ENDPOINTS TO @app.api() ---
# This ensures that endpoints touching ZeroGPU enter the Gradio execution lifecycle.

@app.api(name="similarity")
def similarity_endpoint(req: dict) -> dict:
    request = SimilarityRequest(**req)
    result = platform.similarity.compare(request.text_a, request.text_b)
    if result.get("error"):
        return {"error": result["error"]}
    return result

@app.api(name="search")
def search_endpoint(req: dict) -> dict:
    request = SearchRequest(**req)
    result = platform._hybrid_search(request.query, top_k=request.top_k, filters=request.filters)
    if result.get("error"):
        return {"error": result["error"]}
    return result

@app.api(name="ask")
def ask_endpoint(req: dict) -> dict:
    request = AskRequest(**req)
    return platform.orchestrator.run(mode="auto", query=request.query, top_k=request.top_k)

@app.api(name="chat_message")
def chat_message_endpoint(req: dict) -> dict:
    # Since Gradio API doesn't provide the raw Starlette Request object natively
    # via dependency injection the same way FastAPI does, we just pass None for user_id
    # or handle it differently if needed. For now, backend user auth relies on the 
    # original chat_message route. The user instructions say "DO NOT add PostgreSQL yet.
    # Preserve existing in-memory ConversationStore."
    request = ChatMessageRequest(**req)
    result = platform.chat(
        query=request.query,
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        top_k=request.top_k,
        debug=request.debug,
        user_id=None,
    )
    from research_ai.api.schemas import SourcePaper, ChatMessageResponse
    sources = [
        SourcePaper(
            title=s.get("title", ""),
            paper_id=s.get("paper_id", ""),
            year=s.get("year", ""),
            category=s.get("category", ""),
            abstract_snippet=s.get("abstract_snippet", ""),
            score=float(s.get("score", 0.0)),
            arxiv_url=s.get("arxiv_url", ""),
        )
        for s in result.get("sources", [])
    ]
    resp = ChatMessageResponse(
        answer=result.get("answer", ""),
        conversation_id=result.get("conversation_id", ""),
        session_id=result.get("session_id", ""),
        sources=sources,
    )
    return resp.model_dump()

if __name__ == "__main__":
    print(f"[{os.getpid()}] Executing app.launch()")
    app.launch(server_name="0.0.0.0", server_port=7860)
