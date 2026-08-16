"""FastAPI application — Research AI Intelligence Platform v3.1.

API STRUCTURE
-------------
/health, /stats          — Operational health and index statistics
/classify, /search,      — Direct ML model endpoints (no orchestrator)
/summarize, /similarity
/metadata/*, /citation/* — Research intelligence (metadata, citation graphs)
/knowledge-graph/*       — Session-scoped concept tracking
/pipeline/*              — Pre-built multi-step analysis pipelines
/ask, /agent/run,        — Orchestrated agentic endpoints (Plan→Execute→Evaluate→Synthesize)
/agent/run/stream
/chat/*                  — Full-paper ingestion and per-session chat
/execution/python        — Sandboxed Python code execution (disabled by default)

PRODUCTION HARDENING NOTES
--------------------------
CORS:
  allow_origins=["*"] is safe for a local development server but MUST be
  restricted in production to your specific frontend domain(s).
  Set ALLOWED_ORIGINS env var to a comma-separated list, e.g.:
    ALLOWED_ORIGINS=https://yourapp.com,https://api.yourapp.com

PDF UPLOAD SIZE LIMIT:
  The /chat/upload endpoint now enforces a MAX_UPLOAD_BYTES limit (default 50 MB).
  Without this limit, a 500 MB PDF upload would be read entirely into memory,
  potentially exhausting the server's RAM.  Set via MAX_UPLOAD_MB env var.

RATE LIMITING:
  No rate limiting is implemented here.  For production, add slowapi or a
  reverse-proxy-level limiter (nginx, Cloudflare) in front of this service.

AUTHENTICATION:
  No authentication is implemented.  For production, add OAuth2/API-key
  middleware or use a gateway (Kong, AWS API Gateway) in front of this service.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

# Load .env from project root before any settings are read
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        # dotenv not installed — parse manually
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from research_ai.api.schemas import (
    AgentRequest,
    ArxivLoadRequest,
    AskRequest,
    BulkChatRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    CitationProxyRequest,
    ClassifyRequest,
    MediatedAgentResponse,
    MetadataAnalyseRequest,
    ModelsListResponse,
    PaperChatRequest,
    PipelineRequest,
    PythonExecutionRequest,
    SearchRequest,
    SimilarityRequest,
    SummarizeRequest,
)
from research_ai.common.text import redact_secrets
from research_ai.configs.settings import load_settings
from research_ai.platform import ResearchAIPlatform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = load_settings()
platform = ResearchAIPlatform(settings)

# ---------------------------------------------------------------------------
# Upload size limit
# Protects against memory exhaustion from very large PDF uploads.
# Default: 50 MB.  Override with MAX_UPLOAD_MB environment variable.
# ---------------------------------------------------------------------------
_MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# CORS allowed origins
# In development, allow all origins for convenience.
# In production, set ALLOWED_ORIGINS to restrict access.
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
_allowed_origins: list[str] = (
    ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)
if "*" in _allowed_origins:
    logger.warning(
        "CORS is open to ALL origins (ALLOWED_ORIGINS=*). "
        "Set ALLOWED_ORIGINS=https://yourapp.com in production."
    )

app = FastAPI(
    title="Research AI Intelligence Platform",
    version="3.1.0",
    description=(
        "Agentic scientific research intelligence platform with local ML, "
        "BM25+FAISS hybrid retrieval, citation intelligence, and sandboxed execution."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.paths.frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.paths.frontend_dir), name="static")


# ---------------------------------------------------------------------------
# Request logging middleware
# Logs method, path, status code, and latency for every request.
# Production: replace with a proper structured logging library (structlog).
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "[%s] %s %s → %d  (%.1fms)",
        request_id, request.method, request.url.path,
        response.status_code, latency_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_text(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("final_answer", "answer", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        executor = payload.get("executor_output")
        if isinstance(executor, dict):
            for value in executor.values():
                if isinstance(value, dict):
                    text = value.get("answer") or value.get("summary") or value.get("final_answer")
                    if isinstance(text, str) and text.strip():
                        return text
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    index_path = settings.paths.frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. Visit /docs for API reference."}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = settings.paths.frontend_dir / "favicon.ico"
    return FileResponse(path) if path.exists() else Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.1.0",
        "architecture": "research_ai_agentic_platform",
        "components": {
            "classifier": platform.classifier.ready,
            "hybrid_retrieval": platform.retriever.ready,
            "summarizer": platform.summarizer.ready,
            "paper_chat": True,
            "python_execution": settings.execution.enabled,
            "knowledge_graph": True,
            "citation_engine": True,
            "pipeline_runner": True,
        },
        "llm_backend": settings.llm.backend,
        "llm_provider": settings.llm.provider,
    }


@app.get("/stats")
def stats():
    return {
        "indexed_papers": platform.indexed_paper_count,
        "active_chat_sessions": len(platform.paper_chat.sessions),
        "active_conversations": platform.conversation_store.count,
        "classifier_ready": platform.classifier.ready,
        "retrieval_ready": platform.retriever.ready,
        "embedding_model": platform.embedding_service.model_name,
        "knowledge_graph": platform.knowledge_graph.summary(),
        "available_pipelines": platform.pipeline_runner.available_pipelines(),
        "ollama": platform.ollama_manager.health_check(),
    }


# ---------------------------------------------------------------------------
# ML model routes
# ---------------------------------------------------------------------------

@app.post("/classify")
def classify(req: ClassifyRequest):
    title = req.title or req.abstract
    abstract = req.abstract or req.title
    if not (title or "").strip():
        raise HTTPException(status_code=422, detail="Provide at least a title or abstract.")
    result = platform.classifier.classify(title, abstract)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.post("/search")
def search(req: SearchRequest):
    result = platform.retriever.search(req.query, top_k=req.top_k, filters=req.filters)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    try:
        summary = platform.summarizer.summarize(req.text)
        return {"summary": summary, "word_count": len(summary.split())}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Summarization failed: {redact_secrets(str(exc))}")


@app.post("/summarize-paper")
def summarize_paper(req: ArxivLoadRequest):
    try:
        clean_id = req.arxiv_id.strip().lower().replace("arxiv:", "")
        if platform.retriever.ready:
            docs = platform.retriever.search(clean_id, top_k=10).get("results", [])
            for doc in docs:
                pid = str(doc.get("paper_id", "")).lower()
                if pid == clean_id or pid.endswith(clean_id):
                    text = f"Title: {doc.get('title', '')}\n\nAbstract: {doc.get('abstract', '')}"
                    return {
                        "arxiv_id": req.arxiv_id,
                        "title": doc.get("title", ""),
                        "summary": platform.summarizer.summarize(text),
                    }
        meta = platform.paper_chat.create_or_get_session_from_arxiv_id(req.arxiv_id)
        session = platform.paper_chat.sessions.get(meta["session_id"])
        if session and session.chunks:
            return {
                "arxiv_id": req.arxiv_id,
                "session_id": meta["session_id"],
                "summary": platform.summarizer.summarize(" ".join(session.chunks[:3])),
            }
        raise HTTPException(status_code=404, detail="Could not retrieve paper content.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Summarize paper failed: {redact_secrets(str(exc))}")


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    result = platform.similarity.compare(req.text_a, req.text_b)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Research intelligence routes
# ---------------------------------------------------------------------------

@app.post("/metadata/analyse")
def metadata_analyse(req: MetadataAnalyseRequest):
    """Analyse author, category, year, and abstract quality of a paper list."""
    return platform.metadata_service.analyse(req.papers)


@app.post("/citation/proxy")
def citation_proxy(req: CitationProxyRequest):
    """Derive proxy citation relations from paper metadata."""
    return platform.citation_engine.proxy_citations(req.papers)


@app.post("/citation/clusters")
def citation_clusters(req: CitationProxyRequest):
    """Group papers into co-citation topic clusters."""
    return platform.citation_engine.co_citation_clusters(req.papers)


@app.post("/citation/timeline")
def citation_timeline(req: CitationProxyRequest):
    """Return papers ordered by year as an influence timeline."""
    return platform.citation_engine.influence_timeline(req.papers)


@app.get("/knowledge-graph")
def knowledge_graph_summary():
    """Return current knowledge graph concept summary."""
    return platform.knowledge_graph.summary()


@app.get("/knowledge-graph/concepts")
def top_concepts(n: int = 20):
    """Return the top N concepts tracked across sessions."""
    return {"concepts": platform.knowledge_graph.top_concepts(n)}


# ---------------------------------------------------------------------------
# Pipeline routes
# ---------------------------------------------------------------------------

@app.post("/pipeline/run")
def run_pipeline(req: PipelineRequest):
    """Execute a named research analysis pipeline."""
    result = platform.pipeline_runner.run(req.pipeline_name, req.query)
    if result.errors and not result.steps_ok:
        raise HTTPException(status_code=503, detail=result.errors[0])
    return result.to_dict()


@app.get("/pipeline/list")
def list_pipelines():
    """List all available named research analysis pipelines."""
    return {"pipelines": platform.pipeline_runner.available_pipelines()}


# ---------------------------------------------------------------------------
# Orchestrator / agent routes
# ---------------------------------------------------------------------------

@app.post("/ask")
def ask(req: AskRequest):
    return platform.orchestrator.run(mode="auto", query=req.query, top_k=req.top_k)


@app.post("/agent/run", response_model=MediatedAgentResponse)
def run_agent(req: AgentRequest):
    return platform.orchestrator.run(
        mode=req.mode,
        query=req.query,
        title=req.title,
        abstract=req.abstract,
        top_k=req.top_k,
        text=req.text,
        session_id=req.session_id,
    )


@app.post("/agent/run/stream")
async def run_agent_stream(req: AgentRequest):
    async def event_generator():
        # Run the blocking orchestration in a thread so the event loop stays responsive.
        task = asyncio.create_task(run_in_threadpool(run_agent, req))
        while not task.done():
            yield ": keepalive\n\n"
            await asyncio.sleep(1.0)
        try:
            out = await task
        except Exception as exc:
            err = json.dumps({"event": "error", "message": redact_secrets(str(exc))})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
            return

        text = _primary_text(out)
        request_id = out.get("request_id", "")
        mode = out.get("mode", req.mode)

        yield f"data: {json.dumps({'event': 'start', 'request_id': request_id, 'mode': mode})}\n\n"
        step = max(1, len(text) // 100)
        for i in range(0, len(text), step):
            yield f"data: {json.dumps({'delta': text[i:i + step]}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.008)
        yield (
            f"data: {json.dumps({'event': 'end', 'request_id': request_id, 'mode': mode, 'latency_ms': out.get('latency_ms')})}\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Unified AI chat endpoint — the main user-facing interface
#
# This is the "ChatGPT-like" endpoint. The user sends a natural-language
# query; the AI orchestrator automatically decides:
#   - Which tools to invoke (retrieval, classification, summarization, etc.)
#   - Which model to use (fast for simple tasks, stronger for complex ones)
#   - How to retrieve, rerank, and synthesize evidence
#   - How to cite sources and express confidence
#
# The user NEVER needs to know about /search, /classify, /summarize, etc.
# Those endpoints remain for direct access but the AI handles them internally.
# ---------------------------------------------------------------------------

@app.post("/chat/message", response_model=ChatMessageResponse)
def chat_message(req: ChatMessageRequest):
    """Unified conversational AI endpoint — the primary user interface.

    Orchestrates the full pipeline automatically:
      intent detection → retrieval → reranking → synthesis → validation → citation

    Returns a structured response with answer text, source papers, confidence
    score, and conversation ID for multi-turn continuity.
    """
    try:
        result = platform.chat(
            query=req.query,
            conversation_id=req.conversation_id,
            session_id=req.session_id,
            top_k=req.top_k,
            debug=req.debug,
        )
        # Build structured source list for the response model
        from research_ai.api.schemas import SourcePaper
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
        return ChatMessageResponse(
            answer=result["answer"],
            sources=sources,
            confidence=float(result.get("confidence", 0.0)),
            conversation_id=result["conversation_id"],
            intent=result.get("intent", "research_analysis"),
            tools_used=result.get("tools_used", []),
            model_used=result.get("model_used", ""),
            latency_ms=float(result.get("latency_ms", 0.0)),
            debug_trace=result.get("debug_trace") if req.debug else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=redact_secrets(str(exc)))


@app.post("/chat/stream")
async def chat_stream(req: ChatMessageRequest):
    """Streaming version of /chat/message using Server-Sent Events.

    Delivers the answer token-by-token for a responsive streaming UI.
    Also emits structured metadata (sources, confidence) as a final event.

    SSE event format:
      data: {"delta": "<text chunk>"}              — incremental text
      data: {"event": "sources", "sources": [...]} — final sources list
      data: {"event": "done", "confidence": 0.85, "conversation_id": "..."}
      data: [DONE]
    """
    async def event_generator():
        # Run the blocking pipeline in a thread; emit keepalives while waiting.
        task = asyncio.create_task(
            run_in_threadpool(
                platform.chat,
                query=req.query,
                conversation_id=req.conversation_id,
                session_id=req.session_id,
                top_k=req.top_k,
                debug=False,
            )
        )
        while not task.done():
            yield ": keepalive\n\n"
            await asyncio.sleep(1.0)
        try:
            result = await task
        except Exception as exc:
            err = json.dumps({"event": "error", "message": redact_secrets(str(exc))})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
            return

        text = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        conversation_id = result.get("conversation_id", "")
        intent = result.get("intent", "research_analysis")
        latency_ms = result.get("latency_ms", 0.0)

        # Start event
        yield f"data: {json.dumps({'event': 'start', 'intent': intent, 'conversation_id': conversation_id})}\n\n"

        # Stream answer in small chunks to simulate real-time generation
        # Real streaming requires Ollama's streaming API — this simulates it
        # for compatibility with both cloud and local providers.
        chunk_size = max(1, len(text) // 80)
        for i in range(0, len(text), chunk_size):
            yield f"data: {json.dumps({'delta': text[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
        yield f"data: {json.dumps({'event': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'done', 'confidence': confidence, 'conversation_id': conversation_id, 'latency_ms': latency_ms})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Ollama model management
# ---------------------------------------------------------------------------

@app.get("/models/list", response_model=ModelsListResponse)
def list_models():
    """List locally available Ollama models with their speed tier.

    Tier 1 = fastest (<4B params), Tier 2 = balanced, Tier 3 = most capable.
    Returns empty list if Ollama is not running.
    """
    mgr = platform.ollama_manager
    # Refresh model list on each call so newly pulled models appear immediately
    mgr.discover()
    from research_ai.api.schemas import ModelInfo as ModelInfoSchema
    return ModelsListResponse(
        available=mgr.available,
        models=[ModelInfoSchema(**m) for m in mgr.models_list()],
        default_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Return the turn history for a conversation (for history panel rendering)."""
    conv = platform.conversation_store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation_id": conversation_id,
        "turn_count": conv.turn_count,
        "created_at": conv.created_at,
        "last_active": conv.last_active,
        "turns": [{"role": t.role, "content": t.content} for t in conv.turns],
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation from memory."""
    deleted = platform.conversation_store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": conversation_id}


# ---------------------------------------------------------------------------
# Paper chat routes
# ---------------------------------------------------------------------------

@app.post("/chat/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Ingest a paper from an uploaded PDF or text file and create a chat session.

    Size limit: MAX_UPLOAD_BYTES (default 50 MB, configurable via MAX_UPLOAD_MB).
    Without a size limit, a 500 MB PDF would be read entirely into memory,
    potentially exhausting RAM on a CPU-only server.

    Returns: {"session_id": "...", "chunk_count": N, "source": "upload:filename"}
    """
    try:
        # Read with explicit size guard.
        # UploadFile.read() has no built-in limit — we impose one here.
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is {_MAX_UPLOAD_MB} MB.",
            )
        filename = file.filename or "uploaded_file"
        if filename.lower().endswith(".pdf"):
            return platform.paper_chat.create_session_from_pdf_bytes(content, source=f"upload:{filename}")
        return platform.paper_chat.create_session_from_text(
            text=content.decode("utf-8", errors="ignore"),
            source=f"upload:{filename}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {redact_secrets(str(exc))}")


@app.post("/chat/load-arxiv")
def load_arxiv(req: ArxivLoadRequest):
    try:
        return platform.paper_chat.create_or_get_session_from_arxiv_id(req.arxiv_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"arXiv load failed: {redact_secrets(str(exc))}")


@app.post("/chat/ask")
def chat_ask(req: PaperChatRequest):
    try:
        return platform.paper_chat.ask(req.session_id, req.question, top_k=req.top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Chat failed: {redact_secrets(str(exc))}")


@app.post("/chat/multi-ask")
def chat_multi_ask(req: PaperChatRequest):
    try:
        session_ids = [item.strip() for item in req.session_id.split(",") if item.strip()]
        if not session_ids:
            raise HTTPException(status_code=422, detail="No valid session IDs provided.")
        return platform.paper_chat.ask_multi(session_ids, req.question, top_k_per_session=req.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Multi-chat failed: {redact_secrets(str(exc))}")


@app.post("/chat/bulk-load")
def bulk_load(req: BulkChatRequest):
    results: list[dict] = []
    for arxiv_id in req.arxiv_ids[:5]:
        try:
            meta = platform.paper_chat.create_or_get_session_from_arxiv_id(arxiv_id)
            results.append({
                "arxiv_id": arxiv_id,
                "session_id": meta["session_id"],
                "chunk_count": meta.get("chunk_count", 0),
                "cached": meta.get("cached", False),
                "status": "ok",
            })
        except Exception as exc:
            results.append({"arxiv_id": arxiv_id, "status": "error", "error": redact_secrets(str(exc))})
    session_ids = [item["session_id"] for item in results if item.get("status") == "ok"]
    first_answer = None
    if req.question.strip() and session_ids:
        try:
            first_answer = platform.paper_chat.ask_multi(session_ids, req.question, top_k_per_session=3)
        except Exception as exc:
            first_answer = {"error": redact_secrets(str(exc))}
    return {
        "papers": results,
        "session_ids": session_ids,
        "total_loaded": len(session_ids),
        "answer": first_answer,
    }


@app.get("/chat/session/{session_id}")
def chat_session_info(session_id: str):
    try:
        return platform.paper_chat.session_info(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Execution routes
# ---------------------------------------------------------------------------

@app.post("/execution/python")
def execute_python(req: PythonExecutionRequest):
    result = platform.python_runner.run(req.code).to_dict()
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Execution failed."))
    return result
