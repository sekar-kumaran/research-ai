from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

from src.api.schemas import (
    AgentRequest,
    ArxivLoadRequest,
    AskRequest,
    ClassifyRequest,
    MediatedAgentResponse,
    PaperChatRequest,
    SearchRequest,
    SummarizeRequest,
)
from src.research_assistant.agents import AssistantAgents
from src.research_assistant.paper_chat import PaperChatService
from src.research_assistant.preprocess import build_full_text, clean_text
from src.research_assistant.rag import RAGAssistant
from src.research_assistant.summarization import Summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Assistant API",
    version="2.0.0",
    description="AI-powered research paper retrieval, classification & summarisation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = Path("artifacts")
CLASSIFIER_DIR = ARTIFACTS / "classification"
SIM_DIR = ARTIFACTS / "similarity"

# --- Load artifacts ---
_classifier = None
_vectorizer = None
if (CLASSIFIER_DIR / "classifier.joblib").exists():
    try:
        _classifier = joblib.load(CLASSIFIER_DIR / "classifier.joblib")
        _vectorizer = joblib.load(CLASSIFIER_DIR / "tfidf_vectorizer.joblib")
        logger.info("✅ Classifier loaded.")
    except Exception as e:
        logger.warning(f"⚠️ Classifier load failed: {e}")

_summarizer = None
try:
    _summarizer = Summarizer()
    logger.info("✅ Summarizer initialized.")
except Exception as e:
    logger.warning(f"⚠️ Summarizer init failed: {e}")

_rag = None
_embed_model = None
if (SIM_DIR / "paper_index.faiss").exists() and (SIM_DIR / "paper_metadata.parquet").exists():
    try:
        from src.research_assistant.similarity import load_similarity_artifacts
        index, metadata = load_similarity_artifacts(SIM_DIR)
        embed_name = "all-MiniLM-L6-v2"
        if (SIM_DIR / "embedding_model_name.joblib").exists():
            embed_name = joblib.load(SIM_DIR / "embedding_model_name.joblib")
        _embed_model = SentenceTransformer(embed_name)
        _rag = RAGAssistant(_embed_model, index, metadata)
        logger.info(f"✅ RAG index loaded ({len(metadata)} papers).")
    except Exception as e:
        logger.warning(f"⚠️ RAG load failed: {e}")

_paper_chat = PaperChatService()

_agents = AssistantAgents(
    classifier=_classifier,
    vectorizer=_vectorizer,
    rag_assistant=_rag,
    summarizer=_summarizer,
    paper_chat=_paper_chat,
)

# --- Frontend ---
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _extract_primary_text(payload: object) -> str:
    if isinstance(payload, dict):
        if payload.get("error"):
            return f"⚠️ {payload['error']}"
        for key in ("final_answer", "agent_output", "summary"):
            if isinstance(payload.get(key), str) and payload[key].strip():
                return payload[key]
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
        if isinstance(answer, dict):
            for k in ("final_answer", "answer"):
                if isinstance(answer.get(k), str):
                    return answer[k]
        if payload.get("predicted_category"):
            return f"Predicted category: **{payload['predicted_category']}**"
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)


@app.get("/")
def home():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. Visit /docs for API reference."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "components": {
            "classifier": _classifier is not None,
            "rag": _rag is not None,
            "summarizer": _summarizer is not None,
            "paper_chat": _paper_chat is not None,
        },
        "llm_backend": os.getenv("LLM_BACKEND", "cloud"),
        "llm_provider": os.getenv("CLOUD_LLM_PROVIDER", "groq"),
    }


@app.get("/stats")
def stats():
    """Return index and model statistics."""
    paper_count = 0
    if _rag is not None:
        try:
            paper_count = len(_rag.metadata)
        except Exception:
            pass
    return {
        "indexed_papers": paper_count,
        "active_chat_sessions": len(_paper_chat.sessions),
        "classifier_ready": _classifier is not None,
        "rag_ready": _rag is not None,
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    out = _agents.run(
        mode="classify",
        query=build_full_text(req.title, req.abstract),
        title=req.title,
        abstract=req.abstract,
        top_k=5,
    )
    result = out.get("executor_output", {})
    if result.get("error"):
        raise HTTPException(status_code=503, detail=str(result["error"]))
    return result


@app.post("/search")
def search(req: SearchRequest):
    out = _agents.run(mode="search", query=req.query, top_k=req.top_k)
    result = out.get("executor_output", {})
    if result.get("error"):
        raise HTTPException(status_code=503, detail=str(result["error"]))
    result["query"] = req.query
    return result


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    out = _agents.run(mode="summarize", query=req.text, text=req.text, top_k=5)
    result = out.get("executor_output", {})
    if result.get("error"):
        raise HTTPException(status_code=503, detail=str(result["error"]))
    summary = str(result.get("summary", ""))
    return {"summary": summary, "word_count": len(summary.split())}


@app.post("/ask")
def ask(req: AskRequest):
    out = _agents.run(mode="ask", query=req.query, top_k=req.top_k)
    return out


@app.post("/agent/run", response_model=MediatedAgentResponse)
def run_agent(req: AgentRequest):
    return _agents.run(
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
    out = _agents.run(
        mode=req.mode,
        query=req.query,
        title=req.title,
        abstract=req.abstract,
        top_k=req.top_k,
        text=req.text,
        session_id=req.session_id,
    )
    text = _extract_primary_text(out)
    request_id = out.get("request_id", "")
    mediated_mode = out.get("mode", req.mode)

    async def event_generator():
        start_evt = {"event": "start", "request_id": request_id, "mode": mediated_mode}
        yield f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"
        if not text:
            yield "data: [DONE]\n\n"
            return
        step = max(1, len(text) // 100)
        for i in range(0, len(text), step):
            delta = text[i: i + step]
            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.008)
        end_evt = {
            "event": "end",
            "request_id": request_id,
            "mode": mediated_mode,
            "latency_ms": out.get("latency_ms"),
        }
        yield f"data: {json.dumps(end_evt, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/upload")
async def upload_paper(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "uploaded_file"
        if filename.lower().endswith(".pdf"):
            meta = _paper_chat.create_session_from_pdf_bytes(content, source=f"upload:{filename}")
        else:
            text = content.decode("utf-8", errors="ignore")
            meta = _paper_chat.create_session_from_text(text=text, source=f"upload:{filename}")
        return meta
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}")


@app.post("/chat/load-arxiv")
def load_arxiv_paper(req: ArxivLoadRequest):
    try:
        return _paper_chat.create_or_get_session_from_arxiv_id(req.arxiv_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"arXiv load failed: {exc}")


@app.post("/chat/ask")
def chat_ask(req: PaperChatRequest):
    try:
        out = _agents.run(
            mode="paper_chat",
            query=req.question,
            top_k=req.top_k,
            session_id=req.session_id,
        )
        result = out.get("executor_output", {})
        if result.get("error"):
            raise HTTPException(status_code=400, detail=str(result["error"]))
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Chat failed: {exc}")


@app.get("/chat/session/{session_id}")
def chat_session_info(session_id: str):
    try:
        return _paper_chat.session_info(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
