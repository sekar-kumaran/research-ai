from __future__ import annotations

import asyncio
import json
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

from src.api.schemas import AgentRequest, ArxivLoadRequest, AskRequest, ClassifyRequest, PaperChatRequest, SearchRequest, SummarizeRequest
from src.research_assistant.agents import AssistantAgents
from src.research_assistant.paper_chat import PaperChatService
from src.research_assistant.preprocess import build_full_text, clean_text
from src.research_assistant.rag import RAGAssistant
from src.research_assistant.similarity import load_similarity_artifacts
from src.research_assistant.summarization import Summarizer

app = FastAPI(title="Research Assistant API", version="1.0.0")

ARTIFACTS = Path("artifacts")
CLASSIFIER_DIR = ARTIFACTS / "classification"
SIM_DIR = ARTIFACTS / "similarity"

_classifier = joblib.load(CLASSIFIER_DIR / "classifier.joblib") if (CLASSIFIER_DIR / "classifier.joblib").exists() else None
_vectorizer = joblib.load(CLASSIFIER_DIR / "tfidf_vectorizer.joblib") if (CLASSIFIER_DIR / "tfidf_vectorizer.joblib").exists() else None

try:
    _summarizer = Summarizer()
except Exception:
    _summarizer = None

_rag = None
if (SIM_DIR / "paper_index.faiss").exists() and (SIM_DIR / "paper_metadata.parquet").exists():
    index, metadata = load_similarity_artifacts(SIM_DIR)
    embed_name = joblib.load(SIM_DIR / "embedding_model_name.joblib") if (SIM_DIR / "embedding_model_name.joblib").exists() else "all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(embed_name)
    _rag = RAGAssistant(embed_model, index, metadata)

_paper_chat = PaperChatService()

_agents = AssistantAgents(
    classifier=_classifier,
    vectorizer=_vectorizer,
    rag_assistant=_rag,
    summarizer=_summarizer,
    paper_chat=_paper_chat,
)


def _extract_primary_text(payload: object) -> str:
    if isinstance(payload, dict):
        if payload.get("error"):
            return f"Error: {payload['error']}"

        answer = payload.get("answer")
        if isinstance(answer, str):
            return answer
        if isinstance(answer, dict):
            final_answer = answer.get("final_answer")
            if isinstance(final_answer, str):
                return final_answer
            nested = answer.get("answer")
            if isinstance(nested, str):
                return nested
            paper_answer = answer.get("paper_answer")
            if isinstance(paper_answer, dict):
                paper_nested = paper_answer.get("answer")
                if isinstance(paper_nested, str):
                    return paper_nested

        if isinstance(payload.get("agent_output"), str):
            return payload["agent_output"]
        if isinstance(payload.get("summary"), str):
            return payload["summary"]
        if payload.get("predicted_category"):
            return f"Predicted category: {payload['predicted_category']}"

    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. Open API docs at /docs."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest):
    if _classifier is None or _vectorizer is None:
        return {"error": "Classifier artifacts missing. Run training first."}

    text = clean_text(build_full_text(req.title, req.abstract))
    x = _vectorizer.transform([text])
    pred = _classifier.predict(x)[0]
    return {"predicted_category": str(pred)}


@app.post("/search")
def search(req: SearchRequest):
    if _rag is None:
        return {"error": "Similarity index missing. Build embeddings first."}
    docs = _rag.retrieve(req.query, top_k=req.top_k)
    return {"results": [d.__dict__ for d in docs]}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if _summarizer is None:
        return {"error": "Summarizer unavailable. Install transformers/torch dependencies."}
    summary = _summarizer.summarize(req.text)
    return {"summary": summary}


@app.post("/ask")
def ask(req: AskRequest):
    return _agents.ask(req.query, top_k=req.top_k)


@app.post("/agent/run")
def run_agent(req: AgentRequest):
    return _agents.run(
        mode=req.mode,
        query=req.query,
        title=req.title,
        abstract=req.abstract,
        top_k=req.top_k,
    )


@app.post("/agent/run/stream")
async def run_agent_stream(req: AgentRequest):
    out = _agents.run(
        mode=req.mode,
        query=req.query,
        title=req.title,
        abstract=req.abstract,
        top_k=req.top_k,
    )
    text = _extract_primary_text(out)

    async def event_generator():
        if not text:
            yield "data: [DONE]\n\n"
            return

        step = max(1, len(text) // 120)
        for i in range(0, len(text), step):
            delta = text[i : i + step]
            yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/upload")
async def upload_paper(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "uploaded_file"
        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
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
        return _paper_chat.create_session_from_arxiv_id(req.arxiv_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"arXiv load failed: {exc}")


@app.post("/chat/ask")
def chat_ask(req: PaperChatRequest):
    try:
        return _paper_chat.ask(session_id=req.session_id, question=req.question, top_k=req.top_k)
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
