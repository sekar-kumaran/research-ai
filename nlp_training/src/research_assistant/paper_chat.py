from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from uuid import uuid4

import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError as exc:
    raise RuntimeError("faiss-cpu is required for paper chat retrieval.") from exc

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    metadata: dict = field(default_factory=dict)


class PaperChatService:
    """Chat with uploaded or arXiv papers using FAISS retrieval + cloud or local LLM."""

    CHAT_SYSTEM = (
        "You are an expert research assistant helping a user understand a specific academic paper. "
        "Answer questions using ONLY the provided paper content. "
        "Be accurate, clear, and cite the relevant section when possible. "
        "If the answer isn't in the provided context, say so explicitly."
    )

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generator_model_name: str = "google/flan-t5-small",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.generator_model_name = generator_model_name
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        self._tokenizer = None
        self._generator = None
        self._cloud = None
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def _ensure_generator(self):
        if self.backend == "cloud":
            if self._cloud is None:
                from .cloud_llm import CloudLLMClient
                self._cloud = CloudLLMClient()
            return
        if self._tokenizer is None or self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(0, end - overlap)
        return chunks

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _build_index(self, chunks: list[str]):
        vectors = self._encode(chunks).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def _generate_answer(self, question: str, context: str, history: list[dict], max_tokens: int = 400) -> str:
        self._ensure_generator()

        if self.backend == "cloud":
            messages = [{"role": "system", "content": self.CHAT_SYSTEM}]
            # Add history (last 3 turns)
            for turn in history[-3:]:
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
            messages.append({
                "role": "user",
                "content": f"Context from paper:\n{context}\n\nQuestion: {question}",
            })
            return self._cloud.chat(messages, max_tokens=max_tokens)

        history_text = "\n".join(
            [f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:]]
        )
        prompt = (
            "You are a research paper assistant. Answer ONLY using the provided paper context.\n\n"
            f"Previous Chat:\n{history_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Paper Context:\n{context}\n\n"
            "Answer clearly and concisely:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self._generator.generate(**inputs, max_new_tokens=220, do_sample=False)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def create_session_from_text(self, text: str, source: str, title: str = "", metadata: dict | None = None) -> dict:
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("No text content found after preprocessing.")

        session_id = str(uuid4())
        index = self._build_index(chunks)
        self.sessions[session_id] = ChatSession(
            session_id=session_id,
            source=source,
            chunks=chunks,
            index=index,
            title=title,
            metadata=metadata or {},
        )
        self.source_to_session[source] = session_id
        return {"session_id": session_id, "source": source, "chunk_count": len(chunks), "title": title}

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
        text = self._extract_pdf_text(data)
        return self.create_session_from_text(text=text, source=source)

    @staticmethod
    def normalize_arxiv_id(raw_id: str) -> str:
        token = (raw_id or "").strip()
        if not token:
            return ""
        if token.startswith("arXiv:"):
            token = token.split(":", 1)[1]
        if token.startswith("http://") or token.startswith("https://"):
            token = token.rstrip("/")
            if "/abs/" in token:
                token = token.split("/abs/")[-1]
            elif "/pdf/" in token:
                token = token.split("/pdf/")[-1]
        return token.replace(".pdf", "")

    def create_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        arxiv_id = arxiv_id.strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=60, headers={"User-Agent": "ResearchAssistant/1.0"})
        response.raise_for_status()
        meta = self.create_session_from_pdf_bytes(response.content, source=f"arxiv:{arxiv_id}")
        meta["links"] = {"abs": f"https://arxiv.org/abs/{arxiv_id}", "pdf": pdf_url}
        return meta

    def create_or_get_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        normalized_id = self.normalize_arxiv_id(arxiv_id)
        if not normalized_id:
            raise ValueError("Invalid arXiv identifier.")
        source = f"arxiv:{normalized_id}"
        existing = self.source_to_session.get(source)
        if existing and existing in self.sessions:
            session = self.sessions[existing]
            return {
                "session_id": session.session_id,
                "source": session.source,
                "chunk_count": len(session.chunks),
                "cached": True,
                "links": {
                    "abs": f"https://arxiv.org/abs/{normalized_id}",
                    "pdf": f"https://arxiv.org/pdf/{normalized_id}.pdf",
                },
            }
        meta = self.create_session_from_arxiv_id(normalized_id)
        meta["cached"] = False
        return meta

    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        if session_id not in self.sessions:
            raise KeyError("Invalid session_id. Upload or load a paper first.")

        session = self.sessions[session_id]
        query_vec = self._encode([question]).astype(np.float32)
        scores, ids = session.index.search(query_vec, top_k)

        selected = []
        context_parts = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk_text = session.chunks[int(idx)]
            selected.append({"chunk_id": int(idx), "score": float(score), "text": chunk_text[:800]})
            context_parts.append(chunk_text)

        context = "\n\n".join(context_parts)
        answer = self._generate_answer(question, context, session.history)
        session.history.append({"question": question, "answer": answer})

        return {
            "session_id": session_id,
            "source": session.source,
            "question": question,
            "answer": answer,
            "citations": selected,
            "turns": len(session.history),
        }

    def ask_multi(self, session_ids: list[str], question: str, top_k_per_session: int = 3) -> dict:
        if not session_ids:
            raise ValueError("No sessions provided.")

        collected: list[dict] = []
        for session_id in session_ids:
            if session_id not in self.sessions:
                continue
            session = self.sessions[session_id]
            query_vec = self._encode([question]).astype(np.float32)
            scores, ids = session.index.search(query_vec, top_k_per_session)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                collected.append({
                    "session_id": session_id,
                    "source": session.source,
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": session.chunks[int(idx)],
                })

        if not collected:
            raise ValueError("No retrievable content found in selected papers.")

        collected.sort(key=lambda x: x["score"], reverse=True)
        selected = collected[:10]
        context = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in selected])
        answer = self._generate_answer(question, context, [])

        return {
            "question": question,
            "answer": answer,
            "citations": [{"session_id": i["session_id"], "source": i["source"], "score": i["score"], "text": i["text"][:600]} for i in selected],
            "paper_count": len({i["source"] for i in selected}),
        }

    def session_info(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            raise KeyError("Invalid session_id")
        s = self.sessions[session_id]
        return {
            "session_id": s.session_id,
            "source": s.source,
            "chunk_count": len(s.chunks),
            "turns": len(s.history),
            "title": s.title,
        }
