from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required for paper chat retrieval.") from exc


@dataclass
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]]


class PaperChatService:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", generator_model_name: str = "google/flan-t5-small"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.generator_model_name = generator_model_name
        self.tokenizer = None
        self.generator = None
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def _ensure_generator(self):
        if self.tokenizer is None or self.generator is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
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

    def create_session_from_text(self, text: str, source: str) -> dict:
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
            history=[],
        )
        self.source_to_session[source] = session_id
        return {
            "session_id": session_id,
            "source": source,
            "chunk_count": len(chunks),
        }

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
        text = self._extract_pdf_text(data)
        return self.create_session_from_text(text=text, source=source)

    def create_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        arxiv_id = arxiv_id.strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        meta = self.create_session_from_pdf_bytes(response.content, source=f"arxiv:{arxiv_id}")
        meta["links"] = {
            "abs": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf": pdf_url,
        }
        return meta

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
            token = token.replace(".pdf", "")
        if token.endswith(".pdf"):
            token = token[:-4]
        return token

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

    def ask_multi(self, session_ids: list[str], question: str, top_k_per_session: int = 3) -> dict:
        if not session_ids:
            raise ValueError("No sessions provided for multi-paper QA.")

        self._ensure_generator()

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
                chunk_text = session.chunks[int(idx)]
                collected.append(
                    {
                        "session_id": session_id,
                        "source": session.source,
                        "chunk_id": int(idx),
                        "score": float(score),
                        "text": chunk_text,
                    }
                )

        if not collected:
            raise ValueError("No retrievable content was found in selected papers.")

        collected.sort(key=lambda x: x["score"], reverse=True)
        selected = collected[: min(10, len(collected))]
        context = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in selected])

        prompt = (
            "You are a research paper assistant. Use only the provided multi-paper context. "
            "If the answer is not in context, say that clearly.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer with concise reasoning and cite source IDs like [arxiv:xxxx.xxxxx]:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.generator.generate(**inputs, max_new_tokens=240, do_sample=False)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "session_id": item["session_id"],
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "text": item["text"][:800],
                }
                for item in selected
            ],
            "paper_count": len({item["source"] for item in selected}),
        }

    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        if session_id not in self.sessions:
            raise KeyError("Invalid session_id. Upload or load a paper first.")

        self._ensure_generator()
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

        history_text = "\n".join(
            [f"User: {h['question']}\nAssistant: {h['answer']}" for h in session.history[-4:]]
        )
        context = "\n\n".join(context_parts)

        prompt = (
            "You are a research paper assistant. Answer ONLY using the provided paper context. "
            "If the answer is not in the context, say that clearly.\n\n"
            f"Previous Chat:\n{history_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Paper Context:\n{context}\n\n"
            "Answer clearly and concisely:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.generator.generate(**inputs, max_new_tokens=220, do_sample=False)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        session.history.append({"question": question, "answer": answer})

        return {
            "session_id": session_id,
            "source": session.source,
            "question": question,
            "answer": answer,
            "citations": selected,
            "turns": len(session.history),
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
        }
