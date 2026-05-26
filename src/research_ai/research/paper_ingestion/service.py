"""Full-paper ingestion, per-session FAISS chunk retrieval, and grounded paper chat.

ARCHITECTURE
------------
Each paper loaded into the system (via PDF, text, or arXiv ID) gets its own
mini-FAISS index (IndexFlatIP) built from its chunk embeddings.  This index is
stored in a ChatSession alongside the chunk text and conversation history.

WHY a per-paper FAISS index?
  - The global FAISS index only has title+abstract embeddings (one vector/paper).
  - Full-paper chat needs section-level retrieval: finding which passage of a
    50-page paper answers a specific question.
  - Per-session indices are tiny (10–100 chunks × 384 dims = <1MB each) and
    allow session isolation — querying one paper never pollutes another.

CHUNKING STRATEGY
-----------------
contextual_chunks() (see retrieval/chunking.py) splits text into 900-word
sentence-boundary-respecting chunks with 150-word overlap.  The overlap means
that a sentence at the boundary of a chunk appears in both adjacent chunks,
so context is never lost at retrieval time.

RETRIEVAL IN ask()
------------------
  1. Embed the user question with EmbeddingService (L2-normalised).
  2. IndexFlatIP.search() → top_k chunks by cosine similarity.
  3. Concatenate chunk texts as context for the LLM.
  4. LLM generates a grounded answer citing only the provided context.

CLOUD FACTORY INJECTION (BUG FIX v3.1.1)
-----------------------------------------
Previously this class called CloudLLMClient() directly, creating a second
client instance and breaking the singleton pattern established in platform.py.
Fix: accept a cloud_factory callable from the composition root.  The factory
returns the shared singleton.  If cloud_factory is None (local-only mode), the
class falls back to the local Flan-T5 generator.
"""
from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from uuid import uuid4

import numpy as np
import requests
from pypdf import PdfReader

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

from research_ai.memory.session_memory import ChatSession, SessionMemory
from research_ai.retrieval.chunking import contextual_chunks

logger = logging.getLogger(__name__)

# ArXiv version suffix pattern: matches "v1", "v2", "v10", etc. at end of ID.
# Example: "2301.04567v2" → "2301.04567"
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


class PaperChatService:
    """Full-paper ingestion, chunk retrieval, and grounded paper chat."""

    CHAT_SYSTEM = (
        "You are an expert research assistant answering questions about academic papers. "
        "Use only the provided paper context. If evidence is missing, say so. "
        "Do NOT add information from outside the provided context."
    )

    def __init__(
        self,
        embedding_service,
        memory: SessionMemory | None = None,
        cloud_factory=None,
    ) -> None:
        self.embedding_service = embedding_service
        self.memory = memory or SessionMemory()
        self.generator_model_name = "google/flan-t5-small"
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        # cloud_factory: zero-arg callable that returns the shared CloudLLMClient
        # singleton.  None means local-only mode → use Flan-T5 generator.
        self._cloud_factory = cloud_factory
        self._tokenizer = None
        self._generator = None
        # Resolved cloud client (cached after first successful factory call)
        self._cloud = None

    @property
    def sessions(self) -> dict[str, ChatSession]:
        return self.memory.sessions

    def _ensure_generator(self) -> None:
        """Lazily initialize the answer-generation backend.

        Cloud path: call the injected factory to get the shared singleton client.
          WHY factory not direct import: keeps the singleton pattern consistent
          with the rest of the platform and allows test injection.

        Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
          WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
          and produces coherent short answers from context prompts.
        """
        if self.backend == "cloud":
            if self._cloud is None:
                if self._cloud_factory is not None:
                    # Use the shared singleton from the composition root
                    self._cloud = self._cloud_factory()
                else:
                    # Fallback: direct import (only if factory wasn't injected)
                    from research_ai.llm import CloudLLMClient
                    self._cloud = CloudLLMClient()
            return
        # Local Flan-T5 path
        if self._tokenizer is None or self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    def _build_index(self, chunks: list[str]):
        vectors = self.embedding_service.encode(chunks).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def create_session_from_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> dict:
        chunks = contextual_chunks(text)
        if not chunks:
            raise ValueError("No text content found after chunking.")
        session = ChatSession(
            session_id=str(uuid4()),
            source=source,
            chunks=chunks,
            index=self._build_index(chunks),
            title=title,
            metadata=metadata or {},
        )
        self.memory.put(session)
        return {
            "session_id": session.session_id,
            "source": source,
            "chunk_count": len(chunks),
            "title": title,
        }

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
        text = self._extract_pdf_text(data)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. It may be scanned/image-only.")
        return self.create_session_from_text(text=text, source=source)

    @staticmethod
    def normalize_arxiv_id(raw_id: str) -> str:
        """Normalize any arXiv identifier form to a bare numeric ID.

        Handles all common input formats:
          "arxiv:2301.04567"    → "2301.04567"
          "2301.04567v2"        → "2301.04567"   ← BUG FIX: version suffix stripped
          "https://arxiv.org/abs/2301.04567v2" → "2301.04567"
          "https://arxiv.org/pdf/2301.04567.pdf" → "2301.04567"
          "2301.04567"          → "2301.04567"   (pass-through)

        WHY strip version suffixes (BUG FIX v3.1.1):
          Without this fix, loading "2301.04567" and "2301.04567v2" would create
          two separate chat sessions for the same paper.  This wastes memory and
          confuses the session cache (find_source finds "arxiv:2301.04567v2" but
          not "arxiv:2301.04567").  By normalizing to the base ID, both resolve
          to the same session.

          The arXiv PDF API serves the latest version when no version is specified,
          so stripping "v2" and fetching the bare ID is safe and gives the same PDF.
        """
        token = (raw_id or "").strip()
        if not token:
            return ""

        # Strip "arxiv:" prefix (case-insensitive)
        if token.lower().startswith("arxiv:"):
            token = token.split(":", 1)[1]

        # Strip URL prefixes — handles abs/, pdf/ paths
        if token.startswith(("http://", "https://")):
            token = token.rstrip("/")
            for marker in ("/abs/", "/pdf/"):
                if marker in token:
                    token = token.split(marker)[-1]
                    break

        # Strip ".pdf" extension
        token = token.replace(".pdf", "").strip()

        # Strip version suffix: "2301.04567v2" → "2301.04567"
        # Applied AFTER all other normalization steps.
        token = _ARXIV_VERSION_RE.sub("", token).strip()

        return token

    def create_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        normalized = self.normalize_arxiv_id(arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{normalized}.pdf"
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "ResearchAI/3.0"})
        resp.raise_for_status()
        meta = self.create_session_from_pdf_bytes(resp.content, source=f"arxiv:{normalized}")
        meta["links"] = {"abs": f"https://arxiv.org/abs/{normalized}", "pdf": pdf_url}
        return meta

    def create_or_get_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        normalized = self.normalize_arxiv_id(arxiv_id)
        if not normalized:
            raise ValueError(f"Invalid arXiv identifier: '{arxiv_id}'")
        source = f"arxiv:{normalized}"
        existing = self.memory.find_source(source)
        if existing:
            return {
                "session_id": existing.session_id,
                "source": existing.source,
                "chunk_count": len(existing.chunks),
                "cached": True,
                "links": {
                    "abs": f"https://arxiv.org/abs/{normalized}",
                    "pdf": f"https://arxiv.org/pdf/{normalized}.pdf",
                },
            }
        meta = self.create_session_from_arxiv_id(normalized)
        meta["cached"] = False
        return meta

    def _generate_answer(self, question: str, context: str, history: list[dict], max_tokens: int = 400) -> str:
        self._ensure_generator()
        if self.backend == "cloud":
            messages: list[dict] = [{"role": "system", "content": self.CHAT_SYSTEM}]
            for turn in history[-3:]:
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
            return self._cloud.chat(messages, max_tokens=max_tokens)  # type: ignore[union-attr]

        history_text = "\n".join(f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:])
        prompt = (
            "Answer only from the provided paper context.\n\n"
            f"Previous Chat:\n{history_text}\n\nQuestion:\n{question}\n\nPaper Context:\n{context}\n\nAnswer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # type: ignore[call-arg]
        output_ids = self._generator.generate(**inputs, max_new_tokens=220, do_sample=False)  # type: ignore[union-attr]
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        session = self.memory.get(session_id)
        query_vec = self.embedding_service.encode([question]).astype(np.float32)
        scores, ids = session.index.search(query_vec, top_k)
        selected: list[dict] = []
        context_parts: list[str] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk_text = session.chunks[int(idx)]
            selected.append({"chunk_id": int(idx), "score": round(float(score), 4), "text": chunk_text[:800]})
            context_parts.append(chunk_text)
        answer = self._generate_answer(question, "\n\n".join(context_parts), session.history)
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
            raise ValueError("No session IDs provided.")
        collected: list[dict] = []
        for session_id in session_ids:
            try:
                session = self.memory.get(session_id)
            except KeyError:
                logger.warning("Session %s not found, skipping.", session_id)
                continue
            query_vec = self.embedding_service.encode([question]).astype(np.float32)
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
            raise ValueError("No retrievable content found across selected sessions.")
        collected.sort(key=lambda item: item["score"], reverse=True)
        top = collected[:10]
        context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in top)
        answer = self._generate_answer(question, context, [])
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "session_id": item["session_id"],
                    "source": item["source"],
                    "score": round(item["score"], 4),
                    "text": item["text"][:600],
                }
                for item in top
            ],
            "paper_count": len({item["source"] for item in top}),
        }

    def session_info(self, session_id: str) -> dict:
        session = self.memory.get(session_id)
        return {
            "session_id": session.session_id,
            "source": session.source,
            "chunk_count": len(session.chunks),
            "turns": len(session.history),
            "title": session.title,
        }

