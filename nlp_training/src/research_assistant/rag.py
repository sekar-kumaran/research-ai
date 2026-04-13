from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from .cloud_llm import CloudLLMClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    paper_id: str
    title: str
    abstract: str
    score: float
    authors: str = ""
    category: str = ""
    year: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": round(self.score, 4),
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }

    @property
    def arxiv_url(self) -> str:
        pid = self.paper_id.replace("arxiv:", "").strip()
        return f"https://arxiv.org/abs/{pid}" if pid else ""


class RAGAssistant:
    """Retrieval-Augmented Generation pipeline using FAISS + cloud/local LLM."""

    RAG_SYSTEM_PROMPT = (
        "You are an expert AI research assistant. Answer the user's question using ONLY "
        "the provided research paper context. Cite paper titles where relevant. "
        "If the context is insufficient, say so clearly. Be concise and structured."
    )

    def __init__(
        self,
        embed_model: SentenceTransformer,
        index,
        metadata,
        generator_model_name: str = "google/flan-t5-base",
    ):
        self.embed_model = embed_model
        self.index = index
        self.metadata = metadata
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        self.generator_model_name = generator_model_name
        self._generator = None
        self._tokenizer = None
        self._cloud: CloudLLMClient | None = None

    def _ensure_generator(self):
        if self.backend == "cloud":
            if self._cloud is None:
                self._cloud = CloudLLMClient()
            return
        if self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Embed query and search FAISS index for top-k papers."""
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.clip(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-12, None)

        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
        scores, ids = scores[0], ids[0]

        docs: list[RetrievedDoc] = []
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[int(idx)]
            docs.append(
                RetrievedDoc(
                    paper_id=str(row.get("id", "")),
                    title=str(row.get("title", "Untitled")).strip(),
                    abstract=str(row.get("abstract", "")).strip(),
                    score=float(score),
                    authors=str(row.get("authors", "")),
                    category=str(row.get("categories", "")),
                    year=str(row.get("update_date", ""))[:4],
                )
            )
        return docs

    def answer(self, query: str, top_k: int = 5, max_new_tokens: int = 512) -> dict:
        """Retrieve relevant papers and generate an LLM answer."""
        self._ensure_generator()
        docs = self.retrieve(query, top_k=top_k)

        if not docs:
            return {"query": query, "answer": "No relevant papers found in the index.", "retrieved": []}

        context_parts = []
        for i, d in enumerate(docs, 1):
            context_parts.append(f"[{i}] **{d.title}** ({d.year})\n{d.abstract[:600]}")
        context = "\n\n".join(context_parts)

        prompt = (
            f"Answer this research question based on the papers below.\n\n"
            f"Question: {query}\n\n"
            f"Papers:\n{context}\n\n"
            f"Answer (cite paper numbers like [1], [2]):"
        )

        if self.backend == "cloud":
            answer_text = self._cloud.generate(prompt, max_tokens=max_new_tokens, system=self.RAG_SYSTEM_PROMPT)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            output_ids = self._generator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            answer_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "query": query,
            "answer": answer_text,
            "retrieved": [d.to_dict() for d in docs],
        }
