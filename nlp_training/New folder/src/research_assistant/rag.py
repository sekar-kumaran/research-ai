from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .cloud_llm import CloudLLMClient
from .similarity import search_similar


@dataclass
class RetrievedDoc:
    paper_id: str
    title: str
    abstract: str
    score: float


class RAGAssistant:
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
        self.backend = os.getenv("LLM_BACKEND", "local").strip().lower()
        self.generator_model_name = generator_model_name
        self.generator = None
        self.tokenizer = None
        self.cloud = None

    def _ensure_generator(self):
        if self.backend == "cloud":
            if self.cloud is None:
                self.cloud = CloudLLMClient()
            return

        if self.generator is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.clip(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-12, None)
        scores, ids = search_similar(self.index, query_vec, top_k=top_k)

        docs: list[RetrievedDoc] = []
        for score, idx in zip(scores, ids):
            row = self.metadata.iloc[int(idx)]
            docs.append(
                RetrievedDoc(
                    paper_id=str(row["id"]),
                    title=str(row["title"]),
                    abstract=str(row["abstract"]),
                    score=float(score),
                )
            )
        return docs

    def answer(self, query: str, top_k: int = 5, max_new_tokens: int = 180) -> dict:
        self._ensure_generator()
        docs = self.retrieve(query, top_k=top_k)
        context = "\n\n".join([f"Title: {d.title}\nAbstract: {d.abstract}" for d in docs])
        prompt = (
            "You are a research assistant. Use only the provided context. "
            "If context is insufficient, say so clearly.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )

        if self.backend == "cloud":
            out = self.cloud.generate(prompt, max_tokens=max_new_tokens)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            output_ids = self.generator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "query": query,
            "answer": out,
            "retrieved": [d.__dict__ for d in docs],
        }
