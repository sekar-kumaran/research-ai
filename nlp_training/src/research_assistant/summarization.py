from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


class Summarizer:
    """Paper summarizer using cloud or local model."""

    SUMMARIZE_SYSTEM = (
        "You are an expert research paper summariser. Create clear, structured summaries "
        "that highlight key contributions, methodology, and findings."
    )

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        self._tokenizer = None
        self._model = None
        self._cloud = None

    def _ensure_loaded(self):
        if self.backend == "cloud":
            if self._cloud is None:
                from .cloud_llm import CloudLLMClient
                self._cloud = CloudLLMClient()
            return
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def summarize(self, text: str, max_length: int = 300, min_length: int = 80) -> str:
        if not text or not text.strip():
            return ""
        self._ensure_loaded()

        if self.backend == "cloud":
            prompt = (
                "Summarise the following research paper text. Structure your response as:\n"
                "**Key Contribution:** (1 sentence)\n"
                "**Methodology:** (2-3 sentences)\n"
                "**Main Findings:** (3-4 bullet points)\n"
                "**Impact:** (1 sentence)\n\n"
                f"Text:\n{text[:3000]}"
            )
            return self._cloud.generate(prompt, max_tokens=max_length, system=self.SUMMARIZE_SYSTEM)

        prompt = f"summarize: {text}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize_batch(self, texts: list[str]) -> list[str]:
        return [self.summarize(t) for t in texts]
