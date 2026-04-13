from __future__ import annotations

import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .cloud_llm import CloudLLMClient


class Summarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.backend = os.getenv("LLM_BACKEND", "local").strip().lower()
        self._tokenizer = None
        self._model = None
        self._cloud = None

    def _ensure_loaded(self):
        if self.backend == "cloud":
            if self._cloud is None:
                self._cloud = CloudLLMClient()
            return

        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def summarize(self, text: str, max_length: int = 160, min_length: int = 50) -> str:
        if not text or not text.strip():
            return ""
        self._ensure_loaded()

        if self.backend == "cloud":
            prompt = (
                "Summarize the following research text in 4-6 concise bullet points with key contributions:\n\n"
                f"{text}"
            )
            return self._cloud.generate(prompt, max_tokens=max_length)

        prompt = f"summarize: {text}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
