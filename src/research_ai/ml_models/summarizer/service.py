from __future__ import annotations

import os


class ScientificSummarizer:
    """Scientific summarizer with lazy local/cloud model loading."""

    SUMMARIZE_SYSTEM = (
        "You are an expert research paper summarizer. Highlight contribution, "
        "methodology, main findings, limitations, and scientific impact."
    )

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        self.model_name = model_name
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        self._tokenizer = None
        self._model = None
        self._cloud = None

    @property
    def ready(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self.backend == "cloud":
            if self._cloud is None:
                from research_ai.llm import CloudLLMClient

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
                "Summarize this scientific text with these sections:\n"
                "Key Contribution, Methodology, Main Findings, Limitations, Impact.\n\n"
                f"Text:\n{text[:3500]}"
            )
            return self._cloud.generate(prompt, max_tokens=max_length, system=self.SUMMARIZE_SYSTEM)  # type: ignore[union-attr]

        prompt = f"summarize: {text}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # type: ignore[call-arg]
        output_ids = self._model.generate(  # type: ignore[union-attr]
            **inputs, max_new_tokens=max_length, min_length=min_length, do_sample=False
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

