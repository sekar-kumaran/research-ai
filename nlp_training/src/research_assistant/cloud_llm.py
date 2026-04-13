from __future__ import annotations

import os
import time
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)


class CloudLLMClient:
    """Robust OpenAI-compatible client with retry logic and streaming."""

    SYSTEM_PROMPT = (
        "You are an expert AI research assistant specialising in academic paper analysis. "
        "You help researchers find, understand, classify, and summarise scientific papers. "
        "Be concise, accurate, and academically rigorous. Format answers with clear structure."
    )

    def __init__(self):
        self.provider = os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower()

        if self.provider == "groq":
            self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif self.provider == "openrouter":
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        elif self.provider == "google":
            self.base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            self.api_key = os.getenv("GOOGLE_API_KEY", "")
            self.model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        else:
            raise ValueError(f"Unsupported CLOUD_LLM_PROVIDER: {self.provider}. Use 'groq', 'openrouter', or 'google'.")

        if not self.api_key:
            raise ValueError(f"API key missing for provider '{self.provider}'. Set the corresponding env var.")

    def _headers(self) -> dict[str, str]:
        if self.provider == "google":
            return {"Content-Type": "application/json"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "research-assistant")
        return headers

    def _post_with_retry(self, url: str, payload: dict, timeout: int = 90, retries: int = 3) -> dict:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as e:
                if response.status_code in (429, 503):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited (attempt {attempt+1}), retrying in {wait}s…")
                    time.sleep(wait)
                    last_exc = e
                else:
                    raise
            except requests.RequestException as e:
                last_exc = e
                time.sleep(1)
        raise last_exc or RuntimeError("Max retries exceeded")

    def generate(self, prompt: str, max_tokens: int = 512, system: str | None = None) -> str:
        sys_msg = system or self.SYSTEM_PROMPT

        if self.provider == "google":
            payload = {
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "systemInstruction": {"parts": [{"text": sys_msg}]},
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            }
            model_candidates = [self.model, "gemini-2.0-flash-lite", "gemini-1.5-flash"]
            for model_name in model_candidates:
                url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
                try:
                    data = self._post_with_retry(url, payload)
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise
            return ""

        # OpenAI-compatible (Groq / OpenRouter)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "max_tokens": max_tokens,
        }
        url = f"{self.base_url}/chat/completions"
        data = self._post_with_retry(url, payload)
        return data["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Multi-turn chat with full message history."""
        if self.provider == "google":
            # Convert to Google format
            google_messages = []
            system_msg = None
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    role = "user" if m["role"] == "user" else "model"
                    google_messages.append({"role": role, "parts": [{"text": m["content"]}]})

            payload = {
                "contents": google_messages,
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            }
            if system_msg:
                payload["systemInstruction"] = {"parts": [{"text": system_msg}]}

            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            data = self._post_with_retry(url, payload)
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": max_tokens,
        }
        url = f"{self.base_url}/chat/completions"
        data = self._post_with_retry(url, payload)
        return data["choices"][0]["message"]["content"].strip()
