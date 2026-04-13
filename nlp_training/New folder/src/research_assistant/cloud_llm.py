from __future__ import annotations

import os

import requests


class CloudLLMClient:
    """Minimal OpenAI-compatible client for free cloud LLM providers."""

    def __init__(self):
        self.provider = os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower()

        if self.provider == "groq":
            self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif self.provider == "openrouter":
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        elif self.provider == "google":
            self.base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            self.api_key = os.getenv("GOOGLE_API_KEY", "")
            self.model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        else:
            raise ValueError("Unsupported CLOUD_LLM_PROVIDER. Use 'groq', 'openrouter', or 'google'.")

        if not self.api_key:
            raise ValueError("Cloud LLM API key is missing. Set provider API key env var.")

    def _headers(self) -> dict[str, str]:
        if self.provider == "google":
            return {"Content-Type": "application/json"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "nlp-research-assistant")
        return headers

    def generate(self, prompt: str, max_tokens: int = 220) -> str:
        if self.provider == "google":
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": max_tokens,
                },
            }
            model_candidates = [
                self.model,
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
            ]

            last_error = None
            for model_name in model_candidates:
                url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
                response = requests.post(url, headers=self._headers(), json=payload, timeout=90)

                if response.status_code == 404:
                    last_error = response
                    continue

                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                text_chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
                return "\n".join([t for t in text_chunks if t]).strip()

            if last_error is not None:
                last_error.raise_for_status()
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful research paper assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=self._headers(), json=payload, timeout=90)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
