"""Cloud LLM client — lazy initialization, retry logic, multi-provider support."""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# Singleton cache: one client per (provider, model) pair, lazily built
_CLIENT_CACHE: dict[str, "CloudLLMClient"] = {}


def get_cloud_client() -> "CloudLLMClient":
    """Return a cached CloudLLMClient or raise ValueError if config is missing."""
    provider = os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower()
    if provider == "gemini":
        provider = "google"
    if provider not in _CLIENT_CACHE:
        _CLIENT_CACHE[provider] = CloudLLMClient()
    return _CLIENT_CACHE[provider]


class CloudLLMClient:
    """OpenAI-compatible client for Groq / OpenRouter / Google Gemini / Ollama.

    Constructed lazily — safe to instantiate even before env vars are set.
    Raises ValueError only on the first actual API call when the key is missing.
    """

    SYSTEM_PROMPT = (
        "You are an expert AI research assistant for scientific analysis. "
        "Be accurate, concise, and explicit about evidence limitations."
    )

    def __init__(self) -> None:
        self.provider = os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower()
        if self.provider == "gemini":
            self.provider = "google"

        if self.provider == "groq":
            self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self._api_key_env = "GROQ_API_KEY"
        elif self.provider == "openrouter":
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
            self._api_key_env = "OPENROUTER_API_KEY"
        elif self.provider == "google":
            self.base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            # GEMINI_MODEL is the canonical HF name; GOOGLE_MODEL kept for backward compat
            self.model = (
                os.getenv("GEMINI_MODEL")
                or os.getenv("GOOGLE_MODEL")
                or "gemini-3.5-flash"
            )
            self._api_key_env = "GEMINI_API_KEY"  # primary; fallback handled in .api_key
        elif self.provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            self._api_key_env = ""  # Ollama needs no key
        else:
            raise ValueError(f"Unsupported CLOUD_LLM_PROVIDER: '{self.provider}'. "
                             f"Choose from: groq, openrouter, google, gemini, ollama.")

    @property
    def api_key(self) -> str:
        if self.provider == "ollama":
            return "ollama"  # Ollama accepts any non-empty string
        if self.provider == "google":
            # Check GEMINI_API_KEY first (canonical HF Secrets name),
            # then fall back to GOOGLE_API_KEY for backward compatibility.
            key = (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. "
                    "Please set it as a Hugging Face Space Secret: "
                    "Settings → Repository secrets → New secret → Name: GEMINI_API_KEY"
                )
            return key
        key = os.getenv(self._api_key_env, "").strip()
        if not key:
            raise ValueError(
                f"Missing API key — set the {self._api_key_env} environment variable."
            )
        return key

    def _headers(self) -> dict[str, str]:
        if self.provider == "google":
            return {"Content-Type": "application/json"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "research-ai")
        return headers

    def _post_with_retry(self, url: str, payload: dict, timeout: int = 90, retries: int = 3) -> dict:
        # Ollama runs locally — use a shorter timeout and don't retry on timeouts
        # (retrying a timed-out Ollama request just queues more work and deadlocks the server)
        if self.provider == "ollama":
            timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
            retries = 1
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code in (429, 503):
                    wait = 2 ** attempt
                    logger.warning("LLM rate-limited/unavailable; retrying in %ss (attempt %d).", wait, attempt + 1)
                    time.sleep(wait)
                    continue
                raise
            except requests.Timeout:
                # Don't retry timeouts — they pile up and deadlock local servers
                raise
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1)
        raise last_exc or RuntimeError("LLM request failed after all retries.")

    def generate(self, prompt: str, max_tokens: int = 512, system: str | None = None) -> str:
        system_prompt = system or self.SYSTEM_PROMPT
        if self.provider == "google":
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            }
            for model_name in (self.model, "gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
                url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
                try:
                    data = self._post_with_retry(url, payload)
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        continue
                    raise
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "max_tokens": max_tokens,
        }
        data = self._post_with_retry(f"{self.base_url}/chat/completions", payload)
        return data["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list[dict], max_tokens: int = 512) -> str:
        if self.provider == "google":
            google_messages: list[dict] = []
            system_msg: str | None = None
            for message in messages:
                if message["role"] == "system":
                    system_msg = message["content"]
                else:
                    role = "user" if message["role"] == "user" else "model"
                    if google_messages and google_messages[-1]["role"] == role:
                        google_messages[-1]["parts"][0]["text"] += "\n\n" + message["content"]
                    else:
                        google_messages.append({"role": role, "parts": [{"text": message["content"]}]})
            payload: dict = {
                "contents": google_messages,
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            }
            if system_msg:
                payload["systemInstruction"] = {"parts": [{"text": system_msg}]}
            for model_name in (self.model, "gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
                url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
                try:
                    data = self._post_with_retry(url, payload)
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        continue
                    raise
            return ""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.15,
            "max_tokens": max_tokens,
        }
        data = self._post_with_retry(f"{self.base_url}/chat/completions", payload)
        return data["choices"][0]["message"]["content"].strip()
