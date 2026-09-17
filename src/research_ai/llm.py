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


class ModelRouter:
    """Central model-role selection for chat, planning, synthesis, and metadata tasks."""

    SIMPLE_MARKERS = ("what is", "define", "explain briefly", "hello", "hi", "thanks")
    COMPLEX_MARKERS = ("compare", "latest", "recent", "papers", "literature", "survey", "limitations", "recommend")
    TECH_MARKERS = ("code", "implementation", "algorithm", "pytorch", "tensorflow", "python")

    def select(self, query: str, task: str = "answer") -> str:
        q = (query or "").lower()
        if task in {"intent", "metadata", "planning"}:
            return os.getenv("FAST_MODEL", "gpt-oss:20b-cloud")
        if any(m in q for m in self.TECH_MARKERS):
            return os.getenv("TECHNICAL_MODEL", os.getenv("REASONING_MODEL", "gpt-oss:120b-cloud"))
        if any(m in q for m in self.COMPLEX_MARKERS):
            return os.getenv("REASONING_MODEL", os.getenv("PRIMARY_MODEL", "gpt-oss:120b-cloud"))
        if any(q.startswith(m) for m in self.SIMPLE_MARKERS) or len(q.split()) <= 8:
            return os.getenv("FAST_MODEL", "gpt-oss:20b-cloud")
        return os.getenv("PRIMARY_MODEL", "gpt-oss:120b-cloud")


class CloudLLMClient:
    """OpenAI-compatible client for Groq / OpenRouter / OmniRouter / Google Gemini / Ollama.

    OmniRouter is a local OpenAI-compatible proxy (http://localhost:20218/v1) that
    routes to many upstream providers (Gemini, Claude, GPT, etc.) automatically.
    Use CLOUD_LLM_PROVIDER=omnirouter to avoid external rate limits.

    Constructed lazily — safe to instantiate even before env vars are set.
    Raises ValueError only on the first actual API call when the key is missing.
    """

    SYSTEM_PROMPT = (
        "You are an expert AI research assistant for scientific analysis. "
        "Be accurate, concise, and explicit about evidence limitations."
    )

    # Ordered cascade: try these models in order when the primary fails with 404 or 503
    GOOGLE_MODEL_CASCADE = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

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
        elif self.provider == "omnirouter":
            # OmniRouter: local OpenAI-compatible proxy that routes to many upstream LLMs.
            # Endpoint defaults to localhost:20218/v1; uses auto/* smart-routing model IDs.
            self.base_url = os.getenv("OMNIROUTER_BASE_URL", "http://localhost:20218/v1")
            self.model = os.getenv("PRIMARY_MODEL", os.getenv("OMNIROUTER_MODEL", "auto/best-chat"))
            self._api_key_env = "OMNIROUTER_API_KEY"
        elif self.provider == "google":
            self.base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            # GEMINI_MODEL is the canonical env var; GOOGLE_MODEL kept for backward compat
            self.model = (
                os.getenv("GEMINI_MODEL")
                or os.getenv("GOOGLE_MODEL")
                or "gemini-3.6-flash"
            )
            self._api_key_env = "GEMINI_API_KEY"  # primary; fallback handled in .api_key
        elif self.provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            self.model = os.getenv("OLLAMA_MODEL", os.getenv("PRIMARY_MODEL", "gpt-oss:120b-cloud"))
            self._api_key_env = "OLLAMA_API_KEY"
        else:
            raise ValueError(f"Unsupported CLOUD_LLM_PROVIDER: '{self.provider}'. "
                             f"Choose from: groq, openrouter, omnirouter, google, gemini, ollama.")

    @property
    def api_key(self) -> str:
        if self.provider == "ollama":
            return os.getenv("OLLAMA_API_KEY", "").strip() or "ollama"
        if self.provider == "omnirouter":
            # OmniRouter requires an API key but it can be any string if running locally
            # with no auth configured — fall back to a placeholder so the app doesn't crash.
            key = os.getenv("OMNIROUTER_API_KEY", "").strip()
            return key if key else "omnirouter"
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
        if self.provider == "omnirouter":
            # OmniRouter accepts standard Bearer auth; no extra headers needed.
            return headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "research-ai")
        return headers

    def _ollama_endpoint(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions" if path == "chat" else f"{base}/{path}"
        if base == "https://ollama.com":
            return f"{base}/api/{path}"
        return f"{base}/api/{path}"

    def _post_with_retry(self, url: str, payload: dict, timeout: int = 30, retries: int = 2) -> dict:
        # Ollama runs locally — use a longer timeout and don't retry on timeouts
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

    def _model_candidates(self) -> list[str]:
        fallbacks = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
        return list(dict.fromkeys([self.model] + fallbacks))

    def _google_cascade(self, payload: dict, retries: int = 2) -> str:
        """Try each model in GOOGLE_MODEL_CASCADE, advancing on 404 or 503.

        Returns the first non-empty response text. Returns "" if all models fail.
        """
        # Always try the configured model first, then the rest of the cascade
        models = list(dict.fromkeys([self.model] + self.GOOGLE_MODEL_CASCADE))
        for model_name in models:
            url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
            try:
                data = self._post_with_retry(url, payload, retries=retries)
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                if text:
                    logger.debug("[LLM] model=%s success text_len=%d", model_name, len(text))
                    return text
                logger.warning("[LLM] model=%s returned empty body, trying next", model_name)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (404, 503):
                    logger.warning("[LLM] model=%s status=%d, trying next model", model_name, status)
                    continue
                raise
        logger.error("[LLM] All Gemini models in cascade exhausted")
        return ""

    def generate(self, prompt: str, max_tokens: int = 512, system: str | None = None, json_mode: bool = False) -> str:
        system_prompt = system or self.SYSTEM_PROMPT
        if self.provider == "google":
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            }
            if json_mode:
                payload["generationConfig"]["responseMimeType"] = "application/json"
            return self._google_cascade(payload)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        last_exc: Exception | None = None
        for model in self._model_candidates():
            try:
                if self.provider == "ollama" and not self.base_url.endswith("/v1"):
                    native_payload = {
                        "model": model.replace("-cloud", "") if self.base_url == "https://ollama.com" else model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.15},
                    }
                    if json_mode:
                        native_payload["format"] = "json"
                    data = self._post_with_retry(self._ollama_endpoint("chat"), native_payload)
                    self.model = model
                    return (data.get("message") or {}).get("content", "").strip()
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.15,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                data = self._post_with_retry(f"{self.base_url}/chat/completions", payload)
                self.model = model
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                logger.warning("[LLM] model=%s failed; trying fallback if configured: %s", model, exc)
        raise last_exc or RuntimeError("LLM request failed.")

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
            return self._google_cascade(payload)

        last_exc: Exception | None = None
        for model in self._model_candidates():
            try:
                if self.provider == "ollama" and not self.base_url.endswith("/v1"):
                    native_payload = {
                        "model": model.replace("-cloud", "") if self.base_url == "https://ollama.com" else model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.15},
                    }
                    data = self._post_with_retry(self._ollama_endpoint("chat"), native_payload)
                    self.model = model
                    return (data.get("message") or {}).get("content", "").strip()
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.15,
                    "max_tokens": max_tokens,
                }
                data = self._post_with_retry(f"{self.base_url}/chat/completions", payload)
                self.model = model
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                logger.warning("[LLM] model=%s failed; trying fallback if configured: %s", model, exc)
        raise last_exc or RuntimeError("LLM request failed.")
