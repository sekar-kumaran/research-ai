from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "data")))
    artifacts_root: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "artifacts")))
    frontend_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "frontend"
    )

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_root / "classification"

    @property
    def similarity_dir(self) -> Path:
        return self.artifacts_root / "similarity"

    @property
    def clustering_dir(self) -> Path:
        return self.artifacts_root / "clustering"

    @property
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))


@dataclass(frozen=True)
class LLMSettings:
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "cloud").strip().lower())
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    # Canonical HF Secrets names — checked first, fall back to GOOGLE_* for compat
    gemini_api_key: str = field(default_factory=lambda: (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ))
    gemini_model: str = field(default_factory=lambda: (
        os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.5-flash"
    ))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

    @property
    def effective_provider(self) -> str:
        if self.backend == "ollama":
            return "ollama"
        return self.provider


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    default_top_k: int = 5
    max_top_k: int = 20
    keyword_weight: float = 0.25
    semantic_weight: float = 0.75


@dataclass(frozen=True)
class ExecutionSettings:
    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PYTHON_EXECUTION", "false").lower() == "true"
    )
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_TIMEOUT", "5")))
    max_code_chars: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_MAX_CHARS", "4000")))


@dataclass(frozen=True)
class ServerSettings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    # 7860 is the standard Hugging Face Spaces port; local dev typically uses 8000
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "7860")))
    allowed_origins: str = field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*"))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    server: ServerSettings = field(default_factory=ServerSettings)


def load_settings() -> Settings:
    return Settings()
