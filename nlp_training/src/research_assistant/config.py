from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path
    artifacts_root: Path

    @property
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_root / "classification"

    @property
    def sim_dir(self) -> Path:
        return self.artifacts_root / "similarity"


@dataclass(frozen=True)
class TextConfig:
    min_token_len: int = 3
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    max_df: float = 0.9
    min_df: int = 5


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    sample_per_class: int = 20000
    target_domains: tuple[str, ...] = ("cs", "math", "physics", "q-bio")


@dataclass(frozen=True)
class LLMConfig:
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "cloud").strip().lower())
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower())
    # Groq
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    # OpenRouter
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    # Google
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))


# Singleton config instances
DEFAULT_PATHS = Paths(
    data_root=Path(os.getenv("DATA_ROOT", "data")),
    artifacts_root=Path(os.getenv("ARTIFACTS_ROOT", "artifacts")),
)
DEFAULT_TEXT_CONFIG = TextConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
