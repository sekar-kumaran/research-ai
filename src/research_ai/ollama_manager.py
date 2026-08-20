"""OllamaModelManager — discovers, profiles, and routes queries to local Ollama models.

ARCHITECTURE
------------
This module solves a core UX problem: instead of always using the same local
model for every task, the platform intelligently routes queries to the most
appropriate model based on task complexity. This mirrors how production systems
like ChatGPT internally use GPT-4o-mini for simple tasks and GPT-4o for complex
reasoning — the user never needs to know which model is running.

ROUTING STRATEGY
----------------
Models are grouped into 3 speed tiers:

  Tier 1 (<4B params) — fastest, <5s response
    Best for: greetings, classification, simple search planning
    Examples: qwen2.5:3b, phi3:mini, llama3.2:3b

  Tier 2 (4–10B params) — balanced, 5–30s response
    Best for: research analysis, RAG synthesis, paper Q&A
    Examples: qwen2.5:7b, mistral:7b, llama3.1:8b

  Tier 3 (>10B params) — most capable, 30–120s response
    Best for: deep reasoning, multi-step analysis, complex synthesis
    Examples: qwen2.5:14b, deepseek-r1:14b, llama3:70b

Each task type maps to a minimum tier. The manager finds the cheapest (fastest)
model that meets that minimum requirement — maximizing speed while ensuring quality.

FALLBACK CHAIN
--------------
If no model at the required tier is available, the manager falls back to the
best (highest-tier) model it has. This ensures the system always responds,
even with only a small model installed.

USAGE
-----
The manager is initialized in platform.py and passed wherever model selection
is needed. Call discover() once at startup, then select_model(task_type) at
request time. The result is a model name string that CloudLLMClient accepts
when provider="ollama".
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speed-tier catalog
# Keys are lowercase model names (with or without tag). Values are tier 1–3.
# The lookup supports prefix matching so "qwen2.5:7b-instruct" → "qwen2.5:7b".
# ---------------------------------------------------------------------------

_MODEL_TIER_MAP: dict[str, int] = {
    # ── Tier 1: fastest (<4B) ──────────────────────────────────────────────
    "phi3:mini":         1, "phi3.5:mini":        1, "phi":               1,
    "phi4-mini":         1,
    "gemma:2b":          1, "gemma2:2b":           1,
    "qwen2.5:0.5b":      1, "qwen2.5:1.5b":        1, "qwen2.5:3b":        1,
    "qwen2:1.5b":        1, "qwen2:0.5b":          1,
    "tinyllama":         1,
    "llama3.2:1b":       1, "llama3.2:3b":         1,
    "deepseek-r1:1.5b":  1,
    "smollm":            1, "smollm2":             1,
    # ── Tier 2: balanced (4–10B) ───────────────────────────────────────────
    "qwen2.5:7b":        2, "qwen2:7b":            2,
    "mistral:7b":        2, "mistral":             2,
    "mistral-nemo":      2,
    "llama3:8b":         2, "llama3.1:8b":         2, "llama3.2:8b":       2,
    "llama3.3":          2,
    "gemma:7b":          2, "gemma2:9b":           2,
    "deepseek-r1:7b":    2, "deepseek-r1:8b":      2,
    "deepseek-coder":    2,
    "codellama":         2,
    "phi4":              2,
    "neural-chat":       2,
    # ── Tier 3: most capable (>10B) ───────────────────────────────────────
    "llama3:70b":        3, "llama3.1:70b":        3, "llama3.3:70b":      3,
    "qwen2.5:14b":       3, "qwen2.5:32b":         3, "qwen2.5:72b":       3,
    "qwen2:72b":         3,
    "deepseek-r1:14b":   3, "deepseek-r1:32b":     3, "deepseek-r1:70b":   3,
    "mistral-large":     3,
    "mixtral":           3, "mixtral:8x7b":        3, "mixtral:8x22b":     3,
    "gemma2:27b":        3,
    "command-r":         3, "command-r-plus":      3,
}


# ---------------------------------------------------------------------------
# Task-type → minimum tier required
# Lower tier = simpler task (speed over quality).
# Higher tier = complex reasoning needed.
# ---------------------------------------------------------------------------

_TASK_MIN_TIER: dict[str, int] = {
    "conversation":       1,  # Small talk / greetings — any model fine
    "classification":     1,  # Single-label category prediction — small model fine
    "summarization":      1,  # Abstractive summarization — small model handles it
    "search":             1,  # Search query planning — small model fine
    "trend_analysis":     2,  # Multi-paper trend synthesis — needs decent model
    "research_analysis":  2,  # Full research analysis — needs decent model
    "citation_analysis":  2,  # Citation graph reasoning — needs decent model
    "paper_chat":         2,  # Paper Q&A against chunks — needs decent model
    "reasoning":          3,  # Multi-step logical reasoning — needs best model
    "deep_analysis":      3,  # Complex multi-source synthesis — needs best model
}


def _detect_tier(model_name: str) -> int:
    """Detect tier from model name using exact-then-prefix matching."""
    lower = model_name.lower()
    # Strip common suffixes that don't affect tier: -instruct, -chat, -q4, etc.
    cleaned = lower.split(":")[0]  # strip version tag like ":7b-instruct"

    if lower in _MODEL_TIER_MAP:
        return _MODEL_TIER_MAP[lower]

    # Try prefix match on the base name without tag
    for key, tier in _MODEL_TIER_MAP.items():
        base_key = key.split(":")[0]
        if cleaned.startswith(base_key):
            return tier

    # Try substring match as last resort
    for key, tier in _MODEL_TIER_MAP.items():
        if key.split(":")[0] in cleaned:
            return tier

    return 2  # Unknown models default to tier 2 (balanced)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Metadata for a discovered Ollama model."""
    name: str
    tier: int
    size_gb: float = 0.0

    @property
    def tier_label(self) -> str:
        return {1: "fast", 2: "balanced", 3: "powerful"}.get(self.tier, "unknown")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "size_gb": self.size_gb,
        }


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------

class OllamaModelManager:
    """Discovers available Ollama models and routes queries to the best one.

    Lifecycle:
        1. manager = OllamaModelManager()          # Safe — no network call
        2. manager.discover()                       # Queries Ollama /api/tags
        3. model = manager.select_model("search")  # Returns best model name
        4. Pass model name to CloudLLMClient        # Use it for generation

    The manager is optional: if Ollama is not running, discover() returns False
    and select_model() returns the configured default model from env vars.
    Everything degrades gracefully.
    """

    def __init__(self, base_url: str | None = None) -> None:
        # Strip /v1 suffix if present — Ollama management API lives at root
        raw = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw.rstrip("/").removesuffix("/v1")

        self._models: dict[str, ModelInfo] = {}
        self._default_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self._available: bool = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> bool:
        """Query Ollama for installed models. Returns True if Ollama is reachable.

        Side-effects: populates self._models with discovered ModelInfo objects.
        Safe to call multiple times — subsequent calls refresh the model list.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()

            self._models.clear()
            for m in data.get("models", []):
                name: str = m.get("name", "").strip()
                if not name:
                    continue
                size_bytes: int = m.get("size", 0)
                size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else 0.0
                self._models[name] = ModelInfo(
                    name=name,
                    tier=_detect_tier(name),
                    size_gb=size_gb,
                )

            self._available = bool(self._models)
            if self._available:
                logger.info(
                    "OllamaModelManager: %d model(s) discovered — %s",
                    len(self._models),
                    [m for m in self._models],
                )
            else:
                logger.info("OllamaModelManager: Ollama reachable but no models installed.")
            return self._available

        except requests.Timeout:
            logger.info("OllamaModelManager: Ollama connection timed out — local routing disabled.")
            self._available = False
            return False
        except Exception as exc:
            logger.info("OllamaModelManager: Ollama not reachable (%s).", type(exc).__name__)
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Model selection (the intelligence)
    # ------------------------------------------------------------------

    def select_model(self, task_type: str = "research_analysis") -> str:
        """Choose the best local model for a task type.

        Selection algorithm:
          1. Get minimum tier required for the task
          2. Check if the configured default model meets the requirement
          3. If yes, return default (user preference respected)
          4. Otherwise, find the lowest-tier model that meets the requirement
             (prefer speed: smallest sufficient model wins)
          5. If nothing meets the requirement, return the highest-tier available
          6. Ultimate fallback: return the env-configured default model name

        This ensures:
          - Simple tasks always use small/fast models (lower latency)
          - Complex tasks get the best available model (higher quality)
          - User's OLLAMA_MODEL preference is honored when sufficient
        """
        if not self._models:
            return self._default_model

        min_tier = _TASK_MIN_TIER.get(task_type, 2)

        # Step 1: Check if the user's default model meets the requirement
        if self._default_model in self._models:
            if self._models[self._default_model].tier >= min_tier:
                return self._default_model

        # Step 2: Find candidates at or above required tier
        candidates = [m for m in self._models.values() if m.tier >= min_tier]
        if candidates:
            # Among qualifying models, pick the smallest/fastest (lowest tier first,
            # then alphabetical as tiebreaker for determinism)
            best = min(candidates, key=lambda m: (m.tier, m.name))
            return best.name

        # Step 3: Nothing meets requirement — fall back to best available
        if self._models:
            best_available = max(self._models.values(), key=lambda m: (m.tier, m.name))
            logger.warning(
                "OllamaModelManager: No model meets tier %d for task '%s'. "
                "Using best available: %s (tier %d).",
                min_tier, task_type, best_available.name, best_available.tier,
            )
            return best_available.name

        return self._default_model

    # ------------------------------------------------------------------
    # Properties and helpers
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if Ollama is running and has at least one model installed."""
        return self._available

    @property
    def model_count(self) -> int:
        return len(self._models)

    def models_list(self) -> list[dict]:
        """Return all discovered models as serializable dicts for the API."""
        return [
            m.to_dict()
            for m in sorted(self._models.values(), key=lambda m: (m.tier, m.name))
        ]

    def health_check(self) -> dict:
        """Quick health probe — does not update the model list."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            online = resp.ok
        except Exception:
            online = False
        return {
            "online": online,
            "model_count": len(self._models),
            "models": [m.name for m in self._models.values()],
        }
