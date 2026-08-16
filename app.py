"""app.py — Hugging Face Gradio Space entrypoint for Research AI v3.1.

WHAT THIS FILE DOES
-------------------
Hugging Face Gradio Spaces require a file called `app.py` at the repository
root. When the Space starts, HF executes this file.

This file does NOT replace the Research AI UI with a Gradio chatbot.
Instead, it:

  1. Calls download_artifacts.py to fetch any large artifacts that cannot
     be committed to the repository (e.g. classifier.joblib at 608 MB).
  2. Adds src/ to PYTHONPATH so `import research_ai` resolves correctly.
  3. Launches the real FastAPI application via uvicorn programmatically,
     binding to 0.0.0.0 and the HF-provided PORT (default 7860).
  4. Wraps that server with a minimal Gradio shim so HF Space health checks
     pass — the existing HTML/CSS/JS frontend remains the primary UI.

WHY NOT `gr.Interface(...)` OR `gr.ChatInterface(...)`?
-------------------------------------------------------
The Research AI frontend (frontend/index.html) is a full custom SPA that
calls the FastAPI backend directly. Replacing it with a Gradio widget would
remove all Research AI functionality. We preserve it by running FastAPI
as the primary server and using Gradio only as the Space launcher shim.

CORS NOTE
---------
ALLOWED_ORIGINS defaults to "*" which is safe because the frontend is served
by the same FastAPI process at the same origin. In production you can restrict
it to your Space URL via the ALLOWED_ORIGINS environment variable / Secret.

STARTUP SEQUENCE
----------------
  1. download_artifacts()  — fetch missing large artifacts (fast if cached)
  2. uvicorn.run()         — starts FastAPI; blocks until the process ends

PORT
----
HF Spaces injects PORT as an environment variable. We read it here and also
forward it to uvicorn. The default is 7860, which is the HF Spaces standard.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PYTHONPATH — make `import research_ai` work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("research_ai.app")

# ---------------------------------------------------------------------------
# 2. Load .env if present (local dev / CI convenience)
# ---------------------------------------------------------------------------
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)  # don't override real HF Secrets
        logger.info("Loaded .env from %s", _env_path)
    except ImportError:
        # Manual parse fallback
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# 3. Runtime safety checks — emit clear errors for missing config
# ---------------------------------------------------------------------------

def _check_config() -> None:
    """Warn loudly about missing required config before the server starts."""
    backend = os.getenv("LLM_BACKEND", "cloud").lower()
    provider = os.getenv("CLOUD_LLM_PROVIDER", "").lower()

    if backend == "cloud":
        if provider in ("gemini", "google"):
            key = (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )
            if not key:
                logger.error(
                    "═══════════════════════════════════════════════════════\n"
                    "  GEMINI_API_KEY is not configured!\n"
                    "  The Research AI LLM features will not work.\n"
                    "  Fix: HF Space → Settings → Repository secrets\n"
                    "       Add secret: GEMINI_API_KEY = <your key>\n"
                    "  Get a free key at: https://aistudio.google.com/\n"
                    "═══════════════════════════════════════════════════════"
                )
        elif provider == "groq":
            if not os.getenv("GROQ_API_KEY", "").strip():
                logger.warning("GROQ_API_KEY is not set — Groq LLM will fail.")
        elif provider == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY", "").strip():
                logger.warning("OPENROUTER_API_KEY is not set — OpenRouter LLM will fail.")

    # Python execution should be off for public deployments
    if os.getenv("ENABLE_PYTHON_EXECUTION", "false").lower() == "true":
        logger.warning(
            "ENABLE_PYTHON_EXECUTION=true — sandboxed Python execution is ON. "
            "Disable this for public Hugging Face deployments."
        )


# ---------------------------------------------------------------------------
# 4. Artifact download — fetch large files from HF Dataset repo if missing
# ---------------------------------------------------------------------------

def _download_artifacts() -> None:
    """Run download_artifacts.py if large artifacts are missing."""
    downloader = _REPO_ROOT / "download_artifacts.py"
    if not downloader.exists():
        return
    artifacts_repo = os.getenv("HF_ARTIFACTS_REPO", "").strip()
    if not artifacts_repo:
        logger.info(
            "HF_ARTIFACTS_REPO not set — skipping artifact download. "
            "Ensure artifacts/ directory is pre-populated."
        )
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("download_artifacts", downloader)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.download_artifacts(artifacts_repo)
    except Exception as exc:
        logger.warning("Artifact download failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# 5. Launch FastAPI via uvicorn
# ---------------------------------------------------------------------------

def _launch() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))

    logger.info(
        "\n"
        "══════════════════════════════════════════════\n"
        "  Research AI Intelligence Platform v3.1\n"
        "══════════════════════════════════════════════\n"
        "  Backend  : %s\n"
        "  Provider : %s\n"
        "  Model    : %s\n"
        "  Host     : %s:%d\n"
        "  Exec     : Python execution = %s\n"
        "══════════════════════════════════════════════",
        os.getenv("LLM_BACKEND", "cloud"),
        os.getenv("CLOUD_LLM_PROVIDER", "gemini"),
        os.getenv("GEMINI_MODEL", os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")),
        host, port,
        os.getenv("ENABLE_PYTHON_EXECUTION", "false"),
    )

    uvicorn.run(
        "research_ai.api.main:app",
        host=host,
        port=port,
        # No --reload in production — it watches the filesystem and wastes RAM
        reload=False,
        # workers=1 keeps memory usage low on ZeroGPU free hardware
        workers=1,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Main / Hugging Face Spaces entrypoint
# ---------------------------------------------------------------------------

try:
    import spaces  # type: ignore
except ImportError:  # pragma: no cover - Spaces runtime only
    spaces = None


def _run_startup() -> None:
    """Run the platform startup exactly once, when the app is invoked."""
    _check_config()
    _download_artifacts()
    _launch()


if spaces is not None and hasattr(spaces, "GPU"):
    @spaces.GPU
    def app() -> None:
        _run_startup()
else:
    def app() -> None:
        _run_startup()


if __name__ == "__main__":
    app()
