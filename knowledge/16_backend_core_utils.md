# Backend: Core Utils & Configuration

## File Paths: `src/research_ai/` (Root files)
## Status: Active / Stable

## Description
These files manage the global configuration, environment variables, and the abstraction layer for interacting with Large Language Models (LLMs).

## File Breakdown

### 1. `src/research_ai/configs/settings.py`
- **Class `Settings`**: Built on Pydantic `BaseSettings`.
- **Functions**:
  - Loads `.env` variables recursively. Defines typed configuration for `HOST`, `PORT`, `HF_SPACE_URL`, and LLM parameters. This ensures the app fails fast on startup if a critical environment variable is missing.

### 2. `src/research_ai/llm.py`
- **Class `LLMClient`**: The cloud LLM abstraction layer.
- **Functions**:
  - `generate(prompt, system)`: Routes the text generation request to the configured provider (e.g., OpenAI, Google Gemini, Anthropic) based on `settings.py`. Handles API keys and retries.

### 3. `src/research_ai/ollama_manager.py`
- **Class `OllamaModelManager`**: Local LLM manager.
- **Functions**:
  - Scans the local machine for a running Ollama daemon.
  - Queries `http://localhost:11434/api/tags` to populate the `ModelsListResponse` for the UI, allowing the user to select local, private models (like `llama3` or `qwen`) instead of cloud providers.

### 4. `src/research_ai/common/text.py`
- **Functions**:
  - `redact_secrets()`: Utility function to scrub API keys or passwords from text before writing it to log files, ensuring security compliance.
