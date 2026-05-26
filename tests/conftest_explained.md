# conftest.py Explained

Generated educational companion for `tests/conftest.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`tests/conftest.py` is a Python module in the Test layer: behavioral, safety, performance, and integration checks. It defines no classes and pytest_configure, sample_papers, mock_embedding_service, mock_faiss_store, mock_cloud_factory.

## Why This File Exists

This file isolates one responsibility in the codebase: Test layer: behavioral, safety, performance, and integration checks. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Test layer: behavioral, safety, performance, and integration checks.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `tests/conftest.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[tests/conftest.py]
  ThisFile --> Downstream[Downstream Service/UI/Result]
```

## Inputs and Outputs

- **Inputs:** function arguments, class constructor dependencies, HTTP payloads, environment variables, filesystem artifacts, DOM events, or test fixtures.
- **Outputs:** return values, dictionaries, Pydantic models, rendered DOM state, API responses, logs, process startup, assertions, or side effects.
- **Serialization:** this project uses JSON for APIs/LLM planning, parquet/joblib/faiss for ML artifacts, and HTML/CSS/JS for the browser surface.

## Imports Explained

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pandas` | Pandas provides dataframe operations for tabular metadata and tests. It is ergonomic for moderate in-memory workloads. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## Global Variables and Config

No major module-level variables are declared. This reduces hidden state and keeps imports lightweight.

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

### `pytest_configure`

- **Line:** 28
- **Kind:** synchronous function
- **Arguments:** config
- **Docstring:** No explicit docstring; infer behavior from call sites and body.

```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_artifacts: test requires built FAISS index and ML artifacts",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')",
    )
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.

### `sample_papers`

- **Line:** 44
- **Kind:** synchronous function
- **Arguments:** none
- **Docstring:** Five realistic paper dicts matching the RetrievedDocument schema.

```python
def sample_papers():
    """Five realistic paper dicts matching the RetrievedDocument schema."""
    return [
        {
            "paper_id": "2106.09685",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "abstract": "We propose LoRA, which freezes the pretrained model weights "
                        "and injects trainable rank decomposition matrices.",
            "authors": "Edward Hu, Yelong Shen",
            "category": "cs.LG",
            "year": "2021",
            "score": 0.92,
        },
        {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, "
                        "based solely on attention mechanisms.",
            "authors": "Vaswani, A et al.",
            "category": "cs.CL",
            "year": "2017",
            "score": 0.88,
        },
        {
            "paper_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, which stands for Bidirectional Encoder "
                        "Representations from Transformers.",
            "authors": "Devlin, J et al.",
            "category": "cs.CL",
            "year": "2018",
            "score": 0.85,
        },
        {
            "paper_id": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "abstract": "We train GPT-3, an autoregressive language model with 175 billion "
                        "parameters, and test its performance in the few-shot setting.",
            "authors": "Tom Brown et al.",
            "category": "cs.CL",
            "year": "2020",
            "score": 0.80,
        },
        {
            "paper_id": "2112.10752",
            "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
            "abstract": "We introduce latent diffusion models (LDMs) by applying diffusion "
                        "models in the latent space of powerful pretrained autoencoders.",
            "authors": "Rombach, R et al.",
            "category": "cs.CV",
            "year": "2021",
            "score": 0.78,
        },
    ]
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.

### `mock_embedding_service`

- **Line:** 105
- **Kind:** synchronous function
- **Arguments:** none
- **Docstring:** EmbeddingService that returns deterministic 384-dim L2-normalised vectors.

```python
def mock_embedding_service():
    """EmbeddingService that returns deterministic 384-dim L2-normalised vectors."""
    svc = MagicMock()
    dim = 384

    def encode(texts, batch_size=128):
        n = len(texts)
        # Fixed seed for determinism; different texts get different but stable vectors
        vecs = np.array(
            [np.sin(np.arange(dim, dtype="float32") + i) for i, _ in enumerate(texts)]
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    svc.encode.side_effect = encode
    svc.model_name = "all-MiniLM-L6-v2"
    return svc
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.

### `mock_faiss_store`

- **Line:** 125
- **Kind:** synchronous function
- **Arguments:** sample_papers
- **Docstring:** FaissVectorStore that returns sample_papers on search.

```python
def mock_faiss_store(sample_papers):
    """FaissVectorStore that returns sample_papers on search."""
    from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore, RetrievedDocument

    store = MagicMock(spec=FaissVectorStore)
    store.ready = True
    store.paper_count = len(sample_papers)
    store.metadata = pd.DataFrame(sample_papers)

    docs = [
        RetrievedDocument(
            paper_id=p["paper_id"],
            title=p["title"],
            abstract=p["abstract"],
            score=p["score"],
            authors=p["authors"],
            category=p["category"],
            year=p["year"],
        )
        for p in sample_papers
    ]
    store.search.return_value = docs
    return store
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.

### `mock_cloud_factory`

- **Line:** 151
- **Kind:** synchronous function
- **Arguments:** none
- **Docstring:** Cloud factory that returns a mock LLM producing predictable text.

```python
def mock_cloud_factory():
    """Cloud factory that returns a mock LLM producing predictable text."""
    mock_client = MagicMock()
    mock_client.generate.return_value = (
        "Based on the retrieved papers, the attention mechanism proposed in [1] "
        "became foundational. BERT [2] extended this with bidirectional pre-training. "
        "GPT-3 [3] demonstrated few-shot capabilities at scale."
    )
    mock_client.chat.return_value = mock_client.generate.return_value
    return lambda: mock_client
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.


## Class-by-Class Breakdown

No classes are defined. The module relies on functions, constants, imports, or package exports.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pandas` | Pandas provides dataframe operations for tabular metadata and tests. It is ergonomic for moderate in-memory workloads. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.

## Performance and Memory Notes

- Avoid eager loading of heavy ML models unless startup latency is acceptable.
- Cache expensive clients, tokenizers, vector stores, and embeddings carefully.
- Use float32 for embedding vectors because it halves memory compared with float64 and matches FAISS/neural inference expectations.
- Bound prompt length, uploaded content, result counts, and token budgets to control latency and memory.
- Watch copies of large metadata frames, embedding matrices, and file buffers.

## Scalability Notes

- In-memory state works for local demos but needs Redis/database/object storage for multi-worker cloud deployment.
- CPU/GPU inference should often be separated from the web API when traffic grows.
- Retrieval can start exact and move to approximate indexes as corpus size grows.
- Batch operations and cache repeated work to improve throughput.
- Add metrics for latency, errors, fallback frequency, retrieval hit rate, and token usage.

## Production Engineering Notes

- Keep interfaces stable because other files may import this module or depend on its response shape.
- Prefer typed/structured data over free-form strings at service boundaries.
- Log operational context without secrets or huge payloads.
- Make fallback behavior explicit so users get useful output even when LLMs or artifacts fail.
- Keep provider-specific logic behind adapters so Groq/OpenRouter/Google/Ollama can be swapped.

## Common Bugs and Failure Cases

- Missing `.env` values, model artifacts, or Ollama models can trigger degraded behavior.
- Type mismatches occur when LLM-generated tool arguments cross into strict Python code.
- Empty retrieval results must not become hallucinated answers.
- Network calls need timeouts and careful retry behavior.
- Frontend IDs/classes and API schemas are contracts; changing one side without the other breaks workflows.

## Security Considerations

- Deals with execution or subprocesses. Maintain AST validation, isolated mode, timeouts, and least privilege.

## Real Industry Usage

- This pattern appears in enterprise RAG assistants, scientific search tools, internal research copilots, and ML platform demos.
- The layered design mirrors production systems: API facade, orchestration, retrieval, evaluation, synthesis, UI, and deployment.
- Clear separation lets teams replace model providers, improve retrieval, harden security, or redesign UI independently.

## Optimization Opportunities

- Add tracing around each workflow step.
- Strengthen schema validation at boundaries.
- Persist conversation/session state outside process memory.
- Add load tests and adversarial tests for prompt injection, empty evidence, and large uploads.
- Consider approximate vector indexes, reranker models, or batching when corpus/traffic grows.

## How This Connects To Other Files

- `tests/conftest.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
- `src/research_ai/platform.py` is the backend composition root.
- `src/research_ai/api/main.py` exposes backend behavior over HTTP.
- Retrieval modules depend on artifacts under `artifacts/`.
- Frontend files depend on stable endpoint and DOM contracts.

## End-to-End Flow Summary

- A user/browser/test/startup event enters the system.
- The relevant layer validates or normalizes input.
- Retrieval, ML, orchestration, execution, or UI rendering happens.
- A structured result, visual state, or process side effect is produced.
- Fallbacks and tests keep behavior understandable when dependencies are unavailable.

## Interview Questions

1. What responsibility does this file own?
2. What inputs and outputs define its contract?
3. Which dependencies are expensive or operationally risky?
4. What breaks if this file changes shape?
5. How would you scale or test this behavior in production?

## Key Takeaways

- `tests/conftest.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Shared pytest configuration and fixtures.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
FIXTURE DESIGN
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
--------------
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
All fixtures that touch the ML models or FAISS index are designed to work
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
WITHOUT the actual artifacts present.  If tests need real artifacts, they
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
should be marked with @pytest.mark.requires_artifacts and skipped in CI.
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Fixtures:
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - mock_embedding_service: returns fixed 384-dim L2-normalised vectors
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - mock_faiss_store: in-memory store with 5 fake papers
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - mock_cloud_factory: returns a mock LLM client
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - sample_papers: list of 5 realistic paper dicts
# L0014: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0015: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Imports a dependency, type, or project module needed by later code in this file.
from unittest.mock import MagicMock
# L0018: Blank line that visually separates logical sections and improves readability.

# L0019: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0020: Imports a dependency, type, or project module needed by later code in this file.
import pandas as pd
# L0021: Imports a dependency, type, or project module needed by later code in this file.
import pytest
# L0022: Blank line that visually separates logical sections and improves readability.

# L0023: Blank line that visually separates logical sections and improves readability.

# L0024: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0025: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Markers
# L0026: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0027: Blank line that visually separates logical sections and improves readability.

# L0028: Defines a function or method; parameters are the input contract and the body implements the workflow.
def pytest_configure(config):
# L0029: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    config.addinivalue_line(
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "markers",
# L0031: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "requires_artifacts: test requires built FAISS index and ML artifacts",
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    config.addinivalue_line(
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "markers",
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "slow: marks tests as slow (deselect with -m 'not slow')",
# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0037: Blank line that visually separates logical sections and improves readability.

# L0038: Blank line that visually separates logical sections and improves readability.

# L0039: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0040: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Sample data
# L0041: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0042: Blank line that visually separates logical sections and improves readability.

# L0043: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.fixture
# L0044: Defines a function or method; parameters are the input contract and the body implements the workflow.
def sample_papers():
# L0045: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Five realistic paper dicts matching the RetrievedDocument schema."""
# L0046: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return [
# L0047: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        {
# L0048: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": "2106.09685",
# L0049: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": "We propose LoRA, which freezes the pretrained model weights "
# L0051: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "and injects trainable rank decomposition matrices.",
# L0052: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": "Edward Hu, Yelong Shen",
# L0053: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": "cs.LG",
# L0054: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": "2021",
# L0055: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": 0.92,
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        },
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        {
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": "1706.03762",
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": "Attention Is All You Need",
# L0060: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": "We propose a new simple network architecture, the Transformer, "
# L0061: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "based solely on attention mechanisms.",
# L0062: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": "Vaswani, A et al.",
# L0063: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": "cs.CL",
# L0064: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": "2017",
# L0065: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": 0.88,
# L0066: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        },
# L0067: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        {
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": "1810.04805",
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": "We introduce BERT, which stands for Bidirectional Encoder "
# L0071: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "Representations from Transformers.",
# L0072: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": "Devlin, J et al.",
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": "cs.CL",
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": "2018",
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": 0.85,
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        },
# L0077: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        {
# L0078: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": "2005.14165",
# L0079: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": "Language Models are Few-Shot Learners",
# L0080: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": "We train GPT-3, an autoregressive language model with 175 billion "
# L0081: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "parameters, and test its performance in the few-shot setting.",
# L0082: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": "Tom Brown et al.",
# L0083: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": "cs.CL",
# L0084: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": "2020",
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": 0.80,
# L0086: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        },
# L0087: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        {
# L0088: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": "2112.10752",
# L0089: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
# L0090: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": "We introduce latent diffusion models (LDMs) by applying diffusion "
# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "models in the latent space of powerful pretrained autoencoders.",
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": "Rombach, R et al.",
# L0093: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": "cs.CV",
# L0094: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": "2021",
# L0095: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": 0.78,
# L0096: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        },
# L0097: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ]
# L0098: Blank line that visually separates logical sections and improves readability.

# L0099: Blank line that visually separates logical sections and improves readability.

# L0100: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0101: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Mock services
# L0102: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0103: Blank line that visually separates logical sections and improves readability.

# L0104: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.fixture
# L0105: Defines a function or method; parameters are the input contract and the body implements the workflow.
def mock_embedding_service():
# L0106: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """EmbeddingService that returns deterministic 384-dim L2-normalised vectors."""
# L0107: Assigns or updates a value used later in the workflow; check mutability and data shape.
    svc = MagicMock()
# L0108: Assigns or updates a value used later in the workflow; check mutability and data shape.
    dim = 384
# L0109: Blank line that visually separates logical sections and improves readability.

# L0110: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def encode(texts, batch_size=128):
# L0111: Assigns or updates a value used later in the workflow; check mutability and data shape.
        n = len(texts)
# L0112: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Fixed seed for determinism; different texts get different but stable vectors
# L0113: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vecs = np.array(
# L0114: Assigns or updates a value used later in the workflow; check mutability and data shape.
            [np.sin(np.arange(dim, dtype="float32") + i) for i, _ in enumerate(texts)]
# L0115: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0116: Assigns or updates a value used later in the workflow; check mutability and data shape.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
# L0117: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return vecs / np.clip(norms, 1e-12, None)
# L0118: Blank line that visually separates logical sections and improves readability.

# L0119: Assigns or updates a value used later in the workflow; check mutability and data shape.
    svc.encode.side_effect = encode
# L0120: Assigns or updates a value used later in the workflow; check mutability and data shape.
    svc.model_name = "all-MiniLM-L6-v2"
# L0121: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return svc
# L0122: Blank line that visually separates logical sections and improves readability.

# L0123: Blank line that visually separates logical sections and improves readability.

# L0124: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.fixture
# L0125: Defines a function or method; parameters are the input contract and the body implements the workflow.
def mock_faiss_store(sample_papers):
# L0126: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """FaissVectorStore that returns sample_papers on search."""
# L0127: Imports a dependency, type, or project module needed by later code in this file.
    from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore, RetrievedDocument
# L0128: Blank line that visually separates logical sections and improves readability.

# L0129: Assigns or updates a value used later in the workflow; check mutability and data shape.
    store = MagicMock(spec=FaissVectorStore)
# L0130: Assigns or updates a value used later in the workflow; check mutability and data shape.
    store.ready = True
# L0131: Assigns or updates a value used later in the workflow; check mutability and data shape.
    store.paper_count = len(sample_papers)
# L0132: Assigns or updates a value used later in the workflow; check mutability and data shape.
    store.metadata = pd.DataFrame(sample_papers)
# L0133: Blank line that visually separates logical sections and improves readability.

# L0134: Assigns or updates a value used later in the workflow; check mutability and data shape.
    docs = [
# L0135: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        RetrievedDocument(
# L0136: Assigns or updates a value used later in the workflow; check mutability and data shape.
            paper_id=p["paper_id"],
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
            title=p["title"],
# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
            abstract=p["abstract"],
# L0139: Assigns or updates a value used later in the workflow; check mutability and data shape.
            score=p["score"],
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
            authors=p["authors"],
# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
            category=p["category"],
# L0142: Assigns or updates a value used later in the workflow; check mutability and data shape.
            year=p["year"],
# L0143: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0144: Iterates over data, retry attempts, files, results, or workflow steps.
        for p in sample_papers
# L0145: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ]
# L0146: Assigns or updates a value used later in the workflow; check mutability and data shape.
    store.search.return_value = docs
# L0147: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return store
# L0148: Blank line that visually separates logical sections and improves readability.

# L0149: Blank line that visually separates logical sections and improves readability.

# L0150: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.fixture
# L0151: Defines a function or method; parameters are the input contract and the body implements the workflow.
def mock_cloud_factory():
# L0152: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Cloud factory that returns a mock LLM producing predictable text."""
# L0153: Assigns or updates a value used later in the workflow; check mutability and data shape.
    mock_client = MagicMock()
# L0154: Assigns or updates a value used later in the workflow; check mutability and data shape.
    mock_client.generate.return_value = (
# L0155: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "Based on the retrieved papers, the attention mechanism proposed in [1] "
# L0156: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "became foundational. BERT [2] extended this with bidirectional pre-training. "
# L0157: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "GPT-3 [3] demonstrated few-shot capabilities at scale."
# L0158: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0159: Assigns or updates a value used later in the workflow; check mutability and data shape.
    mock_client.chat.return_value = mock_client.generate.return_value
# L0160: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return lambda: mock_client
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Shared pytest configuration and fixtures.

FIXTURE DESIGN
--------------
All fixtures that touch the ML models or FAISS index are designed to work
WITHOUT the actual artifacts present.  If tests need real artifacts, they
should be marked with @pytest.mark.requires_artifacts and skipped in CI.

Fixtures:
  - mock_embedding_service: returns fixed 384-dim L2-normalised vectors
  - mock_faiss_store: in-memory store with 5 fake papers
  - mock_cloud_factory: returns a mock LLM client
  - sample_papers: list of 5 realistic paper dicts
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_artifacts: test requires built FAISS index and ML artifacts",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')",
    )


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_papers():
    """Five realistic paper dicts matching the RetrievedDocument schema."""
    return [
        {
            "paper_id": "2106.09685",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "abstract": "We propose LoRA, which freezes the pretrained model weights "
                        "and injects trainable rank decomposition matrices.",
            "authors": "Edward Hu, Yelong Shen",
            "category": "cs.LG",
            "year": "2021",
            "score": 0.92,
        },
        {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, "
                        "based solely on attention mechanisms.",
            "authors": "Vaswani, A et al.",
            "category": "cs.CL",
            "year": "2017",
            "score": 0.88,
        },
        {
            "paper_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, which stands for Bidirectional Encoder "
                        "Representations from Transformers.",
            "authors": "Devlin, J et al.",
            "category": "cs.CL",
            "year": "2018",
            "score": 0.85,
        },
        {
            "paper_id": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "abstract": "We train GPT-3, an autoregressive language model with 175 billion "
                        "parameters, and test its performance in the few-shot setting.",
            "authors": "Tom Brown et al.",
            "category": "cs.CL",
            "year": "2020",
            "score": 0.80,
        },
        {
            "paper_id": "2112.10752",
            "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
            "abstract": "We introduce latent diffusion models (LDMs) by applying diffusion "
                        "models in the latent space of powerful pretrained autoencoders.",
            "authors": "Rombach, R et al.",
            "category": "cs.CV",
            "year": "2021",
            "score": 0.78,
        },
    ]


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_service():
    """EmbeddingService that returns deterministic 384-dim L2-normalised vectors."""
    svc = MagicMock()
    dim = 384

    def encode(texts, batch_size=128):
        n = len(texts)
        # Fixed seed for determinism; different texts get different but stable vectors
        vecs = np.array(
            [np.sin(np.arange(dim, dtype="float32") + i) for i, _ in enumerate(texts)]
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    svc.encode.side_effect = encode
    svc.model_name = "all-MiniLM-L6-v2"
    return svc


@pytest.fixture
def mock_faiss_store(sample_papers):
    """FaissVectorStore that returns sample_papers on search."""
    from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore, RetrievedDocument

    store = MagicMock(spec=FaissVectorStore)
    store.ready = True
    store.paper_count = len(sample_papers)
    store.metadata = pd.DataFrame(sample_papers)

    docs = [
        RetrievedDocument(
            paper_id=p["paper_id"],
            title=p["title"],
            abstract=p["abstract"],
            score=p["score"],
            authors=p["authors"],
            category=p["category"],
            year=p["year"],
        )
        for p in sample_papers
    ]
    store.search.return_value = docs
    return store


@pytest.fixture
def mock_cloud_factory():
    """Cloud factory that returns a mock LLM producing predictable text."""
    mock_client = MagicMock()
    mock_client.generate.return_value = (
        "Based on the retrieved papers, the attention mechanism proposed in [1] "
        "became foundational. BERT [2] extended this with bidirectional pre-training. "
        "GPT-3 [3] demonstrated few-shot capabilities at scale."
    )
    mock_client.chat.return_value = mock_client.generate.return_value
    return lambda: mock_client
```
