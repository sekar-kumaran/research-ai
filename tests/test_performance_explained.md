# test_performance.py Explained

Generated educational companion for `tests/test_performance.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`tests/test_performance.py` is a Python module in the Test layer: behavioral, safety, performance, and integration checks. It defines TestBM25Latency, TestRerankerLatency, TestChunkingLatency, TestSandboxLatency, TestEmbeddingCacheLatency, TestArxivNormalizationLatency and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Test layer: behavioral, safety, performance, and integration checks. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Test layer: behavioral, safety, performance, and integration checks.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `tests/test_performance.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[tests/test_performance.py]
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
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## Global Variables and Config

No major module-level variables are declared. This reduces hidden state and keeps imports lightweight.

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `TestBM25Latency`

- **Line:** 35
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_bm25_over_60_candidates` at line 36: method behavior is described by its body and name
- `test_bm25_construction_60_docs` at line 51: method behavior is described by its body and name

```python
class TestBM25Latency:
    def test_bm25_over_60_candidates(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [
            f"transformer attention mechanism neural network layer {i} " * 15
            for i in range(60)
        ]
        bm25 = _BM25(docs)

        start = time.perf_counter()
        _ = bm25.scores("transformer attention mechanism")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0, f"BM25 over 60 docs took {elapsed_ms:.1f}ms (budget: 5ms)"

    def test_bm25_construction_60_docs(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [f"scientific paper abstract about deep learning {i}" * 10 for i in range(60)]
        start = time.perf_counter()
        _ = _BM25(docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20.0, f"BM25 construction took {elapsed_ms:.1f}ms (budget: 20ms)"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestRerankerLatency`

- **Line:** 67
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_reranker_over_60_docs` at line 68: method behavior is described by its body and name

```python
class TestRerankerLatency:
    def test_reranker_over_60_docs(self):
        from research_ai.retrieval.rerankers.service import MetadataReranker

        reranker = MetadataReranker()
        docs = [
            {
                "title": f"Attention Mechanism Paper {i}",
                "abstract": "We propose a novel attention mechanism for transformers.",
                "score": 0.9 - i * 0.01,
            }
            for i in range(60)
        ]

        start = time.perf_counter()
        _ = reranker.rerank("attention transformer", docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0, f"Reranker over 60 docs took {elapsed_ms:.1f}ms (budget: 10ms)"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestChunkingLatency`

- **Line:** 93
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_chunking_5000_words` at line 94: method behavior is described by its body and name

```python
class TestChunkingLatency:
    def test_chunking_5000_words(self):
        from research_ai.retrieval.chunking import contextual_chunks

        text = " ".join([
            "The transformer architecture relies on attention mechanisms. "
            "BERT introduced bidirectional pre-training for language understanding. "
            "GPT-3 demonstrated few-shot capabilities with 175 billion parameters. "
        ] * 100)  # ~5000 words

        start = time.perf_counter()
        chunks = contextual_chunks(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, f"Chunking 5000 words took {elapsed_ms:.1f}ms (budget: 100ms)"
        assert len(chunks) > 0
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestSandboxLatency`

- **Line:** 116
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_validation_clean_code` at line 117: method behavior is described by its body and name
- `test_validation_rejected_code` at line 130: method behavior is described by its body and name

```python
class TestSandboxLatency:
    def test_validation_clean_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "\n".join([f"x_{i} = {i} * 2 + {i}" for i in range(50)])

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.ok
        assert elapsed_ms < 10.0, f"Sandbox validation took {elapsed_ms:.1f}ms (budget: 10ms)"

    def test_validation_rejected_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "import os\nos.system('rm -rf /')"

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert not result.ok
        assert elapsed_ms < 10.0, f"Sandbox rejection took {elapsed_ms:.1f}ms (budget: 10ms)"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestEmbeddingCacheLatency`

- **Line:** 149
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_cache_hit_is_fast` at line 150: A cache hit should return in microseconds, not milliseconds.

```python
class TestEmbeddingCacheLatency:
    def test_cache_hit_is_fast(self):
        """A cache hit should return in microseconds, not milliseconds."""
        from research_ai.retrieval.embeddings.service import EmbeddingService
        import hashlib
        from collections import OrderedDict

        svc = EmbeddingService("all-MiniLM-L6-v2")
        # Pre-populate cache with a known key
        text = "attention is all you need"
        key = hashlib.sha256(f"all-MiniLM-L6-v2:{text}".encode()).hexdigest()
        vec = np.random.rand(1, 384).astype("float32")
        svc._cache[key] = vec

        start = time.perf_counter()
        result = svc._cache.get(key)
        svc._cache.move_to_end(key)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        assert result is not None
        assert elapsed_us < 100.0, f"Cache hit took {elapsed_us:.1f}µs (budget: 100µs)"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestArxivNormalizationLatency`

- **Line:** 177
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_normalization_is_microseconds` at line 178: method behavior is described by its body and name

```python
class TestArxivNormalizationLatency:
    def test_normalization_is_microseconds(self):
        from research_ai.research.paper_ingestion.service import PaperChatService

        test_ids = [
            "2301.04567v2",
            "arxiv:2301.04567",
            "https://arxiv.org/abs/2301.04567v2",
            "https://arxiv.org/pdf/2301.04567.pdf",
        ]

        start = time.perf_counter()
        for _ in range(1000):
            for raw_id in test_ids:
                PaperChatService.normalize_arxiv_id(raw_id)
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / (1000 * len(test_ids))

        assert elapsed_us < 100.0, f"Normalization took {elapsed_us:.1f}µs avg (budget: 100µs)"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `TestBM25Latency` Methods

#### `TestBM25Latency.test_bm25_over_60_candidates`

- **Line:** 36
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_bm25_over_60_candidates(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [
            f"transformer attention mechanism neural network layer {i} " * 15
            for i in range(60)
        ]
        bm25 = _BM25(docs)

        start = time.perf_counter()
        _ = bm25.scores("transformer attention mechanism")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0, f"BM25 over 60 docs took {elapsed_ms:.1f}ms (budget: 5ms)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Latency.test_bm25_construction_60_docs`

- **Line:** 51
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_bm25_construction_60_docs(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [f"scientific paper abstract about deep learning {i}" * 10 for i in range(60)]
        start = time.perf_counter()
        _ = _BM25(docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20.0, f"BM25 construction took {elapsed_ms:.1f}ms (budget: 20ms)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestRerankerLatency` Methods

#### `TestRerankerLatency.test_reranker_over_60_docs`

- **Line:** 68
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_reranker_over_60_docs(self):
        from research_ai.retrieval.rerankers.service import MetadataReranker

        reranker = MetadataReranker()
        docs = [
            {
                "title": f"Attention Mechanism Paper {i}",
                "abstract": "We propose a novel attention mechanism for transformers.",
                "score": 0.9 - i * 0.01,
            }
            for i in range(60)
        ]

        start = time.perf_counter()
        _ = reranker.rerank("attention transformer", docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0, f"Reranker over 60 docs took {elapsed_ms:.1f}ms (budget: 10ms)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestChunkingLatency` Methods

#### `TestChunkingLatency.test_chunking_5000_words`

- **Line:** 94
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_chunking_5000_words(self):
        from research_ai.retrieval.chunking import contextual_chunks

        text = " ".join([
            "The transformer architecture relies on attention mechanisms. "
            "BERT introduced bidirectional pre-training for language understanding. "
            "GPT-3 demonstrated few-shot capabilities with 175 billion parameters. "
        ] * 100)  # ~5000 words

        start = time.perf_counter()
        chunks = contextual_chunks(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, f"Chunking 5000 words took {elapsed_ms:.1f}ms (budget: 100ms)"
        assert len(chunks) > 0
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestSandboxLatency` Methods

#### `TestSandboxLatency.test_validation_clean_code`

- **Line:** 117
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_validation_clean_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "\n".join([f"x_{i} = {i} * 2 + {i}" for i in range(50)])

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.ok
        assert elapsed_ms < 10.0, f"Sandbox validation took {elapsed_ms:.1f}ms (budget: 10ms)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSandboxLatency.test_validation_rejected_code`

- **Line:** 130
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_validation_rejected_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "import os\nos.system('rm -rf /')"

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert not result.ok
        assert elapsed_ms < 10.0, f"Sandbox rejection took {elapsed_ms:.1f}ms (budget: 10ms)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestEmbeddingCacheLatency` Methods

#### `TestEmbeddingCacheLatency.test_cache_hit_is_fast`

- **Line:** 150
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** A cache hit should return in microseconds, not milliseconds.

```python
    def test_cache_hit_is_fast(self):
        """A cache hit should return in microseconds, not milliseconds."""
        from research_ai.retrieval.embeddings.service import EmbeddingService
        import hashlib
        from collections import OrderedDict

        svc = EmbeddingService("all-MiniLM-L6-v2")
        # Pre-populate cache with a known key
        text = "attention is all you need"
        key = hashlib.sha256(f"all-MiniLM-L6-v2:{text}".encode()).hexdigest()
        vec = np.random.rand(1, 384).astype("float32")
        svc._cache[key] = vec

        start = time.perf_counter()
        result = svc._cache.get(key)
        svc._cache.move_to_end(key)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        assert result is not None
        assert elapsed_us < 100.0, f"Cache hit took {elapsed_us:.1f}µs (budget: 100µs)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestArxivNormalizationLatency` Methods

#### `TestArxivNormalizationLatency.test_normalization_is_microseconds`

- **Line:** 178
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_normalization_is_microseconds(self):
        from research_ai.research.paper_ingestion.service import PaperChatService

        test_ids = [
            "2301.04567v2",
            "arxiv:2301.04567",
            "https://arxiv.org/abs/2301.04567v2",
            "https://arxiv.org/pdf/2301.04567.pdf",
        ]

        start = time.perf_counter()
        for _ in range(1000):
            for raw_id in test_ids:
                PaperChatService.normalize_arxiv_id(raw_id)
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / (1000 * len(test_ids))

        assert elapsed_us < 100.0, f"Normalization took {elapsed_us:.1f}µs avg (budget: 100µs)"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

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

- Touches files or paths. Validate filenames, restrict upload size/type, and prevent traversal.
- Deals with execution or subprocesses. Maintain AST validation, isolated mode, timeouts, and least privilege.
- Performs network I/O. Use timeouts, validate responses, and keep private services such as Ollama off the public internet.

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

- `tests/test_performance.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `tests/test_performance.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Performance benchmark tests — latency budgets for critical pipeline stages.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
These tests measure real latency and fail if a component exceeds its budget.
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
They require actual artifacts and model files, so they are marked:
# L0005: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
  @pytest.mark.requires_artifacts
# L0006: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
  @pytest.mark.slow
# L0007: Blank line that visually separates logical sections and improves readability.

# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Run with: pytest tests/test_performance.py -m "requires_artifacts" -v
# L0009: Blank line that visually separates logical sections and improves readability.

# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
LATENCY BUDGETS (CPU, single-threaded)
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
---------------------------------------
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
BM25 over 60 candidates:     < 5ms
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Reranker over 60 candidates: < 2ms
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Text chunking (5000 words):  < 100ms
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Sandbox validation:          < 10ms
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Embedding cache hit:         < 1ms
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
ArXiv ID normalization:      < 0.1ms
# L0018: Blank line that visually separates logical sections and improves readability.

# L0019: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
These are achievable on a modern laptop CPU without GPU.
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Adjust if running on lower-spec hardware.
# L0021: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0022: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0023: Blank line that visually separates logical sections and improves readability.

# L0024: Imports a dependency, type, or project module needed by later code in this file.
import time
# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0027: Imports a dependency, type, or project module needed by later code in this file.
import pytest
# L0028: Blank line that visually separates logical sections and improves readability.

# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0031: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# BM25 latency
# L0032: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0033: Blank line that visually separates logical sections and improves readability.

# L0034: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0035: Defines a class that groups related state and behavior behind a reusable interface.
class TestBM25Latency:
# L0036: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_bm25_over_60_candidates(self):
# L0037: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.hybrid_search.service import _BM25
# L0038: Blank line that visually separates logical sections and improves readability.

# L0039: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            f"transformer attention mechanism neural network layer {i} " * 15
# L0041: Iterates over data, retry attempts, files, results, or workflow steps.
            for i in range(60)
# L0042: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = _BM25(docs)
# L0044: Blank line that visually separates logical sections and improves readability.

# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
        _ = bm25.scores("transformer attention mechanism")
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0048: Blank line that visually separates logical sections and improves readability.

# L0049: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 5.0, f"BM25 over 60 docs took {elapsed_ms:.1f}ms (budget: 5ms)"
# L0050: Blank line that visually separates logical sections and improves readability.

# L0051: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_bm25_construction_60_docs(self):
# L0052: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.hybrid_search.service import _BM25
# L0053: Blank line that visually separates logical sections and improves readability.

# L0054: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [f"scientific paper abstract about deep learning {i}" * 10 for i in range(60)]
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0056: Assigns or updates a value used later in the workflow; check mutability and data shape.
        _ = _BM25(docs)
# L0057: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0058: Blank line that visually separates logical sections and improves readability.

# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 20.0, f"BM25 construction took {elapsed_ms:.1f}ms (budget: 20ms)"
# L0060: Blank line that visually separates logical sections and improves readability.

# L0061: Blank line that visually separates logical sections and improves readability.

# L0062: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0063: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# MetadataReranker latency
# L0064: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0065: Blank line that visually separates logical sections and improves readability.

# L0066: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0067: Defines a class that groups related state and behavior behind a reusable interface.
class TestRerankerLatency:
# L0068: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_reranker_over_60_docs(self):
# L0069: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.rerankers.service import MetadataReranker
# L0070: Blank line that visually separates logical sections and improves readability.

# L0071: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reranker = MetadataReranker()
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "title": f"Attention Mechanism Paper {i}",
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "abstract": "We propose a novel attention mechanism for transformers.",
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "score": 0.9 - i * 0.01,
# L0077: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0078: Iterates over data, retry attempts, files, results, or workflow steps.
            for i in range(60)
# L0079: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0080: Blank line that visually separates logical sections and improves readability.

# L0081: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0082: Assigns or updates a value used later in the workflow; check mutability and data shape.
        _ = reranker.rerank("attention transformer", docs)
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0084: Blank line that visually separates logical sections and improves readability.

# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 10.0, f"Reranker over 60 docs took {elapsed_ms:.1f}ms (budget: 10ms)"
# L0086: Blank line that visually separates logical sections and improves readability.

# L0087: Blank line that visually separates logical sections and improves readability.

# L0088: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0089: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Chunking latency
# L0090: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0091: Blank line that visually separates logical sections and improves readability.

# L0092: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0093: Defines a class that groups related state and behavior behind a reusable interface.
class TestChunkingLatency:
# L0094: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_chunking_5000_words(self):
# L0095: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.chunking import contextual_chunks
# L0096: Blank line that visually separates logical sections and improves readability.

# L0097: Assigns or updates a value used later in the workflow; check mutability and data shape.
        text = " ".join([
# L0098: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "The transformer architecture relies on attention mechanisms. "
# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "BERT introduced bidirectional pre-training for language understanding. "
# L0100: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "GPT-3 demonstrated few-shot capabilities with 175 billion parameters. "
# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ] * 100)  # ~5000 words
# L0102: Blank line that visually separates logical sections and improves readability.

# L0103: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0104: Assigns or updates a value used later in the workflow; check mutability and data shape.
        chunks = contextual_chunks(text)
# L0105: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0106: Blank line that visually separates logical sections and improves readability.

# L0107: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 100.0, f"Chunking 5000 words took {elapsed_ms:.1f}ms (budget: 100ms)"
# L0108: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert len(chunks) > 0
# L0109: Blank line that visually separates logical sections and improves readability.

# L0110: Blank line that visually separates logical sections and improves readability.

# L0111: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0112: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Sandbox validation latency
# L0113: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0114: Blank line that visually separates logical sections and improves readability.

# L0115: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0116: Defines a class that groups related state and behavior behind a reusable interface.
class TestSandboxLatency:
# L0117: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_validation_clean_code(self):
# L0118: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.execution.sandbox.service import SandboxValidator
# L0119: Blank line that visually separates logical sections and improves readability.

# L0120: Assigns or updates a value used later in the workflow; check mutability and data shape.
        validator = SandboxValidator()
# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
        code = "\n".join([f"x_{i} = {i} * 2 + {i}" for i in range(50)])
# L0122: Blank line that visually separates logical sections and improves readability.

# L0123: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0124: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = validator.validate(code)
# L0125: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0126: Blank line that visually separates logical sections and improves readability.

# L0127: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert result.ok
# L0128: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 10.0, f"Sandbox validation took {elapsed_ms:.1f}ms (budget: 10ms)"
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_validation_rejected_code(self):
# L0131: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.execution.sandbox.service import SandboxValidator
# L0132: Blank line that visually separates logical sections and improves readability.

# L0133: Assigns or updates a value used later in the workflow; check mutability and data shape.
        validator = SandboxValidator()
# L0134: Assigns or updates a value used later in the workflow; check mutability and data shape.
        code = "import os\nos.system('rm -rf /')"
# L0135: Blank line that visually separates logical sections and improves readability.

# L0136: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = validator.validate(code)
# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_ms = (time.perf_counter() - start) * 1000
# L0139: Blank line that visually separates logical sections and improves readability.

# L0140: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert not result.ok
# L0141: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_ms < 10.0, f"Sandbox rejection took {elapsed_ms:.1f}ms (budget: 10ms)"
# L0142: Blank line that visually separates logical sections and improves readability.

# L0143: Blank line that visually separates logical sections and improves readability.

# L0144: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0145: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Embedding cache hit latency
# L0146: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0147: Blank line that visually separates logical sections and improves readability.

# L0148: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0149: Defines a class that groups related state and behavior behind a reusable interface.
class TestEmbeddingCacheLatency:
# L0150: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cache_hit_is_fast(self):
# L0151: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """A cache hit should return in microseconds, not milliseconds."""
# L0152: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.embeddings.service import EmbeddingService
# L0153: Imports a dependency, type, or project module needed by later code in this file.
        import hashlib
# L0154: Imports a dependency, type, or project module needed by later code in this file.
        from collections import OrderedDict
# L0155: Blank line that visually separates logical sections and improves readability.

# L0156: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc = EmbeddingService("all-MiniLM-L6-v2")
# L0157: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Pre-populate cache with a known key
# L0158: Assigns or updates a value used later in the workflow; check mutability and data shape.
        text = "attention is all you need"
# L0159: Assigns or updates a value used later in the workflow; check mutability and data shape.
        key = hashlib.sha256(f"all-MiniLM-L6-v2:{text}".encode()).hexdigest()
# L0160: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vec = np.random.rand(1, 384).astype("float32")
# L0161: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc._cache[key] = vec
# L0162: Blank line that visually separates logical sections and improves readability.

# L0163: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0164: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = svc._cache.get(key)
# L0165: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        svc._cache.move_to_end(key)
# L0166: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_us = (time.perf_counter() - start) * 1_000_000
# L0167: Blank line that visually separates logical sections and improves readability.

# L0168: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert result is not None
# L0169: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_us < 100.0, f"Cache hit took {elapsed_us:.1f}µs (budget: 100µs)"
# L0170: Blank line that visually separates logical sections and improves readability.

# L0171: Blank line that visually separates logical sections and improves readability.

# L0172: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0173: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ArXiv ID normalization latency
# L0174: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0175: Blank line that visually separates logical sections and improves readability.

# L0176: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.mark.slow
# L0177: Defines a class that groups related state and behavior behind a reusable interface.
class TestArxivNormalizationLatency:
# L0178: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_normalization_is_microseconds(self):
# L0179: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.research.paper_ingestion.service import PaperChatService
# L0180: Blank line that visually separates logical sections and improves readability.

# L0181: Assigns or updates a value used later in the workflow; check mutability and data shape.
        test_ids = [
# L0182: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "2301.04567v2",
# L0183: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "arxiv:2301.04567",
# L0184: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "https://arxiv.org/abs/2301.04567v2",
# L0185: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "https://arxiv.org/pdf/2301.04567.pdf",
# L0186: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0187: Blank line that visually separates logical sections and improves readability.

# L0188: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = time.perf_counter()
# L0189: Iterates over data, retry attempts, files, results, or workflow steps.
        for _ in range(1000):
# L0190: Iterates over data, retry attempts, files, results, or workflow steps.
            for raw_id in test_ids:
# L0191: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                PaperChatService.normalize_arxiv_id(raw_id)
# L0192: Assigns or updates a value used later in the workflow; check mutability and data shape.
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / (1000 * len(test_ids))
# L0193: Blank line that visually separates logical sections and improves readability.

# L0194: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert elapsed_us < 100.0, f"Normalization took {elapsed_us:.1f}µs avg (budget: 100µs)"
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Performance benchmark tests — latency budgets for critical pipeline stages.

These tests measure real latency and fail if a component exceeds its budget.
They require actual artifacts and model files, so they are marked:
  @pytest.mark.requires_artifacts
  @pytest.mark.slow

Run with: pytest tests/test_performance.py -m "requires_artifacts" -v

LATENCY BUDGETS (CPU, single-threaded)
---------------------------------------
BM25 over 60 candidates:     < 5ms
Reranker over 60 candidates: < 2ms
Text chunking (5000 words):  < 100ms
Sandbox validation:          < 10ms
Embedding cache hit:         < 1ms
ArXiv ID normalization:      < 0.1ms

These are achievable on a modern laptop CPU without GPU.
Adjust if running on lower-spec hardware.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BM25 latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBM25Latency:
    def test_bm25_over_60_candidates(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [
            f"transformer attention mechanism neural network layer {i} " * 15
            for i in range(60)
        ]
        bm25 = _BM25(docs)

        start = time.perf_counter()
        _ = bm25.scores("transformer attention mechanism")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0, f"BM25 over 60 docs took {elapsed_ms:.1f}ms (budget: 5ms)"

    def test_bm25_construction_60_docs(self):
        from research_ai.retrieval.hybrid_search.service import _BM25

        docs = [f"scientific paper abstract about deep learning {i}" * 10 for i in range(60)]
        start = time.perf_counter()
        _ = _BM25(docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20.0, f"BM25 construction took {elapsed_ms:.1f}ms (budget: 20ms)"


# ---------------------------------------------------------------------------
# MetadataReranker latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRerankerLatency:
    def test_reranker_over_60_docs(self):
        from research_ai.retrieval.rerankers.service import MetadataReranker

        reranker = MetadataReranker()
        docs = [
            {
                "title": f"Attention Mechanism Paper {i}",
                "abstract": "We propose a novel attention mechanism for transformers.",
                "score": 0.9 - i * 0.01,
            }
            for i in range(60)
        ]

        start = time.perf_counter()
        _ = reranker.rerank("attention transformer", docs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0, f"Reranker over 60 docs took {elapsed_ms:.1f}ms (budget: 10ms)"


# ---------------------------------------------------------------------------
# Chunking latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestChunkingLatency:
    def test_chunking_5000_words(self):
        from research_ai.retrieval.chunking import contextual_chunks

        text = " ".join([
            "The transformer architecture relies on attention mechanisms. "
            "BERT introduced bidirectional pre-training for language understanding. "
            "GPT-3 demonstrated few-shot capabilities with 175 billion parameters. "
        ] * 100)  # ~5000 words

        start = time.perf_counter()
        chunks = contextual_chunks(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, f"Chunking 5000 words took {elapsed_ms:.1f}ms (budget: 100ms)"
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Sandbox validation latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSandboxLatency:
    def test_validation_clean_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "\n".join([f"x_{i} = {i} * 2 + {i}" for i in range(50)])

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.ok
        assert elapsed_ms < 10.0, f"Sandbox validation took {elapsed_ms:.1f}ms (budget: 10ms)"

    def test_validation_rejected_code(self):
        from research_ai.execution.sandbox.service import SandboxValidator

        validator = SandboxValidator()
        code = "import os\nos.system('rm -rf /')"

        start = time.perf_counter()
        result = validator.validate(code)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert not result.ok
        assert elapsed_ms < 10.0, f"Sandbox rejection took {elapsed_ms:.1f}ms (budget: 10ms)"


# ---------------------------------------------------------------------------
# Embedding cache hit latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEmbeddingCacheLatency:
    def test_cache_hit_is_fast(self):
        """A cache hit should return in microseconds, not milliseconds."""
        from research_ai.retrieval.embeddings.service import EmbeddingService
        import hashlib
        from collections import OrderedDict

        svc = EmbeddingService("all-MiniLM-L6-v2")
        # Pre-populate cache with a known key
        text = "attention is all you need"
        key = hashlib.sha256(f"all-MiniLM-L6-v2:{text}".encode()).hexdigest()
        vec = np.random.rand(1, 384).astype("float32")
        svc._cache[key] = vec

        start = time.perf_counter()
        result = svc._cache.get(key)
        svc._cache.move_to_end(key)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        assert result is not None
        assert elapsed_us < 100.0, f"Cache hit took {elapsed_us:.1f}µs (budget: 100µs)"


# ---------------------------------------------------------------------------
# ArXiv ID normalization latency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestArxivNormalizationLatency:
    def test_normalization_is_microseconds(self):
        from research_ai.research.paper_ingestion.service import PaperChatService

        test_ids = [
            "2301.04567v2",
            "arxiv:2301.04567",
            "https://arxiv.org/abs/2301.04567v2",
            "https://arxiv.org/pdf/2301.04567.pdf",
        ]

        start = time.perf_counter()
        for _ in range(1000):
            for raw_id in test_ids:
                PaperChatService.normalize_arxiv_id(raw_id)
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / (1000 * len(test_ids))

        assert elapsed_us < 100.0, f"Normalization took {elapsed_us:.1f}µs avg (budget: 100µs)"
```
