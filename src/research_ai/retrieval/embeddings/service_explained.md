# service.py Explained

Generated educational companion for `src/research_ai/retrieval/embeddings/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/retrieval/embeddings/service.py` is a Python module in the Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking. It defines EmbeddingService and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/retrieval/embeddings/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/retrieval/embeddings/service.py]
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
| `collections` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `hashlib` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 50 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |
| `_CACHE_MAX` | 55 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `EmbeddingService`

- **Line:** 58
- **Base classes:** `object`
- **Docstring:** Lazy SentenceTransformer embedding service with L2-normalised output.

Features:
- Model is loaded on first encode() call (no startup delay)
- Query-level LRU cache using OrderedDict (O(1) hit and eviction)
- Explicit batch_size parameter for GPU/CPU tuning
- read-only model_name property guards against accidental mutation
- warm_up() pre-loads the model at server startup to avoid first-call latency

**Methods:**
- `__init__` at line 69: method behavior is described by its body and name
- `model_name` at line 77: Read-only model name (set at construction time).
- `model` at line 82: Lazily loaded SentenceTransformer instance.
- `encode` at line 95: Encode texts into L2-normalised embedding vectors.

Single-text inputs use the LRU cache (typical search query path).
Multi-text inputs bypass the cache (typical batch indexing path).

Args:
    texts:      List of strings to encode.  Must be non-empty.
    batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
                128–256 on GPU.

Returns:
    np.ndarray of shape (len(texts), embedding_dim), dtype float32,
    with each row L2-normalised to unit length.
- `warm_up` at line 114: Pre-load the embedding model at server startup.

Without warm_up(), the first encode() call loads the model (1–3 seconds
on CPU), causing the first user request to time out or feel sluggish.
Call this from the FastAPI lifespan or after constructing the platform.
- `_encode_single_cached` at line 127: Encode a single text with LRU caching.
- `_encode_batch` at line 148: Encode a batch of texts and L2-normalise the output vectors.

L2 normalisation: v_norm = v / max(||v||, ε)
  - Ensures all vectors lie on the unit hypersphere.
  - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
  - ε = 1e-12 prevents division by zero for degenerate zero vectors.

```python
class EmbeddingService:
    """Lazy SentenceTransformer embedding service with L2-normalised output.

    Features:
    - Model is loaded on first encode() call (no startup delay)
    - Query-level LRU cache using OrderedDict (O(1) hit and eviction)
    - Explicit batch_size parameter for GPU/CPU tuning
    - read-only model_name property guards against accidental mutation
    - warm_up() pre-loads the model at server startup to avoid first-call latency
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        # OrderedDict as LRU: insertion order is maintained; popitem(last=False)
        # removes the oldest entry in O(1).  move_to_end() promotes on hit in O(1).
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def model_name(self) -> str:
        """Read-only model name (set at construction time)."""
        return self._model_name

    @property
    def model(self):
        """Lazily loaded SentenceTransformer instance."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding model loaded: %s (dim=%d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts into L2-normalised embedding vectors.

        Single-text inputs use the LRU cache (typical search query path).
        Multi-text inputs bypass the cache (typical batch indexing path).

        Args:
            texts:      List of strings to encode.  Must be non-empty.
            batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
                        128–256 on GPU.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
            with each row L2-normalised to unit length.
        """
        if len(texts) == 1:
            return self._encode_single_cached(texts[0], batch_size)
        return self._encode_batch(texts, batch_size)

    def warm_up(self) -> None:
        """Pre-load the embedding model at server startup.

        Without warm_up(), the first encode() call loads the model (1–3 seconds
        on CPU), causing the first user request to time out or feel sluggish.
        Call this from the FastAPI lifespan or after constructing the platform.
        """
        _ = self.model  # triggers lazy load via the property

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _encode_single_cached(self, text: str, batch_size: int) -> np.ndarray:
        """Encode a single text with LRU caching."""
        # SHA-256 over "model_name:text" ensures cache isolation across models
        # and strong collision resistance for adversarial inputs.
        key = hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()

        if key in self._cache:
            # Cache hit: promote to most-recently-used end (O(1))
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cache miss: encode and store
        vec = self._encode_batch([text], batch_size)
        self._cache[key] = vec

        # Evict oldest entry if over capacity (O(1) with OrderedDict)
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)  # removes the first (oldest) item

        return vec

    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode a batch of texts and L2-normalise the output vectors.

        L2 normalisation: v_norm = v / max(||v||, ε)
          - Ensures all vectors lie on the unit hypersphere.
          - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
          - ε = 1e-12 prevents division by zero for degenerate zero vectors.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `EmbeddingService` Methods

#### `EmbeddingService.__init__`

- **Line:** 69
- **Kind:** synchronous method
- **Arguments:** self, model_name
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        # OrderedDict as LRU: insertion order is maintained; popitem(last=False)
        # removes the oldest entry in O(1).  move_to_end() promotes on hit in O(1).
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService.model_name`

- **Line:** 77
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Read-only model name (set at construction time).

```python
    def model_name(self) -> str:
        """Read-only model name (set at construction time)."""
        return self._model_name
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService.model`

- **Line:** 82
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Lazily loaded SentenceTransformer instance.

```python
    def model(self):
        """Lazily loaded SentenceTransformer instance."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding model loaded: %s (dim=%d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService.encode`

- **Line:** 95
- **Kind:** synchronous method
- **Arguments:** self, texts, batch_size
- **Docstring:** Encode texts into L2-normalised embedding vectors.

Single-text inputs use the LRU cache (typical search query path).
Multi-text inputs bypass the cache (typical batch indexing path).

Args:
    texts:      List of strings to encode.  Must be non-empty.
    batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
                128–256 on GPU.

Returns:
    np.ndarray of shape (len(texts), embedding_dim), dtype float32,
    with each row L2-normalised to unit length.

```python
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts into L2-normalised embedding vectors.

        Single-text inputs use the LRU cache (typical search query path).
        Multi-text inputs bypass the cache (typical batch indexing path).

        Args:
            texts:      List of strings to encode.  Must be non-empty.
            batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
                        128–256 on GPU.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
            with each row L2-normalised to unit length.
        """
        if len(texts) == 1:
            return self._encode_single_cached(texts[0], batch_size)
        return self._encode_batch(texts, batch_size)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService.warm_up`

- **Line:** 114
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Pre-load the embedding model at server startup.

Without warm_up(), the first encode() call loads the model (1–3 seconds
on CPU), causing the first user request to time out or feel sluggish.
Call this from the FastAPI lifespan or after constructing the platform.

```python
    def warm_up(self) -> None:
        """Pre-load the embedding model at server startup.

        Without warm_up(), the first encode() call loads the model (1–3 seconds
        on CPU), causing the first user request to time out or feel sluggish.
        Call this from the FastAPI lifespan or after constructing the platform.
        """
        _ = self.model  # triggers lazy load via the property
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService._encode_single_cached`

- **Line:** 127
- **Kind:** synchronous method
- **Arguments:** self, text, batch_size
- **Docstring:** Encode a single text with LRU caching.

```python
    def _encode_single_cached(self, text: str, batch_size: int) -> np.ndarray:
        """Encode a single text with LRU caching."""
        # SHA-256 over "model_name:text" ensures cache isolation across models
        # and strong collision resistance for adversarial inputs.
        key = hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()

        if key in self._cache:
            # Cache hit: promote to most-recently-used end (O(1))
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cache miss: encode and store
        vec = self._encode_batch([text], batch_size)
        self._cache[key] = vec

        # Evict oldest entry if over capacity (O(1) with OrderedDict)
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)  # removes the first (oldest) item

        return vec
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `EmbeddingService._encode_batch`

- **Line:** 148
- **Kind:** synchronous method
- **Arguments:** self, texts, batch_size
- **Docstring:** Encode a batch of texts and L2-normalise the output vectors.

L2 normalisation: v_norm = v / max(||v||, ε)
  - Ensures all vectors lie on the unit hypersphere.
  - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
  - ε = 1e-12 prevents division by zero for degenerate zero vectors.

```python
    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode a batch of texts and L2-normalise the output vectors.

        L2 normalisation: v_norm = v / max(||v||, ε)
          - Ensures all vectors lie on the unit hypersphere.
          - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
          - ε = 1e-12 prevents division by zero for degenerate zero vectors.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `collections` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `hashlib` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
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

- Handles credentials or environment configuration. Keep secrets in environment variables and redact them from logs.

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

- `src/research_ai/retrieval/embeddings/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/retrieval/embeddings/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Lazy SentenceTransformer embedding service with L2-normalisation and LRU caching.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
DESIGN DECISIONS
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
----------------
# L0005: Blank line that visually separates logical sections and improves readability.

# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
WHY L2 NORMALISATION?
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  FAISS IndexFlatIP computes inner products (dot products).  For two unit-length
# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  vectors, inner product ≡ cosine similarity ∈ [-1, 1].  Normalising all vectors
# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  to unit length before indexing and before querying means we get cosine similarity
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  "for free" using the fastest FAISS index type.
# L0011: Blank line that visually separates logical sections and improves readability.

# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  Without normalisation, IndexFlatIP returns raw dot products which are scale-
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  dependent and meaningless for similarity ranking.
# L0014: Blank line that visually separates logical sections and improves readability.

# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  The small epsilon (1e-12) in the denominator prevents division by zero for
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  pathologically short texts (empty strings after tokenization).
# L0017: Blank line that visually separates logical sections and improves readability.

# L0018: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
WHY CACHE ONLY SINGLE-TEXT QUERIES?
# L0019: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  Document indexing (batch encode) is done once at build time — caching those
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  vectors would waste memory for no benefit.  In contrast, search queries are
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  repeated frequently (same query from multiple users, same question rephrased
# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  identically) — caching those saves 50–100ms of model inference per hit.
# L0023: Blank line that visually separates logical sections and improves readability.

# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
LRU CACHE IMPLEMENTATION (BUG FIX v3.1.1)
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------------------------------
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Original: used a list[str] for eviction order with list.pop(0) — O(n) time.
# L0027: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  At 512 entries this costs ~512 pointer comparisons per eviction, which is
# L0028: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  negligible in isolation but adds up under concurrent load.
# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Fix: use collections.OrderedDict which maintains insertion order internally
# L0031: Assigns or updates a value used later in the workflow; check mutability and data shape.
  and supports O(1) move-to-end (via move_to_end) and O(1) popitem(last=False).
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  This is the standard Python LRU pattern.
# L0033: Blank line that visually separates logical sections and improves readability.

# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
CACHE KEY
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
---------
# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
SHA-256 of "{model_name}:{text}" — includes the model name so that if the
# L0037: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
service is ever reconstructed with a different model, old cache entries are
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
automatically invalidated (different keys → cache miss → fresh embedding).
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
MD5 was previously used but SHA-256 is preferred for its stronger collision
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
resistance (relevant if adversarial inputs are a concern).
# L0041: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0042: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0043: Blank line that visually separates logical sections and improves readability.

# L0044: Imports a dependency, type, or project module needed by later code in this file.
import hashlib
# L0045: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0046: Imports a dependency, type, or project module needed by later code in this file.
from collections import OrderedDict
# L0047: Blank line that visually separates logical sections and improves readability.

# L0048: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0049: Blank line that visually separates logical sections and improves readability.

# L0050: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0051: Blank line that visually separates logical sections and improves readability.

# L0052: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Maximum number of query embeddings held in the LRU cache.
# L0053: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# 512 entries × 384 floats × 4 bytes ≈ 768 KB — well within typical RAM budgets.
# L0054: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Increase if your workload has many unique repeated queries.
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
_CACHE_MAX = 512
# L0056: Blank line that visually separates logical sections and improves readability.

# L0057: Blank line that visually separates logical sections and improves readability.

# L0058: Defines a class that groups related state and behavior behind a reusable interface.
class EmbeddingService:
# L0059: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Lazy SentenceTransformer embedding service with L2-normalised output.
# L0060: Blank line that visually separates logical sections and improves readability.

# L0061: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Features:
# L0062: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Model is loaded on first encode() call (no startup delay)
# L0063: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Query-level LRU cache using OrderedDict (O(1) hit and eviction)
# L0064: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Explicit batch_size parameter for GPU/CPU tuning
# L0065: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - read-only model_name property guards against accidental mutation
# L0066: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - warm_up() pre-loads the model at server startup to avoid first-call latency
# L0067: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0068: Blank line that visually separates logical sections and improves readability.

# L0069: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
# L0070: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._model_name = model_name
# L0071: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._model = None
# L0072: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # OrderedDict as LRU: insertion order is maintained; popitem(last=False)
# L0073: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # removes the oldest entry in O(1).  move_to_end() promotes on hit in O(1).
# L0074: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
# L0075: Blank line that visually separates logical sections and improves readability.

# L0076: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0077: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def model_name(self) -> str:
# L0078: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Read-only model name (set at construction time)."""
# L0079: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self._model_name
# L0080: Blank line that visually separates logical sections and improves readability.

# L0081: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0082: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def model(self):
# L0083: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Lazily loaded SentenceTransformer instance."""
# L0084: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self._model is None:
# L0085: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.info("Loading embedding model: %s", self._model_name)
# L0086: Imports a dependency, type, or project module needed by later code in this file.
            from sentence_transformers import SentenceTransformer
# L0087: Assigns or updates a value used later in the workflow; check mutability and data shape.
            self._model = SentenceTransformer(self._model_name)
# L0088: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.info(
# L0089: Assigns or updates a value used later in the workflow; check mutability and data shape.
                "Embedding model loaded: %s (dim=%d)",
# L0090: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                self._model_name,
# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                self._model.get_sentence_embedding_dimension(),
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0093: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self._model
# L0094: Blank line that visually separates logical sections and improves readability.

# L0095: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
# L0096: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Encode texts into L2-normalised embedding vectors.
# L0097: Blank line that visually separates logical sections and improves readability.

# L0098: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Single-text inputs use the LRU cache (typical search query path).
# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Multi-text inputs bypass the cache (typical batch indexing path).
# L0100: Blank line that visually separates logical sections and improves readability.

# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Args:
# L0102: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            texts:      List of strings to encode.  Must be non-empty.
# L0103: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        128–256 on GPU.
# L0105: Blank line that visually separates logical sections and improves readability.

# L0106: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Returns:
# L0107: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
# L0108: Uses a context manager to guarantee setup/cleanup around files, locks, or managed resources.
            with each row L2-normalised to unit length.
# L0109: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0110: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if len(texts) == 1:
# L0111: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return self._encode_single_cached(texts[0], batch_size)
# L0112: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self._encode_batch(texts, batch_size)
# L0113: Blank line that visually separates logical sections and improves readability.

# L0114: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def warm_up(self) -> None:
# L0115: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Pre-load the embedding model at server startup.
# L0116: Blank line that visually separates logical sections and improves readability.

# L0117: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Without warm_up(), the first encode() call loads the model (1–3 seconds
# L0118: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        on CPU), causing the first user request to time out or feel sluggish.
# L0119: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Call this from the FastAPI lifespan or after constructing the platform.
# L0120: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
        _ = self.model  # triggers lazy load via the property
# L0122: Blank line that visually separates logical sections and improves readability.

# L0123: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0124: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Private
# L0125: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0126: Blank line that visually separates logical sections and improves readability.

# L0127: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _encode_single_cached(self, text: str, batch_size: int) -> np.ndarray:
# L0128: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Encode a single text with LRU caching."""
# L0129: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # SHA-256 over "model_name:text" ensures cache isolation across models
# L0130: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # and strong collision resistance for adversarial inputs.
# L0131: Assigns or updates a value used later in the workflow; check mutability and data shape.
        key = hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()
# L0132: Blank line that visually separates logical sections and improves readability.

# L0133: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if key in self._cache:
# L0134: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # Cache hit: promote to most-recently-used end (O(1))
# L0135: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            self._cache.move_to_end(key)
# L0136: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return self._cache[key]
# L0137: Blank line that visually separates logical sections and improves readability.

# L0138: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Cache miss: encode and store
# L0139: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vec = self._encode_batch([text], batch_size)
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._cache[key] = vec
# L0141: Blank line that visually separates logical sections and improves readability.

# L0142: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Evict oldest entry if over capacity (O(1) with OrderedDict)
# L0143: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if len(self._cache) > _CACHE_MAX:
# L0144: Assigns or updates a value used later in the workflow; check mutability and data shape.
            self._cache.popitem(last=False)  # removes the first (oldest) item
# L0145: Blank line that visually separates logical sections and improves readability.

# L0146: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return vec
# L0147: Blank line that visually separates logical sections and improves readability.

# L0148: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
# L0149: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Encode a batch of texts and L2-normalise the output vectors.
# L0150: Blank line that visually separates logical sections and improves readability.

# L0151: Assigns or updates a value used later in the workflow; check mutability and data shape.
        L2 normalisation: v_norm = v / max(||v||, ε)
# L0152: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          - Ensures all vectors lie on the unit hypersphere.
# L0153: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
# L0154: Assigns or updates a value used later in the workflow; check mutability and data shape.
          - ε = 1e-12 prevents division by zero for degenerate zero vectors.
# L0155: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0156: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vectors = self.model.encode(
# L0157: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            texts,
# L0158: Assigns or updates a value used later in the workflow; check mutability and data shape.
            batch_size=batch_size,
# L0159: Assigns or updates a value used later in the workflow; check mutability and data shape.
            show_progress_bar=False,
# L0160: Assigns or updates a value used later in the workflow; check mutability and data shape.
            convert_to_numpy=True,
# L0161: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0162: Assigns or updates a value used later in the workflow; check mutability and data shape.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
# L0163: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return vectors / np.clip(norms, 1e-12, None)
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Lazy SentenceTransformer embedding service with L2-normalisation and LRU caching.

DESIGN DECISIONS
----------------

WHY L2 NORMALISATION?
  FAISS IndexFlatIP computes inner products (dot products).  For two unit-length
  vectors, inner product ≡ cosine similarity ∈ [-1, 1].  Normalising all vectors
  to unit length before indexing and before querying means we get cosine similarity
  "for free" using the fastest FAISS index type.

  Without normalisation, IndexFlatIP returns raw dot products which are scale-
  dependent and meaningless for similarity ranking.

  The small epsilon (1e-12) in the denominator prevents division by zero for
  pathologically short texts (empty strings after tokenization).

WHY CACHE ONLY SINGLE-TEXT QUERIES?
  Document indexing (batch encode) is done once at build time — caching those
  vectors would waste memory for no benefit.  In contrast, search queries are
  repeated frequently (same query from multiple users, same question rephrased
  identically) — caching those saves 50–100ms of model inference per hit.

LRU CACHE IMPLEMENTATION (BUG FIX v3.1.1)
------------------------------------------
Original: used a list[str] for eviction order with list.pop(0) — O(n) time.
  At 512 entries this costs ~512 pointer comparisons per eviction, which is
  negligible in isolation but adds up under concurrent load.

Fix: use collections.OrderedDict which maintains insertion order internally
  and supports O(1) move-to-end (via move_to_end) and O(1) popitem(last=False).
  This is the standard Python LRU pattern.

CACHE KEY
---------
SHA-256 of "{model_name}:{text}" — includes the model name so that if the
service is ever reconstructed with a different model, old cache entries are
automatically invalidated (different keys → cache miss → fresh embedding).
MD5 was previously used but SHA-256 is preferred for its stronger collision
resistance (relevant if adversarial inputs are a concern).
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of query embeddings held in the LRU cache.
# 512 entries × 384 floats × 4 bytes ≈ 768 KB — well within typical RAM budgets.
# Increase if your workload has many unique repeated queries.
_CACHE_MAX = 512


class EmbeddingService:
    """Lazy SentenceTransformer embedding service with L2-normalised output.

    Features:
    - Model is loaded on first encode() call (no startup delay)
    - Query-level LRU cache using OrderedDict (O(1) hit and eviction)
    - Explicit batch_size parameter for GPU/CPU tuning
    - read-only model_name property guards against accidental mutation
    - warm_up() pre-loads the model at server startup to avoid first-call latency
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        # OrderedDict as LRU: insertion order is maintained; popitem(last=False)
        # removes the oldest entry in O(1).  move_to_end() promotes on hit in O(1).
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def model_name(self) -> str:
        """Read-only model name (set at construction time)."""
        return self._model_name

    @property
    def model(self):
        """Lazily loaded SentenceTransformer instance."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding model loaded: %s (dim=%d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts into L2-normalised embedding vectors.

        Single-text inputs use the LRU cache (typical search query path).
        Multi-text inputs bypass the cache (typical batch indexing path).

        Args:
            texts:      List of strings to encode.  Must be non-empty.
            batch_size: SentenceTransformer batch size.  Use 32–64 on CPU,
                        128–256 on GPU.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32,
            with each row L2-normalised to unit length.
        """
        if len(texts) == 1:
            return self._encode_single_cached(texts[0], batch_size)
        return self._encode_batch(texts, batch_size)

    def warm_up(self) -> None:
        """Pre-load the embedding model at server startup.

        Without warm_up(), the first encode() call loads the model (1–3 seconds
        on CPU), causing the first user request to time out or feel sluggish.
        Call this from the FastAPI lifespan or after constructing the platform.
        """
        _ = self.model  # triggers lazy load via the property

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _encode_single_cached(self, text: str, batch_size: int) -> np.ndarray:
        """Encode a single text with LRU caching."""
        # SHA-256 over "model_name:text" ensures cache isolation across models
        # and strong collision resistance for adversarial inputs.
        key = hashlib.sha256(f"{self._model_name}:{text}".encode()).hexdigest()

        if key in self._cache:
            # Cache hit: promote to most-recently-used end (O(1))
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cache miss: encode and store
        vec = self._encode_batch([text], batch_size)
        self._cache[key] = vec

        # Evict oldest entry if over capacity (O(1) with OrderedDict)
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)  # removes the first (oldest) item

        return vec

    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode a batch of texts and L2-normalise the output vectors.

        L2 normalisation: v_norm = v / max(||v||, ε)
          - Ensures all vectors lie on the unit hypersphere.
          - Makes inner product ≡ cosine similarity in FAISS IndexFlatIP.
          - ε = 1e-12 prevents division by zero for degenerate zero vectors.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)
```
