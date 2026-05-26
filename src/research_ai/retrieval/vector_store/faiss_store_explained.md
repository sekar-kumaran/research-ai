# faiss_store.py Explained

Generated educational companion for `src/research_ai/retrieval/vector_store/faiss_store.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/retrieval/vector_store/faiss_store.py` is a Python module in the Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking. It defines RetrievedDocument, FaissVectorStore and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Retrieval layer: chunking, embeddings, FAISS, hybrid search, and reranking.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/retrieval/vector_store/faiss_store.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/retrieval/vector_store/faiss_store.py]
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
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pandas` | Pandas provides dataframe operations for tabular metadata and tests. It is ergonomic for moderate in-memory workloads. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |
| `pyarrow` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 61 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `RetrievedDocument`

- **Line:** 65
- **Base classes:** `object`
- **Docstring:** A single paper retrieved from the FAISS index.

score: inner-product similarity ∈ [0, 1] for L2-normalised vectors.
       Higher is more similar.  Equivalent to cosine similarity.
year:  extracted from the first 4 chars of update_date (arXiv convention).

**Methods:**
- `to_dict` at line 80: method behavior is described by its body and name

```python
class RetrievedDocument:
    """A single paper retrieved from the FAISS index.

    score: inner-product similarity ∈ [0, 1] for L2-normalised vectors.
           Higher is more similar.  Equivalent to cosine similarity.
    year:  extracted from the first 4 chars of update_date (arXiv convention).
    """
    paper_id: str
    title: str
    abstract: str
    score: float
    authors: str = ""
    category: str = ""
    year: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": round(self.score, 4),
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `FaissVectorStore`

- **Line:** 92
- **Base classes:** `object`
- **Docstring:** Wraps a FAISS flat index + aligned Parquet metadata for paper retrieval.

Typical usage:
    store = FaissVectorStore.from_artifacts(Path("artifacts/similarity"))
    results = store.search(query_vec, top_k=10)

The store is safe to share across threads — FAISS IndexFlatIP is read-only
after loading and pandas DataFrames are immutable for .iloc access.

**Methods:**
- `__init__` at line 103: method behavior is described by its body and name
- `from_artifacts` at line 114: Create a lazy store backed by artifact files (loaded on first search).
- `ready` at line 119: True if the store is either already loaded or its artifact files exist.
- `paper_count` at line 131: Number of indexed papers.  Reads Parquet footer only (no data load).
- `_ensure_loaded` at line 147: Load FAISS index and metadata from disk (idempotent after first call).

Raises RuntimeError if artifacts are missing or corrupted.
- `search` at line 178: Search for the top_k most similar papers to query_vec.

Args:
    query_vec: L2-normalised query embedding, shape (1, d) or (d,).
               MUST match the dimension used when building the index.
    top_k:     Number of nearest neighbours to return.

Returns:
    List of RetrievedDocument, sorted by score descending.
    Empty list if the store is not ready or all FAISS IDs are -1.

Raises:
    RuntimeError: if query_vec dimension does not match index dimension.
                  This is a configuration error — rebuild with the
                  correct embedding model.

```python
class FaissVectorStore:
    """Wraps a FAISS flat index + aligned Parquet metadata for paper retrieval.

    Typical usage:
        store = FaissVectorStore.from_artifacts(Path("artifacts/similarity"))
        results = store.search(query_vec, top_k=10)

    The store is safe to share across threads — FAISS IndexFlatIP is read-only
    after loading and pandas DataFrames are immutable for .iloc access.
    """

    def __init__(
        self,
        index=None,
        metadata: pd.DataFrame | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.index = index
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "FaissVectorStore":
        """Create a lazy store backed by artifact files (loaded on first search)."""
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        """True if the store is either already loaded or its artifact files exist."""
        if self.index is not None and not self.metadata.empty:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "paper_index.faiss").exists()
            and (self.artifact_dir / "paper_metadata.parquet").exists()
        )

    @property
    def paper_count(self) -> int:
        """Number of indexed papers.  Reads Parquet footer only (no data load)."""
        if not self.metadata.empty:
            return int(len(self.metadata))
        if self.artifact_dir is None:
            return 0
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not metadata_path.exists():
            return 0
        try:
            # pq.ParquetFile.metadata.num_rows reads only the file footer —
            # extremely fast even for multi-GB Parquet files.
            return int(pq.ParquetFile(metadata_path).metadata.num_rows)
        except Exception:
            return 0

    def _ensure_loaded(self) -> None:
        """Load FAISS index and metadata from disk (idempotent after first call).

        Raises RuntimeError if artifacts are missing or corrupted.
        """
        if self.index is not None and not self.metadata.empty:
            return  # already loaded
        if self.artifact_dir is None:
            raise RuntimeError("Vector store artifact directory is not configured.")
        index_path = self.artifact_dir / "paper_index.faiss"
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not index_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                f"Vector store artifacts are missing in {self.artifact_dir}. "
                "Run the embedding/indexing pipeline first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)

        # Validate positional alignment: FAISS ntotal must equal metadata rows.
        # A mismatch means the index and metadata were built from different data.
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"FAISS index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} rows.  Rebuild the similarity artifacts."
            )
        logger.info(
            "FaissVectorStore loaded: %d papers, dim=%d",
            self.index.ntotal, self.index.d,
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedDocument]:
        """Search for the top_k most similar papers to query_vec.

        Args:
            query_vec: L2-normalised query embedding, shape (1, d) or (d,).
                       MUST match the dimension used when building the index.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of RetrievedDocument, sorted by score descending.
            Empty list if the store is not ready or all FAISS IDs are -1.

        Raises:
            RuntimeError: if query_vec dimension does not match index dimension.
                          This is a configuration error — rebuild with the
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `RetrievedDocument` Methods

#### `RetrievedDocument.to_dict`

- **Line:** 80
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": round(self.score, 4),
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `FaissVectorStore` Methods

#### `FaissVectorStore.__init__`

- **Line:** 103
- **Kind:** synchronous method
- **Arguments:** self, index, metadata, artifact_dir
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(
        self,
        index=None,
        metadata: pd.DataFrame | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.index = index
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.artifact_dir = artifact_dir
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `FaissVectorStore.from_artifacts`

- **Line:** 114
- **Kind:** synchronous method
- **Arguments:** cls, artifact_dir
- **Docstring:** Create a lazy store backed by artifact files (loaded on first search).

```python
    def from_artifacts(cls, artifact_dir: Path) -> "FaissVectorStore":
        """Create a lazy store backed by artifact files (loaded on first search)."""
        return cls(artifact_dir=artifact_dir)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `FaissVectorStore.ready`

- **Line:** 119
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** True if the store is either already loaded or its artifact files exist.

```python
    def ready(self) -> bool:
        """True if the store is either already loaded or its artifact files exist."""
        if self.index is not None and not self.metadata.empty:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "paper_index.faiss").exists()
            and (self.artifact_dir / "paper_metadata.parquet").exists()
        )
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `FaissVectorStore.paper_count`

- **Line:** 131
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Number of indexed papers.  Reads Parquet footer only (no data load).

```python
    def paper_count(self) -> int:
        """Number of indexed papers.  Reads Parquet footer only (no data load)."""
        if not self.metadata.empty:
            return int(len(self.metadata))
        if self.artifact_dir is None:
            return 0
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not metadata_path.exists():
            return 0
        try:
            # pq.ParquetFile.metadata.num_rows reads only the file footer —
            # extremely fast even for multi-GB Parquet files.
            return int(pq.ParquetFile(metadata_path).metadata.num_rows)
        except Exception:
            return 0
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `FaissVectorStore._ensure_loaded`

- **Line:** 147
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Load FAISS index and metadata from disk (idempotent after first call).

Raises RuntimeError if artifacts are missing or corrupted.

```python
    def _ensure_loaded(self) -> None:
        """Load FAISS index and metadata from disk (idempotent after first call).

        Raises RuntimeError if artifacts are missing or corrupted.
        """
        if self.index is not None and not self.metadata.empty:
            return  # already loaded
        if self.artifact_dir is None:
            raise RuntimeError("Vector store artifact directory is not configured.")
        index_path = self.artifact_dir / "paper_index.faiss"
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not index_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                f"Vector store artifacts are missing in {self.artifact_dir}. "
                "Run the embedding/indexing pipeline first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)

        # Validate positional alignment: FAISS ntotal must equal metadata rows.
        # A mismatch means the index and metadata were built from different data.
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"FAISS index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} rows.  Rebuild the similarity artifacts."
            )
        logger.info(
            "FaissVectorStore loaded: %d papers, dim=%d",
            self.index.ntotal, self.index.d,
        )
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `FaissVectorStore.search`

- **Line:** 178
- **Kind:** synchronous method
- **Arguments:** self, query_vec, top_k
- **Docstring:** Search for the top_k most similar papers to query_vec.

Args:
    query_vec: L2-normalised query embedding, shape (1, d) or (d,).
               MUST match the dimension used when building the index.
    top_k:     Number of nearest neighbours to return.

Returns:
    List of RetrievedDocument, sorted by score descending.
    Empty list if the store is not ready or all FAISS IDs are -1.

Raises:
    RuntimeError: if query_vec dimension does not match index dimension.
                  This is a configuration error — rebuild with the
                  correct embedding model.

```python
    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedDocument]:
        """Search for the top_k most similar papers to query_vec.

        Args:
            query_vec: L2-normalised query embedding, shape (1, d) or (d,).
                       MUST match the dimension used when building the index.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of RetrievedDocument, sorted by score descending.
            Empty list if the store is not ready or all FAISS IDs are -1.

        Raises:
            RuntimeError: if query_vec dimension does not match index dimension.
                          This is a configuration error — rebuild with the
                          correct embedding model.
        """
        if not self.ready:
            return []
        self._ensure_loaded()

        # DIMENSION VALIDATION (BUG FIX v3.1.1):
        # FAISS stores the embedding dimension as index.d.  If the embedding
        # model was changed after index construction, query_vec.shape[-1] will
        # differ from index.d, causing silent wrong results or a cryptic FAISS
        # error.  Detect this early and give a clear actionable error message.
        query_dim = query_vec.shape[-1]
        if query_dim != self.index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query vector has {query_dim} dims "
                f"but FAISS index was built with {self.index.d} dims.  "
                "Change EMBEDDING_MODEL back to the model used during indexing, "
                "or rebuild the FAISS index with the new model."
            )

        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
        docs: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], ids[0]):
            # FAISS returns -1 for "no result" when fewer than top_k docs exist
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[int(idx)]
            docs.append(
                RetrievedDocument(
                    paper_id=str(row.get("id", "")),
                    title=str(row.get("title", "Untitled")).strip(),
                    abstract=str(row.get("abstract", "")).strip(),
                    score=float(score),
                    authors=str(row.get("authors", "")),
                    category=str(row.get("categories", "")),
                    year=str(row.get("update_date", ""))[:4],
                )
            )
        return docs
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.
- **Parquet**: Parquet is a compressed columnar format that is efficient for large metadata because readers can scan selected columns.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pandas` | Pandas provides dataframe operations for tabular metadata and tests. It is ergonomic for moderate in-memory workloads. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |
| `pyarrow` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.
- **Parquet**: Parquet is a compressed columnar format that is efficient for large metadata because readers can scan selected columns.

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

- `src/research_ai/retrieval/vector_store/faiss_store.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/retrieval/vector_store/faiss_store.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""FAISS vector store wrapper — semantic paper retrieval over indexed embeddings.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
DATA MODEL
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
----------
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Two artifacts form the store:
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  paper_index.faiss     — FAISS IndexFlatIP (inner product, i.e. cosine for
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                          L2-normalised vectors). One vector per paper.
# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  paper_metadata.parquet— One row per paper, positionally aligned to the FAISS
# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                          index.  Row i ↔ vector i.
# L0010: Blank line that visually separates logical sections and improves readability.

# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
POSITIONAL ALIGNMENT INVARIANT
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
--------------------------------
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
The FAISS index stores raw integer row indices (0, 1, 2, …).  search() maps
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
these back to metadata rows via `metadata.iloc[idx]`.  This means the
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
metadata DataFrame MUST be ordered identically to the embedding order used
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
when building the index.  Never sort, filter, or reset_index on the metadata
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
after index construction without rebuilding the FAISS index.
# L0018: Blank line that visually separates logical sections and improves readability.

# L0019: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
LAZY LOADING
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Artifacts are not read from disk until the first search() call.  This keeps
# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
startup fast and allows the web server to accept health-check requests before
# L0023: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
the 1–2 GB index is fully loaded into RAM.
# L0024: Blank line that visually separates logical sections and improves readability.

# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
DIMENSION VALIDATION (BUG FIX v3.1.1)
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
---------------------------------------
# L0027: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Previously there was no check that the query vector dimension matched the
# L0028: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
FAISS index dimension.  If an administrator changed EMBEDDING_MODEL in .env
# L0029: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
after building the index, all searches would silently return wrong results or
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
crash with an unintelligible FAISS error.
# L0031: Blank line that visually separates logical sections and improves readability.

# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Fix: _ensure_loaded() now reads index.d (the stored embedding dimension) and
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
raises a clear RuntimeError if a query vector of a different dimension is seen
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
in search().  This turns a silent data-corruption bug into a fail-fast error.
# L0035: Blank line that visually separates logical sections and improves readability.

# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
INDEX TYPE CHOICE
# L0037: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
-----------------
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
IndexFlatIP is an exact (non-approximate) brute-force inner-product search.
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
- Pros: 100% recall, no training required, deterministic results.
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
- Cons: O(N × d) per query — linear in corpus size.
# L0041: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
- For 100k papers × 384 dims, this is ~40M multiplications, typically <5ms
# L0042: Uses a context manager to guarantee setup/cleanup around files, locks, or managed resources.
  with FAISS's BLAS-backed implementation on a modern CPU.
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
- If the corpus grows to >500k papers, consider IndexHNSWFlat (approximate,
# L0044: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  sub-linear) or IndexIVFFlat with nlist~100 clusters.
# L0045: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0046: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0047: Blank line that visually separates logical sections and improves readability.

# L0048: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0049: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass
# L0050: Imports a dependency, type, or project module needed by later code in this file.
from pathlib import Path
# L0051: Blank line that visually separates logical sections and improves readability.

# L0052: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0053: Imports a dependency, type, or project module needed by later code in this file.
import pandas as pd
# L0054: Imports a dependency, type, or project module needed by later code in this file.
import pyarrow.parquet as pq
# L0055: Blank line that visually separates logical sections and improves readability.

# L0056: Begins protected execution so failures can be handled without crashing the whole request path.
try:
# L0057: Imports a dependency, type, or project module needed by later code in this file.
    import faiss
# L0058: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
except ImportError as exc:  # pragma: no cover
# L0059: Raises an explicit error when the function cannot safely continue.
    raise RuntimeError("faiss-cpu is required for vector retrieval.") from exc
# L0060: Blank line that visually separates logical sections and improves readability.

# L0061: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0062: Blank line that visually separates logical sections and improves readability.

# L0063: Blank line that visually separates logical sections and improves readability.

# L0064: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0065: Defines a class that groups related state and behavior behind a reusable interface.
class RetrievedDocument:
# L0066: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """A single paper retrieved from the FAISS index.
# L0067: Blank line that visually separates logical sections and improves readability.

# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    score: inner-product similarity ∈ [0, 1] for L2-normalised vectors.
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
           Higher is more similar.  Equivalent to cosine similarity.
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    year:  extracted from the first 4 chars of update_date (arXiv convention).
# L0071: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0072: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    paper_id: str
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    title: str
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    abstract: str
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    score: float
# L0076: Assigns or updates a value used later in the workflow; check mutability and data shape.
    authors: str = ""
# L0077: Assigns or updates a value used later in the workflow; check mutability and data shape.
    category: str = ""
# L0078: Assigns or updates a value used later in the workflow; check mutability and data shape.
    year: str = ""
# L0079: Blank line that visually separates logical sections and improves readability.

# L0080: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def to_dict(self) -> dict:
# L0081: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0082: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_id": self.paper_id,
# L0083: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": self.title,
# L0084: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": self.abstract,
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "score": round(self.score, 4),
# L0086: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": self.authors,
# L0087: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "category": self.category,
# L0088: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "year": self.year,
# L0089: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0090: Blank line that visually separates logical sections and improves readability.

# L0091: Blank line that visually separates logical sections and improves readability.

# L0092: Defines a class that groups related state and behavior behind a reusable interface.
class FaissVectorStore:
# L0093: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Wraps a FAISS flat index + aligned Parquet metadata for paper retrieval.
# L0094: Blank line that visually separates logical sections and improves readability.

# L0095: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Typical usage:
# L0096: Assigns or updates a value used later in the workflow; check mutability and data shape.
        store = FaissVectorStore.from_artifacts(Path("artifacts/similarity"))
# L0097: Assigns or updates a value used later in the workflow; check mutability and data shape.
        results = store.search(query_vec, top_k=10)
# L0098: Blank line that visually separates logical sections and improves readability.

# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    The store is safe to share across threads — FAISS IndexFlatIP is read-only
# L0100: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    after loading and pandas DataFrames are immutable for .iloc access.
# L0101: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0102: Blank line that visually separates logical sections and improves readability.

# L0103: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0105: Assigns or updates a value used later in the workflow; check mutability and data shape.
        index=None,
# L0106: Assigns or updates a value used later in the workflow; check mutability and data shape.
        metadata: pd.DataFrame | None = None,
# L0107: Assigns or updates a value used later in the workflow; check mutability and data shape.
        artifact_dir: Path | None = None,
# L0108: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> None:
# L0109: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.index = index
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.metadata = metadata if metadata is not None else pd.DataFrame()
# L0111: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.artifact_dir = artifact_dir
# L0112: Blank line that visually separates logical sections and improves readability.

# L0113: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @classmethod
# L0114: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def from_artifacts(cls, artifact_dir: Path) -> "FaissVectorStore":
# L0115: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Create a lazy store backed by artifact files (loaded on first search)."""
# L0116: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return cls(artifact_dir=artifact_dir)
# L0117: Blank line that visually separates logical sections and improves readability.

# L0118: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0119: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def ready(self) -> bool:
# L0120: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """True if the store is either already loaded or its artifact files exist."""
# L0121: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.index is not None and not self.metadata.empty:
# L0122: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return True
# L0123: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.artifact_dir is None:
# L0124: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return False
# L0125: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return (
# L0126: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            (self.artifact_dir / "paper_index.faiss").exists()
# L0127: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            and (self.artifact_dir / "paper_metadata.parquet").exists()
# L0128: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0131: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def paper_count(self) -> int:
# L0132: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Number of indexed papers.  Reads Parquet footer only (no data load)."""
# L0133: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not self.metadata.empty:
# L0134: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return int(len(self.metadata))
# L0135: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.artifact_dir is None:
# L0136: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return 0
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
# L0138: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not metadata_path.exists():
# L0139: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return 0
# L0140: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0141: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # pq.ParquetFile.metadata.num_rows reads only the file footer —
# L0142: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # extremely fast even for multi-GB Parquet files.
# L0143: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return int(pq.ParquetFile(metadata_path).metadata.num_rows)
# L0144: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception:
# L0145: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return 0
# L0146: Blank line that visually separates logical sections and improves readability.

# L0147: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _ensure_loaded(self) -> None:
# L0148: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Load FAISS index and metadata from disk (idempotent after first call).
# L0149: Blank line that visually separates logical sections and improves readability.

# L0150: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Raises RuntimeError if artifacts are missing or corrupted.
# L0151: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0152: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.index is not None and not self.metadata.empty:
# L0153: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return  # already loaded
# L0154: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.artifact_dir is None:
# L0155: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError("Vector store artifact directory is not configured.")
# L0156: Assigns or updates a value used later in the workflow; check mutability and data shape.
        index_path = self.artifact_dir / "paper_index.faiss"
# L0157: Assigns or updates a value used later in the workflow; check mutability and data shape.
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
# L0158: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not index_path.exists() or not metadata_path.exists():
# L0159: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError(
# L0160: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                f"Vector store artifacts are missing in {self.artifact_dir}. "
# L0161: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "Run the embedding/indexing pipeline first."
# L0162: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0163: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.index = faiss.read_index(str(index_path))
# L0164: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.metadata = pd.read_parquet(metadata_path)
# L0165: Blank line that visually separates logical sections and improves readability.

# L0166: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Validate positional alignment: FAISS ntotal must equal metadata rows.
# L0167: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # A mismatch means the index and metadata were built from different data.
# L0168: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.index.ntotal != len(self.metadata):
# L0169: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError(
# L0170: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                f"FAISS index has {self.index.ntotal} vectors but metadata has "
# L0171: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                f"{len(self.metadata)} rows.  Rebuild the similarity artifacts."
# L0172: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0173: Emits structured operational information for debugging, monitoring, or failure diagnosis.
        logger.info(
# L0174: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "FaissVectorStore loaded: %d papers, dim=%d",
# L0175: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            self.index.ntotal, self.index.d,
# L0176: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0177: Blank line that visually separates logical sections and improves readability.

# L0178: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedDocument]:
# L0179: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Search for the top_k most similar papers to query_vec.
# L0180: Blank line that visually separates logical sections and improves readability.

# L0181: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Args:
# L0182: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            query_vec: L2-normalised query embedding, shape (1, d) or (d,).
# L0183: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                       MUST match the dimension used when building the index.
# L0184: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            top_k:     Number of nearest neighbours to return.
# L0185: Blank line that visually separates logical sections and improves readability.

# L0186: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Returns:
# L0187: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            List of RetrievedDocument, sorted by score descending.
# L0188: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            Empty list if the store is not ready or all FAISS IDs are -1.
# L0189: Blank line that visually separates logical sections and improves readability.

# L0190: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Raises:
# L0191: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            RuntimeError: if query_vec dimension does not match index dimension.
# L0192: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                          This is a configuration error — rebuild with the
# L0193: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                          correct embedding model.
# L0194: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0195: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not self.ready:
# L0196: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return []
# L0197: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self._ensure_loaded()
# L0198: Blank line that visually separates logical sections and improves readability.

# L0199: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # DIMENSION VALIDATION (BUG FIX v3.1.1):
# L0200: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # FAISS stores the embedding dimension as index.d.  If the embedding
# L0201: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # model was changed after index construction, query_vec.shape[-1] will
# L0202: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # differ from index.d, causing silent wrong results or a cryptic FAISS
# L0203: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # error.  Detect this early and give a clear actionable error message.
# L0204: Assigns or updates a value used later in the workflow; check mutability and data shape.
        query_dim = query_vec.shape[-1]
# L0205: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if query_dim != self.index.d:
# L0206: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError(
# L0207: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                f"Embedding dimension mismatch: query vector has {query_dim} dims "
# L0208: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                f"but FAISS index was built with {self.index.d} dims.  "
# L0209: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "Change EMBEDDING_MODEL back to the model used during indexing, "
# L0210: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "or rebuild the FAISS index with the new model."
# L0211: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0212: Blank line that visually separates logical sections and improves readability.

# L0213: Assigns or updates a value used later in the workflow; check mutability and data shape.
        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
# L0214: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs: list[RetrievedDocument] = []
# L0215: Iterates over data, retry attempts, files, results, or workflow steps.
        for score, idx in zip(scores[0], ids[0]):
# L0216: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # FAISS returns -1 for "no result" when fewer than top_k docs exist
# L0217: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if idx < 0 or idx >= len(self.metadata):
# L0218: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0219: Assigns or updates a value used later in the workflow; check mutability and data shape.
            row = self.metadata.iloc[int(idx)]
# L0220: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            docs.append(
# L0221: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                RetrievedDocument(
# L0222: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    paper_id=str(row.get("id", "")),
# L0223: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    title=str(row.get("title", "Untitled")).strip(),
# L0224: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    abstract=str(row.get("abstract", "")).strip(),
# L0225: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    score=float(score),
# L0226: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    authors=str(row.get("authors", "")),
# L0227: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    category=str(row.get("categories", "")),
# L0228: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    year=str(row.get("update_date", ""))[:4],
# L0229: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                )
# L0230: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0231: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return docs
```

## Source Walkthrough

This file is large, so the opening and closing sections are included here. Use the class/function breakdown above to navigate the middle of the file.

### Opening Section

```python
"""FAISS vector store wrapper — semantic paper retrieval over indexed embeddings.

DATA MODEL
----------
Two artifacts form the store:
  paper_index.faiss     — FAISS IndexFlatIP (inner product, i.e. cosine for
                          L2-normalised vectors). One vector per paper.
  paper_metadata.parquet— One row per paper, positionally aligned to the FAISS
                          index.  Row i ↔ vector i.

POSITIONAL ALIGNMENT INVARIANT
--------------------------------
The FAISS index stores raw integer row indices (0, 1, 2, …).  search() maps
these back to metadata rows via `metadata.iloc[idx]`.  This means the
metadata DataFrame MUST be ordered identically to the embedding order used
when building the index.  Never sort, filter, or reset_index on the metadata
after index construction without rebuilding the FAISS index.

LAZY LOADING
------------
Artifacts are not read from disk until the first search() call.  This keeps
startup fast and allows the web server to accept health-check requests before
the 1–2 GB index is fully loaded into RAM.

DIMENSION VALIDATION (BUG FIX v3.1.1)
---------------------------------------
Previously there was no check that the query vector dimension matched the
FAISS index dimension.  If an administrator changed EMBEDDING_MODEL in .env
after building the index, all searches would silently return wrong results or
crash with an unintelligible FAISS error.

Fix: _ensure_loaded() now reads index.d (the stored embedding dimension) and
raises a clear RuntimeError if a query vector of a different dimension is seen
in search().  This turns a silent data-corruption bug into a fail-fast error.

INDEX TYPE CHOICE
-----------------
IndexFlatIP is an exact (non-approximate) brute-force inner-product search.
- Pros: 100% recall, no training required, deterministic results.
- Cons: O(N × d) per query — linear in corpus size.
- For 100k papers × 384 dims, this is ~40M multiplications, typically <5ms
  with FAISS's BLAS-backed implementation on a modern CPU.
- If the corpus grows to >500k papers, consider IndexHNSWFlat (approximate,
  sub-linear) or IndexIVFFlat with nlist~100 clusters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required for vector retrieval.") from exc

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A single paper retrieved from the FAISS index.

    score: inner-product similarity ∈ [0, 1] for L2-normalised vectors.
           Higher is more similar.  Equivalent to cosine similarity.
    year:  extracted from the first 4 chars of update_date (arXiv convention).
    """
    paper_id: str
    title: str
    abstract: str
    score: float
    authors: str = ""
    category: str = ""
    year: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": round(self.score, 4),
            "authors": self.authors,
            "category": self.category,
            "year": self.year,
        }


class FaissVectorStore:
    """Wraps a FAISS flat index + aligned Parquet metadata for paper retrieval.

    Typical usage:
        store = FaissVectorStore.from_artifacts(Path("artifacts/similarity"))
        results = store.search(query_vec, top_k=10)

    The store is safe to share across threads — FAISS IndexFlatIP is read-only
    after loading and pandas DataFrames are immutable for .iloc access.
    """

    def __init__(
        self,
        index=None,
        metadata: pd.DataFrame | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.index = index
        self.metadata = metadata if metadata is not None else pd.DataFrame()
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "FaissVectorStore":
        """Create a lazy store backed by artifact files (loaded on first search)."""
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        """True if the store is either already loaded or its artifact files exist."""
```

### Closing Section

```python
        if self.index is not None and not self.metadata.empty:
            return  # already loaded
        if self.artifact_dir is None:
            raise RuntimeError("Vector store artifact directory is not configured.")
        index_path = self.artifact_dir / "paper_index.faiss"
        metadata_path = self.artifact_dir / "paper_metadata.parquet"
        if not index_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                f"Vector store artifacts are missing in {self.artifact_dir}. "
                "Run the embedding/indexing pipeline first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)

        # Validate positional alignment: FAISS ntotal must equal metadata rows.
        # A mismatch means the index and metadata were built from different data.
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"FAISS index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} rows.  Rebuild the similarity artifacts."
            )
        logger.info(
            "FaissVectorStore loaded: %d papers, dim=%d",
            self.index.ntotal, self.index.d,
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedDocument]:
        """Search for the top_k most similar papers to query_vec.

        Args:
            query_vec: L2-normalised query embedding, shape (1, d) or (d,).
                       MUST match the dimension used when building the index.
            top_k:     Number of nearest neighbours to return.

        Returns:
            List of RetrievedDocument, sorted by score descending.
            Empty list if the store is not ready or all FAISS IDs are -1.

        Raises:
            RuntimeError: if query_vec dimension does not match index dimension.
                          This is a configuration error — rebuild with the
                          correct embedding model.
        """
        if not self.ready:
            return []
        self._ensure_loaded()

        # DIMENSION VALIDATION (BUG FIX v3.1.1):
        # FAISS stores the embedding dimension as index.d.  If the embedding
        # model was changed after index construction, query_vec.shape[-1] will
        # differ from index.d, causing silent wrong results or a cryptic FAISS
        # error.  Detect this early and give a clear actionable error message.
        query_dim = query_vec.shape[-1]
        if query_dim != self.index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query vector has {query_dim} dims "
                f"but FAISS index was built with {self.index.d} dims.  "
                "Change EMBEDDING_MODEL back to the model used during indexing, "
                "or rebuild the FAISS index with the new model."
            )

        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
        docs: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], ids[0]):
            # FAISS returns -1 for "no result" when fewer than top_k docs exist
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[int(idx)]
            docs.append(
                RetrievedDocument(
                    paper_id=str(row.get("id", "")),
                    title=str(row.get("title", "Untitled")).strip(),
                    abstract=str(row.get("abstract", "")).strip(),
                    score=float(score),
                    authors=str(row.get("authors", "")),
                    category=str(row.get("categories", "")),
                    year=str(row.get("update_date", ""))[:4],
                )
            )
        return docs
```
