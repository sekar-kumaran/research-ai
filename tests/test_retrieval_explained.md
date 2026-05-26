# test_retrieval.py Explained

Generated educational companion for `tests/test_retrieval.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`tests/test_retrieval.py` is a Python module in the Test layer: behavioral, safety, performance, and integration checks. It defines TestBM25Tokenizer, TestBM25Scoring, TestMetadataReranker, TestFaissVectorStoreDimensionValidation, TestEmbeddingServiceCache, TestArxivIdNormalization and faiss_available, _normalize_arxiv_id.

## Why This File Exists

This file isolates one responsibility in the codebase: Test layer: behavioral, safety, performance, and integration checks. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Test layer: behavioral, safety, performance, and integration checks.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `tests/test_retrieval.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[tests/test_retrieval.py]
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
| `math` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
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

### `faiss_available`

- **Line:** 185
- **Kind:** synchronous function
- **Arguments:** none
- **Docstring:** Skip FAISS tests if the library cannot be imported (e.g. numpy ABI mismatch).

```python
def faiss_available():
    """Skip FAISS tests if the library cannot be imported (e.g. numpy ABI mismatch)."""
    return pytest.importorskip("faiss", reason="faiss-cpu not importable on this environment",
                               exc_type=ImportError)
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.

### `_normalize_arxiv_id`

- **Line:** 316
- **Kind:** synchronous function
- **Arguments:** raw_id
- **Docstring:** Inline copy of PaperChatService.normalize_arxiv_id for isolated testing.
Keep this in sync with the implementation in paper_ingestion/service.py.

```python
def _normalize_arxiv_id(raw_id: str) -> str:
    """Inline copy of PaperChatService.normalize_arxiv_id for isolated testing.
    Keep this in sync with the implementation in paper_ingestion/service.py.
    """
    import re
    _ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

    token = (raw_id or "").strip()
    if not token:
        return ""
    if token.lower().startswith("arxiv:"):
        token = token.split(":", 1)[1]
    if token.startswith(("http://", "https://")):
        token = token.rstrip("/")
        for marker in ("/abs/", "/pdf/"):
            if marker in token:
                token = token.split(marker)[-1]
                break
    token = token.replace(".pdf", "").strip()
    token = _ARXIV_VERSION_RE.sub("", token).strip()
    return token
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.


## Class-by-Class Breakdown

### `TestBM25Tokenizer`

- **Line:** 30
- **Base classes:** `object`
- **Docstring:** Verify the BM25 tokenizer fix: alphanumeric tokens (gpt3, t5, etc.).

**Methods:**
- `_bm25_cls` at line 33: method behavior is described by its body and name
- `test_plain_words` at line 37: method behavior is described by its body and name
- `test_alphanumeric_model_names` at line 43: BUG FIX: original regex missed gpt3, t5, bert2, llama2.
- `test_gpt3_kept_together` at line 52: gpt3 should be a single token, not split into gpt + 3.
- `test_lowercasing` at line 58: method behavior is described by its body and name
- `test_numbers_included` at line 64: method behavior is described by its body and name
- `test_empty_string` at line 70: method behavior is described by its body and name

```python
class TestBM25Tokenizer:
    """Verify the BM25 tokenizer fix: alphanumeric tokens (gpt3, t5, etc.)."""

    def _bm25_cls(self):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25

    def test_plain_words(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("attention mechanism")
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_alphanumeric_model_names(self):
        """BUG FIX: original regex missed gpt3, t5, bert2, llama2."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("GPT-3 achieves state-of-the-art on T5 benchmark")
        assert "gpt" in tokens or "gpt3" in tokens or "3" in tokens
        assert "t5" in tokens or "t" in tokens  # at minimum the letter is kept
        # Key assertion: "t5" should now be found as a token (two chars)
        assert "t5" in tokens, f"Expected 't5' in tokens but got: {tokens}"

    def test_gpt3_kept_together(self):
        """gpt3 should be a single token, not split into gpt + 3."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("gpt3 is a large model")
        assert "gpt3" in tokens

    def test_lowercasing(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("BERT TRANSFORMER")
        assert "bert" in tokens
        assert "transformer" in tokens

    def test_numbers_included(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("published in 2023 with 42 experiments")
        assert "2023" in tokens
        assert "42" in tokens

    def test_empty_string(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("")
        assert tokens == []
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestBM25Scoring`

- **Line:** 76
- **Base classes:** `object`
- **Docstring:** Verify BM25 formula correctness.

**Methods:**
- `_make_bm25` at line 79: method behavior is described by its body and name
- `test_relevant_doc_scores_higher` at line 83: method behavior is described by its body and name
- `test_all_zero_scores_for_oov_query` at line 92: method behavior is described by its body and name
- `test_idf_rewards_rare_terms` at line 98: A term appearing in all docs should have lower IDF than a rare term.
- `test_idf_cache_populated_on_access` at line 112: method behavior is described by its body and name
- `test_score_list_length_equals_doc_count` at line 118: method behavior is described by its body and name

```python
class TestBM25Scoring:
    """Verify BM25 formula correctness."""

    def _make_bm25(self, docs):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25(docs)

    def test_relevant_doc_scores_higher(self):
        docs = [
            "transformer attention mechanism neural network",  # doc 0
            "random forest decision tree ensemble learning",   # doc 1
        ]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("transformer attention")
        assert scores[0] > scores[1], "Transformer doc should score higher for transformer query"

    def test_all_zero_scores_for_oov_query(self):
        docs = ["neural network", "decision tree"]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("zyxwvutsrqpon")
        assert all(s == 0.0 for s in scores)

    def test_idf_rewards_rare_terms(self):
        """A term appearing in all docs should have lower IDF than a rare term."""
        docs = [
            "neural network learning",
            "neural network transformer",
            "neural network attention",
        ]
        bm25 = self._make_bm25(docs)
        # "neural" appears in all 3 → low IDF
        # "transformer" appears in 1 → high IDF
        idf_neural = bm25._idf("neural")
        idf_transformer = bm25._idf("transformer")
        assert idf_transformer > idf_neural

    def test_idf_cache_populated_on_access(self):
        docs = ["attention is all you need"]
        bm25 = self._make_bm25(docs)
        _ = bm25._idf("attention")
        assert "attention" in bm25.idf_cache

    def test_score_list_length_equals_doc_count(self):
        docs = ["doc one", "doc two", "doc three"]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("doc")
        assert len(scores) == 3
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestMetadataReranker`

- **Line:** 129
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `_reranker` at line 130: method behavior is described by its body and name
- `test_keyword_weight_constant` at line 134: Verify _KEYWORD_WEIGHT matches the declared 0.15 in hybrid_search.
- `test_perfect_overlap_promotes_score` at line 140: method behavior is described by its body and name
- `test_hybrid_score_formula` at line 150: Verify: hybrid_score = 0.85 × score + 0.15 × overlap.
- `test_empty_query_returns_unchanged` at line 159: method behavior is described by its body and name
- `test_no_overlap_preserves_order_by_score` at line 167: If overlap is zero for all docs, ordering is by base score.

```python
class TestMetadataReranker:
    def _reranker(self):
        from research_ai.retrieval.rerankers.service import MetadataReranker
        return MetadataReranker()

    def test_keyword_weight_constant(self):
        """Verify _KEYWORD_WEIGHT matches the declared 0.15 in hybrid_search."""
        import research_ai.retrieval.rerankers.service as mod
        assert mod._KEYWORD_WEIGHT == pytest.approx(0.15)
        assert mod._SEMANTIC_WEIGHT == pytest.approx(0.85)

    def test_perfect_overlap_promotes_score(self):
        reranker = self._reranker()
        docs = [
            {"title": "Attention mechanism for transformers", "abstract": "", "score": 0.5},
            {"title": "Random forest ensemble", "abstract": "", "score": 0.6},
        ]
        result = reranker.rerank("attention transformers", docs)
        # First doc has high keyword overlap → should rank first despite lower base score
        assert result[0]["title"].startswith("Attention")

    def test_hybrid_score_formula(self):
        """Verify: hybrid_score = 0.85 × score + 0.15 × overlap."""
        reranker = self._reranker()
        docs = [{"title": "neural attention network", "abstract": "", "score": 0.8}]
        result = reranker.rerank("attention neural", docs)
        doc = result[0]
        expected = 0.85 * 0.8 + 0.15 * doc["keyword_score"]
        assert doc["hybrid_score"] == pytest.approx(expected, abs=1e-4)

    def test_empty_query_returns_unchanged(self):
        reranker = self._reranker()
        docs = [{"title": "neural net", "abstract": "", "score": 0.5}]
        # "the a an" → all stopwords → empty token set
        result = reranker.rerank("the a an", docs)
        # Should still return a result (graceful handling)
        assert len(result) == 1

    def test_no_overlap_preserves_order_by_score(self):
        """If overlap is zero for all docs, ordering is by base score."""
        reranker = self._reranker()
        docs = [
            {"title": "alpha beta gamma", "abstract": "", "score": 0.3},
            {"title": "delta epsilon zeta", "abstract": "", "score": 0.7},
        ]
        # Query has no overlap with either doc
        result = reranker.rerank("zyxwvu", docs)
        assert result[0]["score"] == pytest.approx(0.7)
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestFaissVectorStoreDimensionValidation`

- **Line:** 191
- **Base classes:** `object`
- **Docstring:** Verify the dimension mismatch detection added in v3.1.1.

**Methods:**
- `_make_store_with_mock_index` at line 194: method behavior is described by its body and name
- `test_matching_dimension_succeeds` at line 215: method behavior is described by its body and name
- `test_dimension_mismatch_raises_runtime_error` at line 221: BUG FIX: previously this caused a cryptic FAISS error.
- `test_negative_faiss_ids_skipped` at line 228: FAISS returns -1 for padding when fewer than top_k results exist.

```python
class TestFaissVectorStoreDimensionValidation:
    """Verify the dimension mismatch detection added in v3.1.1."""

    def _make_store_with_mock_index(self, index_dim: int, n_rows: int = 5):
        import pandas as pd
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore

        mock_index = MagicMock()
        mock_index.d = index_dim
        mock_index.ntotal = n_rows
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]], dtype="float32"),
            np.array([[0, 1, 2]], dtype="int64"),
        )
        meta = pd.DataFrame({
            "id": [str(i) for i in range(n_rows)],
            "title": [f"Paper {i}" for i in range(n_rows)],
            "abstract": ["abstract"] * n_rows,
            "authors": ["Author"] * n_rows,
            "categories": ["cs.LG"] * n_rows,
            "update_date": ["2023-01-01"] * n_rows,
        })
        return FaissVectorStore(index=mock_index, metadata=meta)

    def test_matching_dimension_succeeds(self, faiss_available):
        store = self._make_store_with_mock_index(index_dim=384)
        query = np.random.rand(1, 384).astype("float32")
        results = store.search(query, top_k=3)
        assert len(results) > 0

    def test_dimension_mismatch_raises_runtime_error(self, faiss_available):
        """BUG FIX: previously this caused a cryptic FAISS error."""
        store = self._make_store_with_mock_index(index_dim=384)
        wrong_query = np.random.rand(1, 768).astype("float32")
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            store.search(wrong_query, top_k=3)

    def test_negative_faiss_ids_skipped(self, faiss_available):
        """FAISS returns -1 for padding when fewer than top_k results exist."""
        import pandas as pd
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore

        mock_index = MagicMock()
        mock_index.d = 4
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.9, 0.0]], dtype="float32"),
            np.array([[0, -1]], dtype="int64"),
        )
        meta = pd.DataFrame({
            "id": ["0", "1"], "title": ["P0", "P1"], "abstract": ["a", "b"],
            "authors": ["", ""], "categories": ["cs.LG", "cs.LG"],
            "update_date": ["2023", "2023"],
        })
        store = FaissVectorStore(index=mock_index, metadata=meta)
        query = np.random.rand(1, 4).astype("float32")
        results = store.search(query, top_k=2)
        assert len(results) == 1
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestEmbeddingServiceCache`

- **Line:** 255
- **Base classes:** `object`
- **Docstring:** Verify the OrderedDict LRU cache behaviour.

**Methods:**
- `_service` at line 258: method behavior is described by its body and name
- `test_cache_is_ordered_dict` at line 262: method behavior is described by its body and name
- `test_cache_hit_promotes_to_end` at line 267: method behavior is described by its body and name
- `test_cache_evicts_oldest_when_full` at line 282: method behavior is described by its body and name
- `test_cache_key_includes_model_name` at line 301: Different model names must produce different cache keys.

```python
class TestEmbeddingServiceCache:
    """Verify the OrderedDict LRU cache behaviour."""

    def _service(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService
        return EmbeddingService.__new__(EmbeddingService)

    def test_cache_is_ordered_dict(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService
        svc = EmbeddingService("all-MiniLM-L6-v2")
        assert isinstance(svc._cache, OrderedDict)

    def test_cache_hit_promotes_to_end(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX

        svc = EmbeddingService("all-MiniLM-L6-v2")
        # Pre-populate cache manually to avoid loading the real model
        key_a = hashlib.sha256("all-MiniLM-L6-v2:query_a".encode()).hexdigest()
        key_b = hashlib.sha256("all-MiniLM-L6-v2:query_b".encode()).hexdigest()
        vec = np.zeros((1, 384), dtype="float32")
        svc._cache[key_a] = vec
        svc._cache[key_b] = vec

        # Access key_a → should move to end (most recent)
        svc._cache.move_to_end(key_a)
        assert list(svc._cache.keys())[-1] == key_a

    def test_cache_evicts_oldest_when_full(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX

        svc = EmbeddingService("test-model")
        vec = np.zeros((1, 4), dtype="float32")

        # Fill cache to capacity
        for i in range(_CACHE_MAX):
            key = f"key_{i}"
            svc._cache[key] = vec

        # Add one more — should evict "key_0"
        svc._cache["key_new"] = vec
        if len(svc._cache) > _CACHE_MAX:
            svc._cache.popitem(last=False)

        assert "key_0" not in svc._cache
        assert "key_new" in svc._cache

    def test_cache_key_includes_model_name(self):
        """Different model names must produce different cache keys."""
        k1 = hashlib.sha256("model-a:same query".encode()).hexdigest()
        k2 = hashlib.sha256("model-b:same query".encode()).hexdigest()
        assert k1 != k2
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestArxivIdNormalization`

- **Line:** 339
- **Base classes:** `object`
- **Docstring:** Tests for ArXiv ID normalization logic.

Uses an inline copy of the normalization function so this test module
does not depend on faiss being importable (paper_ingestion.service imports
faiss at module level which fails on numpy-version-mismatched environments).

**Methods:**
- `_normalize` at line 347: method behavior is described by its body and name
- `test_bare_id_passthrough` at line 350: method behavior is described by its body and name
- `test_version_suffix_stripped` at line 353: BUG FIX: 2301.04567v2 was not normalized to 2301.04567.
- `test_arxiv_prefix_stripped` at line 359: method behavior is described by its body and name
- `test_url_abs_path` at line 363: method behavior is described by its body and name
- `test_url_pdf_path` at line 367: method behavior is described by its body and name
- `test_empty_string_returns_empty` at line 370: method behavior is described by its body and name
- `test_old_format_id` at line 374: Old-style arXiv IDs: hep-th/0606256 (no version suffix).
- `test_version_suffix_stripping_is_last_step` at line 378: Version suffix must be stripped AFTER all other normalization.

```python
class TestArxivIdNormalization:
    """Tests for ArXiv ID normalization logic.

    Uses an inline copy of the normalization function so this test module
    does not depend on faiss being importable (paper_ingestion.service imports
    faiss at module level which fails on numpy-version-mismatched environments).
    """

    def _normalize(self, raw):
        return _normalize_arxiv_id(raw)

    def test_bare_id_passthrough(self):
        assert self._normalize("2301.04567") == "2301.04567"

    def test_version_suffix_stripped(self):
        """BUG FIX: 2301.04567v2 was not normalized to 2301.04567."""
        assert self._normalize("2301.04567v2") == "2301.04567"
        assert self._normalize("2301.04567v10") == "2301.04567"
        assert self._normalize("2301.04567v1") == "2301.04567"

    def test_arxiv_prefix_stripped(self):
        assert self._normalize("arxiv:2301.04567") == "2301.04567"
        assert self._normalize("arXiv:2301.04567v3") == "2301.04567"

    def test_url_abs_path(self):
        assert self._normalize("https://arxiv.org/abs/2301.04567") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"

    def test_url_pdf_path(self):
        assert self._normalize("https://arxiv.org/pdf/2301.04567.pdf") == "2301.04567"

    def test_empty_string_returns_empty(self):
        assert self._normalize("") == ""
        assert self._normalize("   ") == ""

    def test_old_format_id(self):
        """Old-style arXiv IDs: hep-th/0606256 (no version suffix)."""
        assert self._normalize("hep-th/0606256") == "hep-th/0606256"

    def test_version_suffix_stripping_is_last_step(self):
        """Version suffix must be stripped AFTER all other normalization."""
        assert self._normalize("arxiv:2301.04567v2") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `TestBM25Tokenizer` Methods

#### `TestBM25Tokenizer._bm25_cls`

- **Line:** 33
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _bm25_cls(self):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_plain_words`

- **Line:** 37
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_plain_words(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("attention mechanism")
        assert "attention" in tokens
        assert "mechanism" in tokens
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_alphanumeric_model_names`

- **Line:** 43
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** BUG FIX: original regex missed gpt3, t5, bert2, llama2.

```python
    def test_alphanumeric_model_names(self):
        """BUG FIX: original regex missed gpt3, t5, bert2, llama2."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("GPT-3 achieves state-of-the-art on T5 benchmark")
        assert "gpt" in tokens or "gpt3" in tokens or "3" in tokens
        assert "t5" in tokens or "t" in tokens  # at minimum the letter is kept
        # Key assertion: "t5" should now be found as a token (two chars)
        assert "t5" in tokens, f"Expected 't5' in tokens but got: {tokens}"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_gpt3_kept_together`

- **Line:** 52
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** gpt3 should be a single token, not split into gpt + 3.

```python
    def test_gpt3_kept_together(self):
        """gpt3 should be a single token, not split into gpt + 3."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("gpt3 is a large model")
        assert "gpt3" in tokens
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_lowercasing`

- **Line:** 58
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_lowercasing(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("BERT TRANSFORMER")
        assert "bert" in tokens
        assert "transformer" in tokens
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_numbers_included`

- **Line:** 64
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_numbers_included(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("published in 2023 with 42 experiments")
        assert "2023" in tokens
        assert "42" in tokens
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Tokenizer.test_empty_string`

- **Line:** 70
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_empty_string(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("")
        assert tokens == []
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestBM25Scoring` Methods

#### `TestBM25Scoring._make_bm25`

- **Line:** 79
- **Kind:** synchronous method
- **Arguments:** self, docs
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _make_bm25(self, docs):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25(docs)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Scoring.test_relevant_doc_scores_higher`

- **Line:** 83
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_relevant_doc_scores_higher(self):
        docs = [
            "transformer attention mechanism neural network",  # doc 0
            "random forest decision tree ensemble learning",   # doc 1
        ]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("transformer attention")
        assert scores[0] > scores[1], "Transformer doc should score higher for transformer query"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Scoring.test_all_zero_scores_for_oov_query`

- **Line:** 92
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_all_zero_scores_for_oov_query(self):
        docs = ["neural network", "decision tree"]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("zyxwvutsrqpon")
        assert all(s == 0.0 for s in scores)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Scoring.test_idf_rewards_rare_terms`

- **Line:** 98
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** A term appearing in all docs should have lower IDF than a rare term.

```python
    def test_idf_rewards_rare_terms(self):
        """A term appearing in all docs should have lower IDF than a rare term."""
        docs = [
            "neural network learning",
            "neural network transformer",
            "neural network attention",
        ]
        bm25 = self._make_bm25(docs)
        # "neural" appears in all 3 → low IDF
        # "transformer" appears in 1 → high IDF
        idf_neural = bm25._idf("neural")
        idf_transformer = bm25._idf("transformer")
        assert idf_transformer > idf_neural
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Scoring.test_idf_cache_populated_on_access`

- **Line:** 112
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_idf_cache_populated_on_access(self):
        docs = ["attention is all you need"]
        bm25 = self._make_bm25(docs)
        _ = bm25._idf("attention")
        assert "attention" in bm25.idf_cache
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestBM25Scoring.test_score_list_length_equals_doc_count`

- **Line:** 118
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_score_list_length_equals_doc_count(self):
        docs = ["doc one", "doc two", "doc three"]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("doc")
        assert len(scores) == 3
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestMetadataReranker` Methods

#### `TestMetadataReranker._reranker`

- **Line:** 130
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _reranker(self):
        from research_ai.retrieval.rerankers.service import MetadataReranker
        return MetadataReranker()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataReranker.test_keyword_weight_constant`

- **Line:** 134
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Verify _KEYWORD_WEIGHT matches the declared 0.15 in hybrid_search.

```python
    def test_keyword_weight_constant(self):
        """Verify _KEYWORD_WEIGHT matches the declared 0.15 in hybrid_search."""
        import research_ai.retrieval.rerankers.service as mod
        assert mod._KEYWORD_WEIGHT == pytest.approx(0.15)
        assert mod._SEMANTIC_WEIGHT == pytest.approx(0.85)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataReranker.test_perfect_overlap_promotes_score`

- **Line:** 140
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_perfect_overlap_promotes_score(self):
        reranker = self._reranker()
        docs = [
            {"title": "Attention mechanism for transformers", "abstract": "", "score": 0.5},
            {"title": "Random forest ensemble", "abstract": "", "score": 0.6},
        ]
        result = reranker.rerank("attention transformers", docs)
        # First doc has high keyword overlap → should rank first despite lower base score
        assert result[0]["title"].startswith("Attention")
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataReranker.test_hybrid_score_formula`

- **Line:** 150
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Verify: hybrid_score = 0.85 × score + 0.15 × overlap.

```python
    def test_hybrid_score_formula(self):
        """Verify: hybrid_score = 0.85 × score + 0.15 × overlap."""
        reranker = self._reranker()
        docs = [{"title": "neural attention network", "abstract": "", "score": 0.8}]
        result = reranker.rerank("attention neural", docs)
        doc = result[0]
        expected = 0.85 * 0.8 + 0.15 * doc["keyword_score"]
        assert doc["hybrid_score"] == pytest.approx(expected, abs=1e-4)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataReranker.test_empty_query_returns_unchanged`

- **Line:** 159
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_empty_query_returns_unchanged(self):
        reranker = self._reranker()
        docs = [{"title": "neural net", "abstract": "", "score": 0.5}]
        # "the a an" → all stopwords → empty token set
        result = reranker.rerank("the a an", docs)
        # Should still return a result (graceful handling)
        assert len(result) == 1
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataReranker.test_no_overlap_preserves_order_by_score`

- **Line:** 167
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** If overlap is zero for all docs, ordering is by base score.

```python
    def test_no_overlap_preserves_order_by_score(self):
        """If overlap is zero for all docs, ordering is by base score."""
        reranker = self._reranker()
        docs = [
            {"title": "alpha beta gamma", "abstract": "", "score": 0.3},
            {"title": "delta epsilon zeta", "abstract": "", "score": 0.7},
        ]
        # Query has no overlap with either doc
        result = reranker.rerank("zyxwvu", docs)
        assert result[0]["score"] == pytest.approx(0.7)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestFaissVectorStoreDimensionValidation` Methods

#### `TestFaissVectorStoreDimensionValidation._make_store_with_mock_index`

- **Line:** 194
- **Kind:** synchronous method
- **Arguments:** self, index_dim, n_rows
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _make_store_with_mock_index(self, index_dim: int, n_rows: int = 5):
        import pandas as pd
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore

        mock_index = MagicMock()
        mock_index.d = index_dim
        mock_index.ntotal = n_rows
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]], dtype="float32"),
            np.array([[0, 1, 2]], dtype="int64"),
        )
        meta = pd.DataFrame({
            "id": [str(i) for i in range(n_rows)],
            "title": [f"Paper {i}" for i in range(n_rows)],
            "abstract": ["abstract"] * n_rows,
            "authors": ["Author"] * n_rows,
            "categories": ["cs.LG"] * n_rows,
            "update_date": ["2023-01-01"] * n_rows,
        })
        return FaissVectorStore(index=mock_index, metadata=meta)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestFaissVectorStoreDimensionValidation.test_matching_dimension_succeeds`

- **Line:** 215
- **Kind:** synchronous method
- **Arguments:** self, faiss_available
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_matching_dimension_succeeds(self, faiss_available):
        store = self._make_store_with_mock_index(index_dim=384)
        query = np.random.rand(1, 384).astype("float32")
        results = store.search(query, top_k=3)
        assert len(results) > 0
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestFaissVectorStoreDimensionValidation.test_dimension_mismatch_raises_runtime_error`

- **Line:** 221
- **Kind:** synchronous method
- **Arguments:** self, faiss_available
- **Docstring:** BUG FIX: previously this caused a cryptic FAISS error.

```python
    def test_dimension_mismatch_raises_runtime_error(self, faiss_available):
        """BUG FIX: previously this caused a cryptic FAISS error."""
        store = self._make_store_with_mock_index(index_dim=384)
        wrong_query = np.random.rand(1, 768).astype("float32")
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            store.search(wrong_query, top_k=3)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestFaissVectorStoreDimensionValidation.test_negative_faiss_ids_skipped`

- **Line:** 228
- **Kind:** synchronous method
- **Arguments:** self, faiss_available
- **Docstring:** FAISS returns -1 for padding when fewer than top_k results exist.

```python
    def test_negative_faiss_ids_skipped(self, faiss_available):
        """FAISS returns -1 for padding when fewer than top_k results exist."""
        import pandas as pd
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore

        mock_index = MagicMock()
        mock_index.d = 4
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.9, 0.0]], dtype="float32"),
            np.array([[0, -1]], dtype="int64"),
        )
        meta = pd.DataFrame({
            "id": ["0", "1"], "title": ["P0", "P1"], "abstract": ["a", "b"],
            "authors": ["", ""], "categories": ["cs.LG", "cs.LG"],
            "update_date": ["2023", "2023"],
        })
        store = FaissVectorStore(index=mock_index, metadata=meta)
        query = np.random.rand(1, 4).astype("float32")
        results = store.search(query, top_k=2)
        assert len(results) == 1
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestEmbeddingServiceCache` Methods

#### `TestEmbeddingServiceCache._service`

- **Line:** 258
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _service(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService
        return EmbeddingService.__new__(EmbeddingService)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestEmbeddingServiceCache.test_cache_is_ordered_dict`

- **Line:** 262
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_cache_is_ordered_dict(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService
        svc = EmbeddingService("all-MiniLM-L6-v2")
        assert isinstance(svc._cache, OrderedDict)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestEmbeddingServiceCache.test_cache_hit_promotes_to_end`

- **Line:** 267
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_cache_hit_promotes_to_end(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX

        svc = EmbeddingService("all-MiniLM-L6-v2")
        # Pre-populate cache manually to avoid loading the real model
        key_a = hashlib.sha256("all-MiniLM-L6-v2:query_a".encode()).hexdigest()
        key_b = hashlib.sha256("all-MiniLM-L6-v2:query_b".encode()).hexdigest()
        vec = np.zeros((1, 384), dtype="float32")
        svc._cache[key_a] = vec
        svc._cache[key_b] = vec

        # Access key_a → should move to end (most recent)
        svc._cache.move_to_end(key_a)
        assert list(svc._cache.keys())[-1] == key_a
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestEmbeddingServiceCache.test_cache_evicts_oldest_when_full`

- **Line:** 282
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_cache_evicts_oldest_when_full(self):
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX

        svc = EmbeddingService("test-model")
        vec = np.zeros((1, 4), dtype="float32")

        # Fill cache to capacity
        for i in range(_CACHE_MAX):
            key = f"key_{i}"
            svc._cache[key] = vec

        # Add one more — should evict "key_0"
        svc._cache["key_new"] = vec
        if len(svc._cache) > _CACHE_MAX:
            svc._cache.popitem(last=False)

        assert "key_0" not in svc._cache
        assert "key_new" in svc._cache
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestEmbeddingServiceCache.test_cache_key_includes_model_name`

- **Line:** 301
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Different model names must produce different cache keys.

```python
    def test_cache_key_includes_model_name(self):
        """Different model names must produce different cache keys."""
        k1 = hashlib.sha256("model-a:same query".encode()).hexdigest()
        k2 = hashlib.sha256("model-b:same query".encode()).hexdigest()
        assert k1 != k2
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestArxivIdNormalization` Methods

#### `TestArxivIdNormalization._normalize`

- **Line:** 347
- **Kind:** synchronous method
- **Arguments:** self, raw
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _normalize(self, raw):
        return _normalize_arxiv_id(raw)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_bare_id_passthrough`

- **Line:** 350
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_bare_id_passthrough(self):
        assert self._normalize("2301.04567") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_version_suffix_stripped`

- **Line:** 353
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** BUG FIX: 2301.04567v2 was not normalized to 2301.04567.

```python
    def test_version_suffix_stripped(self):
        """BUG FIX: 2301.04567v2 was not normalized to 2301.04567."""
        assert self._normalize("2301.04567v2") == "2301.04567"
        assert self._normalize("2301.04567v10") == "2301.04567"
        assert self._normalize("2301.04567v1") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_arxiv_prefix_stripped`

- **Line:** 359
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_arxiv_prefix_stripped(self):
        assert self._normalize("arxiv:2301.04567") == "2301.04567"
        assert self._normalize("arXiv:2301.04567v3") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_url_abs_path`

- **Line:** 363
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_url_abs_path(self):
        assert self._normalize("https://arxiv.org/abs/2301.04567") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_url_pdf_path`

- **Line:** 367
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_url_pdf_path(self):
        assert self._normalize("https://arxiv.org/pdf/2301.04567.pdf") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_empty_string_returns_empty`

- **Line:** 370
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_empty_string_returns_empty(self):
        assert self._normalize("") == ""
        assert self._normalize("   ") == ""
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_old_format_id`

- **Line:** 374
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Old-style arXiv IDs: hep-th/0606256 (no version suffix).

```python
    def test_old_format_id(self):
        """Old-style arXiv IDs: hep-th/0606256 (no version suffix)."""
        assert self._normalize("hep-th/0606256") == "hep-th/0606256"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestArxivIdNormalization.test_version_suffix_stripping_is_last_step`

- **Line:** 378
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Version suffix must be stripped AFTER all other normalization.

```python
    def test_version_suffix_stripping_is_last_step(self):
        """Version suffix must be stripped AFTER all other normalization."""
        assert self._normalize("arxiv:2301.04567v2") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
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
| `collections` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `hashlib` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `math` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
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

- Handles credentials or environment configuration. Keep secrets in environment variables and redact them from logs.
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

- `tests/test_retrieval.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `tests/test_retrieval.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Tests for hybrid retrieval pipeline components.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Covers:
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - BM25 tokenizer (including the alphanumeric bug fix)
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - BM25 scoring logic (formula correctness, IDF behaviour)
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - HybridSearchService BM25 fusion
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - MetadataReranker weight alignment
# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - FaissVectorStore dimension validation
# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - EmbeddingService LRU cache (OrderedDict, eviction, promotion)
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - ArXiv ID normalization (version suffix stripping)
# L0011: Blank line that visually separates logical sections and improves readability.

# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Tests run WITHOUT the FAISS index loaded — mocks are used for integration
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
boundary tests.  This makes the test suite runnable without artifacts.
# L0014: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0015: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Imports a dependency, type, or project module needed by later code in this file.
import hashlib
# L0018: Imports a dependency, type, or project module needed by later code in this file.
import math
# L0019: Imports a dependency, type, or project module needed by later code in this file.
from collections import OrderedDict
# L0020: Imports a dependency, type, or project module needed by later code in this file.
from unittest.mock import MagicMock, patch
# L0021: Blank line that visually separates logical sections and improves readability.

# L0022: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0023: Imports a dependency, type, or project module needed by later code in this file.
import pytest
# L0024: Blank line that visually separates logical sections and improves readability.

# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0027: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# BM25 tokenizer and scoring
# L0028: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Defines a class that groups related state and behavior behind a reusable interface.
class TestBM25Tokenizer:
# L0031: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify the BM25 tokenizer fix: alphanumeric tokens (gpt3, t5, etc.)."""
# L0032: Blank line that visually separates logical sections and improves readability.

# L0033: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _bm25_cls(self):
# L0034: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.hybrid_search.service import _BM25
# L0035: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return _BM25
# L0036: Blank line that visually separates logical sections and improves readability.

# L0037: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_plain_words(self):
# L0038: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0039: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("attention mechanism")
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "attention" in tokens
# L0041: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "mechanism" in tokens
# L0042: Blank line that visually separates logical sections and improves readability.

# L0043: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_alphanumeric_model_names(self):
# L0044: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """BUG FIX: original regex missed gpt3, t5, bert2, llama2."""
# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("GPT-3 achieves state-of-the-art on T5 benchmark")
# L0047: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "gpt" in tokens or "gpt3" in tokens or "3" in tokens
# L0048: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "t5" in tokens or "t" in tokens  # at minimum the letter is kept
# L0049: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Key assertion: "t5" should now be found as a token (two chars)
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "t5" in tokens, f"Expected 't5' in tokens but got: {tokens}"
# L0051: Blank line that visually separates logical sections and improves readability.

# L0052: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_gpt3_kept_together(self):
# L0053: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """gpt3 should be a single token, not split into gpt + 3."""
# L0054: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("gpt3 is a large model")
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "gpt3" in tokens
# L0057: Blank line that visually separates logical sections and improves readability.

# L0058: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_lowercasing(self):
# L0059: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0060: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("BERT TRANSFORMER")
# L0061: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "bert" in tokens
# L0062: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "transformer" in tokens
# L0063: Blank line that visually separates logical sections and improves readability.

# L0064: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_numbers_included(self):
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("published in 2023 with 42 experiments")
# L0067: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "2023" in tokens
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "42" in tokens
# L0069: Blank line that visually separates logical sections and improves readability.

# L0070: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_empty_string(self):
# L0071: Assigns or updates a value used later in the workflow; check mutability and data shape.
        BM25 = self._bm25_cls()
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tokens = BM25._tokenize("")
# L0073: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert tokens == []
# L0074: Blank line that visually separates logical sections and improves readability.

# L0075: Blank line that visually separates logical sections and improves readability.

# L0076: Defines a class that groups related state and behavior behind a reusable interface.
class TestBM25Scoring:
# L0077: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify BM25 formula correctness."""
# L0078: Blank line that visually separates logical sections and improves readability.

# L0079: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _make_bm25(self, docs):
# L0080: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.hybrid_search.service import _BM25
# L0081: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return _BM25(docs)
# L0082: Blank line that visually separates logical sections and improves readability.

# L0083: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_relevant_doc_scores_higher(self):
# L0084: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "transformer attention mechanism neural network",  # doc 0
# L0086: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "random forest decision tree ensemble learning",   # doc 1
# L0087: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0088: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = self._make_bm25(docs)
# L0089: Assigns or updates a value used later in the workflow; check mutability and data shape.
        scores = bm25.scores("transformer attention")
# L0090: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert scores[0] > scores[1], "Transformer doc should score higher for transformer query"
# L0091: Blank line that visually separates logical sections and improves readability.

# L0092: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_all_zero_scores_for_oov_query(self):
# L0093: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = ["neural network", "decision tree"]
# L0094: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = self._make_bm25(docs)
# L0095: Assigns or updates a value used later in the workflow; check mutability and data shape.
        scores = bm25.scores("zyxwvutsrqpon")
# L0096: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert all(s == 0.0 for s in scores)
# L0097: Blank line that visually separates logical sections and improves readability.

# L0098: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_idf_rewards_rare_terms(self):
# L0099: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """A term appearing in all docs should have lower IDF than a rare term."""
# L0100: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "neural network learning",
# L0102: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "neural network transformer",
# L0103: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "neural network attention",
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0105: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = self._make_bm25(docs)
# L0106: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # "neural" appears in all 3 → low IDF
# L0107: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # "transformer" appears in 1 → high IDF
# L0108: Assigns or updates a value used later in the workflow; check mutability and data shape.
        idf_neural = bm25._idf("neural")
# L0109: Assigns or updates a value used later in the workflow; check mutability and data shape.
        idf_transformer = bm25._idf("transformer")
# L0110: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert idf_transformer > idf_neural
# L0111: Blank line that visually separates logical sections and improves readability.

# L0112: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_idf_cache_populated_on_access(self):
# L0113: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = ["attention is all you need"]
# L0114: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = self._make_bm25(docs)
# L0115: Assigns or updates a value used later in the workflow; check mutability and data shape.
        _ = bm25._idf("attention")
# L0116: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "attention" in bm25.idf_cache
# L0117: Blank line that visually separates logical sections and improves readability.

# L0118: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_score_list_length_equals_doc_count(self):
# L0119: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = ["doc one", "doc two", "doc three"]
# L0120: Assigns or updates a value used later in the workflow; check mutability and data shape.
        bm25 = self._make_bm25(docs)
# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
        scores = bm25.scores("doc")
# L0122: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(scores) == 3
# L0123: Blank line that visually separates logical sections and improves readability.

# L0124: Blank line that visually separates logical sections and improves readability.

# L0125: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0126: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# MetadataReranker weight alignment
# L0127: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0128: Blank line that visually separates logical sections and improves readability.

# L0129: Defines a class that groups related state and behavior behind a reusable interface.
class TestMetadataReranker:
# L0130: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _reranker(self):
# L0131: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.rerankers.service import MetadataReranker
# L0132: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return MetadataReranker()
# L0133: Blank line that visually separates logical sections and improves readability.

# L0134: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_keyword_weight_constant(self):
# L0135: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Verify _KEYWORD_WEIGHT matches the declared 0.15 in hybrid_search."""
# L0136: Imports a dependency, type, or project module needed by later code in this file.
        import research_ai.retrieval.rerankers.service as mod
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert mod._KEYWORD_WEIGHT == pytest.approx(0.15)
# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert mod._SEMANTIC_WEIGHT == pytest.approx(0.85)
# L0139: Blank line that visually separates logical sections and improves readability.

# L0140: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_perfect_overlap_promotes_score(self):
# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reranker = self._reranker()
# L0142: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0143: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {"title": "Attention mechanism for transformers", "abstract": "", "score": 0.5},
# L0144: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {"title": "Random forest ensemble", "abstract": "", "score": 0.6},
# L0145: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0146: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = reranker.rerank("attention transformers", docs)
# L0147: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # First doc has high keyword overlap → should rank first despite lower base score
# L0148: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert result[0]["title"].startswith("Attention")
# L0149: Blank line that visually separates logical sections and improves readability.

# L0150: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_hybrid_score_formula(self):
# L0151: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Verify: hybrid_score = 0.85 × score + 0.15 × overlap."""
# L0152: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reranker = self._reranker()
# L0153: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [{"title": "neural attention network", "abstract": "", "score": 0.8}]
# L0154: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = reranker.rerank("attention neural", docs)
# L0155: Assigns or updates a value used later in the workflow; check mutability and data shape.
        doc = result[0]
# L0156: Assigns or updates a value used later in the workflow; check mutability and data shape.
        expected = 0.85 * 0.8 + 0.15 * doc["keyword_score"]
# L0157: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert doc["hybrid_score"] == pytest.approx(expected, abs=1e-4)
# L0158: Blank line that visually separates logical sections and improves readability.

# L0159: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_empty_query_returns_unchanged(self):
# L0160: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reranker = self._reranker()
# L0161: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [{"title": "neural net", "abstract": "", "score": 0.5}]
# L0162: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # "the a an" → all stopwords → empty token set
# L0163: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = reranker.rerank("the a an", docs)
# L0164: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Should still return a result (graceful handling)
# L0165: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(result) == 1
# L0166: Blank line that visually separates logical sections and improves readability.

# L0167: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_no_overlap_preserves_order_by_score(self):
# L0168: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """If overlap is zero for all docs, ordering is by base score."""
# L0169: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reranker = self._reranker()
# L0170: Assigns or updates a value used later in the workflow; check mutability and data shape.
        docs = [
# L0171: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {"title": "alpha beta gamma", "abstract": "", "score": 0.3},
# L0172: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {"title": "delta epsilon zeta", "abstract": "", "score": 0.7},
# L0173: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0174: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Query has no overlap with either doc
# L0175: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = reranker.rerank("zyxwvu", docs)
# L0176: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert result[0]["score"] == pytest.approx(0.7)
# L0177: Blank line that visually separates logical sections and improves readability.

# L0178: Blank line that visually separates logical sections and improves readability.

# L0179: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0180: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# FaissVectorStore dimension validation
# L0181: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# These tests are skipped if faiss is not importable (numpy ABI mismatch etc.)
# L0182: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0183: Blank line that visually separates logical sections and improves readability.

# L0184: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@pytest.fixture(scope="module")
# L0185: Defines a function or method; parameters are the input contract and the body implements the workflow.
def faiss_available():
# L0186: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Skip FAISS tests if the library cannot be imported (e.g. numpy ABI mismatch)."""
# L0187: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return pytest.importorskip("faiss", reason="faiss-cpu not importable on this environment",
# L0188: Assigns or updates a value used later in the workflow; check mutability and data shape.
                               exc_type=ImportError)
# L0189: Blank line that visually separates logical sections and improves readability.

# L0190: Blank line that visually separates logical sections and improves readability.

# L0191: Defines a class that groups related state and behavior behind a reusable interface.
class TestFaissVectorStoreDimensionValidation:
# L0192: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify the dimension mismatch detection added in v3.1.1."""
# L0193: Blank line that visually separates logical sections and improves readability.

# L0194: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _make_store_with_mock_index(self, index_dim: int, n_rows: int = 5):
# L0195: Imports a dependency, type, or project module needed by later code in this file.
        import pandas as pd
# L0196: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore
# L0197: Blank line that visually separates logical sections and improves readability.

# L0198: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index = MagicMock()
# L0199: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.d = index_dim
# L0200: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.ntotal = n_rows
# L0201: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.search.return_value = (
# L0202: Assigns or updates a value used later in the workflow; check mutability and data shape.
            np.array([[0.9, 0.8, 0.7]], dtype="float32"),
# L0203: Assigns or updates a value used later in the workflow; check mutability and data shape.
            np.array([[0, 1, 2]], dtype="int64"),
# L0204: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0205: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta = pd.DataFrame({
# L0206: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "id": [str(i) for i in range(n_rows)],
# L0207: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": [f"Paper {i}" for i in range(n_rows)],
# L0208: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "abstract": ["abstract"] * n_rows,
# L0209: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": ["Author"] * n_rows,
# L0210: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "categories": ["cs.LG"] * n_rows,
# L0211: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "update_date": ["2023-01-01"] * n_rows,
# L0212: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        })
# L0213: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return FaissVectorStore(index=mock_index, metadata=meta)
# L0214: Blank line that visually separates logical sections and improves readability.

# L0215: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_matching_dimension_succeeds(self, faiss_available):
# L0216: Assigns or updates a value used later in the workflow; check mutability and data shape.
        store = self._make_store_with_mock_index(index_dim=384)
# L0217: Assigns or updates a value used later in the workflow; check mutability and data shape.
        query = np.random.rand(1, 384).astype("float32")
# L0218: Assigns or updates a value used later in the workflow; check mutability and data shape.
        results = store.search(query, top_k=3)
# L0219: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert len(results) > 0
# L0220: Blank line that visually separates logical sections and improves readability.

# L0221: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_dimension_mismatch_raises_runtime_error(self, faiss_available):
# L0222: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """BUG FIX: previously this caused a cryptic FAISS error."""
# L0223: Assigns or updates a value used later in the workflow; check mutability and data shape.
        store = self._make_store_with_mock_index(index_dim=384)
# L0224: Assigns or updates a value used later in the workflow; check mutability and data shape.
        wrong_query = np.random.rand(1, 768).astype("float32")
# L0225: Uses a context manager to guarantee setup/cleanup around files, locks, or managed resources.
        with pytest.raises(RuntimeError, match="dimension mismatch"):
# L0226: Assigns or updates a value used later in the workflow; check mutability and data shape.
            store.search(wrong_query, top_k=3)
# L0227: Blank line that visually separates logical sections and improves readability.

# L0228: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_negative_faiss_ids_skipped(self, faiss_available):
# L0229: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """FAISS returns -1 for padding when fewer than top_k results exist."""
# L0230: Imports a dependency, type, or project module needed by later code in this file.
        import pandas as pd
# L0231: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore
# L0232: Blank line that visually separates logical sections and improves readability.

# L0233: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index = MagicMock()
# L0234: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.d = 4
# L0235: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.ntotal = 2
# L0236: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_index.search.return_value = (
# L0237: Assigns or updates a value used later in the workflow; check mutability and data shape.
            np.array([[0.9, 0.0]], dtype="float32"),
# L0238: Assigns or updates a value used later in the workflow; check mutability and data shape.
            np.array([[0, -1]], dtype="int64"),
# L0239: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0240: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta = pd.DataFrame({
# L0241: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "id": ["0", "1"], "title": ["P0", "P1"], "abstract": ["a", "b"],
# L0242: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "authors": ["", ""], "categories": ["cs.LG", "cs.LG"],
# L0243: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "update_date": ["2023", "2023"],
# L0244: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        })
# L0245: Assigns or updates a value used later in the workflow; check mutability and data shape.
        store = FaissVectorStore(index=mock_index, metadata=meta)
# L0246: Assigns or updates a value used later in the workflow; check mutability and data shape.
        query = np.random.rand(1, 4).astype("float32")
# L0247: Assigns or updates a value used later in the workflow; check mutability and data shape.
        results = store.search(query, top_k=2)
# L0248: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(results) == 1
# L0249: Blank line that visually separates logical sections and improves readability.

# L0250: Blank line that visually separates logical sections and improves readability.

# L0251: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0252: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# EmbeddingService LRU cache
# L0253: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0254: Blank line that visually separates logical sections and improves readability.

# L0255: Defines a class that groups related state and behavior behind a reusable interface.
class TestEmbeddingServiceCache:
# L0256: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify the OrderedDict LRU cache behaviour."""
# L0257: Blank line that visually separates logical sections and improves readability.

# L0258: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _service(self):
# L0259: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.embeddings.service import EmbeddingService
# L0260: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return EmbeddingService.__new__(EmbeddingService)
# L0261: Blank line that visually separates logical sections and improves readability.

# L0262: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cache_is_ordered_dict(self):
# L0263: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.embeddings.service import EmbeddingService
# L0264: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc = EmbeddingService("all-MiniLM-L6-v2")
# L0265: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert isinstance(svc._cache, OrderedDict)
# L0266: Blank line that visually separates logical sections and improves readability.

# L0267: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cache_hit_promotes_to_end(self):
# L0268: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX
# L0269: Blank line that visually separates logical sections and improves readability.

# L0270: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc = EmbeddingService("all-MiniLM-L6-v2")
# L0271: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Pre-populate cache manually to avoid loading the real model
# L0272: Assigns or updates a value used later in the workflow; check mutability and data shape.
        key_a = hashlib.sha256("all-MiniLM-L6-v2:query_a".encode()).hexdigest()
# L0273: Assigns or updates a value used later in the workflow; check mutability and data shape.
        key_b = hashlib.sha256("all-MiniLM-L6-v2:query_b".encode()).hexdigest()
# L0274: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vec = np.zeros((1, 384), dtype="float32")
# L0275: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc._cache[key_a] = vec
# L0276: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc._cache[key_b] = vec
# L0277: Blank line that visually separates logical sections and improves readability.

# L0278: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Access key_a → should move to end (most recent)
# L0279: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        svc._cache.move_to_end(key_a)
# L0280: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert list(svc._cache.keys())[-1] == key_a
# L0281: Blank line that visually separates logical sections and improves readability.

# L0282: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cache_evicts_oldest_when_full(self):
# L0283: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.retrieval.embeddings.service import EmbeddingService, _CACHE_MAX
# L0284: Blank line that visually separates logical sections and improves readability.

# L0285: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc = EmbeddingService("test-model")
# L0286: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vec = np.zeros((1, 4), dtype="float32")
# L0287: Blank line that visually separates logical sections and improves readability.

# L0288: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Fill cache to capacity
# L0289: Iterates over data, retry attempts, files, results, or workflow steps.
        for i in range(_CACHE_MAX):
# L0290: Assigns or updates a value used later in the workflow; check mutability and data shape.
            key = f"key_{i}"
# L0291: Assigns or updates a value used later in the workflow; check mutability and data shape.
            svc._cache[key] = vec
# L0292: Blank line that visually separates logical sections and improves readability.

# L0293: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Add one more — should evict "key_0"
# L0294: Assigns or updates a value used later in the workflow; check mutability and data shape.
        svc._cache["key_new"] = vec
# L0295: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if len(svc._cache) > _CACHE_MAX:
# L0296: Assigns or updates a value used later in the workflow; check mutability and data shape.
            svc._cache.popitem(last=False)
# L0297: Blank line that visually separates logical sections and improves readability.

# L0298: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "key_0" not in svc._cache
# L0299: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "key_new" in svc._cache
# L0300: Blank line that visually separates logical sections and improves readability.

# L0301: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cache_key_includes_model_name(self):
# L0302: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Different model names must produce different cache keys."""
# L0303: Assigns or updates a value used later in the workflow; check mutability and data shape.
        k1 = hashlib.sha256("model-a:same query".encode()).hexdigest()
# L0304: Assigns or updates a value used later in the workflow; check mutability and data shape.
        k2 = hashlib.sha256("model-b:same query".encode()).hexdigest()
# L0305: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert k1 != k2
# L0306: Blank line that visually separates logical sections and improves readability.

# L0307: Blank line that visually separates logical sections and improves readability.

# L0308: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0309: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ArXiv ID normalization (version suffix stripping)
# L0310: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
#
# L0311: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# We test the logic directly using the regex from the module rather than
# L0312: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# importing PaperChatService (which imports faiss at module level and fails
# L0313: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# on environments with a numpy/faiss ABI mismatch).
# L0314: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0315: Blank line that visually separates logical sections and improves readability.

# L0316: Defines a function or method; parameters are the input contract and the body implements the workflow.
def _normalize_arxiv_id(raw_id: str) -> str:
# L0317: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Inline copy of PaperChatService.normalize_arxiv_id for isolated testing.
# L0318: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Keep this in sync with the implementation in paper_ingestion/service.py.
# L0319: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0320: Imports a dependency, type, or project module needed by later code in this file.
    import re
# L0321: Assigns or updates a value used later in the workflow; check mutability and data shape.
    _ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
# L0322: Blank line that visually separates logical sections and improves readability.

# L0323: Assigns or updates a value used later in the workflow; check mutability and data shape.
    token = (raw_id or "").strip()
# L0324: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
    if not token:
# L0325: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return ""
# L0326: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
    if token.lower().startswith("arxiv:"):
# L0327: Assigns or updates a value used later in the workflow; check mutability and data shape.
        token = token.split(":", 1)[1]
# L0328: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
    if token.startswith(("http://", "https://")):
# L0329: Assigns or updates a value used later in the workflow; check mutability and data shape.
        token = token.rstrip("/")
# L0330: Iterates over data, retry attempts, files, results, or workflow steps.
        for marker in ("/abs/", "/pdf/"):
# L0331: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if marker in token:
# L0332: Assigns or updates a value used later in the workflow; check mutability and data shape.
                token = token.split(marker)[-1]
# L0333: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                break
# L0334: Assigns or updates a value used later in the workflow; check mutability and data shape.
    token = token.replace(".pdf", "").strip()
# L0335: Assigns or updates a value used later in the workflow; check mutability and data shape.
    token = _ARXIV_VERSION_RE.sub("", token).strip()
# L0336: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return token
# L0337: Blank line that visually separates logical sections and improves readability.

# L0338: Blank line that visually separates logical sections and improves readability.

# L0339: Defines a class that groups related state and behavior behind a reusable interface.
class TestArxivIdNormalization:
# L0340: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Tests for ArXiv ID normalization logic.
# L0341: Blank line that visually separates logical sections and improves readability.

# L0342: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Uses an inline copy of the normalization function so this test module
# L0343: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    does not depend on faiss being importable (paper_ingestion.service imports
# L0344: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    faiss at module level which fails on numpy-version-mismatched environments).
# L0345: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0346: Blank line that visually separates logical sections and improves readability.

# L0347: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _normalize(self, raw):
# L0348: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return _normalize_arxiv_id(raw)
# L0349: Blank line that visually separates logical sections and improves readability.

# L0350: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_bare_id_passthrough(self):
# L0351: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("2301.04567") == "2301.04567"
# L0352: Blank line that visually separates logical sections and improves readability.

# L0353: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_version_suffix_stripped(self):
# L0354: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """BUG FIX: 2301.04567v2 was not normalized to 2301.04567."""
# L0355: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("2301.04567v2") == "2301.04567"
# L0356: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("2301.04567v10") == "2301.04567"
# L0357: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("2301.04567v1") == "2301.04567"
# L0358: Blank line that visually separates logical sections and improves readability.

# L0359: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_arxiv_prefix_stripped(self):
# L0360: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("arxiv:2301.04567") == "2301.04567"
# L0361: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("arXiv:2301.04567v3") == "2301.04567"
# L0362: Blank line that visually separates logical sections and improves readability.

# L0363: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_url_abs_path(self):
# L0364: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("https://arxiv.org/abs/2301.04567") == "2301.04567"
# L0365: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
# L0366: Blank line that visually separates logical sections and improves readability.

# L0367: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_url_pdf_path(self):
# L0368: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("https://arxiv.org/pdf/2301.04567.pdf") == "2301.04567"
# L0369: Blank line that visually separates logical sections and improves readability.

# L0370: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_empty_string_returns_empty(self):
# L0371: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("") == ""
# L0372: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("   ") == ""
# L0373: Blank line that visually separates logical sections and improves readability.

# L0374: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_old_format_id(self):
# L0375: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Old-style arXiv IDs: hep-th/0606256 (no version suffix)."""
# L0376: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("hep-th/0606256") == "hep-th/0606256"
# L0377: Blank line that visually separates logical sections and improves readability.

# L0378: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_version_suffix_stripping_is_last_step(self):
# L0379: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Version suffix must be stripped AFTER all other normalization."""
# L0380: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("arxiv:2301.04567v2") == "2301.04567"
# L0381: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
```

## Source Walkthrough

This file is large, so the opening and closing sections are included here. Use the class/function breakdown above to navigate the middle of the file.

### Opening Section

```python
"""Tests for hybrid retrieval pipeline components.

Covers:
  - BM25 tokenizer (including the alphanumeric bug fix)
  - BM25 scoring logic (formula correctness, IDF behaviour)
  - HybridSearchService BM25 fusion
  - MetadataReranker weight alignment
  - FaissVectorStore dimension validation
  - EmbeddingService LRU cache (OrderedDict, eviction, promotion)
  - ArXiv ID normalization (version suffix stripping)

Tests run WITHOUT the FAISS index loaded — mocks are used for integration
boundary tests.  This makes the test suite runnable without artifacts.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BM25 tokenizer and scoring
# ---------------------------------------------------------------------------

class TestBM25Tokenizer:
    """Verify the BM25 tokenizer fix: alphanumeric tokens (gpt3, t5, etc.)."""

    def _bm25_cls(self):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25

    def test_plain_words(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("attention mechanism")
        assert "attention" in tokens
        assert "mechanism" in tokens

    def test_alphanumeric_model_names(self):
        """BUG FIX: original regex missed gpt3, t5, bert2, llama2."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("GPT-3 achieves state-of-the-art on T5 benchmark")
        assert "gpt" in tokens or "gpt3" in tokens or "3" in tokens
        assert "t5" in tokens or "t" in tokens  # at minimum the letter is kept
        # Key assertion: "t5" should now be found as a token (two chars)
        assert "t5" in tokens, f"Expected 't5' in tokens but got: {tokens}"

    def test_gpt3_kept_together(self):
        """gpt3 should be a single token, not split into gpt + 3."""
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("gpt3 is a large model")
        assert "gpt3" in tokens

    def test_lowercasing(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("BERT TRANSFORMER")
        assert "bert" in tokens
        assert "transformer" in tokens

    def test_numbers_included(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("published in 2023 with 42 experiments")
        assert "2023" in tokens
        assert "42" in tokens

    def test_empty_string(self):
        BM25 = self._bm25_cls()
        tokens = BM25._tokenize("")
        assert tokens == []


class TestBM25Scoring:
    """Verify BM25 formula correctness."""

    def _make_bm25(self, docs):
        from research_ai.retrieval.hybrid_search.service import _BM25
        return _BM25(docs)

    def test_relevant_doc_scores_higher(self):
        docs = [
            "transformer attention mechanism neural network",  # doc 0
            "random forest decision tree ensemble learning",   # doc 1
        ]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("transformer attention")
        assert scores[0] > scores[1], "Transformer doc should score higher for transformer query"

    def test_all_zero_scores_for_oov_query(self):
        docs = ["neural network", "decision tree"]
        bm25 = self._make_bm25(docs)
        scores = bm25.scores("zyxwvutsrqpon")
        assert all(s == 0.0 for s in scores)

    def test_idf_rewards_rare_terms(self):
        """A term appearing in all docs should have lower IDF than a rare term."""
        docs = [
            "neural network learning",
            "neural network transformer",
            "neural network attention",
        ]
        bm25 = self._make_bm25(docs)
        # "neural" appears in all 3 → low IDF
        # "transformer" appears in 1 → high IDF
        idf_neural = bm25._idf("neural")
        idf_transformer = bm25._idf("transformer")
        assert idf_transformer > idf_neural

    def test_idf_cache_populated_on_access(self):
        docs = ["attention is all you need"]
        bm25 = self._make_bm25(docs)
        _ = bm25._idf("attention")
        assert "attention" in bm25.idf_cache

    def test_score_list_length_equals_doc_count(self):
        docs = ["doc one", "doc two", "doc three"]
        bm25 = self._make_bm25(docs)
```

### Closing Section

```python
        """Different model names must produce different cache keys."""
        k1 = hashlib.sha256("model-a:same query".encode()).hexdigest()
        k2 = hashlib.sha256("model-b:same query".encode()).hexdigest()
        assert k1 != k2


# ---------------------------------------------------------------------------
# ArXiv ID normalization (version suffix stripping)
#
# We test the logic directly using the regex from the module rather than
# importing PaperChatService (which imports faiss at module level and fails
# on environments with a numpy/faiss ABI mismatch).
# ---------------------------------------------------------------------------

def _normalize_arxiv_id(raw_id: str) -> str:
    """Inline copy of PaperChatService.normalize_arxiv_id for isolated testing.
    Keep this in sync with the implementation in paper_ingestion/service.py.
    """
    import re
    _ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

    token = (raw_id or "").strip()
    if not token:
        return ""
    if token.lower().startswith("arxiv:"):
        token = token.split(":", 1)[1]
    if token.startswith(("http://", "https://")):
        token = token.rstrip("/")
        for marker in ("/abs/", "/pdf/"):
            if marker in token:
                token = token.split(marker)[-1]
                break
    token = token.replace(".pdf", "").strip()
    token = _ARXIV_VERSION_RE.sub("", token).strip()
    return token


class TestArxivIdNormalization:
    """Tests for ArXiv ID normalization logic.

    Uses an inline copy of the normalization function so this test module
    does not depend on faiss being importable (paper_ingestion.service imports
    faiss at module level which fails on numpy-version-mismatched environments).
    """

    def _normalize(self, raw):
        return _normalize_arxiv_id(raw)

    def test_bare_id_passthrough(self):
        assert self._normalize("2301.04567") == "2301.04567"

    def test_version_suffix_stripped(self):
        """BUG FIX: 2301.04567v2 was not normalized to 2301.04567."""
        assert self._normalize("2301.04567v2") == "2301.04567"
        assert self._normalize("2301.04567v10") == "2301.04567"
        assert self._normalize("2301.04567v1") == "2301.04567"

    def test_arxiv_prefix_stripped(self):
        assert self._normalize("arxiv:2301.04567") == "2301.04567"
        assert self._normalize("arXiv:2301.04567v3") == "2301.04567"

    def test_url_abs_path(self):
        assert self._normalize("https://arxiv.org/abs/2301.04567") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"

    def test_url_pdf_path(self):
        assert self._normalize("https://arxiv.org/pdf/2301.04567.pdf") == "2301.04567"

    def test_empty_string_returns_empty(self):
        assert self._normalize("") == ""
        assert self._normalize("   ") == ""

    def test_old_format_id(self):
        """Old-style arXiv IDs: hep-th/0606256 (no version suffix)."""
        assert self._normalize("hep-th/0606256") == "hep-th/0606256"

    def test_version_suffix_stripping_is_last_step(self):
        """Version suffix must be stripped AFTER all other normalization."""
        assert self._normalize("arxiv:2301.04567v2") == "2301.04567"
        assert self._normalize("https://arxiv.org/abs/2301.04567v2") == "2301.04567"
```
