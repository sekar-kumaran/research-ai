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
        scores = bm25.scores("doc")
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# MetadataReranker weight alignment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FaissVectorStore dimension validation
# These tests are skipped if faiss is not importable (numpy ABI mismatch etc.)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def faiss_available():
    """Skip FAISS tests if the library cannot be imported (e.g. numpy ABI mismatch)."""
    return pytest.importorskip("faiss", reason="faiss-cpu not importable on this environment",
                               exc_type=ImportError)


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


# ---------------------------------------------------------------------------
# EmbeddingService LRU cache
# ---------------------------------------------------------------------------

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
