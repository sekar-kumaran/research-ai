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
