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
