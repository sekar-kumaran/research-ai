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
