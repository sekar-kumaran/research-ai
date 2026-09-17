"""Hybrid retrieval: FAISS semantic + BM25 keyword + metadata reranking.

THREE-STAGE PIPELINE
--------------------
Stage 1 — Semantic FAISS  (weight 0.60):
  Embed the query with SentenceTransformer, run IndexFlatIP (inner product ≡
  cosine similarity for L2-normalised vectors), retrieve ``candidate_k`` docs.
  Fast: O(d × N) with FAISS BLAS, typically <10ms for 100k vectors.

Stage 2 — BM25 keyword fusion  (weight 0.25):
  Apply Okapi BM25 to the candidate set returned by Stage 1.
  IMPORTANT DESIGN NOTE: BM25 runs over the ~60-doc candidate set, NOT the
  full corpus.  This is intentional for two reasons:
    a) Building a full-corpus BM25 index at startup would require loading all
       abstracts into RAM and rebuilding every time the FAISS index changes.
    b) The candidate set is already semantically relevant; BM25 re-ranks within
       that relevant slice, which is exactly what we want.
  The trade-off is that IDF values are computed over 60 docs rather than 100k,
  making rare-in-candidates terms score higher than in full-corpus BM25.
  In practice this is acceptable because the candidates are already on-topic,
  so the IDF signal correctly distinguishes within-topic specificity.

Stage 3 — Metadata reranking  (weight 0.15, via MetadataReranker):
  Keyword overlap between query tokens and doc tokens, re-weighted:
    hybrid_score = 0.85 × fused_score + 0.15 × keyword_overlap
  This gives a small extra boost to docs whose title/abstract literally contain
  query words that the embedding model may have mapped to synonyms.

WEIGHT RATIONALE
----------------
0.60 semantic: embeddings capture meaning across paraphrases and synonyms.
0.25 BM25:     exact-match keyword signals are critical for model names (BERT,
               GPT-4), datasets (CIFAR-10), and specific metric names (BLEU).
0.15 metadata: title-level keyword overlap is the strongest lexical signal but
               is already partially captured by BM25, so it gets less weight.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from research_ai.retrieval.rerankers import MetadataReranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight BM25 implementation (no extra dependencies)
# ---------------------------------------------------------------------------

class _BM25:
    """Okapi BM25 scorer over a small document set.

    Formula (per query token t, document d):
        score(t, d) = IDF(t) × (tf(t,d) × (K1+1)) / (tf(t,d) + K1×(1 - B + B×|d|/avgdl))

    Parameters:
        K1 = 1.5  — term-frequency saturation. Higher K1 gives more weight to
                    high-frequency terms (aggressive TF scaling). Standard range 1.2–2.0.
        B  = 0.75 — document-length normalisation. B=1 full normalisation; B=0 none.
                    0.75 is the widely validated default for document collections.

    IDF formula used here is the smoothed Robertson IDF:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    The +1 inside the log keeps IDF positive even for tokens appearing in every doc.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: list[str]) -> None:
        self.docs: list[list[str]] = [self._tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        # df[token] = number of documents containing the token (for IDF)
        self.df: Counter = Counter()
        for doc in self.docs:
            for token in set(doc):  # set() so each doc contributes 1 per token
                self.df[token] += 1
        self.idf_cache: dict[str, float] = {}

    def _idf(self, token: str) -> float:
        """Robertson smoothed IDF, cached per token."""
        if token not in self.idf_cache:
            df = self.df.get(token, 0)
            self.idf_cache[token] = math.log(
                (self.n - df + 0.5) / (df + 0.5) + 1
            )
        return self.idf_cache[token]

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score for each document against the query."""
        q_tokens = self._tokenize(query)
        result: list[float] = []
        for doc in self.docs:
            dl = len(doc)
            tf_map: Counter = Counter(doc)
            sc = 0.0
            for token in q_tokens:
                tf = tf_map.get(token, 0)
                idf = self._idf(token)
                num = tf * (self.K1 + 1)
                den = tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                sc += idf * num / max(den, 1e-9)
            result.append(sc)
        return result

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize scientific text into lowercase word-like tokens.

        BUG FIX (v3.1.1): original regex r"\\b[a-z]{2,}\\b" missed:
          - Alphanumeric model names: gpt3, t5, bert2, llama2, phi3
          - Pure numbers: 2019, 2023 (publication years used as filters)
          - Hyphenated terms: "pre-training" was split into "pre" and "training"
            (hyphen is not captured, but both parts are tokenized separately —
            this is actually fine for BM25 since both parts get scored)

        Fix: include sequences of 2+ characters that are alphanumeric (letters
        and/or digits), applied after lowercasing.

        Examples:
          "GPT-4 achieves BLEU of 42.3" → ["gpt", "4", "achieves", "bleu", "of", "42", "3"]
          Wait — we need word boundaries with \\b for alphanumeric:
          "gpt3" → ["gpt3"]   (correctly kept together)
          "pre-training" → ["pre", "training"]  (split at hyphen, both kept)
          "BERT" → ["bert"]
          "2023" → ["2023"]
        """
        return re.findall(r"\b[a-z0-9][a-z0-9]{1,}\b", text.lower())


# ---------------------------------------------------------------------------
# Hybrid Search Service
# ---------------------------------------------------------------------------

class HybridSearchService:
    """Three-stage hybrid retrieval: FAISS semantic → BM25 keyword → reranking.

    WEIGHTS (must sum to 1.0):
      SEMANTIC_WEIGHT = 0.60 — primary signal; handles paraphrases/synonyms
      BM25_WEIGHT     = 0.25 — exact-match boost for model names, datasets
      KEYWORD_WEIGHT  = 0.15 — passed to MetadataReranker for title overlap

    The MetadataReranker applies keyword weight as:
        final_score = (1 - KEYWORD_WEIGHT) × fused_score + KEYWORD_WEIGHT × overlap
    i.e.,   final_score = 0.85 × fused_score + 0.15 × keyword_overlap

    CANDIDATE POOL SIZE
    -------------------
    We retrieve ``candidate_k`` docs from FAISS (default: min(60, top_k×5))
    and then re-rank down to ``top_k``.  A larger pool improves recall at the
    cost of more BM25/reranker compute.  60 is the practical sweet spot for
    a typical 100k-paper index on CPU hardware.

    THREAD SAFETY
    -------------
    HybridSearchService holds no mutable state after construction (BM25 is
    built per-request from candidates).  It is safe to share across threads.
    """

    SEMANTIC_WEIGHT = 0.60
    BM25_WEIGHT = 0.25
    KEYWORD_WEIGHT = 0.15   # forwarded to MetadataReranker; must equal 1 - reranker_semantic_weight

    def __init__(
        self,
        embedding_service,
        vector_store,
        reranker: MetadataReranker | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker or MetadataReranker()

    @property
    def ready(self) -> bool:
        return bool(self.vector_store.ready)

    @property
    def metadata(self):
        return self.vector_store.metadata

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        candidate_k: int | None = None,
    ) -> dict:
        """Execute hybrid retrieval and return ranked results.

        Args:
            query:       Natural-language search query.
            top_k:       Final number of results to return.
            filters:     Optional metadata filters: {"category": "cs.LG", "year": "2023"}.
            candidate_k: FAISS candidate pool size before BM25+rerank.
                         Default: min(60, top_k × 5) — enough for good recall.

        Returns:
            dict with keys: query, retrieval_strategy, results, count, candidate_count.
            On failure: {"error": "<message>"}.
        """
        if not self.ready:
            return {"error": "Search index not ready. Build similarity artifacts first."}

        # Candidate pool: retrieve more than top_k from FAISS so BM25 and
        # the reranker have room to promote better-matching docs.
        # Formula: at least top_k, at most 60 (CPU memory/latency budget).
        candidate_count = max(top_k, candidate_k or min(60, max(top_k * 5, top_k)))

        # Stage 1: Semantic FAISS retrieval (cosine similarity via inner product
        # on L2-normalised vectors — see EmbeddingService.encode()).
        query_vec = self.embedding_service.encode([query])
        raw_docs = [doc.to_dict() for doc in self.vector_store.search(query_vec, candidate_count)]

        # Stage 2: BM25 fusion — re-ranks within the semantic candidate set.
        raw_docs = self._apply_bm25_fusion(query, raw_docs)

        # Stage 3: Metadata filter → MetadataReranker → truncate to top_k.
        raw_docs = self._apply_filters(raw_docs, filters or {})
        final_docs = self.reranker.rerank(query, raw_docs)[:top_k]

        return {
            "query": query,
            "retrieval_strategy": "hybrid_faiss_bm25_metadata",
            "results": final_docs,
            "count": len(final_docs),
            "candidate_count": len(raw_docs),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_bm25_fusion(self, query: str, docs: list[dict]) -> list[dict]:
        """Fuse FAISS semantic scores with BM25 keyword scores."""
        if not docs:
            return docs

        # Build BM25 index lazily over the retrieved candidates
        texts = [f"{d.get('title', '')} {d.get('abstract', '')}" for d in docs]
        bm25 = _BM25(texts)
        bm25_scores = bm25.scores(query)

        if not bm25_scores or max(bm25_scores) == 0:
            return docs

        max_bm25 = max(bm25_scores)
        fused: list[dict] = []
        for doc, bm25_score in zip(docs, bm25_scores):
            semantic_score = float(doc.get("score", 0.0))
            normalised_bm25 = bm25_score / max_bm25
            fused_score = (
                self.SEMANTIC_WEIGHT * semantic_score
                + self.BM25_WEIGHT * normalised_bm25
            )
            item = dict(doc)
            item["semantic_score"] = round(semantic_score, 4)
            item["bm25_score"] = round(normalised_bm25, 4)
            item["score"] = round(fused_score, 4)
            fused.append(item)

        return sorted(fused, key=lambda d: d["score"], reverse=True)

    @staticmethod
    def _apply_filters(docs: list[dict], filters: dict) -> list[dict]:
        category = str(filters.get("category", "")).strip().lower()
        year = str(filters.get("year", "")).strip()
        if not category and not year:
            return docs
        out: list[dict] = []
        for doc in docs:
            if category and category not in str(doc.get("category", "")).lower():
                continue
            if year and year != str(doc.get("year", "")):
                continue
            out.append(doc)
        return out
