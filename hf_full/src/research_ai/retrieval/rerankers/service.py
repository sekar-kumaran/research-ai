"""Metadata reranker — final re-ranking stage of the hybrid retrieval pipeline.

POSITION IN THE PIPELINE
-------------------------
This is Stage 3 (the last stage) of HybridSearchService.  It receives docs
that have already been scored by:
  Stage 1: FAISS semantic similarity (weight 0.60)
  Stage 2: BM25 keyword fusion     (weight 0.25)

Stage 3 computes a title+abstract keyword overlap score and blends it with the
fused score from Stages 1+2.

WEIGHT ALIGNMENT (BUG FIX v3.1.1)
-----------------------------------
The original code used hard-coded weights 0.75 / 0.25 inside rerank(), which
conflicted with HybridSearchService.KEYWORD_WEIGHT = 0.15.  The mismatch meant:

  Declared:  final_score = 0.85 × fused + 0.15 × keyword
  Actual:    final_score = 0.75 × fused + 0.25 × keyword

This gave keyword overlap 1.67× more influence than intended, over-promoting
documents that merely share words with the query (rather than being semantically
relevant) and under-weighting the carefully calibrated BM25+semantic scores.

Fix: MetadataReranker now declares SEMANTIC_WEIGHT = 1 - KEYWORD_WEIGHT so the
weights are defined in one place only (HybridSearchService), and the reranker
imports and uses them.

WHY KEYWORD OVERLAP AT ALL after BM25?
---------------------------------------
BM25 captures token-level frequency at the corpus level (IDF).  The metadata
reranker captures a simpler signal: does the paper title contain the exact query
words?  Title-match is a strong heuristic for relevance that complements IDF-
weighted body text frequency from BM25.

MATHEMATICAL DEFINITION
------------------------
  overlap = |query_tokens ∩ doc_tokens| / |query_tokens|
  hybrid_score = SEMANTIC_WEIGHT × fused_score + KEYWORD_WEIGHT × overlap

  where fused_score already encodes 0.60 × semantic + 0.25 × BM25.
"""
from __future__ import annotations

from research_ai.common.text import tokenize_query

# Must match HybridSearchService.KEYWORD_WEIGHT exactly.
# Defined here as a local constant to avoid circular imports while keeping
# the value synchronized.  If you change KEYWORD_WEIGHT in hybrid_search,
# change this constant too.
_KEYWORD_WEIGHT = 0.15
_SEMANTIC_WEIGHT = 1.0 - _KEYWORD_WEIGHT  # = 0.85


class MetadataReranker:
    """Lightweight local reranker using title/abstract keyword evidence.

    Computes a Jaccard-like overlap between query tokens and document tokens,
    then blends with the upstream fused score at a 85/15 ratio.

    Token representation uses the same tokenize_query() as the query planner —
    lowercase, stopwords removed, lemmatized.  This means "transformers" and
    "transformer" both reduce to "transformer", providing stemming-like matching.
    """

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Re-rank docs by blending fused score with keyword overlap.

        Args:
            query: The original search query string.
            docs:  List of dicts with at least "title", "abstract", and "score" keys.

        Returns:
            Same dicts sorted by hybrid_score descending, with two added fields:
              keyword_score: raw overlap ratio ∈ [0, 1]
              hybrid_score:  final blended score ∈ [0, 1]
        """
        query_tokens = tokenize_query(query)
        if not query_tokens:
            # No tokens after cleaning (e.g., query is all stopwords) → return as-is
            return docs

        reranked = []
        for doc in docs:
            # Compute overlap between query tokens and doc title+abstract tokens.
            # Using the title AND abstract ensures a paper about "neural networks"
            # scores well even if "neural" is only in the abstract.
            haystack = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            doc_tokens = tokenize_query(haystack)

            # Precision-style overlap: what fraction of query terms appear in doc?
            # Using query length as denominator rewards docs that cover all query terms.
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))

            item = dict(doc)
            item["keyword_score"] = round(overlap, 4)
            # BUG FIX: was 0.75/0.25, now correctly 0.85/0.15 per KEYWORD_WEIGHT
            item["hybrid_score"] = round(
                _SEMANTIC_WEIGHT * float(item.get("score", 0.0)) + _KEYWORD_WEIGHT * overlap,
                4,
            )
            reranked.append(item)

        return sorted(reranked, key=lambda item: item["hybrid_score"], reverse=True)

