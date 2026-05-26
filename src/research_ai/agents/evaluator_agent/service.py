"""Evaluator agent — assesses output quality and decides on retry/escalation.

The evaluator is the quality gate in the Plan→Execute→Evaluate→Synthesize loop.
It scores tool outputs across four orthogonal dimensions and triggers a wider
search retry when the score falls below RETRY_THRESHOLD.

SCORING MODEL (weights sum to 1.0):
  1. Retrieval hit-rate (0.40) — are there actual paper results?
     Saturates at 5 results (0.08 × count, capped at 0.40).
     WHY 0.40: retrieval quality is the single biggest predictor of answer quality
     in a RAG system. No evidence → no grounded answer → always retry.

  2. Answer completeness (0.30) — does any tool produce a substantive answer?
     Full credit (0.30) if ≥20 words; half credit (0.15) if any text at all.
     WHY 0.30: a short or missing answer is the second-most common failure mode.

  3. Evidence grounding (0.20) — are there methodology signals, citation patterns,
     or a confirmed classification?
     WHY 0.20: rewards depth of evidence, not just raw retrieval.

  4. Error absence (0.10) — did critical tools run without errors?
     WHY 0.10: an error in a critical tool is a signal to retry, but not as
     serious as zero retrieval; a failed classify still leaves hybrid_search.

RETRY_THRESHOLD = 0.35: at this point the retrieval sub-score alone contributes
0 (no results) or at most 0.08 (one paper), suggesting the first pass missed
the corpus almost entirely. A wider top_k search is warranted.

BUG FIX (v3.1.1): the original code only checked outputs["hybrid_search"] for
retrieval quality. When the planner chose smart_retrieve instead of hybrid_search
(e.g., for citation-aware or short queries), the retrieval sub-score was always
0, falsely triggering a retry every time. The fix checks BOTH retrieval tools
and takes the better score.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Both tools that perform retrieval — the evaluator must consider either one.
# smart_retrieve is the RetrievalAgent wrapper around HybridSearchService;
# it returns results in the same {"results": [...], "count": N} structure.
_RETRIEVAL_TOOLS = ("hybrid_search", "smart_retrieve")


class EvaluatorAgent:
    """Evaluates tool output quality and decides whether a retry is warranted.

    Scoring model (0–1):
      - Retrieval hit-rate:    results present and relevant          (0.4)
      - Answer completeness:   final_answer / synthesis present      (0.3)
      - Evidence grounding:    methodology or citation signals found (0.2)
      - Error absence:         no error keys in critical tools       (0.1)

    A score below RETRY_THRESHOLD triggers a retry with a wider search.
    """

    RETRY_THRESHOLD = 0.35
    ESCALATE_THRESHOLD = 0.10

    def evaluate(self, outputs: dict) -> dict:
        """Score the tool outputs and return evaluation metadata.

        Returns a dict with:
          quality_score   — float in [0, 1]
          needs_retry     — bool: score below RETRY_THRESHOLD
          needs_escalation— bool: score below ESCALATE_THRESHOLD (very bad)
          breakdown       — per-dimension subscores
          retry_top_k_multiplier — how much to expand search on retry
          reason          — human-readable retry reason or "sufficient_evidence"
          tool_errors     — dict of tool→error message for any erroring tools
        """
        score, breakdown = self._score(outputs)
        needs_retry = score < self.RETRY_THRESHOLD
        needs_escalation = score < self.ESCALATE_THRESHOLD

        result: dict = {
            "quality_score": round(score, 3),
            "needs_retry": needs_retry,
            "needs_escalation": needs_escalation,
            "breakdown": breakdown,
        }

        if needs_retry:
            # x3 expansion for catastrophic failure (single paper or less),
            # x2 for marginal failure. Both capped at max_top_k=20 in the orchestrator.
            multiplier = 3 if score < 0.15 else 2
            result["retry_top_k_multiplier"] = multiplier
            result["reason"] = self._retry_reason(outputs, breakdown)
            logger.info(
                "EvaluatorAgent: score=%.2f -> retry (x%d) reason=%s",
                score, multiplier, result["reason"],
            )
        else:
            result["reason"] = "sufficient_evidence"

        # Surface all tool errors so the orchestrator can log/expose them
        errors = {
            tool: info["error"]
            for tool, info in outputs.items()
            if isinstance(info, dict) and info.get("error")
        }
        if errors:
            result["tool_errors"] = errors

        return result

    def _score(self, outputs: dict) -> tuple[float, dict]:
        """Compute a composite quality score from tool outputs.

        Returns (score: float, breakdown: dict) where score ∈ [0, 1].
        Each dimension's weight is documented in the module docstring.
        """
        breakdown: dict = {}
        total = 0.0

        # ------------------------------------------------------------------
        # 1. Retrieval hit-rate (max 0.40)
        #
        # BUG FIX: previously only checked outputs["hybrid_search"].
        # When the planner chose smart_retrieve (RetrievalAgent path), this
        # sub-score was always 0, triggering a spurious retry on every single
        # smart_retrieve call.
        #
        # Fix: check BOTH retrieval tools; take whichever produced more results.
        # This is correct because exactly one of the two tools runs per plan.
        # ------------------------------------------------------------------
        retrieval_count = 0
        for tool_name in _RETRIEVAL_TOOLS:
            candidate = outputs.get(tool_name, {})
            if isinstance(candidate, dict) and not candidate.get("error"):
                count = candidate.get("count", 0)
                if count > retrieval_count:
                    retrieval_count = count

        # Saturates at 5 results: 0.08×5 = 0.40 (full marks).
        # Rationale: 5 papers is enough evidence for a grounded answer.
        retrieval_score = min(0.4, 0.08 * retrieval_count) if retrieval_count > 0 else 0.0
        breakdown["retrieval"] = round(retrieval_score, 3)
        total += retrieval_score

        # ------------------------------------------------------------------
        # 2. Answer completeness (max 0.30)
        #
        # Full credit if any synthesis tool produced ≥20 words.
        # Half credit for any non-empty text (e.g., very short summary).
        # WHY: a ≥20-word answer is the minimum viable researcher response.
        # ------------------------------------------------------------------
        answer_score = 0.0
        for key in ("metadata_rag", "paper_chat", "summarize", "conversation"):
            item = outputs.get(key, {})
            if isinstance(item, dict):
                text = item.get("answer") or item.get("summary") or ""
                if isinstance(text, str) and len(text.split()) >= 20:
                    answer_score = 0.3
                    break
                elif isinstance(text, str) and text.strip():
                    answer_score = 0.15  # short but non-empty
        breakdown["answer_completeness"] = round(answer_score, 3)
        total += answer_score

        # ------------------------------------------------------------------
        # 3. Evidence grounding (max 0.20)
        #
        # Methodology signals (+0.10): specific methods/datasets were extracted,
        #   meaning the retrieval was specific enough to contain method text.
        # Citation co-occurrence (+0.05): category patterns found across papers,
        #   meaning multiple papers share a research area.
        # Classification (+0.05): the query was mapped to an arXiv category,
        #   confirming the topic is within the indexed domain.
        # ------------------------------------------------------------------
        evidence_score = 0.0
        methodology = outputs.get("methodology_extract", {})
        if isinstance(methodology, dict) and methodology.get("count", 0) > 0:
            evidence_score += 0.1
        citation = outputs.get("citation_signals", {})
        if isinstance(citation, dict) and citation.get("category_cooccurrence"):
            evidence_score += 0.05
        classify = outputs.get("classify_query", {})
        if isinstance(classify, dict) and classify.get("predicted_category"):
            evidence_score += 0.05
        breakdown["evidence_grounding"] = round(evidence_score, 3)
        total += evidence_score

        # ------------------------------------------------------------------
        # 4. Error absence (max 0.10)
        #
        # Only critical tools are penalised: hybrid_search, smart_retrieve,
        # metadata_rag, classify_query. Optional tools (methodology, citation)
        # can fail without triggering a retry.
        # ------------------------------------------------------------------
        critical_tools = ("hybrid_search", "smart_retrieve", "metadata_rag", "classify_query")
        error_free = all(
            not isinstance(outputs.get(t, {}), dict) or not outputs.get(t, {}).get("error")
            for t in critical_tools
        )
        error_score = 0.1 if error_free else 0.0
        breakdown["error_absence"] = error_score
        total += error_score

        return min(total, 1.0), breakdown

    @staticmethod
    def _retry_reason(outputs: dict, breakdown: dict) -> str:
        """Return the primary reason for triggering a retry (for logging and UI)."""
        if breakdown.get("retrieval", 0) == 0:
            return "no_retrieval_hits"
        if breakdown.get("answer_completeness", 0) == 0:
            return "no_answer_generated"
        if breakdown.get("evidence_grounding", 0) == 0:
            return "insufficient_evidence"
        return "low_overall_quality"
