"""Tests for EvaluatorAgent — quality scoring and retry logic.

Covers:
  - Retrieval quality sub-score for hybrid_search AND smart_retrieve
  - Answer completeness scoring
  - Evidence grounding scoring
  - Error absence scoring
  - Retry threshold logic
  - Retry multiplier (x2 vs x3)
  - Retry reason mapping
  - Tool error surfacing
"""
from __future__ import annotations

import pytest

from research_ai.agents.evaluator_agent import EvaluatorAgent


@pytest.fixture
def ev():
    return EvaluatorAgent()


# ---------------------------------------------------------------------------
# 1. Retrieval quality — the core bug fix
# ---------------------------------------------------------------------------

class TestRetrievalQuality:
    def test_hybrid_search_zero_results_gives_zero(self, ev):
        outputs = {"hybrid_search": {"count": 0, "results": []}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == 0.0

    def test_hybrid_search_five_results_gives_full_score(self, ev):
        outputs = {"hybrid_search": {"count": 5, "results": [{}] * 5}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == pytest.approx(0.4)

    def test_smart_retrieve_counted_correctly(self, ev):
        """BUG FIX: smart_retrieve was previously ignored → always 0 retrieval score."""
        outputs = {"smart_retrieve": {"count": 5, "results": [{}] * 5}}
        result = ev.evaluate(outputs)
        # Should get full retrieval credit from smart_retrieve
        assert result["breakdown"]["retrieval"] == pytest.approx(0.4)

    def test_smart_retrieve_no_spurious_retry(self, ev):
        """Without the fix, smart_retrieve with 5 results would trigger a retry."""
        outputs = {
            "smart_retrieve": {"count": 5, "results": [{}] * 5},
            "metadata_rag": {"answer": "A substantive answer with at least twenty words here yes this counts"},
            "classify_query": {"predicted_category": "cs.LG"},
        }
        result = ev.evaluate(outputs)
        assert result["needs_retry"] is False, (
            f"Spurious retry triggered: score={result['quality_score']}, "
            f"reason={result.get('reason')}"
        )

    def test_takes_max_when_both_tools_present(self, ev):
        """If somehow both appear, take the higher count."""
        outputs = {
            "hybrid_search": {"count": 2, "results": [{}] * 2},
            "smart_retrieve": {"count": 5, "results": [{}] * 5},
        }
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == pytest.approx(0.4)

    def test_retrieval_score_saturates_at_five(self, ev):
        """More than 5 results does not push score above 0.40."""
        outputs = {"hybrid_search": {"count": 20, "results": [{}] * 20}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == pytest.approx(0.4)

    def test_retrieval_partial_score_three_results(self, ev):
        outputs = {"hybrid_search": {"count": 3, "results": [{}] * 3}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == pytest.approx(0.24)  # 0.08 × 3

    def test_errored_retrieval_gives_zero(self, ev):
        outputs = {"hybrid_search": {"error": "Index not ready"}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["retrieval"] == 0.0


# ---------------------------------------------------------------------------
# 2. Answer completeness
# ---------------------------------------------------------------------------

class TestAnswerCompleteness:
    def test_long_answer_gives_full_score(self, ev):
        long_answer = " ".join(["word"] * 25)  # 25 words ≥ 20 threshold
        outputs = {"metadata_rag": {"answer": long_answer}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["answer_completeness"] == pytest.approx(0.3)

    def test_short_answer_gives_half_score(self, ev):
        short_answer = "A brief answer."  # < 20 words
        outputs = {"metadata_rag": {"answer": short_answer}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["answer_completeness"] == pytest.approx(0.15)

    def test_empty_answer_gives_zero(self, ev):
        outputs = {"metadata_rag": {"answer": ""}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["answer_completeness"] == 0.0

    def test_summarize_answer_counts(self, ev):
        long_summary = " ".join(["word"] * 25)
        outputs = {"summarize": {"summary": long_summary}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["answer_completeness"] == pytest.approx(0.3)

    def test_conversation_counts(self, ev):
        long_conv = " ".join(["word"] * 25)
        outputs = {"conversation": {"answer": long_conv}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["answer_completeness"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 3. Evidence grounding
# ---------------------------------------------------------------------------

class TestEvidenceGrounding:
    def test_full_grounding_score(self, ev):
        outputs = {
            "methodology_extract": {"count": 3, "signals": ["transformer"]},
            "citation_signals": {"category_cooccurrence": {"cs.LG": 5}},
            "classify_query": {"predicted_category": "cs.LG"},
        }
        result = ev.evaluate(outputs)
        assert result["breakdown"]["evidence_grounding"] == pytest.approx(0.2)

    def test_methodology_alone_gives_0_1(self, ev):
        outputs = {"methodology_extract": {"count": 1, "signals": ["bert"]}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["evidence_grounding"] == pytest.approx(0.1)

    def test_zero_methodology_count_gives_zero(self, ev):
        outputs = {"methodology_extract": {"count": 0, "signals": []}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["evidence_grounding"] == 0.0


# ---------------------------------------------------------------------------
# 4. Error absence
# ---------------------------------------------------------------------------

class TestErrorAbsence:
    def test_no_errors_gives_full_error_score(self, ev):
        outputs = {"hybrid_search": {"count": 3, "results": []}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["error_absence"] == pytest.approx(0.1)

    def test_critical_tool_error_gives_zero(self, ev):
        outputs = {"hybrid_search": {"error": "Index not ready"}}
        result = ev.evaluate(outputs)
        assert result["breakdown"]["error_absence"] == 0.0

    def test_tool_errors_surfaced_in_result(self, ev):
        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "metadata_rag": {"error": "LLM timeout"},
        }
        result = ev.evaluate(outputs)
        assert "tool_errors" in result
        assert "hybrid_search" in result["tool_errors"]
        assert "metadata_rag" in result["tool_errors"]


# ---------------------------------------------------------------------------
# 5. Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_no_retry_above_threshold(self, ev):
        """A query with 5 results and a decent answer should not retry."""
        outputs = {
            "hybrid_search": {"count": 5, "results": [{}] * 5},
            "metadata_rag": {"answer": " ".join(["word"] * 25)},
            "classify_query": {"predicted_category": "cs.LG"},
        }
        result = ev.evaluate(outputs)
        assert result["needs_retry"] is False

    def test_retry_triggered_below_threshold(self, ev):
        """Zero retrieval hits should always trigger a retry."""
        outputs = {"hybrid_search": {"count": 0, "results": []}}
        result = ev.evaluate(outputs)
        assert result["needs_retry"] is True

    def test_retry_multiplier_x3_for_very_low_score(self, ev):
        """Score < 0.15 → x3 multiplier."""
        outputs = {}  # all zeros → score = 0.0
        result = ev.evaluate(outputs)
        assert result["quality_score"] < 0.15
        assert result.get("retry_top_k_multiplier") == 3

    def test_retry_multiplier_x2_for_moderate_low_score(self, ev):
        """Score ∈ [0.15, 0.35) → x2 multiplier."""
        # 1 result (0.08) + short answer (0.15) + classification (0.05) + no errors (0.1) = 0.38
        # That's above threshold. Let's use: 2 results (0.16) + no answer = 0.16 + 0.1 = 0.26 < 0.35
        outputs = {
            "hybrid_search": {"count": 2, "results": [{}] * 2},
            "classify_query": {"predicted_category": "cs.LG"},
        }
        result = ev.evaluate(outputs)
        assert 0.15 <= result["quality_score"] < 0.35
        assert result.get("retry_top_k_multiplier") == 2

    def test_retry_reason_no_retrieval_hits(self, ev):
        outputs = {"hybrid_search": {"count": 0, "results": []}}
        result = ev.evaluate(outputs)
        assert result["reason"] == "no_retrieval_hits"

    def test_retry_reason_no_answer_generated(self, ev):
        outputs = {"hybrid_search": {"count": 3, "results": [{}] * 3}}
        result = ev.evaluate(outputs)
        assert result["reason"] == "no_answer_generated"

    def test_sufficient_evidence_reason_when_no_retry(self, ev):
        outputs = {
            "hybrid_search": {"count": 5, "results": [{}] * 5},
            "metadata_rag": {"answer": " ".join(["word"] * 25)},
            "classify_query": {"predicted_category": "cs.LG"},
        }
        result = ev.evaluate(outputs)
        assert result["reason"] == "sufficient_evidence"
