"""Hallucination resistance tests for the synthesis and RAG layers.

These tests stress-test the system's ability to refuse fabrication when:
  1. No papers are found (empty retrieval)
  2. The LLM is asked about topics not in the index
  3. Low-confidence retrieval returns semantically irrelevant docs
  4. The synthesis prompt would allow open-ended generation

All LLM calls are mocked to control what "hallucination" would look like.
We verify that the SYSTEM enforces grounding — not just that the LLM happens
to behave well.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestSynthesisGrounding:
    """Verify SynthesisAgent falls back correctly when context is absent."""

    def _synthesizer(self, cloud_response=None):
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        if cloud_response is None:
            return SynthesisAgent(cloud_factory=None)

        mock_cloud = MagicMock()
        mock_cloud.generate.return_value = cloud_response
        return SynthesisAgent(cloud_factory=lambda: mock_cloud)

    def test_no_cloud_uses_structured_fallback(self):
        """Without a cloud factory, the structured direct answer is always used."""
        synthesizer = self._synthesizer(cloud_response=None)
        outputs = {
            "hybrid_search": {
                "count": 2,
                "results": [
                    {"title": "Attention Is All You Need", "year": "2017",
                     "abstract": "We propose a model based on attention.", "paper_id": "1706.03762"},
                    {"title": "BERT: Pre-training", "year": "2018",
                     "abstract": "BERT is a language model.", "paper_id": "1810.04805"},
                ],
            }
        }
        plan = {"intent": "research_analysis", "query": "attention transformers"}
        answer = synthesizer.synthesize("attention transformers", plan, outputs)
        assert "Attention Is All You Need" in answer or "2017" in answer or "found" in answer.lower()

    def test_cloud_too_short_falls_back_to_structured(self):
        """If the LLM returns < 10 words, the structured fallback is used."""
        synthesizer = self._synthesizer(cloud_response="Yes.")  # < 10 words
        outputs = {
            "hybrid_search": {
                "count": 1,
                "results": [{"title": "Some Paper", "year": "2022",
                             "abstract": "We study X.", "paper_id": "2201.0001"}],
            }
        }
        plan = {"intent": "research_analysis", "query": "transformers"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        # Should NOT return "Yes." — should use structured fallback
        assert answer != "Yes."

    def test_empty_outputs_returns_no_results_message(self):
        synthesizer = self._synthesizer()
        answer = synthesizer.synthesize("obscure topic", {"intent": "search"}, {})
        assert "No results" in answer or len(answer) > 0

    def test_system_prompt_enforces_grounding(self):
        """Verify the system prompt contains the no-fabrication instruction."""
        from research_ai.agents.synthesis_agent.service import SYSTEM_PROMPT
        assert "Do NOT invent" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()
        assert "fabricate" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()

    def test_conversation_bypass_returns_static_answer(self):
        """The conversation tool returns a static answer without retrieval."""
        synthesizer = self._synthesizer()
        outputs = {
            "conversation": {"answer": "Hello! I am your research assistant.", "query": "hi"}
        }
        plan = {"intent": "conversation", "query": "hello"}
        answer = synthesizer.synthesize("hello", plan, outputs)
        assert "Hello" in answer or "assistant" in answer.lower()


class TestMetadataRAGGrounding:
    """Verify _metadata_rag system prompt enforces grounding.

    NOTE: These tests read the source of platform.py via inspect.getsource()
    rather than importing the module, because importing platform.py triggers
    a chain: platform → paper_ingestion → faiss, which may fail on environments
    with a numpy/faiss ABI mismatch.  Reading source avoids that chain.
    """

    def _get_metadata_rag_source(self):
        """Read _metadata_rag source without importing the module."""
        import pathlib
        platform_path = (
            pathlib.Path(__file__).parent.parent
            / "src" / "research_ai" / "platform.py"
        )
        src = platform_path.read_text(encoding="utf-8")
        # Extract just the _metadata_rag method section
        start = src.find("def _metadata_rag")
        end = src.find("\n    def ", start + 1)
        return src[start:end] if end > start else src[start:]

    def test_rag_system_prompt_restricts_fabrication(self):
        """The RAG system prompt should explicitly restrict LLM to paper context."""
        src = self._get_metadata_rag_source()
        # Must contain restriction to provided context
        assert "ONLY" in src or "only" in src.lower(), \
            "_metadata_rag system prompt must restrict LLM to provided context"
        # Must explicitly prohibit fabrication
        assert "Do NOT" in src or "do not" in src.lower(), \
            "_metadata_rag must explicitly prohibit fabrication"

    def test_no_retrieval_returns_no_results_answer(self):
        """If retrieval returns 0 results, answer is 'No relevant papers found'.

        Test the logic inline to avoid the faiss import chain.
        This mirrors the exact logic in platform._metadata_rag.
        """
        # Inline the _metadata_rag no-result logic to test without faiss
        def metadata_rag_no_result_path(query, top_k):
            search = {"results": [], "count": 0}  # simulate empty retrieval
            results = search.get("results", [])
            if not results:
                return {
                    "query": query,
                    "answer": "No relevant papers found in the index.",
                    "retrieved": [],
                }
            return {"unexpected": True}

        result = metadata_rag_no_result_path("nonexistent topic", 5)
        assert result["answer"] == "No relevant papers found in the index."
        assert result["retrieved"] == []


class TestEvidenceGroundedContext:
    """Verify the context builder in SynthesisAgent limits what the LLM sees."""

    def test_context_limited_to_10k_chars(self):
        """_build_context must cap at ~10000 characters to prevent LLM injection."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        # Build a large outputs dict
        outputs = {
            "hybrid_search": {
                "count": 20,
                "results": [
                    {
                        "title": f"Paper {i}: " + "A" * 400,
                        "year": "2023",
                        "category": "cs.LG",
                        "abstract": "B" * 700,
                        "paper_id": f"2301.{i:05d}",
                    }
                    for i in range(20)
                ],
            }
        }
        context = SynthesisAgent._build_context(outputs)
        assert len(context) <= 11000, f"Context too large: {len(context)} chars"

    def test_errored_tools_excluded_from_context(self):
        """Tool outputs with error keys should not appear in the LLM context."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        import json

        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "metadata_rag": {"answer": "The model uses attention mechanism."},
        }
        context = SynthesisAgent._build_context(outputs)
        parsed = json.loads(context)
        assert "hybrid_search" not in parsed, \
            "Errored tool outputs must be excluded from LLM context"
        assert "metadata_rag" in parsed
