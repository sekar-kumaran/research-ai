"""SynthesisAgent — turns grounded tool outputs into a coherent researcher-facing answer.

WHAT THIS DOES
--------------
The SynthesisAgent is the final stage of the orchestration loop. It receives
structured outputs from all the ML/retrieval tools (FAISS search results,
methodology extraction, trend analysis, citations, etc.) and produces:

  1. A natural-language answer grounded entirely in retrieved evidence
  2. A list of source papers cited in the answer (for the UI to render)
  3. A confidence score reflecting how well-evidenced the answer is
  4. A list of which tools were actually invoked

HALLUCINATION PREVENTION
------------------------
The system prompt explicitly forbids inventing paper titles, authors, or
results. The LLM is given only the tool outputs as its "world knowledge"
for this request. Any claim it makes must come from the structured context.

If the LLM is unavailable (no API key, Ollama not running), the agent falls
back to a structured text summary built directly from the tool outputs — no
hallucination risk since it's a formatted dump of real data.

CONFIDENCE SCORING
------------------
Confidence is derived from the EvaluatorAgent's quality_score (0–1).
The EvaluatorAgent scores on 4 dimensions: retrieval hit-rate (0.40),
answer completeness (0.30), evidence grounding (0.20), error absence (0.10).
We expose this score directly in the response so the UI can show users a
visual indicator of how well-evidenced the answer is.

STRUCTURED SOURCES
------------------
After synthesis, we extract the cited papers from the tool outputs and
return them as a structured list. The UI renders these as clickable
"source cards" below the answer — like Perplexity's source footnotes.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the synthesis agent for an AI Research Intelligence Platform. "
    "You receive structured outputs from scientific ML and retrieval tools and must write "
    "a clear, accurate, well-structured answer for a researcher.\n\n"
    "RULES:\n"
    "- Ground every claim in the provided tool outputs. Do NOT invent paper titles, authors, or results.\n"
    "- Cite papers by title and year when available. Use [1], [2], ... notation.\n"
    "- If methodology was extracted, include the key methods/datasets found.\n"
    "- If trend analysis was run, summarise the year distribution briefly.\n"
    "- If citation signals were found, mention related areas or influential papers.\n"
    "- If no relevant papers were found, say so clearly — do not fabricate.\n"
    "- Be concise but substantive. Aim for 150–400 words unless a longer answer is clearly needed.\n"
    "- Start with a direct answer to the query, then provide supporting evidence.\n"
    "- Write conversationally — you are a helpful research assistant, not a search engine.\n"
    "- End with a brief 'Sources' list using the [N] notation you used in the body."
)


class SynthesisAgent:
    """LLM-powered synthesis over structured tool outputs.

    Returns a dict with:
      - answer          : The final conversational response text
      - sources         : List of paper dicts cited in the answer
      - confidence      : 0-1 score from the evaluator (or derived from search results)
      - tools_used      : List of tool names that produced non-empty outputs
      - model_used      : The LLM model name that generated the synthesis (if any)
    """

    def __init__(self, cloud_factory=None) -> None:
        self.cloud_factory = cloud_factory

    def synthesize(self, query: str, plan: dict, outputs: dict) -> str:
        """Legacy single-string interface — calls synthesize_structured internally."""
        result = self.synthesize_structured(query, plan, outputs)
        return result["answer"]

    def synthesize_structured(
        self,
        query: str,
        plan: dict,
        outputs: dict,
        quality_score: float | None = None,
    ) -> dict:
        """Produce a fully structured response dict from tool outputs.

        Returns:
            {
                "answer":      str,           # The narrative answer text
                "sources":     list[dict],    # Retrieved papers cited
                "confidence":  float,         # 0–1 evidence quality score
                "tools_used":  list[str],     # Non-empty tool outputs
                "model_used":  str,           # LLM model that synthesized
            }
        """
        # Extract sources from search outputs (always, regardless of LLM availability)
        sources = self._extract_sources(outputs)

        # Determine which tools actually returned useful output
        tools_used = [
            k for k, v in outputs.items()
            if isinstance(v, dict) and not v.get("error") and k != "_retry"
        ]

        # Derive confidence from quality_score or fallback heuristic
        confidence = self._derive_confidence(quality_score, sources, outputs)

        # Try LLM synthesis first; fall back to structured text if unavailable
        direct = self._structured_direct_answer(outputs)
        cloud = self._cloud()
        model_used = ""

        if cloud is None:
            return {
                "answer": direct,
                "sources": sources,
                "confidence": confidence,
                "tools_used": tools_used,
                "model_used": model_used,
            }

        context = self._build_context(outputs)
        prompt = (
            f"User query: {query}\n\n"
            f"Planned intent: {plan.get('intent', 'research_analysis')}\n\n"
            f"Tool outputs:\n{context}\n\n"
            "Write the final answer now."
        )

        # Local Ollama models are slower — cap tokens to keep latency under ~60s
        is_ollama = getattr(cloud, "provider", "") == "ollama"
        max_tok = 300 if is_ollama else 800
        model_used = getattr(cloud, "model", "")

        try:
            answer = cloud.generate(prompt, max_tokens=max_tok, system=SYSTEM_PROMPT).strip()
            if len(answer.split()) < 10:
                answer = direct  # LLM returned garbage — use structured fallback
        except Exception as exc:
            logger.warning("SynthesisAgent LLM call failed: %s", exc)
            answer = direct

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "tools_used": tools_used,
            "model_used": model_used,
        }

    # ------------------------------------------------------------------
    # Source extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(outputs: dict) -> list[dict]:
        """Extract retrieved papers from search tool outputs.

        Papers come from hybrid_search or smart_retrieve. We deduplicate by
        paper_id and cap at 8 sources to avoid overwhelming the UI.
        """
        seen_ids: set[str] = set()
        sources: list[dict] = []

        for key in ("hybrid_search", "smart_retrieve", "metadata_rag"):
            val = outputs.get(key)
            if not isinstance(val, dict):
                continue

            # metadata_rag stores its results under "retrieved"
            results_key = "retrieved" if key == "metadata_rag" else "results"
            papers = val.get(results_key, [])

            for p in papers[:8]:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("paper_id", "")).strip()
                if pid and pid in seen_ids:
                    continue
                if pid:
                    seen_ids.add(pid)

                abstract = str(p.get("abstract", ""))
                snippet = abstract[:250] + ("…" if len(abstract) > 250 else "")
                sources.append({
                    "title":            p.get("title", "Untitled"),
                    "paper_id":         pid,
                    "year":             str(p.get("year", "")),
                    "category":         str(p.get("category", "")),
                    "abstract_snippet": snippet,
                    "score":            round(float(p.get("score", 0.0)), 4),
                    "arxiv_url":        f"https://arxiv.org/abs/{pid}" if pid else "",
                })

            if len(sources) >= 8:
                break

        return sources

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_confidence(
        quality_score: float | None,
        sources: list[dict],
        outputs: dict,
    ) -> float:
        """Derive a 0–1 confidence score for the response.

        Priority:
          1. Use the EvaluatorAgent's quality_score directly (most accurate)
          2. Heuristic: score based on number of sources + LLM answer presence
        """
        if quality_score is not None:
            return round(max(0.0, min(1.0, float(quality_score))), 3)

        # Heuristic scoring when quality_score is unavailable
        score = 0.0

        # Source quality: up to 0.50 based on retrieved paper count and scores
        if sources:
            avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
            source_factor = min(len(sources) / 5.0, 1.0)   # saturates at 5 sources
            score += 0.50 * source_factor * min(avg_score * 2, 1.0)

        # LLM answer present: +0.30
        for key in ("metadata_rag", "paper_chat"):
            val = outputs.get(key, {})
            if isinstance(val, dict) and isinstance(val.get("answer"), str):
                if len(val["answer"].split()) >= 20:
                    score += 0.30
                    break

        # No errors: +0.20
        errors = sum(1 for v in outputs.values() if isinstance(v, dict) and v.get("error"))
        if errors == 0:
            score += 0.20

        return round(min(score, 1.0), 3)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cloud(self):
        if self.cloud_factory is None:
            return None
        try:
            return self.cloud_factory()
        except Exception:
            return None

    @staticmethod
    def _build_context(outputs: dict) -> str:
        """Build a compact JSON context string, prioritising the most useful keys."""
        priority_keys = (
            "hybrid_search", "smart_retrieve", "metadata_rag",
            "methodology_extract", "trend_analysis", "citation_signals",
            "citation_proxy", "classify_query", "metadata_analyse",
            "summarize", "paper_chat",
        )
        selected: dict = {}
        total_chars = 0
        limit = 10_000  # Stay well within LLM context window

        for key in priority_keys:
            if key not in outputs:
                continue
            val = outputs[key]
            if isinstance(val, dict) and val.get("error"):
                continue

            # Compact search results — include only what the LLM needs to cite
            if key in ("hybrid_search", "smart_retrieve") and isinstance(val, dict):
                compact_results = [
                    {
                        "title":            p.get("title", ""),
                        "year":             p.get("year", ""),
                        "category":         p.get("category", ""),
                        "abstract_snippet": str(p.get("abstract", ""))[:300],
                        "paper_id":         p.get("paper_id", ""),
                    }
                    for p in val.get("results", [])[:8]
                ]
                val = {
                    "count":              val.get("count", 0),
                    "results":            compact_results,
                    "retrieval_strategy": val.get("retrieval_strategy", ""),
                }

            chunk = json.dumps({key: val}, ensure_ascii=False)
            if total_chars + len(chunk) > limit:
                break
            selected[key] = val
            total_chars += len(chunk)

        return json.dumps(selected, ensure_ascii=False, indent=2)

    @staticmethod
    def _structured_direct_answer(outputs: dict) -> str:
        """Best-effort readable answer without LLM — used as fallback."""
        # Conversation tool
        conv = outputs.get("conversation")
        if isinstance(conv, dict) and isinstance(conv.get("answer"), str):
            return conv["answer"]

        # LLM-backed tools (RAG, paper chat, summarize)
        for key in ("metadata_rag", "paper_chat", "summarize"):
            val = outputs.get(key, {})
            if isinstance(val, dict):
                text = val.get("answer") or val.get("summary") or val.get("final_answer")
                if isinstance(text, str) and text.strip():
                    return text

        # Search results — build a numbered list
        for key in ("hybrid_search", "smart_retrieve"):
            val = outputs.get(key, {})
            if isinstance(val, dict) and val.get("results"):
                lines = [f"Found {val.get('count', 0)} relevant papers:"]
                for i, p in enumerate(val["results"][:6], 1):
                    pid = p.get("paper_id", "")
                    link = f" — arxiv.org/abs/{pid}" if pid else ""
                    lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
                    if p.get("abstract"):
                        lines.append(f"   {str(p['abstract'])[:200]}…")
                return "\n".join(lines)

        # Classification only
        clf = outputs.get("classify_query", {})
        if isinstance(clf, dict) and clf.get("predicted_category"):
            return f"Predicted arXiv category: {clf['predicted_category']}"

        # Error fallback
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
        if errors:
            return f"Could not complete the request: {errors[0]}"

        return "No results found. Please check that the paper index has been built and try a different query."
