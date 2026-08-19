"""PlannerAgent — converts a user query into a concrete, executable tool plan.

The planner first tries the cloud LLM with a rich system prompt that includes
a full tool catalog (names + argument schemas + purpose) and a few-shot JSON
example. If the cloud LLM is unavailable or produces invalid JSON it falls
back to a deterministic heuristic planner that covers the most common modes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class ResearchPlan:
    intent: str
    query: str
    top_k: int
    calls: list[ToolCall]
    reason: str
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Tool catalog — the LLM reads this to know exactly what to call and how
# ---------------------------------------------------------------------------

TOOL_CATALOG = """
AVAILABLE TOOLS (use ONLY these names):

1.  classify_query
    Purpose : Predict the arXiv category of a paper (cs.LG, cs.CV, cs.NLP, etc.)
    Args    : {"title": "<string>", "abstract": "<string>"}
    When    : User asks what field/category a topic belongs to, or as a first step before search.

2.  hybrid_search
    Purpose : Semantic + BM25 keyword search over the indexed arXiv paper database.
    Args    : {"query": "<string>", "top_k": <int 1-20>, "filters": {"category": "<optional>", "year": "<optional>"}}
    When    : User wants to find papers on a topic. ALWAYS include this for any research question.

3.  smart_retrieve
    Purpose : Like hybrid_search but auto-selects strategy (hybrid / filtered / citation-aware).
              Also expands short acronym queries (e.g. "rl" → full term).
    Args    : {"query": "<string>", "top_k": <int>, "strategy_hint": "auto|filtered|citation_aware"}
    When    : Use instead of hybrid_search when the query is short/ambiguous or citation-focused.

4.  methodology_extract
    Purpose : Extract method names, datasets, metrics, and experiment types from paper abstracts.
    Args    : {"papers": [<list of paper dicts from a prior search result>]}
    When    : User asks how something works, what method was used, or wants methodology details.
              IMPORTANT: Pass the "results" list from a prior hybrid_search output.

5.  citation_signals
    Purpose : Category and year co-occurrence signals — a lightweight proxy citation graph.
    Args    : {"papers": [<list of paper dicts>]}
    When    : User asks about related work, references, or citation relationships.

6.  citation_proxy
    Purpose : Full proxy citation graph with keyword overlap and temporal proximity scoring.
    Args    : {"papers": [<list of paper dicts>]}
    When    : User asks for a deeper citation or influence analysis.

7.  trend_analysis
    Purpose : Year and category distribution statistics over a set of papers.
    Args    : {"papers": [<list of paper dicts>]}
    When    : User asks about trends, recent work, how a field evolved over time.

8.  metadata_analyse
    Purpose : Author network, abstract quality scores, metadata completeness.
    Args    : {"papers": [<list of paper dicts>]}
    When    : User asks who the key authors are, or wants structural metadata analysis.

9.  summarize
    Purpose : Summarise a block of text using a scientific summariser.
    Args    : {"text": "<string to summarize>"}
    When    : User provides text or an abstract and wants a condensed version.

10. paper_chat
    Purpose : Ask a question about a specific paper that has been loaded into a session.
    Args    : {"session_id": "<string>", "question": "<string>", "top_k": <int>}
    When    : A session_id is present in the request context.

11. metadata_rag
    Purpose : Retrieval-augmented generation — searches papers then generates an LLM answer.
    Args    : {"query": "<string>", "top_k": <int>}
    When    : User wants a direct answer grounded in the paper database. Use as FINAL step.

12. run_pipeline
    Purpose : Run a pre-built multi-step analysis pipeline in one call.
    Args    : {"pipeline_name": "full_research_analysis|quick_search_and_summarize|trend_report", "query": "<string>"}
    When    : User wants a comprehensive report — avoids having to plan each step manually.

13. python_execute
    Purpose : Execute a small sandboxed Python script for computation or statistics.
    Args    : {"code": "<python code string>"}
    When    : User explicitly asks to calculate, simulate, or run code.

14. conversation
    Purpose : Respond to greetings, small talk, and non-research queries.
    Args    : {"query": "<string>"}
    When    : Query is a greeting, thanks, or clearly non-research.

15. cluster_papers
    Purpose : Group a set of papers into topics or clusters using a KMeans clustering model.
    Args    : {"papers": [<list of paper dicts>]}
    When    : User asks to group, cluster, or organize papers into distinct topics.
"""

FEW_SHOT = """
EXAMPLE 1 — Research question:
User: "What are the latest transformer architectures for NLP?"
{
  "intent": "research_analysis",
  "query": "transformer architectures NLP",
  "top_k": 8,
  "reason": "search then extract methodology and answer",
  "calls": [
    {"name": "classify_query", "args": {"title": "transformer architectures NLP", "abstract": "transformer architectures NLP"}, "reason": "identify field"},
    {"name": "hybrid_search",  "args": {"query": "transformer architectures NLP", "top_k": 8}, "reason": "find relevant papers"},
    {"name": "methodology_extract", "args": {"from": "search_results"}, "reason": "extract methods from results"},
    {"name": "trend_analysis", "args": {"from": "search_results"}, "reason": "show recent trend"},
    {"name": "metadata_rag",   "args": {"query": "transformer architectures NLP", "top_k": 5}, "reason": "grounded LLM answer"}
  ]
}

EXAMPLE 2 — Citation / influence:
User: "Which papers influenced BERT?"
{
  "intent": "citation_analysis",
  "query": "BERT influential papers",
  "top_k": 10,
  "reason": "citation-aware retrieval then proxy graph",
  "calls": [
    {"name": "smart_retrieve",  "args": {"query": "BERT influential papers", "top_k": 10, "strategy_hint": "citation_aware"}, "reason": "citation-aware search"},
    {"name": "citation_signals","args": {"from": "search_results"}, "reason": "co-occurrence signals"},
    {"name": "citation_proxy",  "args": {"from": "search_results"}, "reason": "full proxy citation graph"},
    {"name": "metadata_rag",    "args": {"query": "papers that influenced BERT", "top_k": 5}, "reason": "synthesis"}
  ]
}

EXAMPLE 3 — Trends:
User: "How has diffusion model research evolved since 2020?"
{
  "intent": "trend_analysis",
  "query": "diffusion model research evolution 2020",
  "top_k": 15,
  "reason": "wide search then trend stats",
  "calls": [
    {"name": "hybrid_search",  "args": {"query": "diffusion model generative", "top_k": 15}, "reason": "broad search"},
    {"name": "trend_analysis", "args": {"from": "search_results"}, "reason": "year/category trends"},
    {"name": "metadata_analyse","args": {"from": "search_results"}, "reason": "author stats"},
    {"name": "metadata_rag",   "args": {"query": "diffusion model research evolution", "top_k": 8}, "reason": "narrative answer"}
  ]
}

EXAMPLE 4 — Summarize:
User: "Summarize this abstract: ..."
{
  "intent": "summarization",
  "query": "summarize abstract",
  "top_k": 5,
  "reason": "direct summarization",
  "calls": [
    {"name": "summarize", "args": {"text": "<the abstract text>"}, "reason": "condense"}
  ]
}

EXAMPLE 5 — Greeting:
User: "Hello!"
{
  "intent": "conversation",
  "query": "Hello!",
  "top_k": 5,
  "reason": "greeting",
  "calls": [
    {"name": "conversation", "args": {"query": "Hello!"}, "reason": "respond to greeting"}
  ]
}

IMPORTANT RULES:
- Use "from": "search_results" as the ONLY arg when a tool should receive results from the immediately preceding hybrid_search or smart_retrieve call.
- Always end a research flow with metadata_rag for a synthesized answer, OR use run_pipeline for a comprehensive one-shot analysis.
- Return ONLY the JSON object. No explanation text outside the JSON.
- Do not invent tool names not listed above.
"""

SYSTEM_PROMPT = (
    "You are the planning agent for an AI Research Intelligence Platform. "
    "Your job is to analyse the user's query and return a JSON tool-execution plan.\n\n"
    + TOOL_CATALOG
    + "\n" + FEW_SHOT
    + "\nReturn ONLY a single valid JSON object. No markdown, no extra text."
)

# Compact variant for local Ollama — minimal prompt to fit small context windows quickly
_OLLAMA_SYSTEM_PROMPT = (
    "You are a research query planner. Return ONLY a JSON object like:\n"
    '{"intent":"research_analysis","query":"<user query>","top_k":5,"reason":"search","calls":['
    '{"name":"hybrid_search","args":{"query":"<user query>","top_k":5},"reason":"find papers"},'
    '{"name":"metadata_rag","args":{"query":"<user query>","top_k":5},"reason":"answer"}]}\n'
    "Tools: hybrid_search, metadata_rag, classify_query, summarize, methodology_extract.\n"
    "Return ONLY valid JSON. No explanation."
)


class PlannerAgent:
    """LLM-powered intent planner with deterministic heuristic fallback."""

    def __init__(self, cloud_factory=None, max_top_k: int = 20) -> None:
        self.cloud_factory = cloud_factory
        self.max_top_k = max_top_k

    def plan(
        self,
        mode: str,
        query: str,
        top_k: int,
        title: str | None = None,
        abstract: str | None = None,
        text: str | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
    ) -> ResearchPlan:
        mode_hint = (mode or "auto").strip().lower()

        # Hard-gate: conversational input never reaches the tool chain.
        # EXCEPTION: if there's conversation history, the "short greeting" might
        # actually be a follow-up ("yes", "ok", "more please") — let those through.
        if self._is_conversation(query) and not conversation_history:
            return ResearchPlan(
                intent="conversation",
                query=(query or "").strip(),
                top_k=self._clamp(top_k),
                calls=[ToolCall("conversation", {"query": (query or "").strip()})],
                reason="conversational_input",
                used_fallback=True,
            )

        # Explicit mode bypass → heuristic planner
        if mode_hint != "auto":
            return self._fallback_plan(mode_hint, query, top_k, title, abstract, text, session_id)

        # Cloud LLM planning
        cloud = self._cloud()
        if cloud is None:
            return self._fallback_plan(mode_hint, query, top_k, title, abstract, text, session_id)

        # Build the planning prompt. Conversation history is injected here so the
        # LLM planner can resolve pronouns and follow-up references automatically.
        # Example: if prior turn retrieved GNN papers, and user says "which of these
        # use attention?", the planner can reformulate the query as
        # "GNN papers with attention mechanisms" instead of treating it as standalone.
        history_block = (
            f"\nConversation history (last few turns):\n{conversation_history}\n"
            if conversation_history else ""
        )
        prompt = (
            f"User query: {query}\n"
            f"Requested mode: {mode}\n"
            f"Requested top_k: {top_k}\n"
            + (f"Title context: {title}\n" if title else "")
            + (f"Abstract context: {abstract}\n" if abstract else "")
            + (f"Text context (first 400 chars): {(text or '')[:400]}\n" if text else "")
            + (f"Session ID: {session_id}\n" if session_id else "")
            + history_block
        )

        # Local Ollama models can't handle the full 2800-token system prompt in time
        # Use a minimal prompt variant that still produces valid JSON plans
        is_ollama = getattr(cloud, "provider", "") == "ollama"
        sys = _OLLAMA_SYSTEM_PROMPT if is_ollama else SYSTEM_PROMPT
        max_tok = 200 if is_ollama else 600
        
        for attempt in range(2):
            try:
                raw = cloud.generate(prompt, max_tokens=max_tok, system=sys)
                parsed = self._parse_json(raw)
                calls = self._parse_calls(parsed)
                if not calls:
                    raise ValueError("LLM planner produced zero valid tool calls.")
                return ResearchPlan(
                    intent=str(parsed.get("intent") or "research_analysis"),
                    query=str(parsed.get("query") or query),
                    top_k=self._clamp(parsed.get("top_k") or top_k),
                    calls=calls,
                    reason=str(parsed.get("reason") or "llm_planner"),
                    used_fallback=False,
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("LLM planner JSON parse error on attempt 1 (%s). Retrying...", exc)
                    prompt += f"\n\nYOUR PREVIOUS OUTPUT FAILED TO PARSE AS JSON: {exc}\nPlease output strictly valid JSON, checking for trailing or missing commas and unescaped quotes."
                else:
                    logger.warning("LLM planner failed on attempt 2 (%s) — using heuristic fallback.", exc)
                    return self._fallback_plan(mode_hint, query, top_k, title, abstract, text, session_id)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _cloud(self):
        if self.cloud_factory is None:
            return None
        try:
            return self.cloud_factory()
        except Exception:
            return None

    @staticmethod
    def _parse_json(text: str) -> dict:
        payload = (text or "").strip().replace("```json", "").replace("```", "").strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        parsed = json.loads(payload[start: end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Planner JSON must be a dict.")
        return parsed

    @staticmethod
    def _parse_calls(parsed: dict) -> list[ToolCall]:
        return [
            ToolCall(
                name=str(item.get("name", "")),
                args=dict(item.get("args", {})),
                reason=str(item.get("reason", "")),
            )
            for item in parsed.get("calls", [])
            if isinstance(item, dict) and item.get("name")
        ]

    def _clamp(self, value) -> int:
        try:
            return max(1, min(self.max_top_k, int(value)))
        except Exception:
            return 5

    @staticmethod
    def _is_conversation(query: str) -> bool:
        import re
        raw = " ".join((query or "").strip().lower().split())
        # Strip punctuation for matching ("Hello!" -> "hello")
        q = re.sub(r"[^\w\s]", "", raw).strip()
        if not q:
            return True
        greetings = {
            "hi", "hello", "hey", "hii", "hiii", "yo", "sup", "thanks",
            "thank you", "ok", "okay", "good morning", "good afternoon",
            "good evening", "cheers", "bye", "goodbye",
        }
        if q in greetings:
            return True
        return len(q.split()) <= 3 and any(
            q.startswith(p) for p in ("hi ", "hello ", "hey ", "thanks ")
        )

    def _fallback_plan(
        self,
        mode: str,
        query: str,
        top_k: int,
        title: str | None,
        abstract: str | None,
        text: str | None,
        session_id: str | None,
    ) -> ResearchPlan:
        """Deterministic heuristic planner used when cloud LLM is unavailable."""
        q = (query or "").strip()
        lower = q.lower()
        k = self._clamp(top_k)

        if mode == "classify":
            return ResearchPlan(
                intent="classification",
                query=q, top_k=k,
                calls=[ToolCall("classify_query", {"title": title or q, "abstract": abstract or q})],
                reason="heuristic_classify", used_fallback=True,
            )
        if mode == "search":
            return ResearchPlan(
                intent="search",
                query=q, top_k=k,
                calls=[ToolCall("hybrid_search", {"query": q, "top_k": k})],
                reason="heuristic_search", used_fallback=True,
            )
        if mode == "summarize":
            return ResearchPlan(
                intent="summarization",
                query=q, top_k=k,
                calls=[ToolCall("summarize", {"text": text or q})],
                reason="heuristic_summarize", used_fallback=True,
            )
        if mode == "paper_chat" or session_id:
            return ResearchPlan(
                intent="paper_chat",
                query=q, top_k=k,
                calls=[ToolCall("paper_chat", {"session_id": session_id or "", "question": q, "top_k": k})],
                reason="heuristic_paper_chat", used_fallback=True,
            )

        # Default: full research analysis pipeline
        calls: list[ToolCall] = [
            ToolCall("classify_query", {"title": title or q, "abstract": abstract or q}),
            ToolCall("hybrid_search", {"query": q, "top_k": k}),
            ToolCall("methodology_extract", {"from": "search_results"}),
            ToolCall("citation_signals", {"from": "search_results"}),
        ]
        if any(t in lower for t in ("trend", "recent", "over time", "year", "evolved", "history")):
            calls.append(ToolCall("trend_analysis", {"from": "search_results"}))
        if any(t in lower for t in ("who", "author", "wrote", "researcher", "group")):
            calls.append(ToolCall("metadata_analyse", {"from": "search_results"}))
        if any(t in lower for t in ("citation", "influence", "cited", "reference", "related work")):
            calls.append(ToolCall("citation_proxy", {"from": "search_results"}))
        if any(t in lower for t in ("cluster", "group", "topic", "categorize")):
            calls.append(ToolCall("cluster_papers", {"from": "search_results"}))
        if any(t in lower for t in ("calculate", "compute", "statistic", "plot", "simulate", "run code")):
            calls.append(ToolCall("python_execute", {"code": text or ""}))
        calls.append(ToolCall("metadata_rag", {"query": q, "top_k": k}))

        return ResearchPlan(
            intent="research_analysis",
            query=q, top_k=k,
            calls=calls,
            reason="heuristic_auto_plan",
            used_fallback=True,
        )
