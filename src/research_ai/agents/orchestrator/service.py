"""ResearchOrchestrator — Plan → Execute → Evaluate → Retry? → Synthesize."""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from uuid import uuid4

from research_ai.agents.evaluator_agent import EvaluatorAgent
from research_ai.agents.ml_execution_agent import MLExecutionAgent
from research_ai.agents.planner import PlannerAgent, ResearchPlan
from research_ai.agents.synthesis_agent import SynthesisAgent
from research_ai.common.text import redact_secrets

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Principal agentic orchestration layer.

    Loop:
        1. PlannerAgent  →  ResearchPlan (list of ToolCalls)
        2. MLExecutionAgent  →  tool outputs dict
        3. EvaluatorAgent  →  quality score + retry decision
        4. (optional retry with wider top_k)
        5. SynthesisAgent  →  final grounded answer string
    """

    def __init__(
        self,
        planner: PlannerAgent,
        executor: MLExecutionAgent,
        evaluator: EvaluatorAgent,
        synthesizer: SynthesisAgent,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.synthesizer = synthesizer

    def run(
        self,
        mode: str,
        query: str,
        top_k: int = 5,
        title: str | None = None,
        abstract: str | None = None,
        text: str | None = None,
        session_id: str | None = None,
        conversation_history: str | None = None,
    ) -> dict:
        """Execute the full Plan→Execute→Evaluate→Synthesize loop.

        Args:
            mode               : Routing hint ("auto" lets the planner decide).
            query              : The user's current message.
            top_k              : Max documents to retrieve.
            title/abstract/text: Optional extra context for classification/summarization.
            session_id         : Active paper-chat session ID (optional).
            conversation_history: Recent turns as a plain-text summary (optional).
                                   Provided by the ConversationStore so the planner
                                   can understand follow-up questions like "tell me
                                   more about that" or "which was fastest?".
        """
        started = time.perf_counter()
        request_id = str(uuid4())

        # 1. Plan — pass conversation history so the planner can resolve references
        plan = self.planner.plan(
            mode, query, top_k, title, abstract, text, session_id,
            conversation_history=conversation_history,
        )
        logger.info(
            "Orchestrator [%s]: intent=%s tools=%s fallback=%s",
            request_id[:8],
            plan.intent,
            [c.name for c in plan.calls],
            plan.used_fallback,
        )

        # 2. Execute — use the improved execute_plan that handles data-flow
        outputs = self.executor.execute_plan(plan.calls)

        # 3. Evaluate
        evaluation = self.evaluator.evaluate(outputs)
        logger.info(
            "Evaluator [%s]: score=%.2f retry=%s reason=%s",
            request_id[:8],
            evaluation["quality_score"],
            evaluation["needs_retry"],
            evaluation.get("reason"),
        )

        # 4. Retry if needed
        if evaluation.get("needs_retry") and plan.intent != "conversation":
            retry_plan = self._build_retry_plan(plan, evaluation)
            retry_outputs = self.executor.execute_plan(retry_plan.calls)
            retry_search_count = retry_outputs.get("hybrid_search", {}).get("count", 0)
            orig_search_count = outputs.get("hybrid_search", {}).get("count", 0)
            if retry_search_count > orig_search_count:
                outputs.update(retry_outputs)
            outputs["_retry"] = {
                "triggered": True,
                "reason": evaluation.get("reason"),
                "retry_top_k": retry_plan.top_k,
            }

        # 5. Synthesize
        try:
            final_answer = self.synthesizer.synthesize(query, asdict(plan), outputs)
        except Exception as exc:
            logger.warning("SynthesisAgent failed: %s", exc)
            final_answer = self._safe_fallback_answer(outputs)

        return {
            "request_id": request_id,
            "mode": plan.intent,
            "mediator": {"reason": plan.reason, "used_fallback": plan.used_fallback},
            "plan": asdict(plan),
            "executor_output": outputs,
            "evaluation": evaluation,
            "final_answer": final_answer,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_retry_plan(plan: ResearchPlan, evaluation: dict) -> ResearchPlan:
        multiplier = int(evaluation.get("retry_top_k_multiplier", 2))
        new_top_k = min(20, plan.top_k * multiplier)
        # Clone the call list, bump top_k on search calls
        retry_calls = []
        for call in plan.calls:
            import copy
            new_call = copy.deepcopy(call)
            if new_call.name in ("hybrid_search", "smart_retrieve"):
                new_call.args["top_k"] = new_top_k
                new_call.args["candidate_k"] = min(80, new_top_k * 5)
            retry_calls.append(new_call)
        return ResearchPlan(
            intent=plan.intent,
            query=plan.query,
            top_k=new_top_k,
            calls=retry_calls,
            reason=f"retry_after_{evaluation.get('reason', 'low_score')}",
            used_fallback=plan.used_fallback,
        )

    @staticmethod
    def _safe_fallback_answer(outputs: dict) -> str:
        """Last-resort readable answer when synthesis agent itself fails."""
        for key in ("metadata_rag", "paper_chat", "summarize"):
            val = outputs.get(key, {})
            if isinstance(val, dict):
                text = val.get("answer") or val.get("summary") or ""
                if isinstance(text, str) and text.strip():
                    return text
        search = outputs.get("hybrid_search", {})
        if isinstance(search, dict) and search.get("results"):
            lines = [f"Found {search.get('count', 0)} relevant papers:"]
            for i, p in enumerate(search["results"][:5], 1):
                pid = p.get("paper_id", "")
                link = f" — https://arxiv.org/abs/{pid}" if pid else ""
                lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
            return "\n".join(lines)
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
        if errors:
            return redact_secrets(f"Could not complete request: {errors[0]}")
        return "No results found. Please try a more specific query or check that the paper index has been built."
