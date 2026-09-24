"""Composable execution pipelines for multi-step research analysis workflows.

Each pipeline is a named sequence of tool steps with data-flow contracts.
The PipelineRunner executes steps in order, threading outputs from one step
into the inputs of the next. Steps can be conditional, and any step failure
triggers graceful degradation rather than hard failure.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single named step in a pipeline."""
    name: str
    tool: str
    args_template: dict = field(default_factory=dict)
    depends_on: str | None = None   # key from prior step output to inject as input
    optional: bool = False


@dataclass
class PipelineResult:
    name: str
    steps_run: int
    steps_ok: int
    outputs: dict
    latency_ms: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pipeline": self.name,
            "steps_run": self.steps_run,
            "steps_ok": self.steps_ok,
            "outputs": self.outputs,
            "latency_ms": round(self.latency_ms, 2),
            "errors": self.errors,
        }


# Pre-defined research analysis pipelines
PIPELINES: dict[str, list[PipelineStep]] = {
    "full_research_analysis": [
        PipelineStep("classify", "classify_query", {"title": "{query}", "abstract": "{query}"}),
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 8}),
        PipelineStep(
            "methodology", "methodology_extract",
            args_template={"from": "search_results"},
            depends_on="retrieve",
        ),
        PipelineStep(
            "citations", "citation_signals",
            args_template={"from": "search_results"},
            depends_on="retrieve",
            optional=True,
        ),
        PipelineStep(
            "trends", "trend_analysis",
            args_template={"from": "search_results"},
            depends_on="retrieve",
            optional=True,
        ),
        PipelineStep("synthesize", "metadata_rag", {"query": "{query}", "top_k": 5}),
    ],
    "quick_search_and_summarize": [
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 5}),
        PipelineStep("synthesize", "metadata_rag", {"query": "{query}", "top_k": 5}),
    ],
    "classify_and_find_similar": [
        PipelineStep("classify", "classify_query", {"title": "{query}", "abstract": "{query}"}),
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 10}),
    ],
    "trend_report": [
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 15}),
        PipelineStep(
            "trends", "trend_analysis",
            args_template={"from": "search_results"},
            depends_on="retrieve",
        ),
        PipelineStep(
            "citations", "citation_signals",
            args_template={"from": "search_results"},
            depends_on="retrieve",
            optional=True,
        ),
    ],
}


class PipelineRunner:
    """Executes named pipelines by delegating each step to the ML execution agent.

    Data-flow: the output of a step tagged with ``depends_on`` is resolved
    so that ``args_template`` values referencing ``{from: search_results}``
    are replaced with the ``results`` list from the prior step.
    """

    def __init__(self, executor) -> None:
        """
        Args:
            executor: An object with an ``execute(tool_name, args)`` method —
                      typically the ``MLExecutionAgent``.
        """
        self.executor = executor

    def run(self, pipeline_name: str, query: str, extra_args: dict | None = None) -> PipelineResult:
        """Execute a named pipeline and return structured outputs."""
        steps = PIPELINES.get(pipeline_name)
        if steps is None:
            return PipelineResult(
                name=pipeline_name,
                steps_run=0,
                steps_ok=0,
                outputs={},
                latency_ms=0,
                errors=[f"Unknown pipeline: '{pipeline_name}'. Available: {sorted(PIPELINES)}"],
            )

        started = time.perf_counter()
        outputs: dict = {}
        errors: list[str] = []
        extra = extra_args or {}

        for step in steps:
            args = self._resolve_args(step, query, outputs, extra)
            try:
                result = self.executor.execute(step.tool, args)
                outputs[step.name] = result
                if result.get("error") and not step.optional:
                    errors.append(f"Step '{step.name}' error: {result['error']}")
            except Exception as exc:
                error_msg = f"Step '{step.name}' raised: {exc!s}"
                logger.warning(error_msg)
                errors.append(error_msg)
                outputs[step.name] = {"error": str(exc)}
                if not step.optional:
                    break   # non-optional failure → abort pipeline

        return PipelineResult(
            name=pipeline_name,
            steps_run=len(steps),
            steps_ok=sum(1 for s in steps if not outputs.get(s.name, {}).get("error")),
            outputs=outputs,
            latency_ms=(time.perf_counter() - started) * 1000,
            errors=errors,
        )

    def available_pipelines(self) -> list[str]:
        return sorted(PIPELINES.keys())

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_args(
        step: PipelineStep, query: str, outputs: dict, extra: dict
    ) -> dict:
        args: dict = {}
        for key, value in step.args_template.items():
            if isinstance(value, str):
                args[key] = value.replace("{query}", query)
            else:
                args[key] = value

        # Inject results from a prior step when args_template says "from: search_results"
        if args.get("from") == "search_results" and step.depends_on:
            prior = outputs.get(step.depends_on, {})
            args = {"papers": prior.get("results", [])}

        args.update(extra)
        return args
