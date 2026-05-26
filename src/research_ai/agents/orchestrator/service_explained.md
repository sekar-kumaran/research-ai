# service.py Explained

Generated educational companion for `src/research_ai/agents/orchestrator/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/agents/orchestrator/service.py` is a Python module in the Orchestration layer: coordinates Plan -> Execute -> Evaluate -> Synthesize. It defines ResearchOrchestrator and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Orchestration layer: coordinates Plan -> Execute -> Evaluate -> Synthesize. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Orchestration layer: coordinates Plan -> Execute -> Evaluate -> Synthesize.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/agents/orchestrator/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/agents/orchestrator/service.py]
  ThisFile --> Downstream[Downstream Service/UI/Result]
```

## Inputs and Outputs

- **Inputs:** function arguments, class constructor dependencies, HTTP payloads, environment variables, filesystem artifacts, DOM events, or test fixtures.
- **Outputs:** return values, dictionaries, Pydantic models, rendered DOM state, API responses, logs, process startup, assertions, or side effects.
- **Serialization:** this project uses JSON for APIs/LLM planning, parquet/joblib/faiss for ML artifacts, and HTML/CSS/JS for the browser surface.

## Imports Explained

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |
| `uuid` | uuid creates unique IDs for sessions, conversations, and uploaded-document references. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 15 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `ResearchOrchestrator`

- **Line:** 18
- **Base classes:** `object`
- **Docstring:** Principal agentic orchestration layer.

Loop:
    1. PlannerAgent  →  ResearchPlan (list of ToolCalls)
    2. MLExecutionAgent  →  tool outputs dict
    3. EvaluatorAgent  →  quality score + retry decision
    4. (optional retry with wider top_k)
    5. SynthesisAgent  →  final grounded answer string

**Methods:**
- `__init__` at line 29: method behavior is described by its body and name
- `run` at line 41: Execute the full Plan→Execute→Evaluate→Synthesize loop.

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
- `_build_retry_plan` at line 131: method behavior is described by its body and name
- `_safe_fallback_answer` at line 153: Last-resort readable answer when synthesis agent itself fails.

```python
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
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `ResearchOrchestrator` Methods

#### `ResearchOrchestrator.__init__`

- **Line:** 29
- **Kind:** synchronous method
- **Arguments:** self, planner, executor, evaluator, synthesizer
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ResearchOrchestrator.run`

- **Line:** 41
- **Kind:** synchronous method
- **Arguments:** self, mode, query, top_k, title, abstract, text, session_id, conversation_history
- **Docstring:** Execute the full Plan→Execute→Evaluate→Synthesize loop.

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

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ResearchOrchestrator._build_retry_plan`

- **Line:** 131
- **Kind:** synchronous method
- **Arguments:** plan, evaluation
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ResearchOrchestrator._safe_fallback_answer`

- **Line:** 153
- **Kind:** synchronous method
- **Arguments:** outputs
- **Docstring:** Last-resort readable answer when synthesis agent itself fails.

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |
| `uuid` | uuid creates unique IDs for sessions, conversations, and uploaded-document references. |

## ML Concepts Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Performance and Memory Notes

- Avoid eager loading of heavy ML models unless startup latency is acceptable.
- Cache expensive clients, tokenizers, vector stores, and embeddings carefully.
- Use float32 for embedding vectors because it halves memory compared with float64 and matches FAISS/neural inference expectations.
- Bound prompt length, uploaded content, result counts, and token budgets to control latency and memory.
- Watch copies of large metadata frames, embedding matrices, and file buffers.

## Scalability Notes

- In-memory state works for local demos but needs Redis/database/object storage for multi-worker cloud deployment.
- CPU/GPU inference should often be separated from the web API when traffic grows.
- Retrieval can start exact and move to approximate indexes as corpus size grows.
- Batch operations and cache repeated work to improve throughput.
- Add metrics for latency, errors, fallback frequency, retrieval hit rate, and token usage.

## Production Engineering Notes

- Keep interfaces stable because other files may import this module or depend on its response shape.
- Prefer typed/structured data over free-form strings at service boundaries.
- Log operational context without secrets or huge payloads.
- Make fallback behavior explicit so users get useful output even when LLMs or artifacts fail.
- Keep provider-specific logic behind adapters so Groq/OpenRouter/Google/Ollama can be swapped.

## Common Bugs and Failure Cases

- Missing `.env` values, model artifacts, or Ollama models can trigger degraded behavior.
- Type mismatches occur when LLM-generated tool arguments cross into strict Python code.
- Empty retrieval results must not become hallucinated answers.
- Network calls need timeouts and careful retry behavior.
- Frontend IDs/classes and API schemas are contracts; changing one side without the other breaks workflows.

## Security Considerations

- Handles credentials or environment configuration. Keep secrets in environment variables and redact them from logs.
- Deals with execution or subprocesses. Maintain AST validation, isolated mode, timeouts, and least privilege.
- Performs network I/O. Use timeouts, validate responses, and keep private services such as Ollama off the public internet.

## Real Industry Usage

- This pattern appears in enterprise RAG assistants, scientific search tools, internal research copilots, and ML platform demos.
- The layered design mirrors production systems: API facade, orchestration, retrieval, evaluation, synthesis, UI, and deployment.
- Clear separation lets teams replace model providers, improve retrieval, harden security, or redesign UI independently.

## Optimization Opportunities

- Add tracing around each workflow step.
- Strengthen schema validation at boundaries.
- Persist conversation/session state outside process memory.
- Add load tests and adversarial tests for prompt injection, empty evidence, and large uploads.
- Consider approximate vector indexes, reranker models, or batching when corpus/traffic grows.

## How This Connects To Other Files

- `src/research_ai/agents/orchestrator/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
- `src/research_ai/platform.py` is the backend composition root.
- `src/research_ai/api/main.py` exposes backend behavior over HTTP.
- Retrieval modules depend on artifacts under `artifacts/`.
- Frontend files depend on stable endpoint and DOM contracts.

## End-to-End Flow Summary

- A user/browser/test/startup event enters the system.
- The relevant layer validates or normalizes input.
- Retrieval, ML, orchestration, execution, or UI rendering happens.
- A structured result, visual state, or process side effect is produced.
- Fallbacks and tests keep behavior understandable when dependencies are unavailable.

## Interview Questions

1. What responsibility does this file own?
2. What inputs and outputs define its contract?
3. Which dependencies are expensive or operationally risky?
4. What breaks if this file changes shape?
5. How would you scale or test this behavior in production?

## Key Takeaways

- `src/research_ai/agents/orchestrator/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""ResearchOrchestrator — Plan → Execute → Evaluate → Retry? → Synthesize."""
# L0002: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0003: Blank line that visually separates logical sections and improves readability.

# L0004: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0005: Imports a dependency, type, or project module needed by later code in this file.
import time
# L0006: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import asdict
# L0007: Imports a dependency, type, or project module needed by later code in this file.
from uuid import uuid4
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.agents.evaluator_agent import EvaluatorAgent
# L0010: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.agents.ml_execution_agent import MLExecutionAgent
# L0011: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.agents.planner import PlannerAgent, ResearchPlan
# L0012: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.agents.synthesis_agent import SynthesisAgent
# L0013: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.common.text import redact_secrets
# L0014: Blank line that visually separates logical sections and improves readability.

# L0015: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Blank line that visually separates logical sections and improves readability.

# L0018: Defines a class that groups related state and behavior behind a reusable interface.
class ResearchOrchestrator:
# L0019: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Principal agentic orchestration layer.
# L0020: Blank line that visually separates logical sections and improves readability.

# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Loop:
# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        1. PlannerAgent  →  ResearchPlan (list of ToolCalls)
# L0023: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        2. MLExecutionAgent  →  tool outputs dict
# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        3. EvaluatorAgent  →  quality score + retry decision
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        4. (optional retry with wider top_k)
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        5. SynthesisAgent  →  final grounded answer string
# L0027: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0028: Blank line that visually separates logical sections and improves readability.

# L0029: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0031: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        planner: PlannerAgent,
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        executor: MLExecutionAgent,
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        evaluator: EvaluatorAgent,
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        synthesizer: SynthesisAgent,
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> None:
# L0036: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.planner = planner
# L0037: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.executor = executor
# L0038: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.evaluator = evaluator
# L0039: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.synthesizer = synthesizer
# L0040: Blank line that visually separates logical sections and improves readability.

# L0041: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def run(
# L0042: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        mode: str,
# L0044: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        query: str,
# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
        top_k: int = 5,
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
        title: str | None = None,
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
        abstract: str | None = None,
# L0048: Assigns or updates a value used later in the workflow; check mutability and data shape.
        text: str | None = None,
# L0049: Assigns or updates a value used later in the workflow; check mutability and data shape.
        session_id: str | None = None,
# L0050: Assigns or updates a value used later in the workflow; check mutability and data shape.
        conversation_history: str | None = None,
# L0051: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> dict:
# L0052: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Execute the full Plan→Execute→Evaluate→Synthesize loop.
# L0053: Blank line that visually separates logical sections and improves readability.

# L0054: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Args:
# L0055: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            mode               : Routing hint ("auto" lets the planner decide).
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            query              : The user's current message.
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            top_k              : Max documents to retrieve.
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            title/abstract/text: Optional extra context for classification/summarization.
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            session_id         : Active paper-chat session ID (optional).
# L0060: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            conversation_history: Recent turns as a plain-text summary (optional).
# L0061: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                                   Provided by the ConversationStore so the planner
# L0062: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                                   can understand follow-up questions like "tell me
# L0063: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                                   more about that" or "which was fastest?".
# L0064: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
        started = time.perf_counter()
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
        request_id = str(uuid4())
# L0067: Blank line that visually separates logical sections and improves readability.

# L0068: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # 1. Plan — pass conversation history so the planner can resolve references
# L0069: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = self.planner.plan(
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            mode, query, top_k, title, abstract, text, session_id,
# L0071: Assigns or updates a value used later in the workflow; check mutability and data shape.
            conversation_history=conversation_history,
# L0072: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0073: Emits structured operational information for debugging, monitoring, or failure diagnosis.
        logger.info(
# L0074: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "Orchestrator [%s]: intent=%s tools=%s fallback=%s",
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            request_id[:8],
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            plan.intent,
# L0077: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            [c.name for c in plan.calls],
# L0078: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            plan.used_fallback,
# L0079: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0080: Blank line that visually separates logical sections and improves readability.

# L0081: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # 2. Execute — use the improved execute_plan that handles data-flow
# L0082: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = self.executor.execute_plan(plan.calls)
# L0083: Blank line that visually separates logical sections and improves readability.

# L0084: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # 3. Evaluate
# L0085: Assigns or updates a value used later in the workflow; check mutability and data shape.
        evaluation = self.evaluator.evaluate(outputs)
# L0086: Emits structured operational information for debugging, monitoring, or failure diagnosis.
        logger.info(
# L0087: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "Evaluator [%s]: score=%.2f retry=%s reason=%s",
# L0088: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            request_id[:8],
# L0089: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            evaluation["quality_score"],
# L0090: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            evaluation["needs_retry"],
# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            evaluation.get("reason"),
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0093: Blank line that visually separates logical sections and improves readability.

# L0094: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # 4. Retry if needed
# L0095: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if evaluation.get("needs_retry") and plan.intent != "conversation":
# L0096: Assigns or updates a value used later in the workflow; check mutability and data shape.
            retry_plan = self._build_retry_plan(plan, evaluation)
# L0097: Assigns or updates a value used later in the workflow; check mutability and data shape.
            retry_outputs = self.executor.execute_plan(retry_plan.calls)
# L0098: Assigns or updates a value used later in the workflow; check mutability and data shape.
            retry_search_count = retry_outputs.get("hybrid_search", {}).get("count", 0)
# L0099: Assigns or updates a value used later in the workflow; check mutability and data shape.
            orig_search_count = outputs.get("hybrid_search", {}).get("count", 0)
# L0100: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if retry_search_count > orig_search_count:
# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                outputs.update(retry_outputs)
# L0102: Assigns or updates a value used later in the workflow; check mutability and data shape.
            outputs["_retry"] = {
# L0103: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "triggered": True,
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "reason": evaluation.get("reason"),
# L0105: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "retry_top_k": retry_plan.top_k,
# L0106: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0107: Blank line that visually separates logical sections and improves readability.

# L0108: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # 5. Synthesize
# L0109: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
            final_answer = self.synthesizer.synthesize(query, asdict(plan), outputs)
# L0111: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception as exc:
# L0112: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.warning("SynthesisAgent failed: %s", exc)
# L0113: Assigns or updates a value used later in the workflow; check mutability and data shape.
            final_answer = self._safe_fallback_answer(outputs)
# L0114: Blank line that visually separates logical sections and improves readability.

# L0115: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0116: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "request_id": request_id,
# L0117: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "mode": plan.intent,
# L0118: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "mediator": {"reason": plan.reason, "used_fallback": plan.used_fallback},
# L0119: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "plan": asdict(plan),
# L0120: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "executor_output": outputs,
# L0121: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "evaluation": evaluation,
# L0122: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "final_answer": final_answer,
# L0123: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
# L0124: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0125: Blank line that visually separates logical sections and improves readability.

# L0126: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0127: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Private
# L0128: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0131: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _build_retry_plan(plan: ResearchPlan, evaluation: dict) -> ResearchPlan:
# L0132: Assigns or updates a value used later in the workflow; check mutability and data shape.
        multiplier = int(evaluation.get("retry_top_k_multiplier", 2))
# L0133: Assigns or updates a value used later in the workflow; check mutability and data shape.
        new_top_k = min(20, plan.top_k * multiplier)
# L0134: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Clone the call list, bump top_k on search calls
# L0135: Assigns or updates a value used later in the workflow; check mutability and data shape.
        retry_calls = []
# L0136: Iterates over data, retry attempts, files, results, or workflow steps.
        for call in plan.calls:
# L0137: Imports a dependency, type, or project module needed by later code in this file.
            import copy
# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
            new_call = copy.deepcopy(call)
# L0139: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if new_call.name in ("hybrid_search", "smart_retrieve"):
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
                new_call.args["top_k"] = new_top_k
# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
                new_call.args["candidate_k"] = min(80, new_top_k * 5)
# L0142: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            retry_calls.append(new_call)
# L0143: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return ResearchPlan(
# L0144: Assigns or updates a value used later in the workflow; check mutability and data shape.
            intent=plan.intent,
# L0145: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query=plan.query,
# L0146: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=new_top_k,
# L0147: Assigns or updates a value used later in the workflow; check mutability and data shape.
            calls=retry_calls,
# L0148: Assigns or updates a value used later in the workflow; check mutability and data shape.
            reason=f"retry_after_{evaluation.get('reason', 'low_score')}",
# L0149: Assigns or updates a value used later in the workflow; check mutability and data shape.
            used_fallback=plan.used_fallback,
# L0150: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0151: Blank line that visually separates logical sections and improves readability.

# L0152: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0153: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _safe_fallback_answer(outputs: dict) -> str:
# L0154: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Last-resort readable answer when synthesis agent itself fails."""
# L0155: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in ("metadata_rag", "paper_chat", "summarize"):
# L0156: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs.get(key, {})
# L0157: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(val, dict):
# L0158: Assigns or updates a value used later in the workflow; check mutability and data shape.
                text = val.get("answer") or val.get("summary") or ""
# L0159: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if isinstance(text, str) and text.strip():
# L0160: Returns the computed result to the caller; this shape becomes part of the downstream contract.
                    return text
# L0161: Assigns or updates a value used later in the workflow; check mutability and data shape.
        search = outputs.get("hybrid_search", {})
# L0162: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if isinstance(search, dict) and search.get("results"):
# L0163: Assigns or updates a value used later in the workflow; check mutability and data shape.
            lines = [f"Found {search.get('count', 0)} relevant papers:"]
# L0164: Iterates over data, retry attempts, files, results, or workflow steps.
            for i, p in enumerate(search["results"][:5], 1):
# L0165: Assigns or updates a value used later in the workflow; check mutability and data shape.
                pid = p.get("paper_id", "")
# L0166: Assigns or updates a value used later in the workflow; check mutability and data shape.
                link = f" — https://arxiv.org/abs/{pid}" if pid else ""
# L0167: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
# L0168: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return "\n".join(lines)
# L0169: Assigns or updates a value used later in the workflow; check mutability and data shape.
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
# L0170: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if errors:
# L0171: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return redact_secrets(f"Could not complete request: {errors[0]}")
# L0172: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return "No results found. Please try a more specific query or check that the paper index has been built."
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
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
```
