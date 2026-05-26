# service.py Explained

Generated educational companion for `src/research_ai/execution/pipelines/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/execution/pipelines/service.py` is a Python module in the Execution layer: sandboxed Python and predefined analysis pipelines. It defines PipelineStep, PipelineResult, PipelineRunner and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Execution layer: sandboxed Python and predefined analysis pipelines. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Execution layer: sandboxed Python and predefined analysis pipelines.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/execution/pipelines/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/execution/pipelines/service.py]
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
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 14 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |
| `PIPELINES` | 48 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `PipelineStep`

- **Line:** 18
- **Base classes:** `object`
- **Docstring:** A single named step in a pipeline.

**Methods:**
- No methods beyond inherited behavior.

```python
class PipelineStep:
    """A single named step in a pipeline."""
    name: str
    tool: str
    args_template: dict = field(default_factory=dict)
    depends_on: str | None = None   # key from prior step output to inject as input
    optional: bool = False
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `PipelineResult`

- **Line:** 28
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `to_dict` at line 36: method behavior is described by its body and name

```python
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
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `PipelineRunner`

- **Line:** 96
- **Base classes:** `object`
- **Docstring:** Executes named pipelines by delegating each step to the ML execution agent.

Data-flow: the output of a step tagged with ``depends_on`` is resolved
so that ``args_template`` values referencing ``{from: search_results}``
are replaced with the ``results`` list from the prior step.

**Methods:**
- `__init__` at line 104: Args:
    executor: An object with an ``execute(tool_name, args)`` method —
              typically the ``MLExecutionAgent``.
- `run` at line 112: Execute a named pipeline and return structured outputs.
- `available_pipelines` at line 154: method behavior is described by its body and name
- `_resolve_args` at line 162: method behavior is described by its body and name

```python
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
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `PipelineResult` Methods

#### `PipelineResult.to_dict`

- **Line:** 36
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def to_dict(self) -> dict:
        return {
            "pipeline": self.name,
            "steps_run": self.steps_run,
            "steps_ok": self.steps_ok,
            "outputs": self.outputs,
            "latency_ms": round(self.latency_ms, 2),
            "errors": self.errors,
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `PipelineRunner` Methods

#### `PipelineRunner.__init__`

- **Line:** 104
- **Kind:** synchronous method
- **Arguments:** self, executor
- **Docstring:** Args:
    executor: An object with an ``execute(tool_name, args)`` method —
              typically the ``MLExecutionAgent``.

```python
    def __init__(self, executor) -> None:
        """
        Args:
            executor: An object with an ``execute(tool_name, args)`` method —
                      typically the ``MLExecutionAgent``.
        """
        self.executor = executor
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PipelineRunner.run`

- **Line:** 112
- **Kind:** synchronous method
- **Arguments:** self, pipeline_name, query, extra_args
- **Docstring:** Execute a named pipeline and return structured outputs.

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PipelineRunner.available_pipelines`

- **Line:** 154
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def available_pipelines(self) -> list[str]:
        return sorted(PIPELINES.keys())
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PipelineRunner._resolve_args`

- **Line:** 162
- **Kind:** synchronous method
- **Arguments:** step, query, outputs, extra
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
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
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## ML Concepts Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
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

- Deals with execution or subprocesses. Maintain AST validation, isolated mode, timeouts, and least privilege.

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

- `src/research_ai/execution/pipelines/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/execution/pipelines/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Composable execution pipelines for multi-step research analysis workflows.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Each pipeline is a named sequence of tool steps with data-flow contracts.
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
The PipelineRunner executes steps in order, threading outputs from one step
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
into the inputs of the next. Steps can be conditional, and any step failure
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
triggers graceful degradation rather than hard failure.
# L0007: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0008: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0009: Blank line that visually separates logical sections and improves readability.

# L0010: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0011: Imports a dependency, type, or project module needed by later code in this file.
import time
# L0012: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass, field
# L0013: Blank line that visually separates logical sections and improves readability.

# L0014: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0015: Blank line that visually separates logical sections and improves readability.

# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0018: Defines a class that groups related state and behavior behind a reusable interface.
class PipelineStep:
# L0019: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """A single named step in a pipeline."""
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    name: str
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    tool: str
# L0022: Assigns or updates a value used later in the workflow; check mutability and data shape.
    args_template: dict = field(default_factory=dict)
# L0023: Assigns or updates a value used later in the workflow; check mutability and data shape.
    depends_on: str | None = None   # key from prior step output to inject as input
# L0024: Assigns or updates a value used later in the workflow; check mutability and data shape.
    optional: bool = False
# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Blank line that visually separates logical sections and improves readability.

# L0027: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0028: Defines a class that groups related state and behavior behind a reusable interface.
class PipelineResult:
# L0029: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    name: str
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    steps_run: int
# L0031: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    steps_ok: int
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    outputs: dict
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    latency_ms: float
# L0034: Assigns or updates a value used later in the workflow; check mutability and data shape.
    errors: list[str] = field(default_factory=list)
# L0035: Blank line that visually separates logical sections and improves readability.

# L0036: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def to_dict(self) -> dict:
# L0037: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "pipeline": self.name,
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "steps_run": self.steps_run,
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "steps_ok": self.steps_ok,
# L0041: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "outputs": self.outputs,
# L0042: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "latency_ms": round(self.latency_ms, 2),
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "errors": self.errors,
# L0044: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0045: Blank line that visually separates logical sections and improves readability.

# L0046: Blank line that visually separates logical sections and improves readability.

# L0047: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Pre-defined research analysis pipelines
# L0048: Assigns or updates a value used later in the workflow; check mutability and data shape.
PIPELINES: dict[str, list[PipelineStep]] = {
# L0049: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "full_research_analysis": [
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("classify", "classify_query", {"title": "{query}", "abstract": "{query}"}),
# L0051: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 8}),
# L0052: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep(
# L0053: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "methodology", "methodology_extract",
# L0054: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args_template={"from": "search_results"},
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
            depends_on="retrieve",
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ),
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep(
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "citations", "citation_signals",
# L0059: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args_template={"from": "search_results"},
# L0060: Assigns or updates a value used later in the workflow; check mutability and data shape.
            depends_on="retrieve",
# L0061: Assigns or updates a value used later in the workflow; check mutability and data shape.
            optional=True,
# L0062: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ),
# L0063: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep(
# L0064: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "trends", "trend_analysis",
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args_template={"from": "search_results"},
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
            depends_on="retrieve",
# L0067: Assigns or updates a value used later in the workflow; check mutability and data shape.
            optional=True,
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ),
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("synthesize", "metadata_rag", {"query": "{query}", "top_k": 5}),
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ],
# L0071: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "quick_search_and_summarize": [
# L0072: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 5}),
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("synthesize", "metadata_rag", {"query": "{query}", "top_k": 5}),
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ],
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "classify_and_find_similar": [
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("classify", "classify_query", {"title": "{query}", "abstract": "{query}"}),
# L0077: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 10}),
# L0078: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ],
# L0079: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "trend_report": [
# L0080: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep("retrieve", "hybrid_search", {"query": "{query}", "top_k": 15}),
# L0081: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep(
# L0082: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "trends", "trend_analysis",
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args_template={"from": "search_results"},
# L0084: Assigns or updates a value used later in the workflow; check mutability and data shape.
            depends_on="retrieve",
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ),
# L0086: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        PipelineStep(
# L0087: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "citations", "citation_signals",
# L0088: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args_template={"from": "search_results"},
# L0089: Assigns or updates a value used later in the workflow; check mutability and data shape.
            depends_on="retrieve",
# L0090: Assigns or updates a value used later in the workflow; check mutability and data shape.
            optional=True,
# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ),
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ],
# L0093: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
}
# L0094: Blank line that visually separates logical sections and improves readability.

# L0095: Blank line that visually separates logical sections and improves readability.

# L0096: Defines a class that groups related state and behavior behind a reusable interface.
class PipelineRunner:
# L0097: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Executes named pipelines by delegating each step to the ML execution agent.
# L0098: Blank line that visually separates logical sections and improves readability.

# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Data-flow: the output of a step tagged with ``depends_on`` is resolved
# L0100: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    so that ``args_template`` values referencing ``{from: search_results}``
# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    are replaced with the ``results`` list from the prior step.
# L0102: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0103: Blank line that visually separates logical sections and improves readability.

# L0104: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self, executor) -> None:
# L0105: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0106: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Args:
# L0107: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            executor: An object with an ``execute(tool_name, args)`` method —
# L0108: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                      typically the ``MLExecutionAgent``.
# L0109: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.executor = executor
# L0111: Blank line that visually separates logical sections and improves readability.

# L0112: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def run(self, pipeline_name: str, query: str, extra_args: dict | None = None) -> PipelineResult:
# L0113: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Execute a named pipeline and return structured outputs."""
# L0114: Assigns or updates a value used later in the workflow; check mutability and data shape.
        steps = PIPELINES.get(pipeline_name)
# L0115: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if steps is None:
# L0116: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PipelineResult(
# L0117: Assigns or updates a value used later in the workflow; check mutability and data shape.
                name=pipeline_name,
# L0118: Assigns or updates a value used later in the workflow; check mutability and data shape.
                steps_run=0,
# L0119: Assigns or updates a value used later in the workflow; check mutability and data shape.
                steps_ok=0,
# L0120: Assigns or updates a value used later in the workflow; check mutability and data shape.
                outputs={},
# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
                latency_ms=0,
# L0122: Assigns or updates a value used later in the workflow; check mutability and data shape.
                errors=[f"Unknown pipeline: '{pipeline_name}'. Available: {sorted(PIPELINES)}"],
# L0123: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0124: Blank line that visually separates logical sections and improves readability.

# L0125: Assigns or updates a value used later in the workflow; check mutability and data shape.
        started = time.perf_counter()
# L0126: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs: dict = {}
# L0127: Assigns or updates a value used later in the workflow; check mutability and data shape.
        errors: list[str] = []
# L0128: Assigns or updates a value used later in the workflow; check mutability and data shape.
        extra = extra_args or {}
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Iterates over data, retry attempts, files, results, or workflow steps.
        for step in steps:
# L0131: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args = self._resolve_args(step, query, outputs, extra)
# L0132: Begins protected execution so failures can be handled without crashing the whole request path.
            try:
# L0133: Assigns or updates a value used later in the workflow; check mutability and data shape.
                result = self.executor.execute(step.tool, args)
# L0134: Assigns or updates a value used later in the workflow; check mutability and data shape.
                outputs[step.name] = result
# L0135: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if result.get("error") and not step.optional:
# L0136: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    errors.append(f"Step '{step.name}' error: {result['error']}")
# L0137: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
            except Exception as exc:
# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
                error_msg = f"Step '{step.name}' raised: {exc!s}"
# L0139: Emits structured operational information for debugging, monitoring, or failure diagnosis.
                logger.warning(error_msg)
# L0140: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                errors.append(error_msg)
# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
                outputs[step.name] = {"error": str(exc)}
# L0142: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if not step.optional:
# L0143: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    break   # non-optional failure → abort pipeline
# L0144: Blank line that visually separates logical sections and improves readability.

# L0145: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return PipelineResult(
# L0146: Assigns or updates a value used later in the workflow; check mutability and data shape.
            name=pipeline_name,
# L0147: Assigns or updates a value used later in the workflow; check mutability and data shape.
            steps_run=len(steps),
# L0148: Assigns or updates a value used later in the workflow; check mutability and data shape.
            steps_ok=sum(1 for s in steps if not outputs.get(s.name, {}).get("error")),
# L0149: Assigns or updates a value used later in the workflow; check mutability and data shape.
            outputs=outputs,
# L0150: Assigns or updates a value used later in the workflow; check mutability and data shape.
            latency_ms=(time.perf_counter() - started) * 1000,
# L0151: Assigns or updates a value used later in the workflow; check mutability and data shape.
            errors=errors,
# L0152: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0153: Blank line that visually separates logical sections and improves readability.

# L0154: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def available_pipelines(self) -> list[str]:
# L0155: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return sorted(PIPELINES.keys())
# L0156: Blank line that visually separates logical sections and improves readability.

# L0157: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0158: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Private
# L0159: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0160: Blank line that visually separates logical sections and improves readability.

# L0161: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0162: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _resolve_args(
# L0163: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        step: PipelineStep, query: str, outputs: dict, extra: dict
# L0164: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> dict:
# L0165: Assigns or updates a value used later in the workflow; check mutability and data shape.
        args: dict = {}
# L0166: Iterates over data, retry attempts, files, results, or workflow steps.
        for key, value in step.args_template.items():
# L0167: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(value, str):
# L0168: Assigns or updates a value used later in the workflow; check mutability and data shape.
                args[key] = value.replace("{query}", query)
# L0169: Continues conditional control flow for alternate cases or default fallback behavior.
            else:
# L0170: Assigns or updates a value used later in the workflow; check mutability and data shape.
                args[key] = value
# L0171: Blank line that visually separates logical sections and improves readability.

# L0172: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Inject results from a prior step when args_template says "from: search_results"
# L0173: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if args.get("from") == "search_results" and step.depends_on:
# L0174: Assigns or updates a value used later in the workflow; check mutability and data shape.
            prior = outputs.get(step.depends_on, {})
# L0175: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args = {"papers": prior.get("results", [])}
# L0176: Blank line that visually separates logical sections and improves readability.

# L0177: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        args.update(extra)
# L0178: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return args
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
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
```
