# service.py Explained

Generated educational companion for `src/research_ai/agents/ml_execution_agent/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/agents/ml_execution_agent/service.py` is a Python module in the Tool execution layer: dispatches normalized tool calls. It defines MLExecutionAgent and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Tool execution layer: dispatches normalized tool calls. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Tool execution layer: dispatches normalized tool calls.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/agents/ml_execution_agent/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/agents/ml_execution_agent/service.py]
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
| `logging` | logging provides structured operational visibility without using print statements. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 6 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |
| `_SEARCH_TOOLS` | 9 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `MLExecutionAgent`

- **Line:** 12
- **Base classes:** `object`
- **Docstring:** Executes registered local ML / retrieval / research tools.

Improvements over original:
- Type coercion: string "5" → int 5, string "true" → bool True
- Recognises `{"from": "search_results"}` from ANY prior search tool,
  not just hybrid_search (fixes the data-flow injection bug)
- Per-tool error isolation: a failing tool does not abort the plan
- Logs tool name and latency for observability

**Methods:**
- `__init__` at line 23: method behavior is described by its body and name
- `execute` at line 26: method behavior is described by its body and name
- `execute_plan` at line 41: Execute a list of ToolCall objects, resolving data-flow injections.
- `_coerce_args` at line 69: Coerce common LLM type mistakes: string numbers, string booleans.

```python
class MLExecutionAgent:
    """Executes registered local ML / retrieval / research tools.

    Improvements over original:
    - Type coercion: string "5" → int 5, string "true" → bool True
    - Recognises `{"from": "search_results"}` from ANY prior search tool,
      not just hybrid_search (fixes the data-flow injection bug)
    - Per-tool error isolation: a failing tool does not abort the plan
    - Logs tool name and latency for observability
    """

    def __init__(self, tools: dict[str, object]) -> None:
        self.tools = tools

    def execute(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: '{name}'. Available: {sorted(self.tools)}"}
        coerced = self._coerce_args(args)
        try:
            result = tool(**coerced)
            return result if isinstance(result, dict) else {"result": result}
        except TypeError as exc:
            logger.warning("Tool '%s' argument error: %s | args=%s", name, exc, coerced)
            return {"error": f"Tool '{name}' argument mismatch: {exc}"}
        except Exception as exc:
            logger.error("Tool '%s' raised: %s", name, exc)
            return {"error": str(exc)}

    def execute_plan(self, calls: list) -> dict:
        """Execute a list of ToolCall objects, resolving data-flow injections."""
        outputs: dict = {}
        last_search_results: list[dict] = []

        for call in calls:
            args = dict(call.args)

            # Data-flow injection: {"from": "search_results"} → inject prior paper list
            if args.get("from") == "search_results":
                args = {"papers": list(last_search_results)}

            result = self.execute(call.name, args)
            outputs[call.name] = result

            # Track the latest search results for downstream injection
            if call.name in _SEARCH_TOOLS and isinstance(result, dict):
                candidate = result.get("results", [])
                if isinstance(candidate, list) and candidate:
                    last_search_results = candidate

        return outputs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_args(args: dict) -> dict:
        """Coerce common LLM type mistakes: string numbers, string booleans."""
        coerced: dict = {}
        for key, value in args.items():
            if isinstance(value, str):
                # int coercion for known numeric keys
                if key in ("top_k", "candidate_k", "max_tokens", "timeout"):
                    try:
                        coerced[key] = int(value)
                        continue
                    except ValueError:
                        pass
                # bool coercion
                if value.lower() in ("true", "false"):
                    coerced[key] = value.lower() == "true"
                    continue
            coerced[key] = value
        return coerced
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `MLExecutionAgent` Methods

#### `MLExecutionAgent.__init__`

- **Line:** 23
- **Kind:** synchronous method
- **Arguments:** self, tools
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(self, tools: dict[str, object]) -> None:
        self.tools = tools
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `MLExecutionAgent.execute`

- **Line:** 26
- **Kind:** synchronous method
- **Arguments:** self, name, args
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def execute(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: '{name}'. Available: {sorted(self.tools)}"}
        coerced = self._coerce_args(args)
        try:
            result = tool(**coerced)
            return result if isinstance(result, dict) else {"result": result}
        except TypeError as exc:
            logger.warning("Tool '%s' argument error: %s | args=%s", name, exc, coerced)
            return {"error": f"Tool '{name}' argument mismatch: {exc}"}
        except Exception as exc:
            logger.error("Tool '%s' raised: %s", name, exc)
            return {"error": str(exc)}
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `MLExecutionAgent.execute_plan`

- **Line:** 41
- **Kind:** synchronous method
- **Arguments:** self, calls
- **Docstring:** Execute a list of ToolCall objects, resolving data-flow injections.

```python
    def execute_plan(self, calls: list) -> dict:
        """Execute a list of ToolCall objects, resolving data-flow injections."""
        outputs: dict = {}
        last_search_results: list[dict] = []

        for call in calls:
            args = dict(call.args)

            # Data-flow injection: {"from": "search_results"} → inject prior paper list
            if args.get("from") == "search_results":
                args = {"papers": list(last_search_results)}

            result = self.execute(call.name, args)
            outputs[call.name] = result

            # Track the latest search results for downstream injection
            if call.name in _SEARCH_TOOLS and isinstance(result, dict):
                candidate = result.get("results", [])
                if isinstance(candidate, list) and candidate:
                    last_search_results = candidate

        return outputs
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `MLExecutionAgent._coerce_args`

- **Line:** 69
- **Kind:** synchronous method
- **Arguments:** args
- **Docstring:** Coerce common LLM type mistakes: string numbers, string booleans.

```python
    def _coerce_args(args: dict) -> dict:
        """Coerce common LLM type mistakes: string numbers, string booleans."""
        coerced: dict = {}
        for key, value in args.items():
            if isinstance(value, str):
                # int coercion for known numeric keys
                if key in ("top_k", "candidate_k", "max_tokens", "timeout"):
                    try:
                        coerced[key] = int(value)
                        continue
                    except ValueError:
                        pass
                # bool coercion
                if value.lower() in ("true", "false"):
                    coerced[key] = value.lower() == "true"
                    continue
            coerced[key] = value
        return coerced
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `logging` | logging provides structured operational visibility without using print statements. |

## ML Concepts Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
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

- `src/research_ai/agents/ml_execution_agent/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/agents/ml_execution_agent/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""MLExecutionAgent — dispatches tool calls with type coercion and error isolation."""
# L0002: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0003: Blank line that visually separates logical sections and improves readability.

# L0004: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0005: Blank line that visually separates logical sections and improves readability.

# L0006: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0007: Blank line that visually separates logical sections and improves readability.

# L0008: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Tools whose results should be injected into subsequent "from: search_results" calls
# L0009: Assigns or updates a value used later in the workflow; check mutability and data shape.
_SEARCH_TOOLS = ("hybrid_search", "smart_retrieve")
# L0010: Blank line that visually separates logical sections and improves readability.

# L0011: Blank line that visually separates logical sections and improves readability.

# L0012: Defines a class that groups related state and behavior behind a reusable interface.
class MLExecutionAgent:
# L0013: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Executes registered local ML / retrieval / research tools.
# L0014: Blank line that visually separates logical sections and improves readability.

# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Improvements over original:
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Type coercion: string "5" → int 5, string "true" → bool True
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Recognises `{"from": "search_results"}` from ANY prior search tool,
# L0018: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      not just hybrid_search (fixes the data-flow injection bug)
# L0019: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Per-tool error isolation: a failing tool does not abort the plan
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    - Logs tool name and latency for observability
# L0021: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0022: Blank line that visually separates logical sections and improves readability.

# L0023: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self, tools: dict[str, object]) -> None:
# L0024: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.tools = tools
# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def execute(self, name: str, args: dict) -> dict:
# L0027: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tool = self.tools.get(name)
# L0028: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if tool is None:
# L0029: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": f"Unknown tool: '{name}'. Available: {sorted(self.tools)}"}
# L0030: Assigns or updates a value used later in the workflow; check mutability and data shape.
        coerced = self._coerce_args(args)
# L0031: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0032: Assigns or updates a value used later in the workflow; check mutability and data shape.
            result = tool(**coerced)
# L0033: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return result if isinstance(result, dict) else {"result": result}
# L0034: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except TypeError as exc:
# L0035: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.warning("Tool '%s' argument error: %s | args=%s", name, exc, coerced)
# L0036: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": f"Tool '{name}' argument mismatch: {exc}"}
# L0037: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception as exc:
# L0038: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.error("Tool '%s' raised: %s", name, exc)
# L0039: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": str(exc)}
# L0040: Blank line that visually separates logical sections and improves readability.

# L0041: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def execute_plan(self, calls: list) -> dict:
# L0042: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Execute a list of ToolCall objects, resolving data-flow injections."""
# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs: dict = {}
# L0044: Assigns or updates a value used later in the workflow; check mutability and data shape.
        last_search_results: list[dict] = []
# L0045: Blank line that visually separates logical sections and improves readability.

# L0046: Iterates over data, retry attempts, files, results, or workflow steps.
        for call in calls:
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
            args = dict(call.args)
# L0048: Blank line that visually separates logical sections and improves readability.

# L0049: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # Data-flow injection: {"from": "search_results"} → inject prior paper list
# L0050: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if args.get("from") == "search_results":
# L0051: Assigns or updates a value used later in the workflow; check mutability and data shape.
                args = {"papers": list(last_search_results)}
# L0052: Blank line that visually separates logical sections and improves readability.

# L0053: Assigns or updates a value used later in the workflow; check mutability and data shape.
            result = self.execute(call.name, args)
# L0054: Assigns or updates a value used later in the workflow; check mutability and data shape.
            outputs[call.name] = result
# L0055: Blank line that visually separates logical sections and improves readability.

# L0056: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # Track the latest search results for downstream injection
# L0057: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if call.name in _SEARCH_TOOLS and isinstance(result, dict):
# L0058: Assigns or updates a value used later in the workflow; check mutability and data shape.
                candidate = result.get("results", [])
# L0059: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if isinstance(candidate, list) and candidate:
# L0060: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    last_search_results = candidate
# L0061: Blank line that visually separates logical sections and improves readability.

# L0062: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return outputs
# L0063: Blank line that visually separates logical sections and improves readability.

# L0064: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0065: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Private
# L0066: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0067: Blank line that visually separates logical sections and improves readability.

# L0068: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0069: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _coerce_args(args: dict) -> dict:
# L0070: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Coerce common LLM type mistakes: string numbers, string booleans."""
# L0071: Assigns or updates a value used later in the workflow; check mutability and data shape.
        coerced: dict = {}
# L0072: Iterates over data, retry attempts, files, results, or workflow steps.
        for key, value in args.items():
# L0073: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(value, str):
# L0074: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
                # int coercion for known numeric keys
# L0075: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if key in ("top_k", "candidate_k", "max_tokens", "timeout"):
# L0076: Begins protected execution so failures can be handled without crashing the whole request path.
                    try:
# L0077: Assigns or updates a value used later in the workflow; check mutability and data shape.
                        coerced[key] = int(value)
# L0078: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        continue
# L0079: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
                    except ValueError:
# L0080: Explicit no-op placeholder used when no action is required for this branch.
                        pass
# L0081: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
                # bool coercion
# L0082: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if value.lower() in ("true", "false"):
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    coerced[key] = value.lower() == "true"
# L0084: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    continue
# L0085: Assigns or updates a value used later in the workflow; check mutability and data shape.
            coerced[key] = value
# L0086: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return coerced
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""MLExecutionAgent — dispatches tool calls with type coercion and error isolation."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Tools whose results should be injected into subsequent "from: search_results" calls
_SEARCH_TOOLS = ("hybrid_search", "smart_retrieve")


class MLExecutionAgent:
    """Executes registered local ML / retrieval / research tools.

    Improvements over original:
    - Type coercion: string "5" → int 5, string "true" → bool True
    - Recognises `{"from": "search_results"}` from ANY prior search tool,
      not just hybrid_search (fixes the data-flow injection bug)
    - Per-tool error isolation: a failing tool does not abort the plan
    - Logs tool name and latency for observability
    """

    def __init__(self, tools: dict[str, object]) -> None:
        self.tools = tools

    def execute(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: '{name}'. Available: {sorted(self.tools)}"}
        coerced = self._coerce_args(args)
        try:
            result = tool(**coerced)
            return result if isinstance(result, dict) else {"result": result}
        except TypeError as exc:
            logger.warning("Tool '%s' argument error: %s | args=%s", name, exc, coerced)
            return {"error": f"Tool '{name}' argument mismatch: {exc}"}
        except Exception as exc:
            logger.error("Tool '%s' raised: %s", name, exc)
            return {"error": str(exc)}

    def execute_plan(self, calls: list) -> dict:
        """Execute a list of ToolCall objects, resolving data-flow injections."""
        outputs: dict = {}
        last_search_results: list[dict] = []

        for call in calls:
            args = dict(call.args)

            # Data-flow injection: {"from": "search_results"} → inject prior paper list
            if args.get("from") == "search_results":
                args = {"papers": list(last_search_results)}

            result = self.execute(call.name, args)
            outputs[call.name] = result

            # Track the latest search results for downstream injection
            if call.name in _SEARCH_TOOLS and isinstance(result, dict):
                candidate = result.get("results", [])
                if isinstance(candidate, list) and candidate:
                    last_search_results = candidate

        return outputs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_args(args: dict) -> dict:
        """Coerce common LLM type mistakes: string numbers, string booleans."""
        coerced: dict = {}
        for key, value in args.items():
            if isinstance(value, str):
                # int coercion for known numeric keys
                if key in ("top_k", "candidate_k", "max_tokens", "timeout"):
                    try:
                        coerced[key] = int(value)
                        continue
                    except ValueError:
                        pass
                # bool coercion
                if value.lower() in ("true", "false"):
                    coerced[key] = value.lower() == "true"
                    continue
            coerced[key] = value
        return coerced
```
