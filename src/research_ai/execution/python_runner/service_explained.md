# service.py Explained

Generated educational companion for `src/research_ai/execution/python_runner/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/execution/python_runner/service.py` is a Python module in the Execution layer: sandboxed Python and predefined analysis pipelines. It defines PythonExecutionResult, PythonRunner and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Execution layer: sandboxed Python and predefined analysis pipelines. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Execution layer: sandboxed Python and predefined analysis pipelines.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/execution/python_runner/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/execution/python_runner/service.py]
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
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `subprocess` | subprocess runs child processes, especially isolated Python execution for sandboxed code. |
| `sys` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## Global Variables and Config

No major module-level variables are declared. This reduces hidden state and keeps imports lightweight.

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `PythonExecutionResult`

- **Line:** 13
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `to_dict` at line 20: method behavior is described by its body and name

```python
class PythonExecutionResult:
    ok: bool
    stdout: str
    error: str = ""
    latency_ms: float = 0.0
    validation_issues: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            **({"validation_issues": self.validation_issues} if self.validation_issues else {}),
        }
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `PythonRunner`

- **Line:** 30
- **Base classes:** `object`
- **Docstring:** Restricted subprocess Python runner for small scientific calculations.

Two-layer safety:
1. SandboxValidator performs AST-level static analysis before execution.
2. The subprocess runs with ``-I`` (isolated mode) and only a tiny
   builtins surface exposed through exec().

The runner is disabled unless ENABLE_PYTHON_EXECUTION=true. Even when
enabled, this is NOT a container boundary — container-level isolation
(Docker, gVisor) is the recommended production approach.

**Methods:**
- `__init__` at line 49: method behavior is described by its body and name
- `run` at line 60: method behavior is described by its body and name

```python
class PythonRunner:
    """Restricted subprocess Python runner for small scientific calculations.

    Two-layer safety:
    1. SandboxValidator performs AST-level static analysis before execution.
    2. The subprocess runs with ``-I`` (isolated mode) and only a tiny
       builtins surface exposed through exec().

    The runner is disabled unless ENABLE_PYTHON_EXECUTION=true. Even when
    enabled, this is NOT a container boundary — container-level isolation
    (Docker, gVisor) is the recommended production approach.
    """

    SAFE_BUILTINS = (
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "len", "list", "max", "min", "pow", "print", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    )

    def __init__(
        self,
        enabled: bool,
        max_code_chars: int = 4000,
        timeout_seconds: int = 5,
    ) -> None:
        self.enabled = enabled
        self.max_code_chars = max_code_chars
        self.timeout_seconds = timeout_seconds
        self._validator = SandboxValidator()

    def run(self, code: str) -> PythonExecutionResult:
        started = time.perf_counter()

        if not self.enabled:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Python execution is disabled. Set ENABLE_PYTHON_EXECUTION=true to enable it.",
            )

        if len(code) > self.max_code_chars:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Code exceeds size limit ({self.max_code_chars} chars).",
            )

        # Layer 1: AST static analysis
        validation = self._validator.validate(code)
        if not validation.ok:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Code failed sandbox validation.",
                latency_ms=(time.perf_counter() - started) * 1000,
                validation_issues=validation.issues,
            )

        # Layer 2: subprocess execution with restricted builtins
        wrapper = (
            "import math, statistics\n"
            f"_safe = {self.SAFE_BUILTINS}\n"
            "_allowed = {name: getattr(__builtins__, name, None) for name in _safe}\n"
            "_allowed = {k: v for k, v in _allowed.items() if v is not None}\n"
            "_g = {'__builtins__': _allowed, 'math': math, 'statistics': statistics}\n"
            "exec(" + repr(code) + ", _g, {})\n"
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            latency = (time.perf_counter() - started) * 1000

            if completed.returncode != 0:
                return PythonExecutionResult(
                    ok=False,
                    stdout=completed.stdout,
                    error=completed.stderr.strip() or f"Process exited with {completed.returncode}",
                    latency_ms=latency,
                )

            sanitized = self._validator.sanitize_output(completed.stdout)
            return PythonExecutionResult(ok=True, stdout=sanitized, latency_ms=latency)

        except subprocess.TimeoutExpired:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Execution timed out after {self.timeout_seconds}s.",
            )
        except Exception as exc:
            return PythonExecutionResult(
                ok=False, stdout="", error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `PythonExecutionResult` Methods

#### `PythonExecutionResult.to_dict`

- **Line:** 20
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            **({"validation_issues": self.validation_issues} if self.validation_issues else {}),
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `PythonRunner` Methods

#### `PythonRunner.__init__`

- **Line:** 49
- **Kind:** synchronous method
- **Arguments:** self, enabled, max_code_chars, timeout_seconds
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(
        self,
        enabled: bool,
        max_code_chars: int = 4000,
        timeout_seconds: int = 5,
    ) -> None:
        self.enabled = enabled
        self.max_code_chars = max_code_chars
        self.timeout_seconds = timeout_seconds
        self._validator = SandboxValidator()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PythonRunner.run`

- **Line:** 60
- **Kind:** synchronous method
- **Arguments:** self, code
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def run(self, code: str) -> PythonExecutionResult:
        started = time.perf_counter()

        if not self.enabled:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Python execution is disabled. Set ENABLE_PYTHON_EXECUTION=true to enable it.",
            )

        if len(code) > self.max_code_chars:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Code exceeds size limit ({self.max_code_chars} chars).",
            )

        # Layer 1: AST static analysis
        validation = self._validator.validate(code)
        if not validation.ok:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Code failed sandbox validation.",
                latency_ms=(time.perf_counter() - started) * 1000,
                validation_issues=validation.issues,
            )

        # Layer 2: subprocess execution with restricted builtins
        wrapper = (
            "import math, statistics\n"
            f"_safe = {self.SAFE_BUILTINS}\n"
            "_allowed = {name: getattr(__builtins__, name, None) for name in _safe}\n"
            "_allowed = {k: v for k, v in _allowed.items() if v is not None}\n"
            "_g = {'__builtins__': _allowed, 'math': math, 'statistics': statistics}\n"
            "exec(" + repr(code) + ", _g, {})\n"
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            latency = (time.perf_counter() - started) * 1000

            if completed.returncode != 0:
                return PythonExecutionResult(
                    ok=False,
                    stdout=completed.stdout,
                    error=completed.stderr.strip() or f"Process exited with {completed.returncode}",
                    latency_ms=latency,
                )

            sanitized = self._validator.sanitize_output(completed.stdout)
            return PythonExecutionResult(ok=True, stdout=sanitized, latency_ms=latency)

        except subprocess.TimeoutExpired:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Execution timed out after {self.timeout_seconds}s.",
            )
        except Exception as exc:
            return PythonExecutionResult(
                ok=False, stdout="", error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `subprocess` | subprocess runs child processes, especially isolated Python execution for sandboxed code. |
| `sys` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `time` | time measures latency, retry delays, and elapsed operation duration. |

## ML Concepts Used

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

- `src/research_ai/execution/python_runner/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/execution/python_runner/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Sandboxed subprocess Python runner for small scientific calculations."""
# L0002: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0003: Blank line that visually separates logical sections and improves readability.

# L0004: Imports a dependency, type, or project module needed by later code in this file.
import subprocess
# L0005: Imports a dependency, type, or project module needed by later code in this file.
import sys
# L0006: Imports a dependency, type, or project module needed by later code in this file.
import time
# L0007: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.execution.sandbox import SandboxValidator
# L0010: Blank line that visually separates logical sections and improves readability.

# L0011: Blank line that visually separates logical sections and improves readability.

# L0012: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0013: Defines a class that groups related state and behavior behind a reusable interface.
class PythonExecutionResult:
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ok: bool
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    stdout: str
# L0016: Assigns or updates a value used later in the workflow; check mutability and data shape.
    error: str = ""
# L0017: Assigns or updates a value used later in the workflow; check mutability and data shape.
    latency_ms: float = 0.0
# L0018: Assigns or updates a value used later in the workflow; check mutability and data shape.
    validation_issues: list[str] | None = None
# L0019: Blank line that visually separates logical sections and improves readability.

# L0020: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def to_dict(self) -> dict:
# L0021: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "ok": self.ok,
# L0023: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "stdout": self.stdout,
# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "error": self.error,
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "latency_ms": round(self.latency_ms, 2),
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            **({"validation_issues": self.validation_issues} if self.validation_issues else {}),
# L0027: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0028: Blank line that visually separates logical sections and improves readability.

# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Defines a class that groups related state and behavior behind a reusable interface.
class PythonRunner:
# L0031: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Restricted subprocess Python runner for small scientific calculations.
# L0032: Blank line that visually separates logical sections and improves readability.

# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Two-layer safety:
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    1. SandboxValidator performs AST-level static analysis before execution.
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    2. The subprocess runs with ``-I`` (isolated mode) and only a tiny
# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
       builtins surface exposed through exec().
# L0037: Blank line that visually separates logical sections and improves readability.

# L0038: Assigns or updates a value used later in the workflow; check mutability and data shape.
    The runner is disabled unless ENABLE_PYTHON_EXECUTION=true. Even when
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    enabled, this is NOT a container boundary — container-level isolation
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    (Docker, gVisor) is the recommended production approach.
# L0041: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0042: Blank line that visually separates logical sections and improves readability.

# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
    SAFE_BUILTINS = (
# L0044: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
# L0045: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "int", "len", "list", "max", "min", "pow", "print", "range",
# L0046: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
# L0047: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0048: Blank line that visually separates logical sections and improves readability.

# L0049: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0051: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        enabled: bool,
# L0052: Assigns or updates a value used later in the workflow; check mutability and data shape.
        max_code_chars: int = 4000,
# L0053: Assigns or updates a value used later in the workflow; check mutability and data shape.
        timeout_seconds: int = 5,
# L0054: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> None:
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.enabled = enabled
# L0056: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.max_code_chars = max_code_chars
# L0057: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.timeout_seconds = timeout_seconds
# L0058: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._validator = SandboxValidator()
# L0059: Blank line that visually separates logical sections and improves readability.

# L0060: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def run(self, code: str) -> PythonExecutionResult:
# L0061: Assigns or updates a value used later in the workflow; check mutability and data shape.
        started = time.perf_counter()
# L0062: Blank line that visually separates logical sections and improves readability.

# L0063: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not self.enabled:
# L0064: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
                ok=False,
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
                stdout="",
# L0067: Assigns or updates a value used later in the workflow; check mutability and data shape.
                error="Python execution is disabled. Set ENABLE_PYTHON_EXECUTION=true to enable it.",
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0069: Blank line that visually separates logical sections and improves readability.

# L0070: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if len(code) > self.max_code_chars:
# L0071: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
                ok=False, stdout="",
# L0073: Assigns or updates a value used later in the workflow; check mutability and data shape.
                error=f"Code exceeds size limit ({self.max_code_chars} chars).",
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0075: Blank line that visually separates logical sections and improves readability.

# L0076: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Layer 1: AST static analysis
# L0077: Assigns or updates a value used later in the workflow; check mutability and data shape.
        validation = self._validator.validate(code)
# L0078: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not validation.ok:
# L0079: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(
# L0080: Assigns or updates a value used later in the workflow; check mutability and data shape.
                ok=False,
# L0081: Assigns or updates a value used later in the workflow; check mutability and data shape.
                stdout="",
# L0082: Assigns or updates a value used later in the workflow; check mutability and data shape.
                error="Code failed sandbox validation.",
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
                latency_ms=(time.perf_counter() - started) * 1000,
# L0084: Assigns or updates a value used later in the workflow; check mutability and data shape.
                validation_issues=validation.issues,
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0086: Blank line that visually separates logical sections and improves readability.

# L0087: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Layer 2: subprocess execution with restricted builtins
# L0088: Assigns or updates a value used later in the workflow; check mutability and data shape.
        wrapper = (
# L0089: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "import math, statistics\n"
# L0090: Assigns or updates a value used later in the workflow; check mutability and data shape.
            f"_safe = {self.SAFE_BUILTINS}\n"
# L0091: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "_allowed = {name: getattr(__builtins__, name, None) for name in _safe}\n"
# L0092: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "_allowed = {k: v for k, v in _allowed.items() if v is not None}\n"
# L0093: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "_g = {'__builtins__': _allowed, 'math': math, 'statistics': statistics}\n"
# L0094: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "exec(" + repr(code) + ", _g, {})\n"
# L0095: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0096: Blank line that visually separates logical sections and improves readability.

# L0097: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0098: Assigns or updates a value used later in the workflow; check mutability and data shape.
            completed = subprocess.run(
# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                [sys.executable, "-I", "-c", wrapper],
# L0100: Assigns or updates a value used later in the workflow; check mutability and data shape.
                capture_output=True,
# L0101: Assigns or updates a value used later in the workflow; check mutability and data shape.
                text=True,
# L0102: Assigns or updates a value used later in the workflow; check mutability and data shape.
                timeout=self.timeout_seconds,
# L0103: Assigns or updates a value used later in the workflow; check mutability and data shape.
                check=False,
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0105: Assigns or updates a value used later in the workflow; check mutability and data shape.
            latency = (time.perf_counter() - started) * 1000
# L0106: Blank line that visually separates logical sections and improves readability.

# L0107: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if completed.returncode != 0:
# L0108: Returns the computed result to the caller; this shape becomes part of the downstream contract.
                return PythonExecutionResult(
# L0109: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    ok=False,
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    stdout=completed.stdout,
# L0111: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    error=completed.stderr.strip() or f"Process exited with {completed.returncode}",
# L0112: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    latency_ms=latency,
# L0113: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                )
# L0114: Blank line that visually separates logical sections and improves readability.

# L0115: Assigns or updates a value used later in the workflow; check mutability and data shape.
            sanitized = self._validator.sanitize_output(completed.stdout)
# L0116: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(ok=True, stdout=sanitized, latency_ms=latency)
# L0117: Blank line that visually separates logical sections and improves readability.

# L0118: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except subprocess.TimeoutExpired:
# L0119: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(
# L0120: Assigns or updates a value used later in the workflow; check mutability and data shape.
                ok=False, stdout="",
# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
                error=f"Execution timed out after {self.timeout_seconds}s.",
# L0122: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0123: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception as exc:
# L0124: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return PythonExecutionResult(
# L0125: Assigns or updates a value used later in the workflow; check mutability and data shape.
                ok=False, stdout="", error=str(exc),
# L0126: Assigns or updates a value used later in the workflow; check mutability and data shape.
                latency_ms=(time.perf_counter() - started) * 1000,
# L0127: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Sandboxed subprocess Python runner for small scientific calculations."""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

from research_ai.execution.sandbox import SandboxValidator


@dataclass
class PythonExecutionResult:
    ok: bool
    stdout: str
    error: str = ""
    latency_ms: float = 0.0
    validation_issues: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            **({"validation_issues": self.validation_issues} if self.validation_issues else {}),
        }


class PythonRunner:
    """Restricted subprocess Python runner for small scientific calculations.

    Two-layer safety:
    1. SandboxValidator performs AST-level static analysis before execution.
    2. The subprocess runs with ``-I`` (isolated mode) and only a tiny
       builtins surface exposed through exec().

    The runner is disabled unless ENABLE_PYTHON_EXECUTION=true. Even when
    enabled, this is NOT a container boundary — container-level isolation
    (Docker, gVisor) is the recommended production approach.
    """

    SAFE_BUILTINS = (
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "len", "list", "max", "min", "pow", "print", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    )

    def __init__(
        self,
        enabled: bool,
        max_code_chars: int = 4000,
        timeout_seconds: int = 5,
    ) -> None:
        self.enabled = enabled
        self.max_code_chars = max_code_chars
        self.timeout_seconds = timeout_seconds
        self._validator = SandboxValidator()

    def run(self, code: str) -> PythonExecutionResult:
        started = time.perf_counter()

        if not self.enabled:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Python execution is disabled. Set ENABLE_PYTHON_EXECUTION=true to enable it.",
            )

        if len(code) > self.max_code_chars:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Code exceeds size limit ({self.max_code_chars} chars).",
            )

        # Layer 1: AST static analysis
        validation = self._validator.validate(code)
        if not validation.ok:
            return PythonExecutionResult(
                ok=False,
                stdout="",
                error="Code failed sandbox validation.",
                latency_ms=(time.perf_counter() - started) * 1000,
                validation_issues=validation.issues,
            )

        # Layer 2: subprocess execution with restricted builtins
        wrapper = (
            "import math, statistics\n"
            f"_safe = {self.SAFE_BUILTINS}\n"
            "_allowed = {name: getattr(__builtins__, name, None) for name in _safe}\n"
            "_allowed = {k: v for k, v in _allowed.items() if v is not None}\n"
            "_g = {'__builtins__': _allowed, 'math': math, 'statistics': statistics}\n"
            "exec(" + repr(code) + ", _g, {})\n"
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            latency = (time.perf_counter() - started) * 1000

            if completed.returncode != 0:
                return PythonExecutionResult(
                    ok=False,
                    stdout=completed.stdout,
                    error=completed.stderr.strip() or f"Process exited with {completed.returncode}",
                    latency_ms=latency,
                )

            sanitized = self._validator.sanitize_output(completed.stdout)
            return PythonExecutionResult(ok=True, stdout=sanitized, latency_ms=latency)

        except subprocess.TimeoutExpired:
            return PythonExecutionResult(
                ok=False, stdout="",
                error=f"Execution timed out after {self.timeout_seconds}s.",
            )
        except Exception as exc:
            return PythonExecutionResult(
                ok=False, stdout="", error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
```
