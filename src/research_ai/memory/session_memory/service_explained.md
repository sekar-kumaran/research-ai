# service.py Explained

Generated educational companion for `src/research_ai/memory/session_memory/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/memory/session_memory/service.py` is a Python module in the Memory layer: conversation, session, and knowledge graph state. It defines ChatSession, SessionMemory and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Memory layer: conversation, session, and knowledge graph state. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Memory layer: conversation, session, and knowledge graph state.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/memory/session_memory/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/memory/session_memory/service.py]
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

### `ChatSession`

- **Line:** 7
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    metadata: dict = field(default_factory=dict)
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `SessionMemory`

- **Line:** 17
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `__init__` at line 18: method behavior is described by its body and name
- `get` at line 22: method behavior is described by its body and name
- `put` at line 27: method behavior is described by its body and name
- `find_source` at line 31: method behavior is described by its body and name

```python
class SessionMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Load a paper first.")
        return self.sessions[session_id]

    def put(self, session: ChatSession) -> None:
        self.sessions[session.session_id] = session
        self.source_to_session[session.source] = session.session_id

    def find_source(self, source: str) -> ChatSession | None:
        session_id = self.source_to_session.get(source)
        if not session_id:
            return None
        return self.sessions.get(session_id)
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `SessionMemory` Methods

#### `SessionMemory.__init__`

- **Line:** 18
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SessionMemory.get`

- **Line:** 22
- **Kind:** synchronous method
- **Arguments:** self, session_id
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Load a paper first.")
        return self.sessions[session_id]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SessionMemory.put`

- **Line:** 27
- **Kind:** synchronous method
- **Arguments:** self, session
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def put(self, session: ChatSession) -> None:
        self.sessions[session.session_id] = session
        self.source_to_session[session.source] = session.session_id
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SessionMemory.find_source`

- **Line:** 31
- **Kind:** synchronous method
- **Arguments:** self, source
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def find_source(self, source: str) -> ChatSession | None:
        session_id = self.source_to_session.get(source)
        if not session_id:
            return None
        return self.sessions.get(session_id)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |

## ML Concepts Used

- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.

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

- No direct high-risk boundary is visible, but caller-side validation and defensive error handling still matter.

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

- `src/research_ai/memory/session_memory/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/memory/session_memory/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass, field
# L0004: Blank line that visually separates logical sections and improves readability.

# L0005: Blank line that visually separates logical sections and improves readability.

# L0006: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0007: Defines a class that groups related state and behavior behind a reusable interface.
class ChatSession:
# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    session_id: str
# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    source: str
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    chunks: list[str]
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    index: object
# L0012: Assigns or updates a value used later in the workflow; check mutability and data shape.
    history: list[dict[str, str]] = field(default_factory=list)
# L0013: Assigns or updates a value used later in the workflow; check mutability and data shape.
    title: str = ""
# L0014: Assigns or updates a value used later in the workflow; check mutability and data shape.
    metadata: dict = field(default_factory=dict)
# L0015: Blank line that visually separates logical sections and improves readability.

# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Defines a class that groups related state and behavior behind a reusable interface.
class SessionMemory:
# L0018: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self) -> None:
# L0019: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.sessions: dict[str, ChatSession] = {}
# L0020: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.source_to_session: dict[str, str] = {}
# L0021: Blank line that visually separates logical sections and improves readability.

# L0022: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def get(self, session_id: str) -> ChatSession:
# L0023: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if session_id not in self.sessions:
# L0024: Raises an explicit error when the function cannot safely continue.
            raise KeyError(f"Session '{session_id}' not found. Load a paper first.")
# L0025: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.sessions[session_id]
# L0026: Blank line that visually separates logical sections and improves readability.

# L0027: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def put(self, session: ChatSession) -> None:
# L0028: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.sessions[session.session_id] = session
# L0029: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.source_to_session[session.source] = session.session_id
# L0030: Blank line that visually separates logical sections and improves readability.

# L0031: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def find_source(self, source: str) -> ChatSession | None:
# L0032: Assigns or updates a value used later in the workflow; check mutability and data shape.
        session_id = self.source_to_session.get(source)
# L0033: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not session_id:
# L0034: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return None
# L0035: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.sessions.get(session_id)
# L0036: Blank line that visually separates logical sections and improves readability.

```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    metadata: dict = field(default_factory=dict)


class SessionMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Load a paper first.")
        return self.sessions[session_id]

    def put(self, session: ChatSession) -> None:
        self.sessions[session.session_id] = session
        self.source_to_session[session.source] = session.session_id

    def find_source(self, source: str) -> ChatSession | None:
        session_id = self.source_to_session.get(source)
        if not session_id:
            return None
        return self.sessions.get(session_id)
```
