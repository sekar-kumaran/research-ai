# service.py Explained

Generated educational companion for `src/research_ai/agents/retrieval_agent/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/agents/retrieval_agent/service.py` is a Python module in the Retrieval-agent layer: chooses retrieval strategies. It defines RetrievalStrategy, RetrievalAgent and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Retrieval-agent layer: chooses retrieval strategies. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Retrieval-agent layer: chooses retrieval strategies.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/agents/retrieval_agent/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/agents/retrieval_agent/service.py]
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

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 7 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `RetrievalStrategy`

- **Line:** 11
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class RetrievalStrategy:
    mode: str          # "semantic" | "hybrid" | "filtered" | "citation_aware"
    query: str
    top_k: int
    filters: dict
    expand_query: bool
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `RetrievalAgent`

- **Line:** 19
- **Base classes:** `object`
- **Docstring:** Intelligent retrieval specialist that selects strategy based on query analysis.

Sits between the Orchestrator and the HybridSearchService. Analyses the
query to determine whether to use pure-semantic, hybrid, or metadata-filtered
retrieval, and whether the query should be expanded with domain terminology
before embedding. Falls back to hybrid search on any error.

**Methods:**
- `__init__` at line 38: method behavior is described by its body and name
- `retrieve` at line 41: Execute the optimal retrieval strategy for the given query.
- `_select_strategy` at line 75: method behavior is described by its body and name
- `_expand_query` at line 113: method behavior is described by its body and name

```python
class RetrievalAgent:
    """Intelligent retrieval specialist that selects strategy based on query analysis.

    Sits between the Orchestrator and the HybridSearchService. Analyses the
    query to determine whether to use pure-semantic, hybrid, or metadata-filtered
    retrieval, and whether the query should be expanded with domain terminology
    before embedding. Falls back to hybrid search on any error.
    """

    # Domain expansions for sparse queries
    DOMAIN_EXPANSIONS: dict[str, list[str]] = {
        "llm": ["large language model", "transformer", "gpt", "attention"],
        "rl": ["reinforcement learning", "reward", "policy", "agent"],
        "cv": ["computer vision", "image", "convolutional", "object detection"],
        "nlp": ["natural language processing", "text", "bert", "language model"],
        "gnn": ["graph neural network", "node embedding", "message passing"],
        "diffusion": ["diffusion model", "denoising", "score matching", "generative"],
    }

    def __init__(self, search_service) -> None:
        self.search_service = search_service

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        strategy_hint: str = "auto",
    ) -> dict:
        """Execute the optimal retrieval strategy for the given query."""
        if not self.search_service.ready:
            return {"error": "Search index not ready. Build similarity artifacts first."}

        strategy = self._select_strategy(query, top_k, filters or {}, strategy_hint)
        effective_query = self._expand_query(strategy)

        logger.info(
            "RetrievalAgent: mode=%s expand=%s top_k=%d query='%s'",
            strategy.mode,
            strategy.expand_query,
            strategy.top_k,
            effective_query[:60],
        )

        result = self.search_service.search(
            effective_query,
            top_k=strategy.top_k,
            filters=strategy.filters if strategy.mode == "filtered" else None,
            candidate_k=min(strategy.top_k * 6, 80),
        )
        result["retrieval_strategy"] = strategy.mode
        result["query_expanded"] = effective_query != query
        if strategy.expand_query:
            result["original_query"] = query
        return result

    def _select_strategy(
        self, query: str, top_k: int, filters: dict, hint: str
    ) -> RetrievalStrategy:
        q_lower = query.lower()
        words = q_lower.split()

        # Explicit filters present → metadata-filtered retrieval
        if filters:
            return RetrievalStrategy(
                mode="filtered",
                query=query,
                top_k=top_k,
                filters=filters,
                expand_query=False,
            )

        # Very short queries (1–2 words) → expand before embedding
        expand = len(words) <= 2 and any(w in self.DOMAIN_EXPANSIONS for w in words)

        # Citation-aware: user asks about references or related work
        if any(token in q_lower for token in ("cited by", "references", "related work", "influenced")):
            return RetrievalStrategy(
                mode="citation_aware",
                query=query,
                top_k=min(top_k * 2, 20),
                filters={},
                expand_query=expand,
            )

        # Default to hybrid
        return RetrievalStrategy(
            mode="hybrid",
            query=query,
            top_k=top_k,
            filters={},
            expand_query=expand,
        )

    def _expand_query(self, strategy: RetrievalStrategy) -> str:
        if not strategy.expand_query:
            return strategy.query
        words = strategy.query.lower().split()
        expansions: list[str] = []
        for word in words:
            expansions.extend(self.DOMAIN_EXPANSIONS.get(word, []))
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `RetrievalAgent` Methods

#### `RetrievalAgent.__init__`

- **Line:** 38
- **Kind:** synchronous method
- **Arguments:** self, search_service
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(self, search_service) -> None:
        self.search_service = search_service
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `RetrievalAgent.retrieve`

- **Line:** 41
- **Kind:** synchronous method
- **Arguments:** self, query, top_k, filters, strategy_hint
- **Docstring:** Execute the optimal retrieval strategy for the given query.

```python
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        strategy_hint: str = "auto",
    ) -> dict:
        """Execute the optimal retrieval strategy for the given query."""
        if not self.search_service.ready:
            return {"error": "Search index not ready. Build similarity artifacts first."}

        strategy = self._select_strategy(query, top_k, filters or {}, strategy_hint)
        effective_query = self._expand_query(strategy)

        logger.info(
            "RetrievalAgent: mode=%s expand=%s top_k=%d query='%s'",
            strategy.mode,
            strategy.expand_query,
            strategy.top_k,
            effective_query[:60],
        )

        result = self.search_service.search(
            effective_query,
            top_k=strategy.top_k,
            filters=strategy.filters if strategy.mode == "filtered" else None,
            candidate_k=min(strategy.top_k * 6, 80),
        )
        result["retrieval_strategy"] = strategy.mode
        result["query_expanded"] = effective_query != query
        if strategy.expand_query:
            result["original_query"] = query
        return result
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `RetrievalAgent._select_strategy`

- **Line:** 75
- **Kind:** synchronous method
- **Arguments:** self, query, top_k, filters, hint
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _select_strategy(
        self, query: str, top_k: int, filters: dict, hint: str
    ) -> RetrievalStrategy:
        q_lower = query.lower()
        words = q_lower.split()

        # Explicit filters present → metadata-filtered retrieval
        if filters:
            return RetrievalStrategy(
                mode="filtered",
                query=query,
                top_k=top_k,
                filters=filters,
                expand_query=False,
            )

        # Very short queries (1–2 words) → expand before embedding
        expand = len(words) <= 2 and any(w in self.DOMAIN_EXPANSIONS for w in words)

        # Citation-aware: user asks about references or related work
        if any(token in q_lower for token in ("cited by", "references", "related work", "influenced")):
            return RetrievalStrategy(
                mode="citation_aware",
                query=query,
                top_k=min(top_k * 2, 20),
                filters={},
                expand_query=expand,
            )

        # Default to hybrid
        return RetrievalStrategy(
            mode="hybrid",
            query=query,
            top_k=top_k,
            filters={},
            expand_query=expand,
        )
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `RetrievalAgent._expand_query`

- **Line:** 113
- **Kind:** synchronous method
- **Arguments:** self, strategy
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _expand_query(self, strategy: RetrievalStrategy) -> str:
        if not strategy.expand_query:
            return strategy.query
        words = strategy.query.lower().split()
        expansions: list[str] = []
        for word in words:
            expansions.extend(self.DOMAIN_EXPANSIONS.get(word, []))
        if expansions:
            return f"{strategy.query} {' '.join(expansions[:8])}"
        return strategy.query
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Sparse Matrices**: Sparse matrices store only non-zero features, which is essential for high-dimensional token vectors where almost all vocabulary terms are absent.
- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `logging` | logging provides structured operational visibility without using print statements. |

## ML Concepts Used

- **Sparse Matrices**: Sparse matrices store only non-zero features, which is essential for high-dimensional token vectors where almost all vocabulary terms are absent.
- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
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

- `src/research_ai/agents/retrieval_agent/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/agents/retrieval_agent/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Retrieval-specialized agent that selects and executes the optimal retrieval strategy."""
# L0002: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0003: Blank line that visually separates logical sections and improves readability.

# L0004: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0005: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass
# L0006: Blank line that visually separates logical sections and improves readability.

# L0007: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Blank line that visually separates logical sections and improves readability.

# L0010: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass
# L0011: Defines a class that groups related state and behavior behind a reusable interface.
class RetrievalStrategy:
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    mode: str          # "semantic" | "hybrid" | "filtered" | "citation_aware"
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    query: str
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    top_k: int
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    filters: dict
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    expand_query: bool
# L0017: Blank line that visually separates logical sections and improves readability.

# L0018: Blank line that visually separates logical sections and improves readability.

# L0019: Defines a class that groups related state and behavior behind a reusable interface.
class RetrievalAgent:
# L0020: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Intelligent retrieval specialist that selects strategy based on query analysis.
# L0021: Blank line that visually separates logical sections and improves readability.

# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Sits between the Orchestrator and the HybridSearchService. Analyses the
# L0023: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    query to determine whether to use pure-semantic, hybrid, or metadata-filtered
# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    retrieval, and whether the query should be expanded with domain terminology
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    before embedding. Falls back to hybrid search on any error.
# L0026: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0027: Blank line that visually separates logical sections and improves readability.

# L0028: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Domain expansions for sparse queries
# L0029: Assigns or updates a value used later in the workflow; check mutability and data shape.
    DOMAIN_EXPANSIONS: dict[str, list[str]] = {
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "llm": ["large language model", "transformer", "gpt", "attention"],
# L0031: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "rl": ["reinforcement learning", "reward", "policy", "agent"],
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "cv": ["computer vision", "image", "convolutional", "object detection"],
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "nlp": ["natural language processing", "text", "bert", "language model"],
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "gnn": ["graph neural network", "node embedding", "message passing"],
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "diffusion": ["diffusion model", "denoising", "score matching", "generative"],
# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    }
# L0037: Blank line that visually separates logical sections and improves readability.

# L0038: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self, search_service) -> None:
# L0039: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.search_service = search_service
# L0040: Blank line that visually separates logical sections and improves readability.

# L0041: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def retrieve(
# L0042: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        query: str,
# L0044: Assigns or updates a value used later in the workflow; check mutability and data shape.
        top_k: int = 5,
# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
        filters: dict | None = None,
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
        strategy_hint: str = "auto",
# L0047: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> dict:
# L0048: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Execute the optimal retrieval strategy for the given query."""
# L0049: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not self.search_service.ready:
# L0050: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": "Search index not ready. Build similarity artifacts first."}
# L0051: Blank line that visually separates logical sections and improves readability.

# L0052: Assigns or updates a value used later in the workflow; check mutability and data shape.
        strategy = self._select_strategy(query, top_k, filters or {}, strategy_hint)
# L0053: Assigns or updates a value used later in the workflow; check mutability and data shape.
        effective_query = self._expand_query(strategy)
# L0054: Blank line that visually separates logical sections and improves readability.

# L0055: Emits structured operational information for debugging, monitoring, or failure diagnosis.
        logger.info(
# L0056: Assigns or updates a value used later in the workflow; check mutability and data shape.
            "RetrievalAgent: mode=%s expand=%s top_k=%d query='%s'",
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            strategy.mode,
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            strategy.expand_query,
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            strategy.top_k,
# L0060: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            effective_query[:60],
# L0061: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0062: Blank line that visually separates logical sections and improves readability.

# L0063: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = self.search_service.search(
# L0064: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            effective_query,
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=strategy.top_k,
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
            filters=strategy.filters if strategy.mode == "filtered" else None,
# L0067: Assigns or updates a value used later in the workflow; check mutability and data shape.
            candidate_k=min(strategy.top_k * 6, 80),
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0069: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result["retrieval_strategy"] = strategy.mode
# L0070: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result["query_expanded"] = effective_query != query
# L0071: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if strategy.expand_query:
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
            result["original_query"] = query
# L0073: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return result
# L0074: Blank line that visually separates logical sections and improves readability.

# L0075: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _select_strategy(
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self, query: str, top_k: int, filters: dict, hint: str
# L0077: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> RetrievalStrategy:
# L0078: Assigns or updates a value used later in the workflow; check mutability and data shape.
        q_lower = query.lower()
# L0079: Assigns or updates a value used later in the workflow; check mutability and data shape.
        words = q_lower.split()
# L0080: Blank line that visually separates logical sections and improves readability.

# L0081: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Explicit filters present → metadata-filtered retrieval
# L0082: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if filters:
# L0083: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return RetrievalStrategy(
# L0084: Assigns or updates a value used later in the workflow; check mutability and data shape.
                mode="filtered",
# L0085: Assigns or updates a value used later in the workflow; check mutability and data shape.
                query=query,
# L0086: Assigns or updates a value used later in the workflow; check mutability and data shape.
                top_k=top_k,
# L0087: Assigns or updates a value used later in the workflow; check mutability and data shape.
                filters=filters,
# L0088: Assigns or updates a value used later in the workflow; check mutability and data shape.
                expand_query=False,
# L0089: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0090: Blank line that visually separates logical sections and improves readability.

# L0091: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Very short queries (1–2 words) → expand before embedding
# L0092: Assigns or updates a value used later in the workflow; check mutability and data shape.
        expand = len(words) <= 2 and any(w in self.DOMAIN_EXPANSIONS for w in words)
# L0093: Blank line that visually separates logical sections and improves readability.

# L0094: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Citation-aware: user asks about references or related work
# L0095: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if any(token in q_lower for token in ("cited by", "references", "related work", "influenced")):
# L0096: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return RetrievalStrategy(
# L0097: Assigns or updates a value used later in the workflow; check mutability and data shape.
                mode="citation_aware",
# L0098: Assigns or updates a value used later in the workflow; check mutability and data shape.
                query=query,
# L0099: Assigns or updates a value used later in the workflow; check mutability and data shape.
                top_k=min(top_k * 2, 20),
# L0100: Assigns or updates a value used later in the workflow; check mutability and data shape.
                filters={},
# L0101: Assigns or updates a value used later in the workflow; check mutability and data shape.
                expand_query=expand,
# L0102: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            )
# L0103: Blank line that visually separates logical sections and improves readability.

# L0104: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Default to hybrid
# L0105: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return RetrievalStrategy(
# L0106: Assigns or updates a value used later in the workflow; check mutability and data shape.
            mode="hybrid",
# L0107: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query=query,
# L0108: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=top_k,
# L0109: Assigns or updates a value used later in the workflow; check mutability and data shape.
            filters={},
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
            expand_query=expand,
# L0111: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0112: Blank line that visually separates logical sections and improves readability.

# L0113: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _expand_query(self, strategy: RetrievalStrategy) -> str:
# L0114: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not strategy.expand_query:
# L0115: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return strategy.query
# L0116: Assigns or updates a value used later in the workflow; check mutability and data shape.
        words = strategy.query.lower().split()
# L0117: Assigns or updates a value used later in the workflow; check mutability and data shape.
        expansions: list[str] = []
# L0118: Iterates over data, retry attempts, files, results, or workflow steps.
        for word in words:
# L0119: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            expansions.extend(self.DOMAIN_EXPANSIONS.get(word, []))
# L0120: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if expansions:
# L0121: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return f"{strategy.query} {' '.join(expansions[:8])}"
# L0122: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return strategy.query
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Retrieval-specialized agent that selects and executes the optimal retrieval strategy."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    mode: str          # "semantic" | "hybrid" | "filtered" | "citation_aware"
    query: str
    top_k: int
    filters: dict
    expand_query: bool


class RetrievalAgent:
    """Intelligent retrieval specialist that selects strategy based on query analysis.

    Sits between the Orchestrator and the HybridSearchService. Analyses the
    query to determine whether to use pure-semantic, hybrid, or metadata-filtered
    retrieval, and whether the query should be expanded with domain terminology
    before embedding. Falls back to hybrid search on any error.
    """

    # Domain expansions for sparse queries
    DOMAIN_EXPANSIONS: dict[str, list[str]] = {
        "llm": ["large language model", "transformer", "gpt", "attention"],
        "rl": ["reinforcement learning", "reward", "policy", "agent"],
        "cv": ["computer vision", "image", "convolutional", "object detection"],
        "nlp": ["natural language processing", "text", "bert", "language model"],
        "gnn": ["graph neural network", "node embedding", "message passing"],
        "diffusion": ["diffusion model", "denoising", "score matching", "generative"],
    }

    def __init__(self, search_service) -> None:
        self.search_service = search_service

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        strategy_hint: str = "auto",
    ) -> dict:
        """Execute the optimal retrieval strategy for the given query."""
        if not self.search_service.ready:
            return {"error": "Search index not ready. Build similarity artifacts first."}

        strategy = self._select_strategy(query, top_k, filters or {}, strategy_hint)
        effective_query = self._expand_query(strategy)

        logger.info(
            "RetrievalAgent: mode=%s expand=%s top_k=%d query='%s'",
            strategy.mode,
            strategy.expand_query,
            strategy.top_k,
            effective_query[:60],
        )

        result = self.search_service.search(
            effective_query,
            top_k=strategy.top_k,
            filters=strategy.filters if strategy.mode == "filtered" else None,
            candidate_k=min(strategy.top_k * 6, 80),
        )
        result["retrieval_strategy"] = strategy.mode
        result["query_expanded"] = effective_query != query
        if strategy.expand_query:
            result["original_query"] = query
        return result

    def _select_strategy(
        self, query: str, top_k: int, filters: dict, hint: str
    ) -> RetrievalStrategy:
        q_lower = query.lower()
        words = q_lower.split()

        # Explicit filters present → metadata-filtered retrieval
        if filters:
            return RetrievalStrategy(
                mode="filtered",
                query=query,
                top_k=top_k,
                filters=filters,
                expand_query=False,
            )

        # Very short queries (1–2 words) → expand before embedding
        expand = len(words) <= 2 and any(w in self.DOMAIN_EXPANSIONS for w in words)

        # Citation-aware: user asks about references or related work
        if any(token in q_lower for token in ("cited by", "references", "related work", "influenced")):
            return RetrievalStrategy(
                mode="citation_aware",
                query=query,
                top_k=min(top_k * 2, 20),
                filters={},
                expand_query=expand,
            )

        # Default to hybrid
        return RetrievalStrategy(
            mode="hybrid",
            query=query,
            top_k=top_k,
            filters={},
            expand_query=expand,
        )

    def _expand_query(self, strategy: RetrievalStrategy) -> str:
        if not strategy.expand_query:
            return strategy.query
        words = strategy.query.lower().split()
        expansions: list[str] = []
        for word in words:
            expansions.extend(self.DOMAIN_EXPANSIONS.get(word, []))
        if expansions:
            return f"{strategy.query} {' '.join(expansions[:8])}"
        return strategy.query
```
