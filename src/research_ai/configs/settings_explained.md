# settings.py Explained

Generated educational companion for `src/research_ai/configs/settings.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/configs/settings.py` is a Python module in the Configuration layer: typed settings backed by environment variables. It defines Paths, LLMSettings, RetrievalSettings, ExecutionSettings, Settings and load_settings.

## Why This File Exists

This file isolates one responsibility in the codebase: Configuration layer: typed settings backed by environment variables. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Configuration layer: typed settings backed by environment variables.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/configs/settings.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/configs/settings.py]
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
| `os` | os reads environment variables and process/runtime configuration. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |

## Global Variables and Config

No major module-level variables are declared. This reduces hidden state and keeps imports lightweight.

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

### `load_settings`

- **Line:** 78
- **Kind:** synchronous function
- **Arguments:** none
- **Docstring:** No explicit docstring; infer behavior from call sites and body.

```python
def load_settings() -> Settings:
    return Settings()
```

This function's parameters define its input contract. Its return value or side effect defines how downstream code uses it. Review error handling, resource usage, and whether the function performs CPU work, I/O, model inference, or pure transformation.


## Class-by-Class Breakdown

### `Paths`

- **Line:** 9
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `classifier_dir` at line 17: method behavior is described by its body and name
- `similarity_dir` at line 21: method behavior is described by its body and name
- `clustering_dir` at line 25: method behavior is described by its body and name
- `arxiv_shards` at line 29: method behavior is described by its body and name

```python
class Paths:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "data")))
    artifacts_root: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "artifacts")))
    frontend_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "frontend"
    )

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_root / "classification"

    @property
    def similarity_dir(self) -> Path:
        return self.artifacts_root / "similarity"

    @property
    def clustering_dir(self) -> Path:
        return self.artifacts_root / "clustering"

    @property
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `LLMSettings`

- **Line:** 34
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class LLMSettings:
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "cloud").strip().lower())
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `RetrievalSettings`

- **Line:** 51
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class RetrievalSettings:
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    default_top_k: int = 5
    max_top_k: int = 20
    keyword_weight: float = 0.25
    semantic_weight: float = 0.75
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `ExecutionSettings`

- **Line:** 62
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class ExecutionSettings:
    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PYTHON_EXECUTION", "false").lower() == "true"
    )
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_TIMEOUT", "5")))
    max_code_chars: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_MAX_CHARS", "4000")))
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `Settings`

- **Line:** 71
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- No methods beyond inherited behavior.

```python
class Settings:
    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `Paths` Methods

#### `Paths.classifier_dir`

- **Line:** 17
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def classifier_dir(self) -> Path:
        return self.artifacts_root / "classification"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `Paths.similarity_dir`

- **Line:** 21
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def similarity_dir(self) -> Path:
        return self.artifacts_root / "similarity"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `Paths.clustering_dir`

- **Line:** 25
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def clustering_dir(self) -> Path:
        return self.artifacts_root / "clustering"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `Paths.arxiv_shards`

- **Line:** 29
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.
- **Parquet**: Parquet is a compressed columnar format that is efficient for large metadata because readers can scan selected columns.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `dataclasses` | dataclasses reduce boilerplate for typed configuration/result containers. |
| `os` | os reads environment variables and process/runtime configuration. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.
- **Parquet**: Parquet is a compressed columnar format that is efficient for large metadata because readers can scan selected columns.

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
- Touches files or paths. Validate filenames, restrict upload size/type, and prevent traversal.
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

- `src/research_ai/configs/settings.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/configs/settings.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Imports a dependency, type, or project module needed by later code in this file.
import os
# L0004: Imports a dependency, type, or project module needed by later code in this file.
from dataclasses import dataclass, field
# L0005: Imports a dependency, type, or project module needed by later code in this file.
from pathlib import Path
# L0006: Blank line that visually separates logical sections and improves readability.

# L0007: Blank line that visually separates logical sections and improves readability.

# L0008: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass(frozen=True)
# L0009: Defines a class that groups related state and behavior behind a reusable interface.
class Paths:
# L0010: Assigns or updates a value used later in the workflow; check mutability and data shape.
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "data")))
# L0011: Assigns or updates a value used later in the workflow; check mutability and data shape.
    artifacts_root: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "artifacts")))
# L0012: Assigns or updates a value used later in the workflow; check mutability and data shape.
    frontend_dir: Path = field(
# L0013: Assigns or updates a value used later in the workflow; check mutability and data shape.
        default_factory=lambda: Path(__file__).resolve().parents[3] / "frontend"
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0015: Blank line that visually separates logical sections and improves readability.

# L0016: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0017: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def classifier_dir(self) -> Path:
# L0018: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.artifacts_root / "classification"
# L0019: Blank line that visually separates logical sections and improves readability.

# L0020: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0021: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def similarity_dir(self) -> Path:
# L0022: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.artifacts_root / "similarity"
# L0023: Blank line that visually separates logical sections and improves readability.

# L0024: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0025: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def clustering_dir(self) -> Path:
# L0026: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.artifacts_root / "clustering"
# L0027: Blank line that visually separates logical sections and improves readability.

# L0028: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0029: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def arxiv_shards(self) -> list[Path]:
# L0030: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))
# L0031: Blank line that visually separates logical sections and improves readability.

# L0032: Blank line that visually separates logical sections and improves readability.

# L0033: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass(frozen=True)
# L0034: Defines a class that groups related state and behavior behind a reusable interface.
class LLMSettings:
# L0035: Assigns or updates a value used later in the workflow; check mutability and data shape.
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "cloud").strip().lower())
# L0036: Assigns or updates a value used later in the workflow; check mutability and data shape.
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower())
# L0037: Assigns or updates a value used later in the workflow; check mutability and data shape.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
# L0038: Assigns or updates a value used later in the workflow; check mutability and data shape.
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
# L0039: Assigns or updates a value used later in the workflow; check mutability and data shape.
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
# L0040: Assigns or updates a value used later in the workflow; check mutability and data shape.
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
# L0041: Assigns or updates a value used later in the workflow; check mutability and data shape.
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"))
# L0042: Assigns or updates a value used later in the workflow; check mutability and data shape.
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
# L0044: Assigns or updates a value used later in the workflow; check mutability and data shape.
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
# L0048: Blank line that visually separates logical sections and improves readability.

# L0049: Blank line that visually separates logical sections and improves readability.

# L0050: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass(frozen=True)
# L0051: Defines a class that groups related state and behavior behind a reusable interface.
class RetrievalSettings:
# L0052: Assigns or updates a value used later in the workflow; check mutability and data shape.
    embedding_model_name: str = field(
# L0053: Assigns or updates a value used later in the workflow; check mutability and data shape.
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# L0054: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0055: Assigns or updates a value used later in the workflow; check mutability and data shape.
    default_top_k: int = 5
# L0056: Assigns or updates a value used later in the workflow; check mutability and data shape.
    max_top_k: int = 20
# L0057: Assigns or updates a value used later in the workflow; check mutability and data shape.
    keyword_weight: float = 0.25
# L0058: Assigns or updates a value used later in the workflow; check mutability and data shape.
    semantic_weight: float = 0.75
# L0059: Blank line that visually separates logical sections and improves readability.

# L0060: Blank line that visually separates logical sections and improves readability.

# L0061: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass(frozen=True)
# L0062: Defines a class that groups related state and behavior behind a reusable interface.
class ExecutionSettings:
# L0063: Assigns or updates a value used later in the workflow; check mutability and data shape.
    enabled: bool = field(
# L0064: Assigns or updates a value used later in the workflow; check mutability and data shape.
        default_factory=lambda: os.getenv("ENABLE_PYTHON_EXECUTION", "false").lower() == "true"
# L0065: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_TIMEOUT", "5")))
# L0067: Assigns or updates a value used later in the workflow; check mutability and data shape.
    max_code_chars: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_MAX_CHARS", "4000")))
# L0068: Blank line that visually separates logical sections and improves readability.

# L0069: Blank line that visually separates logical sections and improves readability.

# L0070: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
@dataclass(frozen=True)
# L0071: Defines a class that groups related state and behavior behind a reusable interface.
class Settings:
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
    paths: Paths = field(default_factory=Paths)
# L0073: Assigns or updates a value used later in the workflow; check mutability and data shape.
    llm: LLMSettings = field(default_factory=LLMSettings)
# L0074: Assigns or updates a value used later in the workflow; check mutability and data shape.
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
# L0075: Assigns or updates a value used later in the workflow; check mutability and data shape.
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
# L0076: Blank line that visually separates logical sections and improves readability.

# L0077: Blank line that visually separates logical sections and improves readability.

# L0078: Defines a function or method; parameters are the input contract and the body implements the workflow.
def load_settings() -> Settings:
# L0079: Returns the computed result to the caller; this shape becomes part of the downstream contract.
    return Settings()
# L0080: Blank line that visually separates logical sections and improves readability.

```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "data")))
    artifacts_root: Path = field(default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "artifacts")))
    frontend_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "frontend"
    )

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_root / "classification"

    @property
    def similarity_dir(self) -> Path:
        return self.artifacts_root / "similarity"

    @property
    def clustering_dir(self) -> Path:
        return self.artifacts_root / "clustering"

    @property
    def arxiv_shards(self) -> list[Path]:
        return sorted((self.data_root / "arxiv_chunks").glob("*.parquet"))


@dataclass(frozen=True)
class LLMSettings:
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "cloud").strip().lower())
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))


@dataclass(frozen=True)
class RetrievalSettings:
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    default_top_k: int = 5
    max_top_k: int = 20
    keyword_weight: float = 0.25
    semantic_weight: float = 0.75


@dataclass(frozen=True)
class ExecutionSettings:
    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PYTHON_EXECUTION", "false").lower() == "true"
    )
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_TIMEOUT", "5")))
    max_code_chars: int = field(default_factory=lambda: int(os.getenv("PYTHON_EXEC_MAX_CHARS", "4000")))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)


def load_settings() -> Settings:
    return Settings()
```
