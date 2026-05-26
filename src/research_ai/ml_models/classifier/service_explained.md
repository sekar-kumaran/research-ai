# service.py Explained

Generated educational companion for `src/research_ai/ml_models/classifier/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/ml_models/classifier/service.py` is a Python module in the ML services layer: classifiers, summarizers, similarity, ranking, and extraction. It defines ClassifierService and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: ML services layer: classifiers, summarizers, similarity, ranking, and extraction. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** ML services layer: classifiers, summarizers, similarity, ranking, and extraction.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/ml_models/classifier/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/ml_models/classifier/service.py]
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
| `joblib` | Joblib serializes Python and scikit-learn artifacts such as vectorizers, classifiers, and model metadata. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 10 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `ClassifierService`

- **Line:** 13
- **Base classes:** `object`
- **Docstring:** Local arXiv category classifier backed by trained sklearn artifacts.

**Methods:**
- `__init__` at line 16: method behavior is described by its body and name
- `from_artifacts` at line 27: method behavior is described by its body and name
- `ready` at line 31: method behavior is described by its body and name
- `_ensure_loaded` at line 41: method behavior is described by its body and name
- `classify` at line 54: method behavior is described by its body and name

```python
class ClassifierService:
    """Local arXiv category classifier backed by trained sklearn artifacts."""

    def __init__(
        self,
        classifier: object | None = None,
        vectorizer: object | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "ClassifierService":
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        if self.classifier is not None and self.vectorizer is not None:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "classifier.joblib").exists()
            and (self.artifact_dir / "tfidf_vectorizer.joblib").exists()
        )

    def _ensure_loaded(self) -> None:
        if self.classifier is not None and self.vectorizer is not None:
            return
        if self.artifact_dir is None:
            raise RuntimeError("Classifier artifact directory is not configured.")
        classifier_path = self.artifact_dir / "classifier.joblib"
        vectorizer_path = self.artifact_dir / "tfidf_vectorizer.joblib"
        if not classifier_path.exists() or not vectorizer_path.exists():
            raise RuntimeError("Classifier artifacts are missing.")
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Classifier artifacts loaded from %s", self.artifact_dir)

    def classify(self, title: str, abstract: str) -> dict:
        if not self.ready:
            return {"error": "Classifier not ready. Run the training pipeline first."}
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("Classifier load failed: %s", exc)
            return {"error": f"Classifier load failed: {exc}"}
        text = clean_text(build_full_text(title, abstract))
        if not text:
            return {"error": "No classifiable text provided."}
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        try:
            proba = self.classifier.predict_proba(x)[0]
            classes = self.classifier.classes_
            confidence = {
                str(label): round(float(score), 4)
                for label, score in sorted(zip(classes, proba), key=lambda item: -item[1])[:5]
            }
        except Exception:
            confidence = {}
        return {"predicted_category": str(pred), "confidence": confidence}
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `ClassifierService` Methods

#### `ClassifierService.__init__`

- **Line:** 16
- **Kind:** synchronous method
- **Arguments:** self, classifier, vectorizer, artifact_dir
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(
        self,
        classifier: object | None = None,
        vectorizer: object | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.artifact_dir = artifact_dir
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ClassifierService.from_artifacts`

- **Line:** 27
- **Kind:** synchronous method
- **Arguments:** cls, artifact_dir
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def from_artifacts(cls, artifact_dir: Path) -> "ClassifierService":
        return cls(artifact_dir=artifact_dir)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ClassifierService.ready`

- **Line:** 31
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def ready(self) -> bool:
        if self.classifier is not None and self.vectorizer is not None:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "classifier.joblib").exists()
            and (self.artifact_dir / "tfidf_vectorizer.joblib").exists()
        )
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ClassifierService._ensure_loaded`

- **Line:** 41
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _ensure_loaded(self) -> None:
        if self.classifier is not None and self.vectorizer is not None:
            return
        if self.artifact_dir is None:
            raise RuntimeError("Classifier artifact directory is not configured.")
        classifier_path = self.artifact_dir / "classifier.joblib"
        vectorizer_path = self.artifact_dir / "tfidf_vectorizer.joblib"
        if not classifier_path.exists() or not vectorizer_path.exists():
            raise RuntimeError("Classifier artifacts are missing.")
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Classifier artifacts loaded from %s", self.artifact_dir)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `ClassifierService.classify`

- **Line:** 54
- **Kind:** synchronous method
- **Arguments:** self, title, abstract
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def classify(self, title: str, abstract: str) -> dict:
        if not self.ready:
            return {"error": "Classifier not ready. Run the training pipeline first."}
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("Classifier load failed: %s", exc)
            return {"error": f"Classifier load failed: {exc}"}
        text = clean_text(build_full_text(title, abstract))
        if not text:
            return {"error": "No classifiable text provided."}
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        try:
            proba = self.classifier.predict_proba(x)[0]
            classes = self.classifier.classes_
            confidence = {
                str(label): round(float(score), 4)
                for label, score in sorted(zip(classes, proba), key=lambda item: -item[1])[:5]
            }
        except Exception:
            confidence = {}
        return {"predicted_category": str(pred), "confidence": confidence}
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **TF-IDF**: TF-IDF weights terms by local frequency and global rarity. It is cheap, sparse, interpretable, and strong for lexical scientific categories.
- **Sparse Matrices**: Sparse matrices store only non-zero features, which is essential for high-dimensional token vectors where almost all vocabulary terms are absent.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `joblib` | Joblib serializes Python and scikit-learn artifacts such as vectorizers, classifiers, and model metadata. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `pathlib` | pathlib gives object-oriented paths and reduces path-concatenation bugs across local and cloud deployments. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **TF-IDF**: TF-IDF weights terms by local frequency and global rarity. It is cheap, sparse, interpretable, and strong for lexical scientific categories.
- **Sparse Matrices**: Sparse matrices store only non-zero features, which is essential for high-dimensional token vectors where almost all vocabulary terms are absent.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.
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

- Touches files or paths. Validate filenames, restrict upload size/type, and prevent traversal.

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

- `src/research_ai/ml_models/classifier/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/ml_models/classifier/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0004: Imports a dependency, type, or project module needed by later code in this file.
from pathlib import Path
# L0005: Blank line that visually separates logical sections and improves readability.

# L0006: Imports a dependency, type, or project module needed by later code in this file.
import joblib
# L0007: Blank line that visually separates logical sections and improves readability.

# L0008: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.common.text import build_full_text, clean_text
# L0009: Blank line that visually separates logical sections and improves readability.

# L0010: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0011: Blank line that visually separates logical sections and improves readability.

# L0012: Blank line that visually separates logical sections and improves readability.

# L0013: Defines a class that groups related state and behavior behind a reusable interface.
class ClassifierService:
# L0014: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Local arXiv category classifier backed by trained sklearn artifacts."""
# L0015: Blank line that visually separates logical sections and improves readability.

# L0016: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0018: Assigns or updates a value used later in the workflow; check mutability and data shape.
        classifier: object | None = None,
# L0019: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vectorizer: object | None = None,
# L0020: Assigns or updates a value used later in the workflow; check mutability and data shape.
        artifact_dir: Path | None = None,
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> None:
# L0022: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.classifier = classifier
# L0023: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.vectorizer = vectorizer
# L0024: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.artifact_dir = artifact_dir
# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @classmethod
# L0027: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def from_artifacts(cls, artifact_dir: Path) -> "ClassifierService":
# L0028: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return cls(artifact_dir=artifact_dir)
# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0031: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def ready(self) -> bool:
# L0032: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.classifier is not None and self.vectorizer is not None:
# L0033: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return True
# L0034: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.artifact_dir is None:
# L0035: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return False
# L0036: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return (
# L0037: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            (self.artifact_dir / "classifier.joblib").exists()
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            and (self.artifact_dir / "tfidf_vectorizer.joblib").exists()
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0040: Blank line that visually separates logical sections and improves readability.

# L0041: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _ensure_loaded(self) -> None:
# L0042: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.classifier is not None and self.vectorizer is not None:
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            return
# L0044: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.artifact_dir is None:
# L0045: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError("Classifier artifact directory is not configured.")
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
        classifier_path = self.artifact_dir / "classifier.joblib"
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vectorizer_path = self.artifact_dir / "tfidf_vectorizer.joblib"
# L0048: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not classifier_path.exists() or not vectorizer_path.exists():
# L0049: Raises an explicit error when the function cannot safely continue.
            raise RuntimeError("Classifier artifacts are missing.")
# L0050: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.classifier = joblib.load(classifier_path)
# L0051: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.vectorizer = joblib.load(vectorizer_path)
# L0052: Emits structured operational information for debugging, monitoring, or failure diagnosis.
        logger.info("Classifier artifacts loaded from %s", self.artifact_dir)
# L0053: Blank line that visually separates logical sections and improves readability.

# L0054: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def classify(self, title: str, abstract: str) -> dict:
# L0055: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not self.ready:
# L0056: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": "Classifier not ready. Run the training pipeline first."}
# L0057: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            self._ensure_loaded()
# L0059: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception as exc:
# L0060: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.warning("Classifier load failed: %s", exc)
# L0061: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": f"Classifier load failed: {exc}"}
# L0062: Assigns or updates a value used later in the workflow; check mutability and data shape.
        text = clean_text(build_full_text(title, abstract))
# L0063: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not text:
# L0064: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"error": "No classifiable text provided."}
# L0065: Assigns or updates a value used later in the workflow; check mutability and data shape.
        x = self.vectorizer.transform([text])
# L0066: Assigns or updates a value used later in the workflow; check mutability and data shape.
        pred = self.classifier.predict(x)[0]
# L0067: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0068: Assigns or updates a value used later in the workflow; check mutability and data shape.
            proba = self.classifier.predict_proba(x)[0]
# L0069: Assigns or updates a value used later in the workflow; check mutability and data shape.
            classes = self.classifier.classes_
# L0070: Assigns or updates a value used later in the workflow; check mutability and data shape.
            confidence = {
# L0071: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                str(label): round(float(score), 4)
# L0072: Iterates over data, retry attempts, files, results, or workflow steps.
                for label, score in sorted(zip(classes, proba), key=lambda item: -item[1])[:5]
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0074: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception:
# L0075: Assigns or updates a value used later in the workflow; check mutability and data shape.
            confidence = {}
# L0076: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {"predicted_category": str(pred), "confidence": confidence}
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from research_ai.common.text import build_full_text, clean_text

logger = logging.getLogger(__name__)


class ClassifierService:
    """Local arXiv category classifier backed by trained sklearn artifacts."""

    def __init__(
        self,
        classifier: object | None = None,
        vectorizer: object | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.artifact_dir = artifact_dir

    @classmethod
    def from_artifacts(cls, artifact_dir: Path) -> "ClassifierService":
        return cls(artifact_dir=artifact_dir)

    @property
    def ready(self) -> bool:
        if self.classifier is not None and self.vectorizer is not None:
            return True
        if self.artifact_dir is None:
            return False
        return (
            (self.artifact_dir / "classifier.joblib").exists()
            and (self.artifact_dir / "tfidf_vectorizer.joblib").exists()
        )

    def _ensure_loaded(self) -> None:
        if self.classifier is not None and self.vectorizer is not None:
            return
        if self.artifact_dir is None:
            raise RuntimeError("Classifier artifact directory is not configured.")
        classifier_path = self.artifact_dir / "classifier.joblib"
        vectorizer_path = self.artifact_dir / "tfidf_vectorizer.joblib"
        if not classifier_path.exists() or not vectorizer_path.exists():
            raise RuntimeError("Classifier artifacts are missing.")
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Classifier artifacts loaded from %s", self.artifact_dir)

    def classify(self, title: str, abstract: str) -> dict:
        if not self.ready:
            return {"error": "Classifier not ready. Run the training pipeline first."}
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("Classifier load failed: %s", exc)
            return {"error": f"Classifier load failed: {exc}"}
        text = clean_text(build_full_text(title, abstract))
        if not text:
            return {"error": "No classifiable text provided."}
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        try:
            proba = self.classifier.predict_proba(x)[0]
            classes = self.classifier.classes_
            confidence = {
                str(label): round(float(score), 4)
                for label, score in sorted(zip(classes, proba), key=lambda item: -item[1])[:5]
            }
        except Exception:
            confidence = {}
        return {"predicted_category": str(pred), "confidence": confidence}
```
