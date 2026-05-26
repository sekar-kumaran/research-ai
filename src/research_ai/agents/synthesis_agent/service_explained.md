# service.py Explained

Generated educational companion for `src/research_ai/agents/synthesis_agent/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/agents/synthesis_agent/service.py` is a Python module in the Synthesis layer: turns tool outputs into grounded answers. It defines SynthesisAgent and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Synthesis layer: turns tool outputs into grounded answers. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Synthesis layer: turns tool outputs into grounded answers.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/agents/synthesis_agent/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/agents/synthesis_agent/service.py]
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
| `json` | json serializes/deserializes API payloads, LLM planning output, and artifact metadata. |
| `logging` | logging provides structured operational visibility without using print statements. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 43 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |
| `SYSTEM_PROMPT` | 45 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `SynthesisAgent`

- **Line:** 63
- **Base classes:** `object`
- **Docstring:** LLM-powered synthesis over structured tool outputs.

Returns a dict with:
  - answer          : The final conversational response text
  - sources         : List of paper dicts cited in the answer
  - confidence      : 0-1 score from the evaluator (or derived from search results)
  - tools_used      : List of tool names that produced non-empty outputs
  - model_used      : The LLM model name that generated the synthesis (if any)

**Methods:**
- `__init__` at line 74: method behavior is described by its body and name
- `synthesize` at line 77: Legacy single-string interface — calls synthesize_structured internally.
- `synthesize_structured` at line 82: Produce a fully structured response dict from tool outputs.

Returns:
    {
        "answer":      str,           # The narrative answer text
        "sources":     list[dict],    # Retrieved papers cited
        "confidence":  float,         # 0–1 evidence quality score
        "tools_used":  list[str],     # Non-empty tool outputs
        "model_used":  str,           # LLM model that synthesized
    }
- `_extract_sources` at line 160: Extract retrieved papers from search tool outputs.

Papers come from hybrid_search or smart_retrieve. We deduplicate by
paper_id and cap at 8 sources to avoid overwhelming the UI.
- `_derive_confidence` at line 209: Derive a 0–1 confidence score for the response.

Priority:
  1. Use the EvaluatorAgent's quality_score directly (most accurate)
  2. Heuristic: score based on number of sources + LLM answer presence
- `_cloud` at line 251: method behavior is described by its body and name
- `_build_context` at line 260: Build a compact JSON context string, prioritising the most useful keys.
- `_structured_direct_answer` at line 306: Best-effort readable answer without LLM — used as fallback.

```python
class SynthesisAgent:
    """LLM-powered synthesis over structured tool outputs.

    Returns a dict with:
      - answer          : The final conversational response text
      - sources         : List of paper dicts cited in the answer
      - confidence      : 0-1 score from the evaluator (or derived from search results)
      - tools_used      : List of tool names that produced non-empty outputs
      - model_used      : The LLM model name that generated the synthesis (if any)
    """

    def __init__(self, cloud_factory=None) -> None:
        self.cloud_factory = cloud_factory

    def synthesize(self, query: str, plan: dict, outputs: dict) -> str:
        """Legacy single-string interface — calls synthesize_structured internally."""
        result = self.synthesize_structured(query, plan, outputs)
        return result["answer"]

    def synthesize_structured(
        self,
        query: str,
        plan: dict,
        outputs: dict,
        quality_score: float | None = None,
    ) -> dict:
        """Produce a fully structured response dict from tool outputs.

        Returns:
            {
                "answer":      str,           # The narrative answer text
                "sources":     list[dict],    # Retrieved papers cited
                "confidence":  float,         # 0–1 evidence quality score
                "tools_used":  list[str],     # Non-empty tool outputs
                "model_used":  str,           # LLM model that synthesized
            }
        """
        # Extract sources from search outputs (always, regardless of LLM availability)
        sources = self._extract_sources(outputs)

        # Determine which tools actually returned useful output
        tools_used = [
            k for k, v in outputs.items()
            if isinstance(v, dict) and not v.get("error") and k != "_retry"
        ]

        # Derive confidence from quality_score or fallback heuristic
        confidence = self._derive_confidence(quality_score, sources, outputs)

        # Try LLM synthesis first; fall back to structured text if unavailable
        direct = self._structured_direct_answer(outputs)
        cloud = self._cloud()
        model_used = ""

        if cloud is None:
            return {
                "answer": direct,
                "sources": sources,
                "confidence": confidence,
                "tools_used": tools_used,
                "model_used": model_used,
            }

        context = self._build_context(outputs)
        prompt = (
            f"User query: {query}\n\n"
            f"Planned intent: {plan.get('intent', 'research_analysis')}\n\n"
            f"Tool outputs:\n{context}\n\n"
            "Write the final answer now."
        )

        # Local Ollama models are slower — cap tokens to keep latency under ~60s
        is_ollama = getattr(cloud, "provider", "") == "ollama"
        max_tok = 300 if is_ollama else 800
        model_used = getattr(cloud, "model", "")

        try:
            answer = cloud.generate(prompt, max_tokens=max_tok, system=SYSTEM_PROMPT).strip()
            if len(answer.split()) < 10:
                answer = direct  # LLM returned garbage — use structured fallback
        except Exception as exc:
            logger.warning("SynthesisAgent LLM call failed: %s", exc)
            answer = direct

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "tools_used": tools_used,
            "model_used": model_used,
        }

    # ------------------------------------------------------------------
    # Source extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(outputs: dict) -> list[dict]:
        """Extract retrieved papers from search tool outputs.

        Papers come from hybrid_search or smart_retrieve. We deduplicate by
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `SynthesisAgent` Methods

#### `SynthesisAgent.__init__`

- **Line:** 74
- **Kind:** synchronous method
- **Arguments:** self, cloud_factory
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(self, cloud_factory=None) -> None:
        self.cloud_factory = cloud_factory
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent.synthesize`

- **Line:** 77
- **Kind:** synchronous method
- **Arguments:** self, query, plan, outputs
- **Docstring:** Legacy single-string interface — calls synthesize_structured internally.

```python
    def synthesize(self, query: str, plan: dict, outputs: dict) -> str:
        """Legacy single-string interface — calls synthesize_structured internally."""
        result = self.synthesize_structured(query, plan, outputs)
        return result["answer"]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent.synthesize_structured`

- **Line:** 82
- **Kind:** synchronous method
- **Arguments:** self, query, plan, outputs, quality_score
- **Docstring:** Produce a fully structured response dict from tool outputs.

Returns:
    {
        "answer":      str,           # The narrative answer text
        "sources":     list[dict],    # Retrieved papers cited
        "confidence":  float,         # 0–1 evidence quality score
        "tools_used":  list[str],     # Non-empty tool outputs
        "model_used":  str,           # LLM model that synthesized
    }

```python
    def synthesize_structured(
        self,
        query: str,
        plan: dict,
        outputs: dict,
        quality_score: float | None = None,
    ) -> dict:
        """Produce a fully structured response dict from tool outputs.

        Returns:
            {
                "answer":      str,           # The narrative answer text
                "sources":     list[dict],    # Retrieved papers cited
                "confidence":  float,         # 0–1 evidence quality score
                "tools_used":  list[str],     # Non-empty tool outputs
                "model_used":  str,           # LLM model that synthesized
            }
        """
        # Extract sources from search outputs (always, regardless of LLM availability)
        sources = self._extract_sources(outputs)

        # Determine which tools actually returned useful output
        tools_used = [
            k for k, v in outputs.items()
            if isinstance(v, dict) and not v.get("error") and k != "_retry"
        ]

        # Derive confidence from quality_score or fallback heuristic
        confidence = self._derive_confidence(quality_score, sources, outputs)

        # Try LLM synthesis first; fall back to structured text if unavailable
        direct = self._structured_direct_answer(outputs)
        cloud = self._cloud()
        model_used = ""

        if cloud is None:
            return {
                "answer": direct,
                "sources": sources,
                "confidence": confidence,
                "tools_used": tools_used,
                "model_used": model_used,
            }

        context = self._build_context(outputs)
        prompt = (
            f"User query: {query}\n\n"
            f"Planned intent: {plan.get('intent', 'research_analysis')}\n\n"
            f"Tool outputs:\n{context}\n\n"
            "Write the final answer now."
        )

        # Local Ollama models are slower — cap tokens to keep latency under ~60s
        is_ollama = getattr(cloud, "provider", "") == "ollama"
        max_tok = 300 if is_ollama else 800
        model_used = getattr(cloud, "model", "")

        try:
            answer = cloud.generate(prompt, max_tokens=max_tok, system=SYSTEM_PROMPT).strip()
            if len(answer.split()) < 10:
                answer = direct  # LLM returned garbage — use structured fallback
        except Exception as exc:
            logger.warning("SynthesisAgent LLM call failed: %s", exc)
            answer = direct

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "tools_used": tools_used,
            "model_used": model_used,
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent._extract_sources`

- **Line:** 160
- **Kind:** synchronous method
- **Arguments:** outputs
- **Docstring:** Extract retrieved papers from search tool outputs.

Papers come from hybrid_search or smart_retrieve. We deduplicate by
paper_id and cap at 8 sources to avoid overwhelming the UI.

```python
    def _extract_sources(outputs: dict) -> list[dict]:
        """Extract retrieved papers from search tool outputs.

        Papers come from hybrid_search or smart_retrieve. We deduplicate by
        paper_id and cap at 8 sources to avoid overwhelming the UI.
        """
        seen_ids: set[str] = set()
        sources: list[dict] = []

        for key in ("hybrid_search", "smart_retrieve", "metadata_rag"):
            val = outputs.get(key)
            if not isinstance(val, dict):
                continue

            # metadata_rag stores its results under "retrieved"
            results_key = "retrieved" if key == "metadata_rag" else "results"
            papers = val.get(results_key, [])

            for p in papers[:8]:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("paper_id", "")).strip()
                if pid and pid in seen_ids:
                    continue
                if pid:
                    seen_ids.add(pid)

                abstract = str(p.get("abstract", ""))
                snippet = abstract[:250] + ("…" if len(abstract) > 250 else "")
                sources.append({
                    "title":            p.get("title", "Untitled"),
                    "paper_id":         pid,
                    "year":             str(p.get("year", "")),
                    "category":         str(p.get("category", "")),
                    "abstract_snippet": snippet,
                    "score":            round(float(p.get("score", 0.0)), 4),
                    "arxiv_url":        f"https://arxiv.org/abs/{pid}" if pid else "",
                })

            if len(sources) >= 8:
                break

        return sources
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent._derive_confidence`

- **Line:** 209
- **Kind:** synchronous method
- **Arguments:** quality_score, sources, outputs
- **Docstring:** Derive a 0–1 confidence score for the response.

Priority:
  1. Use the EvaluatorAgent's quality_score directly (most accurate)
  2. Heuristic: score based on number of sources + LLM answer presence

```python
    def _derive_confidence(
        quality_score: float | None,
        sources: list[dict],
        outputs: dict,
    ) -> float:
        """Derive a 0–1 confidence score for the response.

        Priority:
          1. Use the EvaluatorAgent's quality_score directly (most accurate)
          2. Heuristic: score based on number of sources + LLM answer presence
        """
        if quality_score is not None:
            return round(max(0.0, min(1.0, float(quality_score))), 3)

        # Heuristic scoring when quality_score is unavailable
        score = 0.0

        # Source quality: up to 0.50 based on retrieved paper count and scores
        if sources:
            avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
            source_factor = min(len(sources) / 5.0, 1.0)   # saturates at 5 sources
            score += 0.50 * source_factor * min(avg_score * 2, 1.0)

        # LLM answer present: +0.30
        for key in ("metadata_rag", "paper_chat"):
            val = outputs.get(key, {})
            if isinstance(val, dict) and isinstance(val.get("answer"), str):
                if len(val["answer"].split()) >= 20:
                    score += 0.30
                    break

        # No errors: +0.20
        errors = sum(1 for v in outputs.values() if isinstance(v, dict) and v.get("error"))
        if errors == 0:
            score += 0.20

        return round(min(score, 1.0), 3)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent._cloud`

- **Line:** 251
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _cloud(self):
        if self.cloud_factory is None:
            return None
        try:
            return self.cloud_factory()
        except Exception:
            return None
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent._build_context`

- **Line:** 260
- **Kind:** synchronous method
- **Arguments:** outputs
- **Docstring:** Build a compact JSON context string, prioritising the most useful keys.

```python
    def _build_context(outputs: dict) -> str:
        """Build a compact JSON context string, prioritising the most useful keys."""
        priority_keys = (
            "hybrid_search", "smart_retrieve", "metadata_rag",
            "methodology_extract", "trend_analysis", "citation_signals",
            "citation_proxy", "classify_query", "metadata_analyse",
            "summarize", "paper_chat",
        )
        selected: dict = {}
        total_chars = 0
        limit = 10_000  # Stay well within LLM context window

        for key in priority_keys:
            if key not in outputs:
                continue
            val = outputs[key]
            if isinstance(val, dict) and val.get("error"):
                continue

            # Compact search results — include only what the LLM needs to cite
            if key in ("hybrid_search", "smart_retrieve") and isinstance(val, dict):
                compact_results = [
                    {
                        "title":            p.get("title", ""),
                        "year":             p.get("year", ""),
                        "category":         p.get("category", ""),
                        "abstract_snippet": str(p.get("abstract", ""))[:300],
                        "paper_id":         p.get("paper_id", ""),
                    }
                    for p in val.get("results", [])[:8]
                ]
                val = {
                    "count":              val.get("count", 0),
                    "results":            compact_results,
                    "retrieval_strategy": val.get("retrieval_strategy", ""),
                }

            chunk = json.dumps({key: val}, ensure_ascii=False)
            if total_chars + len(chunk) > limit:
                break
            selected[key] = val
            total_chars += len(chunk)

        return json.dumps(selected, ensure_ascii=False, indent=2)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `SynthesisAgent._structured_direct_answer`

- **Line:** 306
- **Kind:** synchronous method
- **Arguments:** outputs
- **Docstring:** Best-effort readable answer without LLM — used as fallback.

```python
    def _structured_direct_answer(outputs: dict) -> str:
        """Best-effort readable answer without LLM — used as fallback."""
        # Conversation tool
        conv = outputs.get("conversation")
        if isinstance(conv, dict) and isinstance(conv.get("answer"), str):
            return conv["answer"]

        # LLM-backed tools (RAG, paper chat, summarize)
        for key in ("metadata_rag", "paper_chat", "summarize"):
            val = outputs.get(key, {})
            if isinstance(val, dict):
                text = val.get("answer") or val.get("summary") or val.get("final_answer")
                if isinstance(text, str) and text.strip():
                    return text

        # Search results — build a numbered list
        for key in ("hybrid_search", "smart_retrieve"):
            val = outputs.get(key, {})
            if isinstance(val, dict) and val.get("results"):
                lines = [f"Found {val.get('count', 0)} relevant papers:"]
                for i, p in enumerate(val["results"][:6], 1):
                    pid = p.get("paper_id", "")
                    link = f" — arxiv.org/abs/{pid}" if pid else ""
                    lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
                    if p.get("abstract"):
                        lines.append(f"   {str(p['abstract'])[:200]}…")
                return "\n".join(lines)

        # Classification only
        clf = outputs.get("classify_query", {})
        if isinstance(clf, dict) and clf.get("predicted_category"):
            return f"Predicted arXiv category: {clf['predicted_category']}"

        # Error fallback
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
        if errors:
            return f"Could not complete the request: {errors[0]}"

        return "No results found. Please check that the paper index has been built and try a different query."
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `json` | json serializes/deserializes API payloads, LLM planning output, and artifact metadata. |
| `logging` | logging provides structured operational visibility without using print statements. |

## ML Concepts Used

- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.

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

- `src/research_ai/agents/synthesis_agent/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `src/research_ai/agents/synthesis_agent/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""SynthesisAgent — turns grounded tool outputs into a coherent researcher-facing answer.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
WHAT THIS DOES
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
--------------
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
The SynthesisAgent is the final stage of the orchestration loop. It receives
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
structured outputs from all the ML/retrieval tools (FAISS search results,
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
methodology extraction, trend analysis, citations, etc.) and produces:
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  1. A natural-language answer grounded entirely in retrieved evidence
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  2. A list of source papers cited in the answer (for the UI to render)
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  3. A confidence score reflecting how well-evidenced the answer is
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  4. A list of which tools were actually invoked
# L0013: Blank line that visually separates logical sections and improves readability.

# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
HALLUCINATION PREVENTION
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------------
# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
The system prompt explicitly forbids inventing paper titles, authors, or
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
results. The LLM is given only the tool outputs as its "world knowledge"
# L0018: Iterates over data, retry attempts, files, results, or workflow steps.
for this request. Any claim it makes must come from the structured context.
# L0019: Blank line that visually separates logical sections and improves readability.

# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
If the LLM is unavailable (no API key, Ollama not running), the agent falls
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
back to a structured text summary built directly from the tool outputs — no
# L0022: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
hallucination risk since it's a formatted dump of real data.
# L0023: Blank line that visually separates logical sections and improves readability.

# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
CONFIDENCE SCORING
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Confidence is derived from the EvaluatorAgent's quality_score (0–1).
# L0027: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
The EvaluatorAgent scores on 4 dimensions: retrieval hit-rate (0.40),
# L0028: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
answer completeness (0.30), evidence grounding (0.20), error absence (0.10).
# L0029: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
We expose this score directly in the response so the UI can show users a
# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
visual indicator of how well-evidenced the answer is.
# L0031: Blank line that visually separates logical sections and improves readability.

# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
STRUCTURED SOURCES
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
After synthesis, we extract the cited papers from the tool outputs and
# L0035: Returns the computed result to the caller; this shape becomes part of the downstream contract.
return them as a structured list. The UI renders these as clickable
# L0036: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
"source cards" below the answer — like Perplexity's source footnotes.
# L0037: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0038: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0039: Blank line that visually separates logical sections and improves readability.

# L0040: Imports a dependency, type, or project module needed by later code in this file.
import json
# L0041: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0042: Blank line that visually separates logical sections and improves readability.

# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0044: Blank line that visually separates logical sections and improves readability.

# L0045: Assigns or updates a value used later in the workflow; check mutability and data shape.
SYSTEM_PROMPT = (
# L0046: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "You are the synthesis agent for an AI Research Intelligence Platform. "
# L0047: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "You receive structured outputs from scientific ML and retrieval tools and must write "
# L0048: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "a clear, accurate, well-structured answer for a researcher.\n\n"
# L0049: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "RULES:\n"
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- Ground every claim in the provided tool outputs. Do NOT invent paper titles, authors, or results.\n"
# L0051: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- Cite papers by title and year when available. Use [1], [2], ... notation.\n"
# L0052: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- If methodology was extracted, include the key methods/datasets found.\n"
# L0053: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- If trend analysis was run, summarise the year distribution briefly.\n"
# L0054: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- If citation signals were found, mention related areas or influential papers.\n"
# L0055: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- If no relevant papers were found, say so clearly — do not fabricate.\n"
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- Be concise but substantive. Aim for 150–400 words unless a longer answer is clearly needed.\n"
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- Start with a direct answer to the query, then provide supporting evidence.\n"
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- Write conversationally — you are a helpful research assistant, not a search engine.\n"
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    "- End with a brief 'Sources' list using the [N] notation you used in the body."
# L0060: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
)
# L0061: Blank line that visually separates logical sections and improves readability.

# L0062: Blank line that visually separates logical sections and improves readability.

# L0063: Defines a class that groups related state and behavior behind a reusable interface.
class SynthesisAgent:
# L0064: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """LLM-powered synthesis over structured tool outputs.
# L0065: Blank line that visually separates logical sections and improves readability.

# L0066: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    Returns a dict with:
# L0067: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      - answer          : The final conversational response text
# L0068: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      - sources         : List of paper dicts cited in the answer
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      - confidence      : 0-1 score from the evaluator (or derived from search results)
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      - tools_used      : List of tool names that produced non-empty outputs
# L0071: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
      - model_used      : The LLM model name that generated the synthesis (if any)
# L0072: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0073: Blank line that visually separates logical sections and improves readability.

# L0074: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(self, cloud_factory=None) -> None:
# L0075: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.cloud_factory = cloud_factory
# L0076: Blank line that visually separates logical sections and improves readability.

# L0077: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def synthesize(self, query: str, plan: dict, outputs: dict) -> str:
# L0078: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Legacy single-string interface — calls synthesize_structured internally."""
# L0079: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = self.synthesize_structured(query, plan, outputs)
# L0080: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return result["answer"]
# L0081: Blank line that visually separates logical sections and improves readability.

# L0082: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def synthesize_structured(
# L0083: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0084: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        query: str,
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        plan: dict,
# L0086: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        outputs: dict,
# L0087: Assigns or updates a value used later in the workflow; check mutability and data shape.
        quality_score: float | None = None,
# L0088: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> dict:
# L0089: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Produce a fully structured response dict from tool outputs.
# L0090: Blank line that visually separates logical sections and improves readability.

# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Returns:
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            {
# L0093: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "answer":      str,           # The narrative answer text
# L0094: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "sources":     list[dict],    # Retrieved papers cited
# L0095: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "confidence":  float,         # 0–1 evidence quality score
# L0096: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "tools_used":  list[str],     # Non-empty tool outputs
# L0097: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "model_used":  str,           # LLM model that synthesized
# L0098: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0099: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0100: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Extract sources from search outputs (always, regardless of LLM availability)
# L0101: Assigns or updates a value used later in the workflow; check mutability and data shape.
        sources = self._extract_sources(outputs)
# L0102: Blank line that visually separates logical sections and improves readability.

# L0103: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Determine which tools actually returned useful output
# L0104: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tools_used = [
# L0105: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            k for k, v in outputs.items()
# L0106: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(v, dict) and not v.get("error") and k != "_retry"
# L0107: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0108: Blank line that visually separates logical sections and improves readability.

# L0109: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Derive confidence from quality_score or fallback heuristic
# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
        confidence = self._derive_confidence(quality_score, sources, outputs)
# L0111: Blank line that visually separates logical sections and improves readability.

# L0112: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Try LLM synthesis first; fall back to structured text if unavailable
# L0113: Assigns or updates a value used later in the workflow; check mutability and data shape.
        direct = self._structured_direct_answer(outputs)
# L0114: Assigns or updates a value used later in the workflow; check mutability and data shape.
        cloud = self._cloud()
# L0115: Assigns or updates a value used later in the workflow; check mutability and data shape.
        model_used = ""
# L0116: Blank line that visually separates logical sections and improves readability.

# L0117: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if cloud is None:
# L0118: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {
# L0119: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "answer": direct,
# L0120: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "sources": sources,
# L0121: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "confidence": confidence,
# L0122: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "tools_used": tools_used,
# L0123: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "model_used": model_used,
# L0124: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0125: Blank line that visually separates logical sections and improves readability.

# L0126: Assigns or updates a value used later in the workflow; check mutability and data shape.
        context = self._build_context(outputs)
# L0127: Assigns or updates a value used later in the workflow; check mutability and data shape.
        prompt = (
# L0128: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            f"User query: {query}\n\n"
# L0129: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            f"Planned intent: {plan.get('intent', 'research_analysis')}\n\n"
# L0130: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            f"Tool outputs:\n{context}\n\n"
# L0131: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "Write the final answer now."
# L0132: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0133: Blank line that visually separates logical sections and improves readability.

# L0134: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Local Ollama models are slower — cap tokens to keep latency under ~60s
# L0135: Assigns or updates a value used later in the workflow; check mutability and data shape.
        is_ollama = getattr(cloud, "provider", "") == "ollama"
# L0136: Assigns or updates a value used later in the workflow; check mutability and data shape.
        max_tok = 300 if is_ollama else 800
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
        model_used = getattr(cloud, "model", "")
# L0138: Blank line that visually separates logical sections and improves readability.

# L0139: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
            answer = cloud.generate(prompt, max_tokens=max_tok, system=SYSTEM_PROMPT).strip()
# L0141: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if len(answer.split()) < 10:
# L0142: Assigns or updates a value used later in the workflow; check mutability and data shape.
                answer = direct  # LLM returned garbage — use structured fallback
# L0143: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception as exc:
# L0144: Emits structured operational information for debugging, monitoring, or failure diagnosis.
            logger.warning("SynthesisAgent LLM call failed: %s", exc)
# L0145: Assigns or updates a value used later in the workflow; check mutability and data shape.
            answer = direct
# L0146: Blank line that visually separates logical sections and improves readability.

# L0147: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0148: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "answer": answer,
# L0149: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "sources": sources,
# L0150: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "confidence": confidence,
# L0151: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "tools_used": tools_used,
# L0152: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "model_used": model_used,
# L0153: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0154: Blank line that visually separates logical sections and improves readability.

# L0155: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0156: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Source extraction
# L0157: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0158: Blank line that visually separates logical sections and improves readability.

# L0159: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0160: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _extract_sources(outputs: dict) -> list[dict]:
# L0161: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Extract retrieved papers from search tool outputs.
# L0162: Blank line that visually separates logical sections and improves readability.

# L0163: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Papers come from hybrid_search or smart_retrieve. We deduplicate by
# L0164: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        paper_id and cap at 8 sources to avoid overwhelming the UI.
# L0165: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0166: Assigns or updates a value used later in the workflow; check mutability and data shape.
        seen_ids: set[str] = set()
# L0167: Assigns or updates a value used later in the workflow; check mutability and data shape.
        sources: list[dict] = []
# L0168: Blank line that visually separates logical sections and improves readability.

# L0169: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in ("hybrid_search", "smart_retrieve", "metadata_rag"):
# L0170: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs.get(key)
# L0171: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if not isinstance(val, dict):
# L0172: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0173: Blank line that visually separates logical sections and improves readability.

# L0174: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # metadata_rag stores its results under "retrieved"
# L0175: Assigns or updates a value used later in the workflow; check mutability and data shape.
            results_key = "retrieved" if key == "metadata_rag" else "results"
# L0176: Assigns or updates a value used later in the workflow; check mutability and data shape.
            papers = val.get(results_key, [])
# L0177: Blank line that visually separates logical sections and improves readability.

# L0178: Iterates over data, retry attempts, files, results, or workflow steps.
            for p in papers[:8]:
# L0179: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if not isinstance(p, dict):
# L0180: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    continue
# L0181: Assigns or updates a value used later in the workflow; check mutability and data shape.
                pid = str(p.get("paper_id", "")).strip()
# L0182: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if pid and pid in seen_ids:
# L0183: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    continue
# L0184: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if pid:
# L0185: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    seen_ids.add(pid)
# L0186: Blank line that visually separates logical sections and improves readability.

# L0187: Assigns or updates a value used later in the workflow; check mutability and data shape.
                abstract = str(p.get("abstract", ""))
# L0188: Assigns or updates a value used later in the workflow; check mutability and data shape.
                snippet = abstract[:250] + ("…" if len(abstract) > 250 else "")
# L0189: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                sources.append({
# L0190: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "title":            p.get("title", "Untitled"),
# L0191: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "paper_id":         pid,
# L0192: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "year":             str(p.get("year", "")),
# L0193: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "category":         str(p.get("category", "")),
# L0194: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "abstract_snippet": snippet,
# L0195: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "score":            round(float(p.get("score", 0.0)), 4),
# L0196: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "arxiv_url":        f"https://arxiv.org/abs/{pid}" if pid else "",
# L0197: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                })
# L0198: Blank line that visually separates logical sections and improves readability.

# L0199: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if len(sources) >= 8:
# L0200: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                break
# L0201: Blank line that visually separates logical sections and improves readability.

# L0202: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return sources
# L0203: Blank line that visually separates logical sections and improves readability.

# L0204: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0205: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Confidence scoring
# L0206: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0207: Blank line that visually separates logical sections and improves readability.

# L0208: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0209: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _derive_confidence(
# L0210: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        quality_score: float | None,
# L0211: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        sources: list[dict],
# L0212: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        outputs: dict,
# L0213: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> float:
# L0214: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Derive a 0–1 confidence score for the response.
# L0215: Blank line that visually separates logical sections and improves readability.

# L0216: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Priority:
# L0217: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          1. Use the EvaluatorAgent's quality_score directly (most accurate)
# L0218: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          2. Heuristic: score based on number of sources + LLM answer presence
# L0219: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0220: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if quality_score is not None:
# L0221: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return round(max(0.0, min(1.0, float(quality_score))), 3)
# L0222: Blank line that visually separates logical sections and improves readability.

# L0223: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Heuristic scoring when quality_score is unavailable
# L0224: Assigns or updates a value used later in the workflow; check mutability and data shape.
        score = 0.0
# L0225: Blank line that visually separates logical sections and improves readability.

# L0226: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Source quality: up to 0.50 based on retrieved paper count and scores
# L0227: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if sources:
# L0228: Assigns or updates a value used later in the workflow; check mutability and data shape.
            avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
# L0229: Assigns or updates a value used later in the workflow; check mutability and data shape.
            source_factor = min(len(sources) / 5.0, 1.0)   # saturates at 5 sources
# L0230: Assigns or updates a value used later in the workflow; check mutability and data shape.
            score += 0.50 * source_factor * min(avg_score * 2, 1.0)
# L0231: Blank line that visually separates logical sections and improves readability.

# L0232: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # LLM answer present: +0.30
# L0233: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in ("metadata_rag", "paper_chat"):
# L0234: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs.get(key, {})
# L0235: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(val, dict) and isinstance(val.get("answer"), str):
# L0236: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if len(val["answer"].split()) >= 20:
# L0237: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    score += 0.30
# L0238: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    break
# L0239: Blank line that visually separates logical sections and improves readability.

# L0240: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # No errors: +0.20
# L0241: Assigns or updates a value used later in the workflow; check mutability and data shape.
        errors = sum(1 for v in outputs.values() if isinstance(v, dict) and v.get("error"))
# L0242: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if errors == 0:
# L0243: Assigns or updates a value used later in the workflow; check mutability and data shape.
            score += 0.20
# L0244: Blank line that visually separates logical sections and improves readability.

# L0245: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return round(min(score, 1.0), 3)
# L0246: Blank line that visually separates logical sections and improves readability.

# L0247: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0248: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # Private helpers
# L0249: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
    # ------------------------------------------------------------------
# L0250: Blank line that visually separates logical sections and improves readability.

# L0251: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _cloud(self):
# L0252: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.cloud_factory is None:
# L0253: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return None
# L0254: Begins protected execution so failures can be handled without crashing the whole request path.
        try:
# L0255: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return self.cloud_factory()
# L0256: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
        except Exception:
# L0257: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return None
# L0258: Blank line that visually separates logical sections and improves readability.

# L0259: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0260: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _build_context(outputs: dict) -> str:
# L0261: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Build a compact JSON context string, prioritising the most useful keys."""
# L0262: Assigns or updates a value used later in the workflow; check mutability and data shape.
        priority_keys = (
# L0263: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search", "smart_retrieve", "metadata_rag",
# L0264: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "methodology_extract", "trend_analysis", "citation_signals",
# L0265: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "citation_proxy", "classify_query", "metadata_analyse",
# L0266: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "summarize", "paper_chat",
# L0267: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0268: Assigns or updates a value used later in the workflow; check mutability and data shape.
        selected: dict = {}
# L0269: Assigns or updates a value used later in the workflow; check mutability and data shape.
        total_chars = 0
# L0270: Assigns or updates a value used later in the workflow; check mutability and data shape.
        limit = 10_000  # Stay well within LLM context window
# L0271: Blank line that visually separates logical sections and improves readability.

# L0272: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in priority_keys:
# L0273: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if key not in outputs:
# L0274: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0275: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs[key]
# L0276: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(val, dict) and val.get("error"):
# L0277: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0278: Blank line that visually separates logical sections and improves readability.

# L0279: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
            # Compact search results — include only what the LLM needs to cite
# L0280: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if key in ("hybrid_search", "smart_retrieve") and isinstance(val, dict):
# L0281: Assigns or updates a value used later in the workflow; check mutability and data shape.
                compact_results = [
# L0282: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    {
# L0283: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "title":            p.get("title", ""),
# L0284: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "year":             p.get("year", ""),
# L0285: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "category":         p.get("category", ""),
# L0286: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "abstract_snippet": str(p.get("abstract", ""))[:300],
# L0287: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "paper_id":         p.get("paper_id", ""),
# L0288: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    }
# L0289: Iterates over data, retry attempts, files, results, or workflow steps.
                    for p in val.get("results", [])[:8]
# L0290: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                ]
# L0291: Assigns or updates a value used later in the workflow; check mutability and data shape.
                val = {
# L0292: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "count":              val.get("count", 0),
# L0293: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "results":            compact_results,
# L0294: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "retrieval_strategy": val.get("retrieval_strategy", ""),
# L0295: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                }
# L0296: Blank line that visually separates logical sections and improves readability.

# L0297: Assigns or updates a value used later in the workflow; check mutability and data shape.
            chunk = json.dumps({key: val}, ensure_ascii=False)
# L0298: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if total_chars + len(chunk) > limit:
# L0299: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                break
# L0300: Assigns or updates a value used later in the workflow; check mutability and data shape.
            selected[key] = val
# L0301: Assigns or updates a value used later in the workflow; check mutability and data shape.
            total_chars += len(chunk)
# L0302: Blank line that visually separates logical sections and improves readability.

# L0303: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return json.dumps(selected, ensure_ascii=False, indent=2)
# L0304: Blank line that visually separates logical sections and improves readability.

# L0305: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0306: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _structured_direct_answer(outputs: dict) -> str:
# L0307: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Best-effort readable answer without LLM — used as fallback."""
# L0308: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Conversation tool
# L0309: Assigns or updates a value used later in the workflow; check mutability and data shape.
        conv = outputs.get("conversation")
# L0310: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if isinstance(conv, dict) and isinstance(conv.get("answer"), str):
# L0311: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return conv["answer"]
# L0312: Blank line that visually separates logical sections and improves readability.

# L0313: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # LLM-backed tools (RAG, paper chat, summarize)
# L0314: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in ("metadata_rag", "paper_chat", "summarize"):
# L0315: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs.get(key, {})
# L0316: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(val, dict):
# L0317: Assigns or updates a value used later in the workflow; check mutability and data shape.
                text = val.get("answer") or val.get("summary") or val.get("final_answer")
# L0318: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if isinstance(text, str) and text.strip():
# L0319: Returns the computed result to the caller; this shape becomes part of the downstream contract.
                    return text
# L0320: Blank line that visually separates logical sections and improves readability.

# L0321: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Search results — build a numbered list
# L0322: Iterates over data, retry attempts, files, results, or workflow steps.
        for key in ("hybrid_search", "smart_retrieve"):
# L0323: Assigns or updates a value used later in the workflow; check mutability and data shape.
            val = outputs.get(key, {})
# L0324: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if isinstance(val, dict) and val.get("results"):
# L0325: Assigns or updates a value used later in the workflow; check mutability and data shape.
                lines = [f"Found {val.get('count', 0)} relevant papers:"]
# L0326: Iterates over data, retry attempts, files, results, or workflow steps.
                for i, p in enumerate(val["results"][:6], 1):
# L0327: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    pid = p.get("paper_id", "")
# L0328: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    link = f" — arxiv.org/abs/{pid}" if pid else ""
# L0329: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
# L0330: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                    if p.get("abstract"):
# L0331: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        lines.append(f"   {str(p['abstract'])[:200]}…")
# L0332: Returns the computed result to the caller; this shape becomes part of the downstream contract.
                return "\n".join(lines)
# L0333: Blank line that visually separates logical sections and improves readability.

# L0334: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Classification only
# L0335: Assigns or updates a value used later in the workflow; check mutability and data shape.
        clf = outputs.get("classify_query", {})
# L0336: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if isinstance(clf, dict) and clf.get("predicted_category"):
# L0337: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return f"Predicted arXiv category: {clf['predicted_category']}"
# L0338: Blank line that visually separates logical sections and improves readability.

# L0339: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Error fallback
# L0340: Assigns or updates a value used later in the workflow; check mutability and data shape.
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
# L0341: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if errors:
# L0342: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return f"Could not complete the request: {errors[0]}"
# L0343: Blank line that visually separates logical sections and improves readability.

# L0344: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return "No results found. Please check that the paper index has been built and try a different query."
```

## Source Walkthrough

This file is large, so the opening and closing sections are included here. Use the class/function breakdown above to navigate the middle of the file.

### Opening Section

```python
"""SynthesisAgent — turns grounded tool outputs into a coherent researcher-facing answer.

WHAT THIS DOES
--------------
The SynthesisAgent is the final stage of the orchestration loop. It receives
structured outputs from all the ML/retrieval tools (FAISS search results,
methodology extraction, trend analysis, citations, etc.) and produces:

  1. A natural-language answer grounded entirely in retrieved evidence
  2. A list of source papers cited in the answer (for the UI to render)
  3. A confidence score reflecting how well-evidenced the answer is
  4. A list of which tools were actually invoked

HALLUCINATION PREVENTION
------------------------
The system prompt explicitly forbids inventing paper titles, authors, or
results. The LLM is given only the tool outputs as its "world knowledge"
for this request. Any claim it makes must come from the structured context.

If the LLM is unavailable (no API key, Ollama not running), the agent falls
back to a structured text summary built directly from the tool outputs — no
hallucination risk since it's a formatted dump of real data.

CONFIDENCE SCORING
------------------
Confidence is derived from the EvaluatorAgent's quality_score (0–1).
The EvaluatorAgent scores on 4 dimensions: retrieval hit-rate (0.40),
answer completeness (0.30), evidence grounding (0.20), error absence (0.10).
We expose this score directly in the response so the UI can show users a
visual indicator of how well-evidenced the answer is.

STRUCTURED SOURCES
------------------
After synthesis, we extract the cited papers from the tool outputs and
return them as a structured list. The UI renders these as clickable
"source cards" below the answer — like Perplexity's source footnotes.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the synthesis agent for an AI Research Intelligence Platform. "
    "You receive structured outputs from scientific ML and retrieval tools and must write "
    "a clear, accurate, well-structured answer for a researcher.\n\n"
    "RULES:\n"
    "- Ground every claim in the provided tool outputs. Do NOT invent paper titles, authors, or results.\n"
    "- Cite papers by title and year when available. Use [1], [2], ... notation.\n"
    "- If methodology was extracted, include the key methods/datasets found.\n"
    "- If trend analysis was run, summarise the year distribution briefly.\n"
    "- If citation signals were found, mention related areas or influential papers.\n"
    "- If no relevant papers were found, say so clearly — do not fabricate.\n"
    "- Be concise but substantive. Aim for 150–400 words unless a longer answer is clearly needed.\n"
    "- Start with a direct answer to the query, then provide supporting evidence.\n"
    "- Write conversationally — you are a helpful research assistant, not a search engine.\n"
    "- End with a brief 'Sources' list using the [N] notation you used in the body."
)


class SynthesisAgent:
    """LLM-powered synthesis over structured tool outputs.

    Returns a dict with:
      - answer          : The final conversational response text
      - sources         : List of paper dicts cited in the answer
      - confidence      : 0-1 score from the evaluator (or derived from search results)
      - tools_used      : List of tool names that produced non-empty outputs
      - model_used      : The LLM model name that generated the synthesis (if any)
    """

    def __init__(self, cloud_factory=None) -> None:
        self.cloud_factory = cloud_factory

    def synthesize(self, query: str, plan: dict, outputs: dict) -> str:
        """Legacy single-string interface — calls synthesize_structured internally."""
        result = self.synthesize_structured(query, plan, outputs)
        return result["answer"]

    def synthesize_structured(
        self,
        query: str,
        plan: dict,
        outputs: dict,
        quality_score: float | None = None,
    ) -> dict:
        """Produce a fully structured response dict from tool outputs.

        Returns:
            {
                "answer":      str,           # The narrative answer text
                "sources":     list[dict],    # Retrieved papers cited
                "confidence":  float,         # 0–1 evidence quality score
                "tools_used":  list[str],     # Non-empty tool outputs
                "model_used":  str,           # LLM model that synthesized
            }
        """
        # Extract sources from search outputs (always, regardless of LLM availability)
        sources = self._extract_sources(outputs)

        # Determine which tools actually returned useful output
        tools_used = [
            k for k, v in outputs.items()
            if isinstance(v, dict) and not v.get("error") and k != "_retry"
        ]

        # Derive confidence from quality_score or fallback heuristic
        confidence = self._derive_confidence(quality_score, sources, outputs)

        # Try LLM synthesis first; fall back to structured text if unavailable
        direct = self._structured_direct_answer(outputs)
        cloud = self._cloud()
        model_used = ""

        if cloud is None:
            return {
                "answer": direct,
                "sources": sources,
```

### Closing Section

```python
            "citation_proxy", "classify_query", "metadata_analyse",
            "summarize", "paper_chat",
        )
        selected: dict = {}
        total_chars = 0
        limit = 10_000  # Stay well within LLM context window

        for key in priority_keys:
            if key not in outputs:
                continue
            val = outputs[key]
            if isinstance(val, dict) and val.get("error"):
                continue

            # Compact search results — include only what the LLM needs to cite
            if key in ("hybrid_search", "smart_retrieve") and isinstance(val, dict):
                compact_results = [
                    {
                        "title":            p.get("title", ""),
                        "year":             p.get("year", ""),
                        "category":         p.get("category", ""),
                        "abstract_snippet": str(p.get("abstract", ""))[:300],
                        "paper_id":         p.get("paper_id", ""),
                    }
                    for p in val.get("results", [])[:8]
                ]
                val = {
                    "count":              val.get("count", 0),
                    "results":            compact_results,
                    "retrieval_strategy": val.get("retrieval_strategy", ""),
                }

            chunk = json.dumps({key: val}, ensure_ascii=False)
            if total_chars + len(chunk) > limit:
                break
            selected[key] = val
            total_chars += len(chunk)

        return json.dumps(selected, ensure_ascii=False, indent=2)

    @staticmethod
    def _structured_direct_answer(outputs: dict) -> str:
        """Best-effort readable answer without LLM — used as fallback."""
        # Conversation tool
        conv = outputs.get("conversation")
        if isinstance(conv, dict) and isinstance(conv.get("answer"), str):
            return conv["answer"]

        # LLM-backed tools (RAG, paper chat, summarize)
        for key in ("metadata_rag", "paper_chat", "summarize"):
            val = outputs.get(key, {})
            if isinstance(val, dict):
                text = val.get("answer") or val.get("summary") or val.get("final_answer")
                if isinstance(text, str) and text.strip():
                    return text

        # Search results — build a numbered list
        for key in ("hybrid_search", "smart_retrieve"):
            val = outputs.get(key, {})
            if isinstance(val, dict) and val.get("results"):
                lines = [f"Found {val.get('count', 0)} relevant papers:"]
                for i, p in enumerate(val["results"][:6], 1):
                    pid = p.get("paper_id", "")
                    link = f" — arxiv.org/abs/{pid}" if pid else ""
                    lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '')}){link}")
                    if p.get("abstract"):
                        lines.append(f"   {str(p['abstract'])[:200]}…")
                return "\n".join(lines)

        # Classification only
        clf = outputs.get("classify_query", {})
        if isinstance(clf, dict) and clf.get("predicted_category"):
            return f"Predicted arXiv category: {clf['predicted_category']}"

        # Error fallback
        errors = [v["error"] for v in outputs.values() if isinstance(v, dict) and v.get("error")]
        if errors:
            return f"Could not complete the request: {errors[0]}"

        return "No results found. Please check that the paper index has been built and try a different query."
```
