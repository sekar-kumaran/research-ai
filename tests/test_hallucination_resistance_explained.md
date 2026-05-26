# test_hallucination_resistance.py Explained

Generated educational companion for `tests/test_hallucination_resistance.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`tests/test_hallucination_resistance.py` is a Python module in the Test layer: behavioral, safety, performance, and integration checks. It defines TestSynthesisGrounding, TestMetadataRAGGrounding, TestEvidenceGroundedContext and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Test layer: behavioral, safety, performance, and integration checks. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Test layer: behavioral, safety, performance, and integration checks.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `tests/test_hallucination_resistance.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[tests/test_hallucination_resistance.py]
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
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

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

### `TestSynthesisGrounding`

- **Line:** 20
- **Base classes:** `object`
- **Docstring:** Verify SynthesisAgent falls back correctly when context is absent.

**Methods:**
- `_synthesizer` at line 23: method behavior is described by its body and name
- `test_no_cloud_uses_structured_fallback` at line 33: Without a cloud factory, the structured direct answer is always used.
- `test_cloud_too_short_falls_back_to_structured` at line 51: If the LLM returns < 10 words, the structured fallback is used.
- `test_empty_outputs_returns_no_results_message` at line 66: method behavior is described by its body and name
- `test_system_prompt_enforces_grounding` at line 71: Verify the system prompt contains the no-fabrication instruction.
- `test_conversation_bypass_returns_static_answer` at line 77: The conversation tool returns a static answer without retrieval.

```python
class TestSynthesisGrounding:
    """Verify SynthesisAgent falls back correctly when context is absent."""

    def _synthesizer(self, cloud_response=None):
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        if cloud_response is None:
            return SynthesisAgent(cloud_factory=None)

        mock_cloud = MagicMock()
        mock_cloud.generate.return_value = cloud_response
        return SynthesisAgent(cloud_factory=lambda: mock_cloud)

    def test_no_cloud_uses_structured_fallback(self):
        """Without a cloud factory, the structured direct answer is always used."""
        synthesizer = self._synthesizer(cloud_response=None)
        outputs = {
            "hybrid_search": {
                "count": 2,
                "results": [
                    {"title": "Attention Is All You Need", "year": "2017",
                     "abstract": "We propose a model based on attention.", "paper_id": "1706.03762"},
                    {"title": "BERT: Pre-training", "year": "2018",
                     "abstract": "BERT is a language model.", "paper_id": "1810.04805"},
                ],
            }
        }
        plan = {"intent": "research_analysis", "query": "attention transformers"}
        answer = synthesizer.synthesize("attention transformers", plan, outputs)
        assert "Attention Is All You Need" in answer or "2017" in answer or "found" in answer.lower()

    def test_cloud_too_short_falls_back_to_structured(self):
        """If the LLM returns < 10 words, the structured fallback is used."""
        synthesizer = self._synthesizer(cloud_response="Yes.")  # < 10 words
        outputs = {
            "hybrid_search": {
                "count": 1,
                "results": [{"title": "Some Paper", "year": "2022",
                             "abstract": "We study X.", "paper_id": "2201.0001"}],
            }
        }
        plan = {"intent": "research_analysis", "query": "transformers"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        # Should NOT return "Yes." — should use structured fallback
        assert answer != "Yes."

    def test_empty_outputs_returns_no_results_message(self):
        synthesizer = self._synthesizer()
        answer = synthesizer.synthesize("obscure topic", {"intent": "search"}, {})
        assert "No results" in answer or len(answer) > 0

    def test_system_prompt_enforces_grounding(self):
        """Verify the system prompt contains the no-fabrication instruction."""
        from research_ai.agents.synthesis_agent.service import SYSTEM_PROMPT
        assert "Do NOT invent" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()
        assert "fabricate" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()

    def test_conversation_bypass_returns_static_answer(self):
        """The conversation tool returns a static answer without retrieval."""
        synthesizer = self._synthesizer()
        outputs = {
            "conversation": {"answer": "Hello! I am your research assistant.", "query": "hi"}
        }
        plan = {"intent": "conversation", "query": "hello"}
        answer = synthesizer.synthesize("hello", plan, outputs)
        assert "Hello" in answer or "assistant" in answer.lower()
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestMetadataRAGGrounding`

- **Line:** 88
- **Base classes:** `object`
- **Docstring:** Verify _metadata_rag system prompt enforces grounding.

NOTE: These tests read the source of platform.py via inspect.getsource()
rather than importing the module, because importing platform.py triggers
a chain: platform → paper_ingestion → faiss, which may fail on environments
with a numpy/faiss ABI mismatch.  Reading source avoids that chain.

**Methods:**
- `_get_metadata_rag_source` at line 97: Read _metadata_rag source without importing the module.
- `test_rag_system_prompt_restricts_fabrication` at line 110: The RAG system prompt should explicitly restrict LLM to paper context.
- `test_no_retrieval_returns_no_results_answer` at line 120: If retrieval returns 0 results, answer is 'No relevant papers found'.

Test the logic inline to avoid the faiss import chain.
This mirrors the exact logic in platform._metadata_rag.

```python
class TestMetadataRAGGrounding:
    """Verify _metadata_rag system prompt enforces grounding.

    NOTE: These tests read the source of platform.py via inspect.getsource()
    rather than importing the module, because importing platform.py triggers
    a chain: platform → paper_ingestion → faiss, which may fail on environments
    with a numpy/faiss ABI mismatch.  Reading source avoids that chain.
    """

    def _get_metadata_rag_source(self):
        """Read _metadata_rag source without importing the module."""
        import pathlib
        platform_path = (
            pathlib.Path(__file__).parent.parent
            / "src" / "research_ai" / "platform.py"
        )
        src = platform_path.read_text(encoding="utf-8")
        # Extract just the _metadata_rag method section
        start = src.find("def _metadata_rag")
        end = src.find("\n    def ", start + 1)
        return src[start:end] if end > start else src[start:]

    def test_rag_system_prompt_restricts_fabrication(self):
        """The RAG system prompt should explicitly restrict LLM to paper context."""
        src = self._get_metadata_rag_source()
        # Must contain restriction to provided context
        assert "ONLY" in src or "only" in src.lower(), \
            "_metadata_rag system prompt must restrict LLM to provided context"
        # Must explicitly prohibit fabrication
        assert "Do NOT" in src or "do not" in src.lower(), \
            "_metadata_rag must explicitly prohibit fabrication"

    def test_no_retrieval_returns_no_results_answer(self):
        """If retrieval returns 0 results, answer is 'No relevant papers found'.

        Test the logic inline to avoid the faiss import chain.
        This mirrors the exact logic in platform._metadata_rag.
        """
        # Inline the _metadata_rag no-result logic to test without faiss
        def metadata_rag_no_result_path(query, top_k):
            search = {"results": [], "count": 0}  # simulate empty retrieval
            results = search.get("results", [])
            if not results:
                return {
                    "query": query,
                    "answer": "No relevant papers found in the index.",
                    "retrieved": [],
                }
            return {"unexpected": True}

        result = metadata_rag_no_result_path("nonexistent topic", 5)
        assert result["answer"] == "No relevant papers found in the index."
        assert result["retrieved"] == []
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestEvidenceGroundedContext`

- **Line:** 143
- **Base classes:** `object`
- **Docstring:** Verify the context builder in SynthesisAgent limits what the LLM sees.

**Methods:**
- `test_context_limited_to_10k_chars` at line 146: _build_context must cap at ~10000 characters to prevent LLM injection.
- `test_errored_tools_excluded_from_context` at line 169: Tool outputs with error keys should not appear in the LLM context.

```python
class TestEvidenceGroundedContext:
    """Verify the context builder in SynthesisAgent limits what the LLM sees."""

    def test_context_limited_to_10k_chars(self):
        """_build_context must cap at ~10000 characters to prevent LLM injection."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        # Build a large outputs dict
        outputs = {
            "hybrid_search": {
                "count": 20,
                "results": [
                    {
                        "title": f"Paper {i}: " + "A" * 400,
                        "year": "2023",
                        "category": "cs.LG",
                        "abstract": "B" * 700,
                        "paper_id": f"2301.{i:05d}",
                    }
                    for i in range(20)
                ],
            }
        }
        context = SynthesisAgent._build_context(outputs)
        assert len(context) <= 11000, f"Context too large: {len(context)} chars"

    def test_errored_tools_excluded_from_context(self):
        """Tool outputs with error keys should not appear in the LLM context."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        import json

        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "metadata_rag": {"answer": "The model uses attention mechanism."},
        }
        context = SynthesisAgent._build_context(outputs)
        parsed = json.loads(context)
        assert "hybrid_search" not in parsed, \
            "Errored tool outputs must be excluded from LLM context"
        assert "metadata_rag" in parsed
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `TestSynthesisGrounding` Methods

#### `TestSynthesisGrounding._synthesizer`

- **Line:** 23
- **Kind:** synchronous method
- **Arguments:** self, cloud_response
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _synthesizer(self, cloud_response=None):
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        if cloud_response is None:
            return SynthesisAgent(cloud_factory=None)

        mock_cloud = MagicMock()
        mock_cloud.generate.return_value = cloud_response
        return SynthesisAgent(cloud_factory=lambda: mock_cloud)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSynthesisGrounding.test_no_cloud_uses_structured_fallback`

- **Line:** 33
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Without a cloud factory, the structured direct answer is always used.

```python
    def test_no_cloud_uses_structured_fallback(self):
        """Without a cloud factory, the structured direct answer is always used."""
        synthesizer = self._synthesizer(cloud_response=None)
        outputs = {
            "hybrid_search": {
                "count": 2,
                "results": [
                    {"title": "Attention Is All You Need", "year": "2017",
                     "abstract": "We propose a model based on attention.", "paper_id": "1706.03762"},
                    {"title": "BERT: Pre-training", "year": "2018",
                     "abstract": "BERT is a language model.", "paper_id": "1810.04805"},
                ],
            }
        }
        plan = {"intent": "research_analysis", "query": "attention transformers"}
        answer = synthesizer.synthesize("attention transformers", plan, outputs)
        assert "Attention Is All You Need" in answer or "2017" in answer or "found" in answer.lower()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSynthesisGrounding.test_cloud_too_short_falls_back_to_structured`

- **Line:** 51
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** If the LLM returns < 10 words, the structured fallback is used.

```python
    def test_cloud_too_short_falls_back_to_structured(self):
        """If the LLM returns < 10 words, the structured fallback is used."""
        synthesizer = self._synthesizer(cloud_response="Yes.")  # < 10 words
        outputs = {
            "hybrid_search": {
                "count": 1,
                "results": [{"title": "Some Paper", "year": "2022",
                             "abstract": "We study X.", "paper_id": "2201.0001"}],
            }
        }
        plan = {"intent": "research_analysis", "query": "transformers"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        # Should NOT return "Yes." — should use structured fallback
        assert answer != "Yes."
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSynthesisGrounding.test_empty_outputs_returns_no_results_message`

- **Line:** 66
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_empty_outputs_returns_no_results_message(self):
        synthesizer = self._synthesizer()
        answer = synthesizer.synthesize("obscure topic", {"intent": "search"}, {})
        assert "No results" in answer or len(answer) > 0
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSynthesisGrounding.test_system_prompt_enforces_grounding`

- **Line:** 71
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Verify the system prompt contains the no-fabrication instruction.

```python
    def test_system_prompt_enforces_grounding(self):
        """Verify the system prompt contains the no-fabrication instruction."""
        from research_ai.agents.synthesis_agent.service import SYSTEM_PROMPT
        assert "Do NOT invent" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()
        assert "fabricate" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestSynthesisGrounding.test_conversation_bypass_returns_static_answer`

- **Line:** 77
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** The conversation tool returns a static answer without retrieval.

```python
    def test_conversation_bypass_returns_static_answer(self):
        """The conversation tool returns a static answer without retrieval."""
        synthesizer = self._synthesizer()
        outputs = {
            "conversation": {"answer": "Hello! I am your research assistant.", "query": "hi"}
        }
        plan = {"intent": "conversation", "query": "hello"}
        answer = synthesizer.synthesize("hello", plan, outputs)
        assert "Hello" in answer or "assistant" in answer.lower()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestMetadataRAGGrounding` Methods

#### `TestMetadataRAGGrounding._get_metadata_rag_source`

- **Line:** 97
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Read _metadata_rag source without importing the module.

```python
    def _get_metadata_rag_source(self):
        """Read _metadata_rag source without importing the module."""
        import pathlib
        platform_path = (
            pathlib.Path(__file__).parent.parent
            / "src" / "research_ai" / "platform.py"
        )
        src = platform_path.read_text(encoding="utf-8")
        # Extract just the _metadata_rag method section
        start = src.find("def _metadata_rag")
        end = src.find("\n    def ", start + 1)
        return src[start:end] if end > start else src[start:]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataRAGGrounding.test_rag_system_prompt_restricts_fabrication`

- **Line:** 110
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** The RAG system prompt should explicitly restrict LLM to paper context.

```python
    def test_rag_system_prompt_restricts_fabrication(self):
        """The RAG system prompt should explicitly restrict LLM to paper context."""
        src = self._get_metadata_rag_source()
        # Must contain restriction to provided context
        assert "ONLY" in src or "only" in src.lower(), \
            "_metadata_rag system prompt must restrict LLM to provided context"
        # Must explicitly prohibit fabrication
        assert "Do NOT" in src or "do not" in src.lower(), \
            "_metadata_rag must explicitly prohibit fabrication"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMetadataRAGGrounding.test_no_retrieval_returns_no_results_answer`

- **Line:** 120
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** If retrieval returns 0 results, answer is 'No relevant papers found'.

Test the logic inline to avoid the faiss import chain.
This mirrors the exact logic in platform._metadata_rag.

```python
    def test_no_retrieval_returns_no_results_answer(self):
        """If retrieval returns 0 results, answer is 'No relevant papers found'.

        Test the logic inline to avoid the faiss import chain.
        This mirrors the exact logic in platform._metadata_rag.
        """
        # Inline the _metadata_rag no-result logic to test without faiss
        def metadata_rag_no_result_path(query, top_k):
            search = {"results": [], "count": 0}  # simulate empty retrieval
            results = search.get("results", [])
            if not results:
                return {
                    "query": query,
                    "answer": "No relevant papers found in the index.",
                    "retrieved": [],
                }
            return {"unexpected": True}

        result = metadata_rag_no_result_path("nonexistent topic", 5)
        assert result["answer"] == "No relevant papers found in the index."
        assert result["retrieved"] == []
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestEvidenceGroundedContext` Methods

#### `TestEvidenceGroundedContext.test_context_limited_to_10k_chars`

- **Line:** 146
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** _build_context must cap at ~10000 characters to prevent LLM injection.

```python
    def test_context_limited_to_10k_chars(self):
        """_build_context must cap at ~10000 characters to prevent LLM injection."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        # Build a large outputs dict
        outputs = {
            "hybrid_search": {
                "count": 20,
                "results": [
                    {
                        "title": f"Paper {i}: " + "A" * 400,
                        "year": "2023",
                        "category": "cs.LG",
                        "abstract": "B" * 700,
                        "paper_id": f"2301.{i:05d}",
                    }
                    for i in range(20)
                ],
            }
        }
        context = SynthesisAgent._build_context(outputs)
        assert len(context) <= 11000, f"Context too large: {len(context)} chars"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestEvidenceGroundedContext.test_errored_tools_excluded_from_context`

- **Line:** 169
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Tool outputs with error keys should not appear in the LLM context.

```python
    def test_errored_tools_excluded_from_context(self):
        """Tool outputs with error keys should not appear in the LLM context."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        import json

        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "metadata_rag": {"answer": "The model uses attention mechanism."},
        }
        context = SynthesisAgent._build_context(outputs)
        parsed = json.loads(context)
        assert "hybrid_search" not in parsed, \
            "Errored tool outputs must be excluded from LLM context"
        assert "metadata_rag" in parsed
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Calibration**: Calibration makes predicted probabilities better match real correctness rates, which matters for user-facing confidence.
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
- Touches files or paths. Validate filenames, restrict upload size/type, and prevent traversal.
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

- `tests/test_hallucination_resistance.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `tests/test_hallucination_resistance.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Hallucination resistance tests for the synthesis and RAG layers.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
These tests stress-test the system's ability to refuse fabrication when:
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  1. No papers are found (empty retrieval)
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  2. The LLM is asked about topics not in the index
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  3. Low-confidence retrieval returns semantically irrelevant docs
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  4. The synthesis prompt would allow open-ended generation
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
All LLM calls are mocked to control what "hallucination" would look like.
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
We verify that the SYSTEM enforces grounding — not just that the LLM happens
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
to behave well.
# L0012: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0013: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0014: Blank line that visually separates logical sections and improves readability.

# L0015: Imports a dependency, type, or project module needed by later code in this file.
from unittest.mock import MagicMock
# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Imports a dependency, type, or project module needed by later code in this file.
import pytest
# L0018: Blank line that visually separates logical sections and improves readability.

# L0019: Blank line that visually separates logical sections and improves readability.

# L0020: Defines a class that groups related state and behavior behind a reusable interface.
class TestSynthesisGrounding:
# L0021: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify SynthesisAgent falls back correctly when context is absent."""
# L0022: Blank line that visually separates logical sections and improves readability.

# L0023: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _synthesizer(self, cloud_response=None):
# L0024: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if cloud_response is None:
# L0027: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return SynthesisAgent(cloud_factory=None)
# L0028: Blank line that visually separates logical sections and improves readability.

# L0029: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_cloud = MagicMock()
# L0030: Assigns or updates a value used later in the workflow; check mutability and data shape.
        mock_cloud.generate.return_value = cloud_response
# L0031: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return SynthesisAgent(cloud_factory=lambda: mock_cloud)
# L0032: Blank line that visually separates logical sections and improves readability.

# L0033: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_no_cloud_uses_structured_fallback(self):
# L0034: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Without a cloud factory, the structured direct answer is always used."""
# L0035: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = self._synthesizer(cloud_response=None)
# L0036: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0037: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "count": 2,
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "results": [
# L0040: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    {"title": "Attention Is All You Need", "year": "2017",
# L0041: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                     "abstract": "We propose a model based on attention.", "paper_id": "1706.03762"},
# L0042: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    {"title": "BERT: Pre-training", "year": "2018",
# L0043: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                     "abstract": "BERT is a language model.", "paper_id": "1810.04805"},
# L0044: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                ],
# L0045: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0046: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0047: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = {"intent": "research_analysis", "query": "attention transformers"}
# L0048: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("attention transformers", plan, outputs)
# L0049: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "Attention Is All You Need" in answer or "2017" in answer or "found" in answer.lower()
# L0050: Blank line that visually separates logical sections and improves readability.

# L0051: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_cloud_too_short_falls_back_to_structured(self):
# L0052: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """If the LLM returns < 10 words, the structured fallback is used."""
# L0053: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = self._synthesizer(cloud_response="Yes.")  # < 10 words
# L0054: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0055: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {
# L0056: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "count": 1,
# L0057: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "results": [{"title": "Some Paper", "year": "2022",
# L0058: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                             "abstract": "We study X.", "paper_id": "2201.0001"}],
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0060: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0061: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = {"intent": "research_analysis", "query": "transformers"}
# L0062: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("transformers", plan, outputs)
# L0063: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Should NOT return "Yes." — should use structured fallback
# L0064: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert answer != "Yes."
# L0065: Blank line that visually separates logical sections and improves readability.

# L0066: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_empty_outputs_returns_no_results_message(self):
# L0067: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = self._synthesizer()
# L0068: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("obscure topic", {"intent": "search"}, {})
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "No results" in answer or len(answer) > 0
# L0070: Blank line that visually separates logical sections and improves readability.

# L0071: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_system_prompt_enforces_grounding(self):
# L0072: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Verify the system prompt contains the no-fabrication instruction."""
# L0073: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SYSTEM_PROMPT
# L0074: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "Do NOT invent" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "fabricate" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()
# L0076: Blank line that visually separates logical sections and improves readability.

# L0077: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_conversation_bypass_returns_static_answer(self):
# L0078: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """The conversation tool returns a static answer without retrieval."""
# L0079: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = self._synthesizer()
# L0080: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0081: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "conversation": {"answer": "Hello! I am your research assistant.", "query": "hi"}
# L0082: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = {"intent": "conversation", "query": "hello"}
# L0084: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("hello", plan, outputs)
# L0085: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "Hello" in answer or "assistant" in answer.lower()
# L0086: Blank line that visually separates logical sections and improves readability.

# L0087: Blank line that visually separates logical sections and improves readability.

# L0088: Defines a class that groups related state and behavior behind a reusable interface.
class TestMetadataRAGGrounding:
# L0089: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify _metadata_rag system prompt enforces grounding.
# L0090: Blank line that visually separates logical sections and improves readability.

# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    NOTE: These tests read the source of platform.py via inspect.getsource()
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    rather than importing the module, because importing platform.py triggers
# L0093: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    a chain: platform → paper_ingestion → faiss, which may fail on environments
# L0094: Uses a context manager to guarantee setup/cleanup around files, locks, or managed resources.
    with a numpy/faiss ABI mismatch.  Reading source avoids that chain.
# L0095: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """
# L0096: Blank line that visually separates logical sections and improves readability.

# L0097: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _get_metadata_rag_source(self):
# L0098: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Read _metadata_rag source without importing the module."""
# L0099: Imports a dependency, type, or project module needed by later code in this file.
        import pathlib
# L0100: Assigns or updates a value used later in the workflow; check mutability and data shape.
        platform_path = (
# L0101: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            pathlib.Path(__file__).parent.parent
# L0102: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            / "src" / "research_ai" / "platform.py"
# L0103: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0104: Assigns or updates a value used later in the workflow; check mutability and data shape.
        src = platform_path.read_text(encoding="utf-8")
# L0105: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Extract just the _metadata_rag method section
# L0106: Assigns or updates a value used later in the workflow; check mutability and data shape.
        start = src.find("def _metadata_rag")
# L0107: Assigns or updates a value used later in the workflow; check mutability and data shape.
        end = src.find("\n    def ", start + 1)
# L0108: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return src[start:end] if end > start else src[start:]
# L0109: Blank line that visually separates logical sections and improves readability.

# L0110: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_rag_system_prompt_restricts_fabrication(self):
# L0111: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """The RAG system prompt should explicitly restrict LLM to paper context."""
# L0112: Assigns or updates a value used later in the workflow; check mutability and data shape.
        src = self._get_metadata_rag_source()
# L0113: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Must contain restriction to provided context
# L0114: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "ONLY" in src or "only" in src.lower(), \
# L0115: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "_metadata_rag system prompt must restrict LLM to provided context"
# L0116: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Must explicitly prohibit fabrication
# L0117: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "Do NOT" in src or "do not" in src.lower(), \
# L0118: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "_metadata_rag must explicitly prohibit fabrication"
# L0119: Blank line that visually separates logical sections and improves readability.

# L0120: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_no_retrieval_returns_no_results_answer(self):
# L0121: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """If retrieval returns 0 results, answer is 'No relevant papers found'.
# L0122: Blank line that visually separates logical sections and improves readability.

# L0123: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Test the logic inline to avoid the faiss import chain.
# L0124: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        This mirrors the exact logic in platform._metadata_rag.
# L0125: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0126: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Inline the _metadata_rag no-result logic to test without faiss
# L0127: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def metadata_rag_no_result_path(query, top_k):
# L0128: Assigns or updates a value used later in the workflow; check mutability and data shape.
            search = {"results": [], "count": 0}  # simulate empty retrieval
# L0129: Assigns or updates a value used later in the workflow; check mutability and data shape.
            results = search.get("results", [])
# L0130: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if not results:
# L0131: Returns the computed result to the caller; this shape becomes part of the downstream contract.
                return {
# L0132: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "query": query,
# L0133: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "answer": "No relevant papers found in the index.",
# L0134: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "retrieved": [],
# L0135: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                }
# L0136: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"unexpected": True}
# L0137: Blank line that visually separates logical sections and improves readability.

# L0138: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = metadata_rag_no_result_path("nonexistent topic", 5)
# L0139: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert result["answer"] == "No relevant papers found in the index."
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert result["retrieved"] == []
# L0141: Blank line that visually separates logical sections and improves readability.

# L0142: Blank line that visually separates logical sections and improves readability.

# L0143: Defines a class that groups related state and behavior behind a reusable interface.
class TestEvidenceGroundedContext:
# L0144: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Verify the context builder in SynthesisAgent limits what the LLM sees."""
# L0145: Blank line that visually separates logical sections and improves readability.

# L0146: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_context_limited_to_10k_chars(self):
# L0147: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """_build_context must cap at ~10000 characters to prevent LLM injection."""
# L0148: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0149: Blank line that visually separates logical sections and improves readability.

# L0150: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Build a large outputs dict
# L0151: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0152: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {
# L0153: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "count": 20,
# L0154: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "results": [
# L0155: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    {
# L0156: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "title": f"Paper {i}: " + "A" * 400,
# L0157: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "year": "2023",
# L0158: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "category": "cs.LG",
# L0159: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "abstract": "B" * 700,
# L0160: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                        "paper_id": f"2301.{i:05d}",
# L0161: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    }
# L0162: Iterates over data, retry attempts, files, results, or workflow steps.
                    for i in range(20)
# L0163: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                ],
# L0164: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0165: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0166: Assigns or updates a value used later in the workflow; check mutability and data shape.
        context = SynthesisAgent._build_context(outputs)
# L0167: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(context) <= 11000, f"Context too large: {len(context)} chars"
# L0168: Blank line that visually separates logical sections and improves readability.

# L0169: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_errored_tools_excluded_from_context(self):
# L0170: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Tool outputs with error keys should not appear in the LLM context."""
# L0171: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0172: Imports a dependency, type, or project module needed by later code in this file.
        import json
# L0173: Blank line that visually separates logical sections and improves readability.

# L0174: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0175: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {"error": "Index not ready"},
# L0176: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "metadata_rag": {"answer": "The model uses attention mechanism."},
# L0177: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0178: Assigns or updates a value used later in the workflow; check mutability and data shape.
        context = SynthesisAgent._build_context(outputs)
# L0179: Assigns or updates a value used later in the workflow; check mutability and data shape.
        parsed = json.loads(context)
# L0180: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "hybrid_search" not in parsed, \
# L0181: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "Errored tool outputs must be excluded from LLM context"
# L0182: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "metadata_rag" in parsed
```

## Source Walkthrough

The complete source is included because the file is short enough to study directly.

```python
"""Hallucination resistance tests for the synthesis and RAG layers.

These tests stress-test the system's ability to refuse fabrication when:
  1. No papers are found (empty retrieval)
  2. The LLM is asked about topics not in the index
  3. Low-confidence retrieval returns semantically irrelevant docs
  4. The synthesis prompt would allow open-ended generation

All LLM calls are mocked to control what "hallucination" would look like.
We verify that the SYSTEM enforces grounding — not just that the LLM happens
to behave well.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestSynthesisGrounding:
    """Verify SynthesisAgent falls back correctly when context is absent."""

    def _synthesizer(self, cloud_response=None):
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        if cloud_response is None:
            return SynthesisAgent(cloud_factory=None)

        mock_cloud = MagicMock()
        mock_cloud.generate.return_value = cloud_response
        return SynthesisAgent(cloud_factory=lambda: mock_cloud)

    def test_no_cloud_uses_structured_fallback(self):
        """Without a cloud factory, the structured direct answer is always used."""
        synthesizer = self._synthesizer(cloud_response=None)
        outputs = {
            "hybrid_search": {
                "count": 2,
                "results": [
                    {"title": "Attention Is All You Need", "year": "2017",
                     "abstract": "We propose a model based on attention.", "paper_id": "1706.03762"},
                    {"title": "BERT: Pre-training", "year": "2018",
                     "abstract": "BERT is a language model.", "paper_id": "1810.04805"},
                ],
            }
        }
        plan = {"intent": "research_analysis", "query": "attention transformers"}
        answer = synthesizer.synthesize("attention transformers", plan, outputs)
        assert "Attention Is All You Need" in answer or "2017" in answer or "found" in answer.lower()

    def test_cloud_too_short_falls_back_to_structured(self):
        """If the LLM returns < 10 words, the structured fallback is used."""
        synthesizer = self._synthesizer(cloud_response="Yes.")  # < 10 words
        outputs = {
            "hybrid_search": {
                "count": 1,
                "results": [{"title": "Some Paper", "year": "2022",
                             "abstract": "We study X.", "paper_id": "2201.0001"}],
            }
        }
        plan = {"intent": "research_analysis", "query": "transformers"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        # Should NOT return "Yes." — should use structured fallback
        assert answer != "Yes."

    def test_empty_outputs_returns_no_results_message(self):
        synthesizer = self._synthesizer()
        answer = synthesizer.synthesize("obscure topic", {"intent": "search"}, {})
        assert "No results" in answer or len(answer) > 0

    def test_system_prompt_enforces_grounding(self):
        """Verify the system prompt contains the no-fabrication instruction."""
        from research_ai.agents.synthesis_agent.service import SYSTEM_PROMPT
        assert "Do NOT invent" in SYSTEM_PROMPT or "not invent" in SYSTEM_PROMPT.lower()
        assert "fabricate" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()

    def test_conversation_bypass_returns_static_answer(self):
        """The conversation tool returns a static answer without retrieval."""
        synthesizer = self._synthesizer()
        outputs = {
            "conversation": {"answer": "Hello! I am your research assistant.", "query": "hi"}
        }
        plan = {"intent": "conversation", "query": "hello"}
        answer = synthesizer.synthesize("hello", plan, outputs)
        assert "Hello" in answer or "assistant" in answer.lower()


class TestMetadataRAGGrounding:
    """Verify _metadata_rag system prompt enforces grounding.

    NOTE: These tests read the source of platform.py via inspect.getsource()
    rather than importing the module, because importing platform.py triggers
    a chain: platform → paper_ingestion → faiss, which may fail on environments
    with a numpy/faiss ABI mismatch.  Reading source avoids that chain.
    """

    def _get_metadata_rag_source(self):
        """Read _metadata_rag source without importing the module."""
        import pathlib
        platform_path = (
            pathlib.Path(__file__).parent.parent
            / "src" / "research_ai" / "platform.py"
        )
        src = platform_path.read_text(encoding="utf-8")
        # Extract just the _metadata_rag method section
        start = src.find("def _metadata_rag")
        end = src.find("\n    def ", start + 1)
        return src[start:end] if end > start else src[start:]

    def test_rag_system_prompt_restricts_fabrication(self):
        """The RAG system prompt should explicitly restrict LLM to paper context."""
        src = self._get_metadata_rag_source()
        # Must contain restriction to provided context
        assert "ONLY" in src or "only" in src.lower(), \
            "_metadata_rag system prompt must restrict LLM to provided context"
        # Must explicitly prohibit fabrication
        assert "Do NOT" in src or "do not" in src.lower(), \
            "_metadata_rag must explicitly prohibit fabrication"

    def test_no_retrieval_returns_no_results_answer(self):
        """If retrieval returns 0 results, answer is 'No relevant papers found'.

        Test the logic inline to avoid the faiss import chain.
        This mirrors the exact logic in platform._metadata_rag.
        """
        # Inline the _metadata_rag no-result logic to test without faiss
        def metadata_rag_no_result_path(query, top_k):
            search = {"results": [], "count": 0}  # simulate empty retrieval
            results = search.get("results", [])
            if not results:
                return {
                    "query": query,
                    "answer": "No relevant papers found in the index.",
                    "retrieved": [],
                }
            return {"unexpected": True}

        result = metadata_rag_no_result_path("nonexistent topic", 5)
        assert result["answer"] == "No relevant papers found in the index."
        assert result["retrieved"] == []


class TestEvidenceGroundedContext:
    """Verify the context builder in SynthesisAgent limits what the LLM sees."""

    def test_context_limited_to_10k_chars(self):
        """_build_context must cap at ~10000 characters to prevent LLM injection."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        # Build a large outputs dict
        outputs = {
            "hybrid_search": {
                "count": 20,
                "results": [
                    {
                        "title": f"Paper {i}: " + "A" * 400,
                        "year": "2023",
                        "category": "cs.LG",
                        "abstract": "B" * 700,
                        "paper_id": f"2301.{i:05d}",
                    }
                    for i in range(20)
                ],
            }
        }
        context = SynthesisAgent._build_context(outputs)
        assert len(context) <= 11000, f"Context too large: {len(context)} chars"

    def test_errored_tools_excluded_from_context(self):
        """Tool outputs with error keys should not appear in the LLM context."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        import json

        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "metadata_rag": {"answer": "The model uses attention mechanism."},
        }
        context = SynthesisAgent._build_context(outputs)
        parsed = json.loads(context)
        assert "hybrid_search" not in parsed, \
            "Errored tool outputs must be excluded from LLM context"
        assert "metadata_rag" in parsed
```
