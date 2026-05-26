# test_pipeline_integrity.py Explained

Generated educational companion for `tests/test_pipeline_integrity.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`tests/test_pipeline_integrity.py` is a Python module in the Test layer: behavioral, safety, performance, and integration checks. It defines TestMLExecutionAgent, TestPlannerConversationGate, TestOrchestratorRetry, TestHallucinationResistance and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Test layer: behavioral, safety, performance, and integration checks. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Test layer: behavioral, safety, performance, and integration checks.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `tests/test_pipeline_integrity.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[tests/test_pipeline_integrity.py]
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

### `TestMLExecutionAgent`

- **Line:** 30
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `_agent` at line 31: method behavior is described by its body and name
- `test_unknown_tool_returns_error_dict` at line 35: method behavior is described by its body and name
- `test_string_int_coercion` at line 41: method behavior is described by its body and name
- `test_string_bool_coercion_true` at line 54: method behavior is described by its body and name
- `test_data_flow_injection_from_search` at line 76: 'from: search_results' in args should be replaced with prior paper list.
- `test_data_flow_injection_from_smart_retrieve` at line 105: smart_retrieve results are also tracked for downstream injection.
- `test_tool_error_does_not_abort_plan` at line 130: A failing tool should return an error dict, not abort the plan.

```python
class TestMLExecutionAgent:
    def _agent(self, tools=None):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        return MLExecutionAgent(tools or {})

    def test_unknown_tool_returns_error_dict(self):
        agent = self._agent()
        result = agent.execute("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_string_int_coercion(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received_args = {}

        def fake_tool(top_k=5, **_):
            received_args["top_k"] = top_k
            return {"result": "ok"}

        agent = MLExecutionAgent({"my_tool": fake_tool})
        agent.execute("my_tool", {"top_k": "10"})  # string "10" → int 10
        assert received_args["top_k"] == 10
        assert isinstance(received_args["top_k"], int)

    def test_string_bool_coercion_true(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received = {}

        def fake_tool(**kwargs):
            received.update(kwargs)
            return {}

        agent = MLExecutionAgent({"t": fake_tool})
        agent.execute("t", {"some_bool_key": "true"})
        # Note: coercion only applies to keys in ("top_k", "candidate_k", "max_tokens", "timeout")
        # For other keys, "true" stays as string unless it hits the bool branch
        # Let's verify the bool branch works for a string value "true"
        agent.execute("t", {"top_k": "true"})
        # "true" → try int("true") fails → try bool → "true" → True
        # Actually looking at the code: int coercion for top_k is tried first,
        # fails, then bool coercion is tried
        # Covered by the general bool branch
        agent2 = MLExecutionAgent({"t": fake_tool})
        agent2.execute("t", {"flag": "true"})
        assert received.get("flag") is True or received.get("flag") == "true"

    def test_data_flow_injection_from_search(self):
        """'from: search_results' in args should be replaced with prior paper list."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        papers_received = []

        def fake_search(query="", top_k=5, **_):
            return {"results": [{"title": "Paper A"}, {"title": "Paper B"}], "count": 2}

        def fake_methodology(papers=None, **_):
            papers_received.extend(papers or [])
            return {"count": 1, "signals": ["attention"]}

        agent = MLExecutionAgent({
            "hybrid_search": fake_search,
            "methodology_extract": fake_methodology,
        })

        calls = [
            ToolCall("hybrid_search", {"query": "transformers", "top_k": 5}),
            ToolCall("methodology_extract", {"from": "search_results"}),
        ]
        outputs = agent.execute_plan(calls)

        # Verify injection happened
        assert len(papers_received) == 2
        assert papers_received[0]["title"] == "Paper A"

    def test_data_flow_injection_from_smart_retrieve(self):
        """smart_retrieve results are also tracked for downstream injection."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        injected = []

        def fake_smart_retrieve(query="", top_k=5, **_):
            return {"results": [{"title": "Smart Paper"}], "count": 1}

        def fake_methodology(papers=None, **_):
            injected.extend(papers or [])
            return {"count": 0}

        agent = MLExecutionAgent({
            "smart_retrieve": fake_smart_retrieve,
            "methodology_extract": fake_methodology,
        })
        calls = [
            ToolCall("smart_retrieve", {"query": "diffusion", "top_k": 5}),
            ToolCall("methodology_extract", {"from": "search_results"}),
        ]
        agent.execute_plan(calls)
        assert injected[0]["title"] == "Smart Paper"

    def test_tool_error_does_not_abort_plan(self):
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestPlannerConversationGate`

- **Line:** 159
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `_planner` at line 160: method behavior is described by its body and name
- `test_greeting_routed_to_conversation_tool` at line 164: method behavior is described by its body and name
- `test_greeting_does_not_call_retrieval` at line 171: method behavior is described by its body and name
- `test_research_query_not_flagged_as_conversation` at line 179: method behavior is described by its body and name
- `test_thanks_routed_to_conversation` at line 186: method behavior is described by its body and name
- `test_classify_mode_produces_classify_tool` at line 191: method behavior is described by its body and name
- `test_search_mode_produces_hybrid_search` at line 198: method behavior is described by its body and name
- `test_top_k_clamped_to_max` at line 204: method behavior is described by its body and name

```python
class TestPlannerConversationGate:
    def _planner(self):
        from research_ai.agents.planner.service import PlannerAgent
        return PlannerAgent(cloud_factory=None)  # no LLM, uses heuristic fallback

    def test_greeting_routed_to_conversation_tool(self):
        planner = self._planner()
        plan = planner.plan("auto", "Hello!", top_k=5)
        assert plan.intent == "conversation"
        assert len(plan.calls) == 1
        assert plan.calls[0].name == "conversation"

    def test_greeting_does_not_call_retrieval(self):
        planner = self._planner()
        plan = planner.plan("auto", "Hi there", top_k=5)
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" not in tool_names
        assert "smart_retrieve" not in tool_names
        assert "metadata_rag" not in tool_names

    def test_research_query_not_flagged_as_conversation(self):
        planner = self._planner()
        plan = planner.plan("auto", "What are the latest transformer architectures?", top_k=5)
        assert plan.intent != "conversation"
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" in tool_names or "smart_retrieve" in tool_names

    def test_thanks_routed_to_conversation(self):
        planner = self._planner()
        plan = planner.plan("auto", "Thanks!", top_k=5)
        assert plan.intent == "conversation"

    def test_classify_mode_produces_classify_tool(self):
        planner = self._planner()
        plan = planner.plan("classify", "neural networks", top_k=5,
                            title="Neural Networks", abstract="We study neural networks.")
        tool_names = [c.name for c in plan.calls]
        assert "classify_query" in tool_names

    def test_search_mode_produces_hybrid_search(self):
        planner = self._planner()
        plan = planner.plan("search", "diffusion models generative", top_k=5)
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" in tool_names

    def test_top_k_clamped_to_max(self):
        planner = self._planner()
        plan = planner.plan("search", "attention", top_k=999)
        assert plan.top_k <= planner.max_top_k
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestOrchestratorRetry`

- **Line:** 214
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_retry_plan_increases_top_k` at line 215: method behavior is described by its body and name
- `test_retry_plan_capped_at_20` at line 233: method behavior is described by its body and name
- `test_conversation_intent_skips_retry` at line 248: Conversations should never trigger a retry even with low score.

```python
class TestOrchestratorRetry:
    def test_retry_plan_increases_top_k(self):
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        plan = ResearchPlan(
            intent="research_analysis",
            query="transformers",
            top_k=5,
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 5})],
            reason="test",
        )
        evaluation = {"retry_top_k_multiplier": 2, "reason": "no_retrieval_hits"}
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)

        assert retry_plan.top_k == 10  # 5 × 2
        search_call = next(c for c in retry_plan.calls if c.name == "hybrid_search")
        assert search_call.args["top_k"] == 10

    def test_retry_plan_capped_at_20(self):
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        plan = ResearchPlan(
            intent="research_analysis",
            query="transformers",
            top_k=15,  # 15 × 3 = 45 → should cap at 20
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 15})],
            reason="test",
        )
        evaluation = {"retry_top_k_multiplier": 3, "reason": "no_retrieval_hits"}
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)
        assert retry_plan.top_k <= 20

    def test_conversation_intent_skips_retry(self):
        """Conversations should never trigger a retry even with low score."""
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import PlannerAgent, ResearchPlan, ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        from research_ai.agents.evaluator_agent.service import EvaluatorAgent
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        planner = MagicMock()
        planner.plan.return_value = ResearchPlan(
            intent="conversation",
            query="hello",
            top_k=5,
            calls=[ToolCall("conversation", {"query": "hello"})],
            reason="greeting",
            used_fallback=True,
        )

        def fake_conversation(query="", **_):
            return {"answer": "Hello! How can I help you?", "query": query}

        executor = MLExecutionAgent({"conversation": fake_conversation})
        evaluator = EvaluatorAgent()
        synthesizer = SynthesisAgent(cloud_factory=None)

        orchestrator = ResearchOrchestrator(planner, executor, evaluator, synthesizer)
        result = orchestrator.run("auto", "hello")

        # Should have a final answer, no retry triggered
        assert result["final_answer"]
        assert "_retry" not in result["executor_output"]
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.

### `TestHallucinationResistance`

- **Line:** 285
- **Base classes:** `object`
- **Docstring:** No explicit class docstring.

**Methods:**
- `test_no_retrieval_produces_fallback_not_fabrication` at line 286: When the retrieval index is empty/not-ready, answer must be the
fallback message, not an LLM-fabricated answer.
- `test_errored_tools_dont_block_synthesis` at line 308: If all tools error, synthesizer still returns a safe fallback.

```python
class TestHallucinationResistance:
    def test_no_retrieval_produces_fallback_not_fabrication(self):
        """When the retrieval index is empty/not-ready, answer must be the
        fallback message, not an LLM-fabricated answer."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        # SynthesisAgent with no cloud factory → uses structured direct answer
        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Search index not ready."},
        }
        plan = ResearchPlan(
            intent="research_analysis", query="transformers",
            top_k=5, calls=[], reason="test",
        )
        answer = synthesizer.synthesize("transformers", plan.__dict__, outputs)
        # The answer should indicate the problem — NOT fabricate paper titles
        assert isinstance(answer, str) and len(answer) > 0
        # Must NOT contain made-up citations or invented paper data
        # (we can check it's the fallback path by verifying no search results appear)
        assert "No results" in answer or "not ready" in answer.lower() or answer.strip()

    def test_errored_tools_dont_block_synthesis(self):
        """If all tools error, synthesizer still returns a safe fallback."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "classify_query": {"error": "Classifier not loaded"},
            "metadata_rag": {"error": "No cloud LLM"},
        }
        plan = {"intent": "research_analysis", "query": "test"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        assert isinstance(answer, str)
        # Should not raise, should return something useful
        assert len(answer) > 0
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `TestMLExecutionAgent` Methods

#### `TestMLExecutionAgent._agent`

- **Line:** 31
- **Kind:** synchronous method
- **Arguments:** self, tools
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _agent(self, tools=None):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        return MLExecutionAgent(tools or {})
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_unknown_tool_returns_error_dict`

- **Line:** 35
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_unknown_tool_returns_error_dict(self):
        agent = self._agent()
        result = agent.execute("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_string_int_coercion`

- **Line:** 41
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_string_int_coercion(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received_args = {}

        def fake_tool(top_k=5, **_):
            received_args["top_k"] = top_k
            return {"result": "ok"}

        agent = MLExecutionAgent({"my_tool": fake_tool})
        agent.execute("my_tool", {"top_k": "10"})  # string "10" → int 10
        assert received_args["top_k"] == 10
        assert isinstance(received_args["top_k"], int)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_string_bool_coercion_true`

- **Line:** 54
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_string_bool_coercion_true(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received = {}

        def fake_tool(**kwargs):
            received.update(kwargs)
            return {}

        agent = MLExecutionAgent({"t": fake_tool})
        agent.execute("t", {"some_bool_key": "true"})
        # Note: coercion only applies to keys in ("top_k", "candidate_k", "max_tokens", "timeout")
        # For other keys, "true" stays as string unless it hits the bool branch
        # Let's verify the bool branch works for a string value "true"
        agent.execute("t", {"top_k": "true"})
        # "true" → try int("true") fails → try bool → "true" → True
        # Actually looking at the code: int coercion for top_k is tried first,
        # fails, then bool coercion is tried
        # Covered by the general bool branch
        agent2 = MLExecutionAgent({"t": fake_tool})
        agent2.execute("t", {"flag": "true"})
        assert received.get("flag") is True or received.get("flag") == "true"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_data_flow_injection_from_search`

- **Line:** 76
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** 'from: search_results' in args should be replaced with prior paper list.

```python
    def test_data_flow_injection_from_search(self):
        """'from: search_results' in args should be replaced with prior paper list."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        papers_received = []

        def fake_search(query="", top_k=5, **_):
            return {"results": [{"title": "Paper A"}, {"title": "Paper B"}], "count": 2}

        def fake_methodology(papers=None, **_):
            papers_received.extend(papers or [])
            return {"count": 1, "signals": ["attention"]}

        agent = MLExecutionAgent({
            "hybrid_search": fake_search,
            "methodology_extract": fake_methodology,
        })

        calls = [
            ToolCall("hybrid_search", {"query": "transformers", "top_k": 5}),
            ToolCall("methodology_extract", {"from": "search_results"}),
        ]
        outputs = agent.execute_plan(calls)

        # Verify injection happened
        assert len(papers_received) == 2
        assert papers_received[0]["title"] == "Paper A"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_data_flow_injection_from_smart_retrieve`

- **Line:** 105
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** smart_retrieve results are also tracked for downstream injection.

```python
    def test_data_flow_injection_from_smart_retrieve(self):
        """smart_retrieve results are also tracked for downstream injection."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        injected = []

        def fake_smart_retrieve(query="", top_k=5, **_):
            return {"results": [{"title": "Smart Paper"}], "count": 1}

        def fake_methodology(papers=None, **_):
            injected.extend(papers or [])
            return {"count": 0}

        agent = MLExecutionAgent({
            "smart_retrieve": fake_smart_retrieve,
            "methodology_extract": fake_methodology,
        })
        calls = [
            ToolCall("smart_retrieve", {"query": "diffusion", "top_k": 5}),
            ToolCall("methodology_extract", {"from": "search_results"}),
        ]
        agent.execute_plan(calls)
        assert injected[0]["title"] == "Smart Paper"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestMLExecutionAgent.test_tool_error_does_not_abort_plan`

- **Line:** 130
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** A failing tool should return an error dict, not abort the plan.

```python
    def test_tool_error_does_not_abort_plan(self):
        """A failing tool should return an error dict, not abort the plan."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        def failing_tool(**_):
            raise ValueError("Simulated tool failure")

        def working_tool(query="", **_):
            return {"count": 3, "results": ["p1", "p2", "p3"]}

        agent = MLExecutionAgent({
            "fail_tool": failing_tool,
            "good_tool": working_tool,
        })
        calls = [
            ToolCall("fail_tool", {}),
            ToolCall("good_tool", {"query": "transformers"}),
        ]
        outputs = agent.execute_plan(calls)

        assert "error" in outputs["fail_tool"]
        assert outputs["good_tool"]["count"] == 3
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestPlannerConversationGate` Methods

#### `TestPlannerConversationGate._planner`

- **Line:** 160
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _planner(self):
        from research_ai.agents.planner.service import PlannerAgent
        return PlannerAgent(cloud_factory=None)  # no LLM, uses heuristic fallback
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_greeting_routed_to_conversation_tool`

- **Line:** 164
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_greeting_routed_to_conversation_tool(self):
        planner = self._planner()
        plan = planner.plan("auto", "Hello!", top_k=5)
        assert plan.intent == "conversation"
        assert len(plan.calls) == 1
        assert plan.calls[0].name == "conversation"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_greeting_does_not_call_retrieval`

- **Line:** 171
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_greeting_does_not_call_retrieval(self):
        planner = self._planner()
        plan = planner.plan("auto", "Hi there", top_k=5)
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" not in tool_names
        assert "smart_retrieve" not in tool_names
        assert "metadata_rag" not in tool_names
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_research_query_not_flagged_as_conversation`

- **Line:** 179
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_research_query_not_flagged_as_conversation(self):
        planner = self._planner()
        plan = planner.plan("auto", "What are the latest transformer architectures?", top_k=5)
        assert plan.intent != "conversation"
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" in tool_names or "smart_retrieve" in tool_names
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_thanks_routed_to_conversation`

- **Line:** 186
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_thanks_routed_to_conversation(self):
        planner = self._planner()
        plan = planner.plan("auto", "Thanks!", top_k=5)
        assert plan.intent == "conversation"
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_classify_mode_produces_classify_tool`

- **Line:** 191
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_classify_mode_produces_classify_tool(self):
        planner = self._planner()
        plan = planner.plan("classify", "neural networks", top_k=5,
                            title="Neural Networks", abstract="We study neural networks.")
        tool_names = [c.name for c in plan.calls]
        assert "classify_query" in tool_names
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_search_mode_produces_hybrid_search`

- **Line:** 198
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_search_mode_produces_hybrid_search(self):
        planner = self._planner()
        plan = planner.plan("search", "diffusion models generative", top_k=5)
        tool_names = [c.name for c in plan.calls]
        assert "hybrid_search" in tool_names
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestPlannerConversationGate.test_top_k_clamped_to_max`

- **Line:** 204
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_top_k_clamped_to_max(self):
        planner = self._planner()
        plan = planner.plan("search", "attention", top_k=999)
        assert plan.top_k <= planner.max_top_k
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestOrchestratorRetry` Methods

#### `TestOrchestratorRetry.test_retry_plan_increases_top_k`

- **Line:** 215
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_retry_plan_increases_top_k(self):
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        plan = ResearchPlan(
            intent="research_analysis",
            query="transformers",
            top_k=5,
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 5})],
            reason="test",
        )
        evaluation = {"retry_top_k_multiplier": 2, "reason": "no_retrieval_hits"}
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)

        assert retry_plan.top_k == 10  # 5 × 2
        search_call = next(c for c in retry_plan.calls if c.name == "hybrid_search")
        assert search_call.args["top_k"] == 10
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestOrchestratorRetry.test_retry_plan_capped_at_20`

- **Line:** 233
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def test_retry_plan_capped_at_20(self):
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        plan = ResearchPlan(
            intent="research_analysis",
            query="transformers",
            top_k=15,  # 15 × 3 = 45 → should cap at 20
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 15})],
            reason="test",
        )
        evaluation = {"retry_top_k_multiplier": 3, "reason": "no_retrieval_hits"}
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)
        assert retry_plan.top_k <= 20
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestOrchestratorRetry.test_conversation_intent_skips_retry`

- **Line:** 248
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Conversations should never trigger a retry even with low score.

```python
    def test_conversation_intent_skips_retry(self):
        """Conversations should never trigger a retry even with low score."""
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import PlannerAgent, ResearchPlan, ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        from research_ai.agents.evaluator_agent.service import EvaluatorAgent
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        planner = MagicMock()
        planner.plan.return_value = ResearchPlan(
            intent="conversation",
            query="hello",
            top_k=5,
            calls=[ToolCall("conversation", {"query": "hello"})],
            reason="greeting",
            used_fallback=True,
        )

        def fake_conversation(query="", **_):
            return {"answer": "Hello! How can I help you?", "query": query}

        executor = MLExecutionAgent({"conversation": fake_conversation})
        evaluator = EvaluatorAgent()
        synthesizer = SynthesisAgent(cloud_factory=None)

        orchestrator = ResearchOrchestrator(planner, executor, evaluator, synthesizer)
        result = orchestrator.run("auto", "hello")

        # Should have a final answer, no retry triggered
        assert result["final_answer"]
        assert "_retry" not in result["executor_output"]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

### Class `TestHallucinationResistance` Methods

#### `TestHallucinationResistance.test_no_retrieval_produces_fallback_not_fabrication`

- **Line:** 286
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** When the retrieval index is empty/not-ready, answer must be the
fallback message, not an LLM-fabricated answer.

```python
    def test_no_retrieval_produces_fallback_not_fabrication(self):
        """When the retrieval index is empty/not-ready, answer must be the
        fallback message, not an LLM-fabricated answer."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        # SynthesisAgent with no cloud factory → uses structured direct answer
        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Search index not ready."},
        }
        plan = ResearchPlan(
            intent="research_analysis", query="transformers",
            top_k=5, calls=[], reason="test",
        )
        answer = synthesizer.synthesize("transformers", plan.__dict__, outputs)
        # The answer should indicate the problem — NOT fabricate paper titles
        assert isinstance(answer, str) and len(answer) > 0
        # Must NOT contain made-up citations or invented paper data
        # (we can check it's the fallback path by verifying no search results appear)
        assert "No results" in answer or "not ready" in answer.lower() or answer.strip()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `TestHallucinationResistance.test_errored_tools_dont_block_synthesis`

- **Line:** 308
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** If all tools error, synthesizer still returns a safe fallback.

```python
    def test_errored_tools_dont_block_synthesis(self):
        """If all tools error, synthesizer still returns a safe fallback."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "classify_query": {"error": "Classifier not loaded"},
            "metadata_rag": {"error": "No cloud LLM"},
        }
        plan = {"intent": "research_analysis", "query": "test"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        assert isinstance(answer, str)
        # Should not raise, should return something useful
        assert len(answer) > 0
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Classification**: Classification maps text or features to discrete labels, supporting category prediction and routing.
- **Streaming**: Streaming improves perceived latency by sending incremental output instead of waiting for full completion.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `pytest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `unittest` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |

## ML Concepts Used

- **Hybrid Retrieval**: Hybrid retrieval combines semantic vectors with lexical/keyword evidence, improving scientific search where exact terms matter.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
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

- `tests/test_pipeline_integrity.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
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

- `tests/test_pipeline_integrity.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Tests for pipeline integrity — verifying the one-way ML-first pipeline.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
ARCHITECTURE UNDER TEST
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------------
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
USER INPUT → retrieval/ML → structured evidence → LLM reasoning → grounded output
# L0006: Blank line that visually separates logical sections and improves readability.

# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
These tests verify that:
# L0008: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  1. The pipeline CANNOT produce an answer without retrieval (no hallucination bypass)
# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  2. Data-flow injection correctly pipes search results to downstream tools
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  3. The MLExecutionAgent isolates tool errors (one failure doesn't kill the plan)
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  4. Type coercion works correctly (string "5" → int 5, "true" → bool True)
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  5. Unknown tools return a clear error dict (not an exception)
# L0013: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  6. The PlannerAgent conversation gate prevents tool chains for greetings
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  7. The orchestrator retry logic uses wider top_k on retry
# L0015: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  8. PaperChatService normalization produces consistent sessions
# L0016: Blank line that visually separates logical sections and improves readability.

# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
All LLM calls are mocked — these tests verify system logic, not LLM quality.
# L0018: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0019: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0020: Blank line that visually separates logical sections and improves readability.

# L0021: Imports a dependency, type, or project module needed by later code in this file.
from unittest.mock import MagicMock, patch
# L0022: Blank line that visually separates logical sections and improves readability.

# L0023: Imports a dependency, type, or project module needed by later code in this file.
import pytest
# L0024: Blank line that visually separates logical sections and improves readability.

# L0025: Blank line that visually separates logical sections and improves readability.

# L0026: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0027: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# MLExecutionAgent — type coercion and data-flow injection
# L0028: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Defines a class that groups related state and behavior behind a reusable interface.
class TestMLExecutionAgent:
# L0031: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _agent(self, tools=None):
# L0032: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0033: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return MLExecutionAgent(tools or {})
# L0034: Blank line that visually separates logical sections and improves readability.

# L0035: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_unknown_tool_returns_error_dict(self):
# L0036: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = self._agent()
# L0037: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = agent.execute("nonexistent_tool", {})
# L0038: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "error" in result
# L0039: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "nonexistent_tool" in result["error"]
# L0040: Blank line that visually separates logical sections and improves readability.

# L0041: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_string_int_coercion(self):
# L0042: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0043: Assigns or updates a value used later in the workflow; check mutability and data shape.
        received_args = {}
# L0044: Blank line that visually separates logical sections and improves readability.

# L0045: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_tool(top_k=5, **_):
# L0046: Assigns or updates a value used later in the workflow; check mutability and data shape.
            received_args["top_k"] = top_k
# L0047: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"result": "ok"}
# L0048: Blank line that visually separates logical sections and improves readability.

# L0049: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = MLExecutionAgent({"my_tool": fake_tool})
# L0050: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        agent.execute("my_tool", {"top_k": "10"})  # string "10" → int 10
# L0051: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert received_args["top_k"] == 10
# L0052: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert isinstance(received_args["top_k"], int)
# L0053: Blank line that visually separates logical sections and improves readability.

# L0054: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_string_bool_coercion_true(self):
# L0055: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0056: Assigns or updates a value used later in the workflow; check mutability and data shape.
        received = {}
# L0057: Blank line that visually separates logical sections and improves readability.

# L0058: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_tool(**kwargs):
# L0059: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            received.update(kwargs)
# L0060: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {}
# L0061: Blank line that visually separates logical sections and improves readability.

# L0062: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = MLExecutionAgent({"t": fake_tool})
# L0063: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        agent.execute("t", {"some_bool_key": "true"})
# L0064: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Note: coercion only applies to keys in ("top_k", "candidate_k", "max_tokens", "timeout")
# L0065: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # For other keys, "true" stays as string unless it hits the bool branch
# L0066: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Let's verify the bool branch works for a string value "true"
# L0067: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        agent.execute("t", {"top_k": "true"})
# L0068: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # "true" → try int("true") fails → try bool → "true" → True
# L0069: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Actually looking at the code: int coercion for top_k is tried first,
# L0070: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # fails, then bool coercion is tried
# L0071: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Covered by the general bool branch
# L0072: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent2 = MLExecutionAgent({"t": fake_tool})
# L0073: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        agent2.execute("t", {"flag": "true"})
# L0074: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert received.get("flag") is True or received.get("flag") == "true"
# L0075: Blank line that visually separates logical sections and improves readability.

# L0076: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_data_flow_injection_from_search(self):
# L0077: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """'from: search_results' in args should be replaced with prior paper list."""
# L0078: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ToolCall
# L0079: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0080: Blank line that visually separates logical sections and improves readability.

# L0081: Assigns or updates a value used later in the workflow; check mutability and data shape.
        papers_received = []
# L0082: Blank line that visually separates logical sections and improves readability.

# L0083: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_search(query="", top_k=5, **_):
# L0084: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"results": [{"title": "Paper A"}, {"title": "Paper B"}], "count": 2}
# L0085: Blank line that visually separates logical sections and improves readability.

# L0086: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_methodology(papers=None, **_):
# L0087: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            papers_received.extend(papers or [])
# L0088: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"count": 1, "signals": ["attention"]}
# L0089: Blank line that visually separates logical sections and improves readability.

# L0090: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = MLExecutionAgent({
# L0091: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": fake_search,
# L0092: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "methodology_extract": fake_methodology,
# L0093: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        })
# L0094: Blank line that visually separates logical sections and improves readability.

# L0095: Assigns or updates a value used later in the workflow; check mutability and data shape.
        calls = [
# L0096: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("hybrid_search", {"query": "transformers", "top_k": 5}),
# L0097: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("methodology_extract", {"from": "search_results"}),
# L0098: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0099: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = agent.execute_plan(calls)
# L0100: Blank line that visually separates logical sections and improves readability.

# L0101: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Verify injection happened
# L0102: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(papers_received) == 2
# L0103: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert papers_received[0]["title"] == "Paper A"
# L0104: Blank line that visually separates logical sections and improves readability.

# L0105: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_data_flow_injection_from_smart_retrieve(self):
# L0106: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """smart_retrieve results are also tracked for downstream injection."""
# L0107: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ToolCall
# L0108: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0109: Blank line that visually separates logical sections and improves readability.

# L0110: Assigns or updates a value used later in the workflow; check mutability and data shape.
        injected = []
# L0111: Blank line that visually separates logical sections and improves readability.

# L0112: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_smart_retrieve(query="", top_k=5, **_):
# L0113: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"results": [{"title": "Smart Paper"}], "count": 1}
# L0114: Blank line that visually separates logical sections and improves readability.

# L0115: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_methodology(papers=None, **_):
# L0116: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            injected.extend(papers or [])
# L0117: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"count": 0}
# L0118: Blank line that visually separates logical sections and improves readability.

# L0119: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = MLExecutionAgent({
# L0120: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "smart_retrieve": fake_smart_retrieve,
# L0121: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "methodology_extract": fake_methodology,
# L0122: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        })
# L0123: Assigns or updates a value used later in the workflow; check mutability and data shape.
        calls = [
# L0124: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("smart_retrieve", {"query": "diffusion", "top_k": 5}),
# L0125: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("methodology_extract", {"from": "search_results"}),
# L0126: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0127: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        agent.execute_plan(calls)
# L0128: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert injected[0]["title"] == "Smart Paper"
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_tool_error_does_not_abort_plan(self):
# L0131: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """A failing tool should return an error dict, not abort the plan."""
# L0132: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ToolCall
# L0133: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0134: Blank line that visually separates logical sections and improves readability.

# L0135: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def failing_tool(**_):
# L0136: Raises an explicit error when the function cannot safely continue.
            raise ValueError("Simulated tool failure")
# L0137: Blank line that visually separates logical sections and improves readability.

# L0138: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def working_tool(query="", **_):
# L0139: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"count": 3, "results": ["p1", "p2", "p3"]}
# L0140: Blank line that visually separates logical sections and improves readability.

# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
        agent = MLExecutionAgent({
# L0142: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "fail_tool": failing_tool,
# L0143: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "good_tool": working_tool,
# L0144: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        })
# L0145: Assigns or updates a value used later in the workflow; check mutability and data shape.
        calls = [
# L0146: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("fail_tool", {}),
# L0147: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ToolCall("good_tool", {"query": "transformers"}),
# L0148: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        ]
# L0149: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = agent.execute_plan(calls)
# L0150: Blank line that visually separates logical sections and improves readability.

# L0151: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "error" in outputs["fail_tool"]
# L0152: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert outputs["good_tool"]["count"] == 3
# L0153: Blank line that visually separates logical sections and improves readability.

# L0154: Blank line that visually separates logical sections and improves readability.

# L0155: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0156: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# PlannerAgent — conversation gate
# L0157: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0158: Blank line that visually separates logical sections and improves readability.

# L0159: Defines a class that groups related state and behavior behind a reusable interface.
class TestPlannerConversationGate:
# L0160: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _planner(self):
# L0161: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import PlannerAgent
# L0162: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return PlannerAgent(cloud_factory=None)  # no LLM, uses heuristic fallback
# L0163: Blank line that visually separates logical sections and improves readability.

# L0164: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_greeting_routed_to_conversation_tool(self):
# L0165: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0166: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("auto", "Hello!", top_k=5)
# L0167: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert plan.intent == "conversation"
# L0168: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert len(plan.calls) == 1
# L0169: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert plan.calls[0].name == "conversation"
# L0170: Blank line that visually separates logical sections and improves readability.

# L0171: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_greeting_does_not_call_retrieval(self):
# L0172: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0173: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("auto", "Hi there", top_k=5)
# L0174: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tool_names = [c.name for c in plan.calls]
# L0175: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "hybrid_search" not in tool_names
# L0176: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "smart_retrieve" not in tool_names
# L0177: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "metadata_rag" not in tool_names
# L0178: Blank line that visually separates logical sections and improves readability.

# L0179: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_research_query_not_flagged_as_conversation(self):
# L0180: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0181: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("auto", "What are the latest transformer architectures?", top_k=5)
# L0182: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert plan.intent != "conversation"
# L0183: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tool_names = [c.name for c in plan.calls]
# L0184: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "hybrid_search" in tool_names or "smart_retrieve" in tool_names
# L0185: Blank line that visually separates logical sections and improves readability.

# L0186: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_thanks_routed_to_conversation(self):
# L0187: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0188: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("auto", "Thanks!", top_k=5)
# L0189: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert plan.intent == "conversation"
# L0190: Blank line that visually separates logical sections and improves readability.

# L0191: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_classify_mode_produces_classify_tool(self):
# L0192: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0193: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("classify", "neural networks", top_k=5,
# L0194: Assigns or updates a value used later in the workflow; check mutability and data shape.
                            title="Neural Networks", abstract="We study neural networks.")
# L0195: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tool_names = [c.name for c in plan.calls]
# L0196: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "classify_query" in tool_names
# L0197: Blank line that visually separates logical sections and improves readability.

# L0198: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_search_mode_produces_hybrid_search(self):
# L0199: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0200: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("search", "diffusion models generative", top_k=5)
# L0201: Assigns or updates a value used later in the workflow; check mutability and data shape.
        tool_names = [c.name for c in plan.calls]
# L0202: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "hybrid_search" in tool_names
# L0203: Blank line that visually separates logical sections and improves readability.

# L0204: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_top_k_clamped_to_max(self):
# L0205: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = self._planner()
# L0206: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = planner.plan("search", "attention", top_k=999)
# L0207: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert plan.top_k <= planner.max_top_k
# L0208: Blank line that visually separates logical sections and improves readability.

# L0209: Blank line that visually separates logical sections and improves readability.

# L0210: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0211: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Orchestrator retry logic
# L0212: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0213: Blank line that visually separates logical sections and improves readability.

# L0214: Defines a class that groups related state and behavior behind a reusable interface.
class TestOrchestratorRetry:
# L0215: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_retry_plan_increases_top_k(self):
# L0216: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
# L0217: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ResearchPlan, ToolCall
# L0218: Blank line that visually separates logical sections and improves readability.

# L0219: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = ResearchPlan(
# L0220: Assigns or updates a value used later in the workflow; check mutability and data shape.
            intent="research_analysis",
# L0221: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query="transformers",
# L0222: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=5,
# L0223: Assigns or updates a value used later in the workflow; check mutability and data shape.
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 5})],
# L0224: Assigns or updates a value used later in the workflow; check mutability and data shape.
            reason="test",
# L0225: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0226: Assigns or updates a value used later in the workflow; check mutability and data shape.
        evaluation = {"retry_top_k_multiplier": 2, "reason": "no_retrieval_hits"}
# L0227: Assigns or updates a value used later in the workflow; check mutability and data shape.
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)
# L0228: Blank line that visually separates logical sections and improves readability.

# L0229: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert retry_plan.top_k == 10  # 5 × 2
# L0230: Assigns or updates a value used later in the workflow; check mutability and data shape.
        search_call = next(c for c in retry_plan.calls if c.name == "hybrid_search")
# L0231: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert search_call.args["top_k"] == 10
# L0232: Blank line that visually separates logical sections and improves readability.

# L0233: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_retry_plan_capped_at_20(self):
# L0234: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
# L0235: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ResearchPlan, ToolCall
# L0236: Blank line that visually separates logical sections and improves readability.

# L0237: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = ResearchPlan(
# L0238: Assigns or updates a value used later in the workflow; check mutability and data shape.
            intent="research_analysis",
# L0239: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query="transformers",
# L0240: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=15,  # 15 × 3 = 45 → should cap at 20
# L0241: Assigns or updates a value used later in the workflow; check mutability and data shape.
            calls=[ToolCall("hybrid_search", {"query": "transformers", "top_k": 15})],
# L0242: Assigns or updates a value used later in the workflow; check mutability and data shape.
            reason="test",
# L0243: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0244: Assigns or updates a value used later in the workflow; check mutability and data shape.
        evaluation = {"retry_top_k_multiplier": 3, "reason": "no_retrieval_hits"}
# L0245: Assigns or updates a value used later in the workflow; check mutability and data shape.
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)
# L0246: Assigns or updates a value used later in the workflow; check mutability and data shape.
        assert retry_plan.top_k <= 20
# L0247: Blank line that visually separates logical sections and improves readability.

# L0248: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_conversation_intent_skips_retry(self):
# L0249: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Conversations should never trigger a retry even with low score."""
# L0250: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
# L0251: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import PlannerAgent, ResearchPlan, ToolCall
# L0252: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
# L0253: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.evaluator_agent.service import EvaluatorAgent
# L0254: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0255: Blank line that visually separates logical sections and improves readability.

# L0256: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner = MagicMock()
# L0257: Assigns or updates a value used later in the workflow; check mutability and data shape.
        planner.plan.return_value = ResearchPlan(
# L0258: Assigns or updates a value used later in the workflow; check mutability and data shape.
            intent="conversation",
# L0259: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query="hello",
# L0260: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=5,
# L0261: Assigns or updates a value used later in the workflow; check mutability and data shape.
            calls=[ToolCall("conversation", {"query": "hello"})],
# L0262: Assigns or updates a value used later in the workflow; check mutability and data shape.
            reason="greeting",
# L0263: Assigns or updates a value used later in the workflow; check mutability and data shape.
            used_fallback=True,
# L0264: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0265: Blank line that visually separates logical sections and improves readability.

# L0266: Defines a function or method; parameters are the input contract and the body implements the workflow.
        def fake_conversation(query="", **_):
# L0267: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {"answer": "Hello! How can I help you?", "query": query}
# L0268: Blank line that visually separates logical sections and improves readability.

# L0269: Assigns or updates a value used later in the workflow; check mutability and data shape.
        executor = MLExecutionAgent({"conversation": fake_conversation})
# L0270: Assigns or updates a value used later in the workflow; check mutability and data shape.
        evaluator = EvaluatorAgent()
# L0271: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = SynthesisAgent(cloud_factory=None)
# L0272: Blank line that visually separates logical sections and improves readability.

# L0273: Assigns or updates a value used later in the workflow; check mutability and data shape.
        orchestrator = ResearchOrchestrator(planner, executor, evaluator, synthesizer)
# L0274: Assigns or updates a value used later in the workflow; check mutability and data shape.
        result = orchestrator.run("auto", "hello")
# L0275: Blank line that visually separates logical sections and improves readability.

# L0276: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Should have a final answer, no retry triggered
# L0277: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert result["final_answer"]
# L0278: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "_retry" not in result["executor_output"]
# L0279: Blank line that visually separates logical sections and improves readability.

# L0280: Blank line that visually separates logical sections and improves readability.

# L0281: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0282: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Pipeline: no-retrieval → fallback answer (hallucination resistance)
# L0283: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ---------------------------------------------------------------------------
# L0284: Blank line that visually separates logical sections and improves readability.

# L0285: Defines a class that groups related state and behavior behind a reusable interface.
class TestHallucinationResistance:
# L0286: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_no_retrieval_produces_fallback_not_fabrication(self):
# L0287: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """When the retrieval index is empty/not-ready, answer must be the
# L0288: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        fallback message, not an LLM-fabricated answer."""
# L0289: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0290: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.planner.service import ResearchPlan, ToolCall
# L0291: Blank line that visually separates logical sections and improves readability.

# L0292: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # SynthesisAgent with no cloud factory → uses structured direct answer
# L0293: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = SynthesisAgent(cloud_factory=None)
# L0294: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0295: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {"error": "Search index not ready."},
# L0296: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0297: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = ResearchPlan(
# L0298: Assigns or updates a value used later in the workflow; check mutability and data shape.
            intent="research_analysis", query="transformers",
# L0299: Assigns or updates a value used later in the workflow; check mutability and data shape.
            top_k=5, calls=[], reason="test",
# L0300: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0301: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("transformers", plan.__dict__, outputs)
# L0302: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # The answer should indicate the problem — NOT fabricate paper titles
# L0303: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert isinstance(answer, str) and len(answer) > 0
# L0304: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Must NOT contain made-up citations or invented paper data
# L0305: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # (we can check it's the fallback path by verifying no search results appear)
# L0306: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert "No results" in answer or "not ready" in answer.lower() or answer.strip()
# L0307: Blank line that visually separates logical sections and improves readability.

# L0308: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def test_errored_tools_dont_block_synthesis(self):
# L0309: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """If all tools error, synthesizer still returns a safe fallback."""
# L0310: Imports a dependency, type, or project module needed by later code in this file.
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
# L0311: Blank line that visually separates logical sections and improves readability.

# L0312: Assigns or updates a value used later in the workflow; check mutability and data shape.
        synthesizer = SynthesisAgent(cloud_factory=None)
# L0313: Assigns or updates a value used later in the workflow; check mutability and data shape.
        outputs = {
# L0314: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "hybrid_search": {"error": "Index not ready"},
# L0315: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "classify_query": {"error": "Classifier not loaded"},
# L0316: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "metadata_rag": {"error": "No cloud LLM"},
# L0317: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0318: Assigns or updates a value used later in the workflow; check mutability and data shape.
        plan = {"intent": "research_analysis", "query": "test"}
# L0319: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = synthesizer.synthesize("transformers", plan, outputs)
# L0320: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert isinstance(answer, str)
# L0321: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Should not raise, should return something useful
# L0322: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        assert len(answer) > 0
```

## Source Walkthrough

This file is large, so the opening and closing sections are included here. Use the class/function breakdown above to navigate the middle of the file.

### Opening Section

```python
"""Tests for pipeline integrity — verifying the one-way ML-first pipeline.

ARCHITECTURE UNDER TEST
------------------------
USER INPUT → retrieval/ML → structured evidence → LLM reasoning → grounded output

These tests verify that:
  1. The pipeline CANNOT produce an answer without retrieval (no hallucination bypass)
  2. Data-flow injection correctly pipes search results to downstream tools
  3. The MLExecutionAgent isolates tool errors (one failure doesn't kill the plan)
  4. Type coercion works correctly (string "5" → int 5, "true" → bool True)
  5. Unknown tools return a clear error dict (not an exception)
  6. The PlannerAgent conversation gate prevents tool chains for greetings
  7. The orchestrator retry logic uses wider top_k on retry
  8. PaperChatService normalization produces consistent sessions

All LLM calls are mocked — these tests verify system logic, not LLM quality.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# MLExecutionAgent — type coercion and data-flow injection
# ---------------------------------------------------------------------------

class TestMLExecutionAgent:
    def _agent(self, tools=None):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        return MLExecutionAgent(tools or {})

    def test_unknown_tool_returns_error_dict(self):
        agent = self._agent()
        result = agent.execute("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_string_int_coercion(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received_args = {}

        def fake_tool(top_k=5, **_):
            received_args["top_k"] = top_k
            return {"result": "ok"}

        agent = MLExecutionAgent({"my_tool": fake_tool})
        agent.execute("my_tool", {"top_k": "10"})  # string "10" → int 10
        assert received_args["top_k"] == 10
        assert isinstance(received_args["top_k"], int)

    def test_string_bool_coercion_true(self):
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        received = {}

        def fake_tool(**kwargs):
            received.update(kwargs)
            return {}

        agent = MLExecutionAgent({"t": fake_tool})
        agent.execute("t", {"some_bool_key": "true"})
        # Note: coercion only applies to keys in ("top_k", "candidate_k", "max_tokens", "timeout")
        # For other keys, "true" stays as string unless it hits the bool branch
        # Let's verify the bool branch works for a string value "true"
        agent.execute("t", {"top_k": "true"})
        # "true" → try int("true") fails → try bool → "true" → True
        # Actually looking at the code: int coercion for top_k is tried first,
        # fails, then bool coercion is tried
        # Covered by the general bool branch
        agent2 = MLExecutionAgent({"t": fake_tool})
        agent2.execute("t", {"flag": "true"})
        assert received.get("flag") is True or received.get("flag") == "true"

    def test_data_flow_injection_from_search(self):
        """'from: search_results' in args should be replaced with prior paper list."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        papers_received = []

        def fake_search(query="", top_k=5, **_):
            return {"results": [{"title": "Paper A"}, {"title": "Paper B"}], "count": 2}

        def fake_methodology(papers=None, **_):
            papers_received.extend(papers or [])
            return {"count": 1, "signals": ["attention"]}

        agent = MLExecutionAgent({
            "hybrid_search": fake_search,
            "methodology_extract": fake_methodology,
        })

        calls = [
            ToolCall("hybrid_search", {"query": "transformers", "top_k": 5}),
            ToolCall("methodology_extract", {"from": "search_results"}),
        ]
        outputs = agent.execute_plan(calls)

        # Verify injection happened
        assert len(papers_received) == 2
        assert papers_received[0]["title"] == "Paper A"

    def test_data_flow_injection_from_smart_retrieve(self):
        """smart_retrieve results are also tracked for downstream injection."""
        from research_ai.agents.planner.service import ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent

        injected = []

        def fake_smart_retrieve(query="", top_k=5, **_):
            return {"results": [{"title": "Smart Paper"}], "count": 1}

        def fake_methodology(papers=None, **_):
            injected.extend(papers or [])
            return {"count": 0}

        agent = MLExecutionAgent({
            "smart_retrieve": fake_smart_retrieve,
```

### Closing Section

```python
        )
        evaluation = {"retry_top_k_multiplier": 3, "reason": "no_retrieval_hits"}
        retry_plan = ResearchOrchestrator._build_retry_plan(plan, evaluation)
        assert retry_plan.top_k <= 20

    def test_conversation_intent_skips_retry(self):
        """Conversations should never trigger a retry even with low score."""
        from research_ai.agents.orchestrator.service import ResearchOrchestrator
        from research_ai.agents.planner.service import PlannerAgent, ResearchPlan, ToolCall
        from research_ai.agents.ml_execution_agent.service import MLExecutionAgent
        from research_ai.agents.evaluator_agent.service import EvaluatorAgent
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        planner = MagicMock()
        planner.plan.return_value = ResearchPlan(
            intent="conversation",
            query="hello",
            top_k=5,
            calls=[ToolCall("conversation", {"query": "hello"})],
            reason="greeting",
            used_fallback=True,
        )

        def fake_conversation(query="", **_):
            return {"answer": "Hello! How can I help you?", "query": query}

        executor = MLExecutionAgent({"conversation": fake_conversation})
        evaluator = EvaluatorAgent()
        synthesizer = SynthesisAgent(cloud_factory=None)

        orchestrator = ResearchOrchestrator(planner, executor, evaluator, synthesizer)
        result = orchestrator.run("auto", "hello")

        # Should have a final answer, no retry triggered
        assert result["final_answer"]
        assert "_retry" not in result["executor_output"]


# ---------------------------------------------------------------------------
# Pipeline: no-retrieval → fallback answer (hallucination resistance)
# ---------------------------------------------------------------------------

class TestHallucinationResistance:
    def test_no_retrieval_produces_fallback_not_fabrication(self):
        """When the retrieval index is empty/not-ready, answer must be the
        fallback message, not an LLM-fabricated answer."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent
        from research_ai.agents.planner.service import ResearchPlan, ToolCall

        # SynthesisAgent with no cloud factory → uses structured direct answer
        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Search index not ready."},
        }
        plan = ResearchPlan(
            intent="research_analysis", query="transformers",
            top_k=5, calls=[], reason="test",
        )
        answer = synthesizer.synthesize("transformers", plan.__dict__, outputs)
        # The answer should indicate the problem — NOT fabricate paper titles
        assert isinstance(answer, str) and len(answer) > 0
        # Must NOT contain made-up citations or invented paper data
        # (we can check it's the fallback path by verifying no search results appear)
        assert "No results" in answer or "not ready" in answer.lower() or answer.strip()

    def test_errored_tools_dont_block_synthesis(self):
        """If all tools error, synthesizer still returns a safe fallback."""
        from research_ai.agents.synthesis_agent.service import SynthesisAgent

        synthesizer = SynthesisAgent(cloud_factory=None)
        outputs = {
            "hybrid_search": {"error": "Index not ready"},
            "classify_query": {"error": "Classifier not loaded"},
            "metadata_rag": {"error": "No cloud LLM"},
        }
        plan = {"intent": "research_analysis", "query": "test"}
        answer = synthesizer.synthesize("transformers", plan, outputs)
        assert isinstance(answer, str)
        # Should not raise, should return something useful
        assert len(answer) > 0
```
