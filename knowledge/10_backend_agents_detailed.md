# Backend: Agents Detailed

## File Paths: `src/research_ai/agents/`
## Status: Active / Stable

## Description
This directory contains the autonomous agents that power the orchestration pipeline. Each agent is isolated in its own folder and provides a `service.py` file exposing a primary action (e.g., `.plan()`, `.execute()`).

## File Breakdown

### 1. `src/research_ai/agents/planner/service.py`
- **Class `PlannerAgent`**: Decides which tools to use.
- **Functions**:
  - `plan(query, session_id, top_k, conversation_history)`: Prompts the LLM with the available tool registry. Forces JSON output.
  - `_fallback_plan()`: Hardcoded logic that runs if the LLM fails to output valid JSON (e.g., defaults to `metadata_rag` if a session is active, or `hybrid_search` otherwise).

### 2. `src/research_ai/agents/ml_execution_agent/service.py`
- **Class `ExecutorAgent`**: Runs the tools requested by the Planner.
- **Functions**:
  - `execute(plan, tool_registry)`: Loops through the `plan["plan"]` array. Looks up the function in the registry and executes it sequentially.
  - `_execute_concurrent(plan, tool_registry)`: (Alternative) Runs independent tools using `asyncio.gather` for speed.

### 3. `src/research_ai/agents/evaluator_agent/service.py`
- **Class `EvaluatorAgent`**: Assesses the quality of the executed tools.
- **Functions**:
  - `evaluate(query, executor_outputs)`: Analyzes retrieval scores (e.g., from FAISS) and determines if the retrieved context is relevant enough to answer the user's query. Returns a `quality_score` (0.0 to 1.0).

### 4. `src/research_ai/agents/synthesis_agent/service.py`
- **Class `SynthesisAgent`**: Formats the final answer for the user.
- **Functions**:
  - `synthesize_structured()`: Main entry point. Returns a dictionary containing `answer`, `sources`, `confidence`, and `tools_used`.
  - `_extract_sources()`: Pulls out the top 8 papers from search results to display in UI citation cards.
  - `_structured_direct_answer()`: A deterministic fallback that formats raw search results into a numbered list if the LLM text generation fails or takes too long.

### 5. `src/research_ai/agents/orchestrator/service.py`
- **Class `Orchestrator`**: The manager that chains the agents together.
- **Functions**:
  - `run()`: Executes the full ReAct loop: `plan` -> `execute` -> `evaluate` (optional). Returns the raw dictionary of outputs which `platform.py` then passes to the `SynthesisAgent`.

### 6. `src/research_ai/agents/retrieval_agent/service.py`
- **Class `RetrievalAgent`**: (Legacy/Specialized). Often bypassed in favor of direct tool execution via `hybrid_search`, but contains logic for rewriting user queries to be more "search engine friendly" before querying FAISS.
