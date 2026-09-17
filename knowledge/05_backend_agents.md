# Backend: Orchestration Agents

## File Paths: `src/research_ai/agents/`
## Status: Active / Stable

## Description
The agent system replaces traditional hardcoded logic trees with LLM-driven decision making. It uses a swarm-style separation of concerns to maintain stability and prevent hallucinations.

## 1. Planner Agent (`planner/service.py`)
- **Role**: The Brain. Takes the user's query and conversation history and decides *what* to do.
- **Input**: User Query, Chat History, Tool Descriptions.
- **Output**: JSON payload defining an `intent` (e.g., `research_analysis`, `casual_chat`) and a `plan` array of tools to invoke.
- **Logic**: It forces the LLM to output rigid JSON. If the LLM fails (e.g., outputs malformed JSON), it catches the exception and falls back to a deterministic heuristic (e.g., if it sees a `session_id`, it forces `metadata_rag`).

## 2. Executor Agent (`executor/service.py`)
- **Role**: The Muscle. Blindly executes the plan created by the Planner.
- **Logic**: Iterates over the `plan` array. Looks up the requested tool in the `tool_registry` dict passed from `platform.py`. 
- **Error Handling**: Wraps each tool execution in a `try/except`. If a tool crashes (e.g., Hugging Face Space timeout), it logs the error and injects `{"error": "..."}` into the output state, allowing the pipeline to continue.

## 3. Evaluator Agent (`evaluator/service.py`)
- **Role**: The Critic. (Currently minimally used).
- **Logic**: Looks at the tool outputs (e.g., the FAISS search scores) and assigns a `quality_score` between 0.0 and 1.0 representing how confident the system is in the retrieved evidence.

## 4. Synthesis Agent (`synthesis_agent/service.py`)
- **Role**: The Voice. Turns raw JSON tool outputs into a human-readable response.
- **Logic**: 
  - `_extract_sources()`: Pulls out `title`, `abstract`, and `arxiv_url` from search payloads to build citation cards for the UI.
  - `_structured_direct_answer()`: A deterministic fallback method. If the search returns empty results, it returns a hardcoded "I searched but found nothing" message. If classification is used, it outputs the category.
  - LLM Generation: If local/cloud LLMs are available, it passes the retrieved JSON text to the LLM to write a fluid, narrative answer citing the papers.
