# Backend: Platform Core

## File Path: `src/research_ai/platform.py`
## Status: Active / Stable

## Description
The `ResearchAIPlatform` class acts as the Composition Root (Dependency Injection container) for the entire application. It instantiates the agents, services, and the remote HF proxy client, tying them all together into a unified registry.

## Core Responsibilities

### 1. Initialization
- Reads `Settings` (from `configs/settings.py`) to configure the system.
- Instantiates `RemoteMLClient` to connect to the Hugging Face ZeroGPU space.
- Initializes all specialized services (e.g., `RemoteHybridSearchService`, `RemoteClassifierService`).
- Instantiates the four core Agents: `PlannerAgent`, `ExecutorAgent`, `EvaluatorAgent`, `SynthesisAgent`.

### 2. The Tool Registry
- Defines a dictionary mapping string names (e.g., `"hybrid_search"`) to internal platform methods (e.g., `self._hybrid_search`).
- This registry is passed to the `PlannerAgent`, restricting the LLM to only invoking permitted, implemented capabilities.

### 3. The `chat()` Orchestration Loop
The `chat()` method is the heart of the system, implementing a ReAct-style (Plan -> Execute -> Evaluate -> Synthesize) pattern:
1. **Memory Retrieval**: Looks up `conversation_id` in `ConversationStore` to get the last N turns.
2. **Contextual Planning**: Passes the query and conversation history to the `PlannerAgent`, which returns a JSON plan (which tools to use).
3. **Execution**: The plan is routed to the `ExecutorAgent`, which sequentially or concurrently runs the requested tools from the registry.
4. **Synthesis**: The tool outputs are passed to the `SynthesisAgent` (specifically `_structured_direct_answer` or LLM generation), which extracts sources, assigns confidence, and formats the final answer.
5. **Memory Storage**: The final assistant answer is stored back into the `ConversationStore`.
6. **Return**: Yields a structured dictionary (`answer`, `sources`, `conversation_id`) consumed by `main.py` for streaming.
