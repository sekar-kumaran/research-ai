# Backend: Memory Detailed

## File Paths: `src/research_ai/memory/`
## Status: Active / Stable

## Description
This module handles all state retention between HTTP requests, enabling a conversational UX and context-aware research.

## File Breakdown

### 1. `src/research_ai/memory/conversation_store.py`
- **Class `ConversationStore`**: Bounded LRU cache (max 500 conversations).
- **Class `Conversation`**: Represents a single session.
- **Functions**:
  - `add(role, content)`: Pushes a message turn. Caps history at 20 pairs to save RAM.
  - `context_summary(last_n)`: Formats the history into a compact plaintext string so the `PlannerAgent` understands pronouns and context (e.g., "tell me more about the second paper").

### 2. `src/research_ai/memory/knowledge_graph/service.py`
- **Class `KnowledgeGraph`**: Tracks entities and concepts extracted during a session.
- **Functions**:
  - `ingest_query(query)`: Parses the user's prompt to extract key entities.
  - `get_graph()`: Returns nodes and edges representing the relationships between topics explored during the chat session. This feeds the UI Knowledge Graph visualizer.

### 3. `src/research_ai/memory/session_memory/service.py`
- **Class `SessionMemory`**: Handles uploaded documents (PDF/TXT).
- **Functions**:
  - `add_session()`: Creates a new `ChatSession` when a user uploads a file.
  - `_cleanup_loop()`: A background thread that enforces a 2-hour TTL (Time-To-Live). Automatically deletes uploaded papers that haven't been queried recently to prevent memory leaks and save storage.
