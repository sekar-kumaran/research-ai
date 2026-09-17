# Backend: Memory and State

## File Paths: `src/research_ai/memory/`
## Status: Active / Stable

## Description
Since HTTP is stateless, the backend must maintain conversational context in memory so the Agent can resolve pronouns (e.g., "Tell me more about *that* paper").

## 1. ConversationStore (`conversation_store.py`)
- An LRU (Least Recently Used) cache built on Python's `collections.OrderedDict`.
- **Concurrency**: Bounded to a maximum of 500 concurrent conversations to prevent memory leaks on long-running servers.
- **Turn Limits**: Bounded to a maximum of 20 turns (40 messages) per conversation. Older messages are silently dropped to keep the LLM context window small.
- **Planner Integration**: Provides a `context_summary()` method that compresses the JSON message history into a compact plaintext block for the `PlannerAgent`.

## 2. SessionMemory (`session_memory/service.py`)
- Used for tracking uploaded documents (e.g., PDFs or arXiv papers).
- When a user uploads a file, it creates a `ChatSession` containing the extracted text chunks.
- Maps a `session_id` to the `ChatSession` object.
- **TTL Eviction**: Implements a 2-hour Time-To-Live (TTL). A background task (`_cleanup_loop`) runs periodically to delete uploaded papers that haven't been accessed recently. This fulfills the requirement to automatically clean up temporary downloaded papers to save storage/RAM.
