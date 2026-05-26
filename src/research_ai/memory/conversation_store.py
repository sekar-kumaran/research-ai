"""ConversationStore — persistent multi-turn conversation memory.

WHY THIS EXISTS
---------------
Without conversation memory, every message the user sends is treated as
independent. The orchestrator cannot understand follow-up questions like:
  "Tell me more about that last paper"
  "Which of these methods is fastest?"
  "Compare that to what you just said"

With conversation memory, the planner agent receives recent turns as context,
allowing it to resolve references, chain related queries, and maintain coherent
multi-turn research dialogues — exactly like ChatGPT's conversation model.

ARCHITECTURE
------------
Each browser session gets a `conversation_id` (UUID). Messages accumulate as
a chronological list of {role, content} turns. The planner receives a compact
text summary of recent turns rather than the full JSON — this keeps the LLM
prompt within token limits while preserving semantic context.

MEMORY BOUNDS
-------------
  - Maximum 20 turns per conversation (40 messages: 20 user + 20 assistant)
  - Maximum 500 concurrent conversations (LRU eviction)
  - In-memory only (cleared on server restart)

For production deployments: replace the OrderedDict store with Redis or
PostgreSQL to survive restarts and scale horizontally.

LRU EVICTION
------------
Uses OrderedDict (same pattern as the embedding cache in retrieval/embeddings/)
to provide O(1) move-to-end on access and O(1) popitem from the left for
eviction. This means memory usage is bounded at O(_MAX_CONVERSATIONS) regardless
of server uptime.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

# Maximum number of (user + assistant) message pairs to keep per conversation.
# Older pairs are silently dropped. 20 pairs = 40 total messages.
_MAX_TURN_PAIRS = 20

# Maximum concurrent conversations before LRU eviction starts.
_MAX_CONVERSATIONS = 500


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single message in a conversation."""
    role: str       # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=_utcnow)


@dataclass
class Conversation:
    """A stateful multi-turn conversation session.

    Tracks all turns chronologically. Provides helpers to:
      - Format turns for OpenAI-style message lists (to_messages)
      - Build a compact context string for the planner prompt (context_summary)
    """
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow)
    last_active: str = field(default_factory=_utcnow)

    def add(self, role: str, content: str) -> None:
        """Append a turn, evicting oldest pairs if over the limit."""
        self.turns.append(Turn(role=role, content=content))
        self.last_active = _utcnow()

        # Enforce the turn limit: keep only the most recent _MAX_TURN_PAIRS pairs
        # This means we keep at most _MAX_TURN_PAIRS * 2 messages total.
        max_messages = _MAX_TURN_PAIRS * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]

    def to_messages(self, last_n_pairs: int = 10) -> list[dict]:
        """Return the last N turn-pairs as OpenAI-style message dicts.

        Used to pass conversation history directly to an LLM API call.
        Format: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        """
        recent = self.turns[-(last_n_pairs * 2):]
        return [{"role": t.role, "content": t.content} for t in recent]

    def context_summary(self, last_n_pairs: int = 6) -> str:
        """Build a compact plain-text context block for the planner prompt.

        Returns an empty string if there's no history (first message in session).
        Assistant responses are truncated to 300 chars to bound prompt size.

        Example output:
            User: What are the best GNN papers for drug discovery?
            Assistant: I found 8 relevant papers. The most cited is...
            User: Which of those use attention mechanisms?
        """
        recent = self.turns[-(last_n_pairs * 2):]
        if not recent:
            return ""

        lines: list[str] = []
        for turn in recent:
            label = "User" if turn.role == "user" else "Assistant"
            # Truncate long assistant responses to prevent context bloat
            content = turn.content
            if turn.role == "assistant" and len(content) > 300:
                content = content[:297] + "…"
            lines.append(f"{label}: {content}")

        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_user_query(self) -> str | None:
        """Return the most recent user message, or None if no turns yet."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ConversationStore:
    """LRU-bounded in-memory store for multi-turn conversations.

    Thread-safety: This implementation is NOT thread-safe. FastAPI's async
    model means event-loop concurrency is fine (no two requests for the same
    conversation_id run simultaneously on one event loop), but for multi-worker
    deployments use Redis as the backing store instead.

    Usage:
        store = ConversationStore()

        # Start or resume a conversation
        cid, conv = store.get_or_create(conversation_id=request.conversation_id)

        # Add user message before orchestration
        conv.add("user", query)

        # Add assistant response after orchestration
        conv.add("assistant", final_answer)

        # Get context for the planner
        context = conv.context_summary(last_n_pairs=6)
    """

    def __init__(self) -> None:
        # OrderedDict gives O(1) move-to-end (LRU update) and O(1) popitem (eviction)
        self._store: OrderedDict[str, Conversation] = OrderedDict()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def create(self) -> str:
        """Create a new conversation and return its ID."""
        cid = str(uuid4())
        self._store[cid] = Conversation(conversation_id=cid)
        self._evict_if_needed()
        return cid

    def get(self, conversation_id: str) -> Conversation | None:
        """Retrieve a conversation by ID, updating its LRU position."""
        if conversation_id not in self._store:
            return None
        self._store.move_to_end(conversation_id)
        return self._store[conversation_id]

    def get_or_create(self, conversation_id: str | None) -> tuple[str, Conversation]:
        """Return (id, conversation), creating a new one if ID is unknown or None.

        Always returns a valid (id, Conversation) pair. Call this at the start
        of every /chat/message request.
        """
        if conversation_id and conversation_id in self._store:
            self._store.move_to_end(conversation_id)
            return conversation_id, self._store[conversation_id]

        # Create new conversation
        cid = str(uuid4())
        conv = Conversation(conversation_id=cid)
        self._store[cid] = conv
        self._evict_if_needed()
        return cid, conv

    def add_turn(self, conversation_id: str, role: str, content: str) -> bool:
        """Add a turn to an existing conversation. Returns False if not found."""
        conv = self.get(conversation_id)
        if conv is None:
            return False
        conv.add(role, content)
        return True

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        if conversation_id in self._store:
            del self._store[conversation_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Properties and introspection
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._store)

    def summary(self) -> dict:
        """Return store statistics for the /stats endpoint."""
        return {
            "active_conversations": self.count,
            "max_conversations": _MAX_CONVERSATIONS,
            "max_turns_per_conversation": _MAX_TURN_PAIRS,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest conversations when the store is over capacity.

        OrderedDict.popitem(last=False) removes from the front (oldest),
        making this O(1) per eviction rather than O(n) with a list.
        """
        while len(self._store) > _MAX_CONVERSATIONS:
            oldest_id, _ = self._store.popitem(last=False)
            logger.debug("ConversationStore: evicted oldest conversation %s", oldest_id)
