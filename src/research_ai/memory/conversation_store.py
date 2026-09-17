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

try:
    from sqlalchemy import select
    from research_ai.database.engine import SessionLocal
    from research_ai.database.models import ConversationRecord, MessageRecord
except Exception:  # pragma: no cover - keeps the store usable in constrained tests
    SessionLocal = None
    ConversationRecord = None
    MessageRecord = None
    select = None

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
    user_id: str | None = None
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

    def __init__(self, persistent: bool = True) -> None:
        # OrderedDict gives O(1) move-to-end (LRU update) and O(1) popitem (eviction)
        self._store: OrderedDict[str, Conversation] = OrderedDict()
        self.persistent = persistent and SessionLocal is not None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def create(self, user_id: str | None = None) -> str:
        """Create a new conversation and return its ID."""
        cid = str(uuid4())
        self._store[cid] = Conversation(conversation_id=cid, user_id=user_id)
        if self.persistent:
            with SessionLocal() as db:
                db.add(ConversationRecord(id=cid, title="New research chat", user_id=user_id))
                db.commit()
        self._evict_if_needed()
        return cid

    def get(self, conversation_id: str) -> Conversation | None:
        """Retrieve a conversation by ID, updating its LRU position."""
        if conversation_id not in self._store:
            loaded = self._load_from_db(conversation_id)
            if loaded is None:
                return None
            self._store[conversation_id] = loaded
        self._store.move_to_end(conversation_id)
        return self._store[conversation_id]

    def get_or_create(self, conversation_id: str | None, user_id: str | None = None) -> tuple[str, Conversation]:
        """Return (id, conversation), creating a new one if ID is unknown or None.

        Always returns a valid (id, Conversation) pair. Call this at the start
        of every /chat/message request. Pass user_id to associate the conversation
        with the authenticated user so list(user_id=...) works correctly.
        """
        if conversation_id and conversation_id in self._store:
            self._store.move_to_end(conversation_id)
            return conversation_id, self._store[conversation_id]
        if conversation_id:
            loaded = self._load_from_db(conversation_id)
            if loaded is not None:
                self._store[conversation_id] = loaded
                self._store.move_to_end(conversation_id)
                return conversation_id, loaded

        # Create new conversation, associating with user if provided
        cid = str(uuid4())
        conv = Conversation(conversation_id=cid, user_id=user_id)
        self._store[cid] = conv
        if self.persistent:
            with SessionLocal() as db:
                db.add(ConversationRecord(id=cid, title="New research chat", user_id=user_id))
                db.commit()
        self._evict_if_needed()
        return cid, conv

    def add_turn(self, conversation_id: str, role: str, content: str) -> bool:
        """Add a turn to an existing conversation. Returns False if not found."""
        conv = self.get(conversation_id)
        if conv is None:
            return False
        conv.add(role, content)
        self._persist_turn(conversation_id, role, content)
        return True

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        if conversation_id in self._store:
            del self._store[conversation_id]
            deleted = True
        else:
            deleted = False
        if self.persistent:
            with SessionLocal() as db:
                record = db.get(ConversationRecord, conversation_id)
                if record is not None:
                    db.delete(record)
                    db.commit()
                    deleted = True
        return deleted

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

    def list(self, search: str = "", limit: int = 50, user_id: str | None = None) -> list[dict]:
        """Return recent conversations from the persistent store when available.

        When user_id is supplied, only that user's conversations are returned.
        This is enforced both in the DB query and in the in-memory fallback.
        """
        if self.persistent:
            with SessionLocal() as db:
                stmt = select(ConversationRecord).order_by(ConversationRecord.updated_at.desc()).limit(limit)
                if user_id:
                    stmt = stmt.where(ConversationRecord.user_id == user_id)
                records = db.scalars(stmt).all()
                out = []
                for record in records:
                    if search and search.lower() not in (record.title or "").lower():
                        continue
                    out.append(
                        {
                            "conversation_id": record.id,
                            "title": record.title,
                            "created_at": record.created_at.isoformat(),
                            "last_active": record.updated_at.isoformat(),
                            "message_count": len(record.messages),
                        }
                    )
                return out
        # In-memory fallback: filter by user_id stored on the Conversation object
        return [
            {
                "conversation_id": cid,
                "title": self._title_from(conv),
                "created_at": conv.created_at,
                "last_active": conv.last_active,
                "message_count": conv.turn_count,
            }
            for cid, conv in list(self._store.items())[-limit:]
            if not user_id or getattr(conv, "user_id", None) == user_id
        ]

    def rename(self, conversation_id: str, title: str) -> bool:
        title = title.strip()[:200]
        if not title:
            return False
        if self.persistent:
            with SessionLocal() as db:
                record = db.get(ConversationRecord, conversation_id)
                if record is None:
                    return False
                record.title = title
                db.commit()
                return True
        return conversation_id in self._store

    def clear(self, conversation_id: str) -> bool:
        conv = self.get(conversation_id)
        if conv is None:
            return False
        conv.turns.clear()
        if self.persistent:
            with SessionLocal() as db:
                record = db.get(ConversationRecord, conversation_id)
                if record is not None:
                    for message in list(record.messages):
                        db.delete(message)
                    db.commit()
        return True

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

    def _persist_turn(self, conversation_id: str, role: str, content: str) -> None:
        if not self.persistent:
            return
        with SessionLocal() as db:
            record = db.get(ConversationRecord, conversation_id)
            if record is None:
                record = ConversationRecord(id=conversation_id, title=self._derive_title(content) if role == "user" else "New research chat")
                db.add(record)
                db.flush()
            if role == "user" and record.title == "New research chat":
                record.title = self._derive_title(content)
            db.add(MessageRecord(conversation_id=conversation_id, role=role, content=content))
            db.commit()

    def _load_from_db(self, conversation_id: str) -> Conversation | None:
        if not self.persistent:
            return None
        with SessionLocal() as db:
            record = db.get(ConversationRecord, conversation_id)
            if record is None:
                return None
            conv = Conversation(
                conversation_id=record.id,
                user_id=record.user_id,
                created_at=record.created_at.isoformat(),
                last_active=record.updated_at.isoformat(),
            )
            for message in record.messages:
                conv.turns.append(
                    Turn(role=message.role, content=message.content, timestamp=message.created_at.isoformat())
                )
            return conv

    def owner_id(self, conversation_id: str) -> str | None:
        conv = self.get(conversation_id)
        if conv is not None:
            return conv.user_id
        return None

    @staticmethod
    def _derive_title(content: str) -> str:
        title = " ".join(content.split())[:80]
        return title or "New research chat"

    @staticmethod
    def _title_from(conv: Conversation) -> str:
        return ConversationStore._derive_title(conv.last_user_query or "New research chat")
