"""Session memory for paper chat sessions.

Papers downloaded (PDF, arXiv) are stored in memory as ChatSession objects.
Each session has a TTL (time-to-live). Sessions not accessed for more than
SESSION_TTL_HOURS are automatically evicted on the next put() or evict_expired()
call to prevent unbounded memory growth on a long-running Render server.

Storage: in-process memory only. Sessions are lost on server restart.
This is intentional — papers should be re-loaded if needed, keeping the
server lightweight and stateless.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sessions unused for this many hours will be evicted
SESSION_TTL_HOURS = 2
SESSION_TTL_SECONDS = SESSION_TTL_HOURS * 3600


@dataclass
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    metadata: dict = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last_accessed timestamp on every use."""
        self.last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_accessed

    @property
    def expired(self) -> bool:
        return self.age_seconds > SESSION_TTL_SECONDS


class SessionMemory:
    """In-memory store for paper chat sessions with TTL-based eviction.
    
    Sessions expire after SESSION_TTL_HOURS of inactivity. This prevents
    memory leaks on long-running servers while keeping recently used papers
    instantly available.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. The paper session may have expired (TTL={SESSION_TTL_HOURS}h). Please reload the paper.")
        session = self.sessions[session_id]
        if session.expired:
            self._evict(session_id)
            raise KeyError(f"Session '{session_id}' has expired after {SESSION_TTL_HOURS}h of inactivity. Please reload the paper.")
        session.touch()
        return session

    def put(self, session: ChatSession) -> None:
        session.touch()
        self.sessions[session.session_id] = session
        self.source_to_session[session.source] = session.session_id
        # Opportunistic cleanup of expired sessions
        self.evict_expired()

    def find_source(self, source: str) -> ChatSession | None:
        session_id = self.source_to_session.get(source)
        if not session_id:
            return None
        session = self.sessions.get(session_id)
        if session and session.expired:
            self._evict(session_id)
            return None
        if session:
            session.touch()
        return session

    def evict_expired(self) -> int:
        """Remove all expired sessions. Returns count of evicted sessions."""
        expired_ids = [sid for sid, s in self.sessions.items() if s.expired]
        for sid in expired_ids:
            self._evict(sid)
        if expired_ids:
            logger.info("SessionMemory: evicted %d expired session(s)", len(expired_ids))
        return len(expired_ids)

    def _evict(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            self.source_to_session.pop(session.source, None)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self.sessions.values() if not s.expired)
