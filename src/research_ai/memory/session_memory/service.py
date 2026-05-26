from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatSession:
    session_id: str
    source: str
    chunks: list[str]
    index: object
    history: list[dict[str, str]] = field(default_factory=list)
    title: str = ""
    metadata: dict = field(default_factory=dict)


class SessionMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}
        self.source_to_session: dict[str, str] = {}

    def get(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Load a paper first.")
        return self.sessions[session_id]

    def put(self, session: ChatSession) -> None:
        self.sessions[session.session_id] = session
        self.source_to_session[session.source] = session.session_id

    def find_source(self, source: str) -> ChatSession | None:
        session_id = self.source_to_session.get(source)
        if not session_id:
            return None
        return self.sessions.get(session_id)

