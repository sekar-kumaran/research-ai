"""In-memory knowledge graph for tracking research concepts across sessions."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    name: str
    frequency: int = 0
    sessions: set[str] = field(default_factory=set)
    related: set[str] = field(default_factory=set)
    category_votes: dict[str, int] = field(default_factory=dict)


class KnowledgeGraph:
    """Session-scoped knowledge graph that tracks concepts, entities, and relations.

    Ingests paper metadata and chat turns to build a lightweight concept network.
    All data is in-memory (per-process) and resets on server restart. The graph
    provides concept frequency, co-occurrence, and session-linkage signals that
    the orchestrator can use for context-aware retrieval and synthesis.

    Designed to be replaced by a persistent graph database (Neo4j, NetworkX on
    disk) without changing the service interface.
    """

    # Regex patterns for extracting domain concepts from scientific text
    CONCEPT_PATTERNS = (
        r"\b(?:transformer|bert|gpt|llm|diffusion model|gnn|cnn|rnn|lstm|vae|gan)\b",
        r"\b(?:reinforcement learning|supervised learning|self-supervised|contrastive)\b",
        r"\b(?:attention mechanism|multi-head attention|positional encoding)\b",
        r"\b(?:fine-tun(?:ing|e)|pre-train(?:ing|ed)|zero-shot|few-shot)\b",
        r"\b(?:benchmark|dataset|evaluation|ablation|baseline)\b",
    )

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self._session_concepts: dict[str, set[str]] = defaultdict(set)
        self._query_log: list[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_papers(self, papers: list[dict], session_id: str = "") -> None:
        """Extract and register concepts from paper metadata."""
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            concepts = self._extract_concepts(text)
            for concept in concepts:
                self._register(concept, session_id)
            # Category → concept links
            for category in str(paper.get("category", "")).split():
                self._register(category.lower(), session_id, is_category=True)

    def ingest_query(self, query: str, session_id: str = "") -> None:
        """Register a user query as a concept signal."""
        self._query_log.append(query)
        for concept in self._extract_concepts(query):
            self._register(concept, session_id)

    def link_session(self, session_id: str, paper_id: str, title: str) -> None:
        """Explicitly link a session to a paper concept."""
        key = self._normalize(title[:60])
        if key:
            self._register(key, session_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def top_concepts(self, n: int = 10) -> list[dict]:
        """Return the most frequently encountered concepts."""
        ranked = sorted(self.concepts.values(), key=lambda c: c.frequency, reverse=True)
        return [
            {
                "concept": c.name,
                "frequency": c.frequency,
                "session_count": len(c.sessions),
                "related": sorted(c.related)[:5],
            }
            for c in ranked[:n]
        ]

    def concepts_for_session(self, session_id: str) -> list[str]:
        """Return concepts encountered in a specific session."""
        return sorted(self._session_concepts.get(session_id, set()))

    def related_concepts(self, concept: str, n: int = 5) -> list[str]:
        """Return concepts that co-occurred with the given concept."""
        key = self._normalize(concept)
        node = self.concepts.get(key)
        if node is None:
            return []
        return sorted(node.related, key=lambda c: self.concepts.get(c, Concept(c)).frequency, reverse=True)[:n]

    def summary(self) -> dict:
        return {
            "total_concepts": len(self.concepts),
            "total_sessions_tracked": len(self._session_concepts),
            "total_queries_logged": len(self._query_log),
            "top_concepts": self.top_concepts(5),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _register(self, name: str, session_id: str, is_category: bool = False) -> None:
        key = self._normalize(name)
        if not key:
            return
        if key not in self.concepts:
            self.concepts[key] = Concept(name=key)
        node = self.concepts[key]
        node.frequency += 1
        if session_id:
            node.sessions.add(session_id)
            self._session_concepts[session_id].add(key)
            # Co-occurrence edges: link to other concepts already in this session
            for other in self._session_concepts[session_id]:
                if other != key:
                    node.related.add(other)
                    self.concepts[other].related.add(key)

    def _extract_concepts(self, text: str) -> list[str]:
        found: list[str] = []
        lower = text.lower()
        for pattern in self.CONCEPT_PATTERNS:
            for match in re.finditer(pattern, lower):
                token = match.group(0).strip()
                if token:
                    found.append(token)
        return found

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "_", text.strip().lower())[:80]
