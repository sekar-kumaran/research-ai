"""Retrieval-specialized agent that selects and executes the optimal retrieval strategy."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    mode: str          # "semantic" | "hybrid" | "filtered" | "citation_aware"
    query: str
    top_k: int
    filters: dict
    expand_query: bool


class RetrievalAgent:
    """Intelligent retrieval specialist that selects strategy based on query analysis.

    Sits between the Orchestrator and the HybridSearchService. Analyses the
    query to determine whether to use pure-semantic, hybrid, or metadata-filtered
    retrieval, and whether the query should be expanded with domain terminology
    before embedding. Falls back to hybrid search on any error.
    """

    # Domain expansions for sparse queries
    DOMAIN_EXPANSIONS: dict[str, list[str]] = {
        "llm": ["large language model", "transformer", "gpt", "attention"],
        "rl": ["reinforcement learning", "reward", "policy", "agent"],
        "cv": ["computer vision", "image", "convolutional", "object detection"],
        "nlp": ["natural language processing", "text", "bert", "language model"],
        "gnn": ["graph neural network", "node embedding", "message passing"],
        "diffusion": ["diffusion model", "denoising", "score matching", "generative"],
    }

    def __init__(self, search_service) -> None:
        self.search_service = search_service

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        strategy_hint: str = "auto",
    ) -> dict:
        """Execute the optimal retrieval strategy for the given query."""
        if not self.search_service.ready:
            return {"error": "Search index not ready. Build similarity artifacts first."}

        strategy = self._select_strategy(query, top_k, filters or {}, strategy_hint)
        effective_query = self._expand_query(strategy)

        logger.info(
            "RetrievalAgent: mode=%s expand=%s top_k=%d query='%s'",
            strategy.mode,
            strategy.expand_query,
            strategy.top_k,
            effective_query[:60],
        )

        result = self.search_service.search(
            effective_query,
            top_k=strategy.top_k,
            filters=strategy.filters if strategy.mode == "filtered" else None,
            candidate_k=min(strategy.top_k * 6, 80),
        )
        result["retrieval_strategy"] = strategy.mode
        result["query_expanded"] = effective_query != query
        if strategy.expand_query:
            result["original_query"] = query
        return result

    def _select_strategy(
        self, query: str, top_k: int, filters: dict, hint: str
    ) -> RetrievalStrategy:
        q_lower = query.lower()
        words = q_lower.split()

        # Explicit filters present → metadata-filtered retrieval
        if filters:
            return RetrievalStrategy(
                mode="filtered",
                query=query,
                top_k=top_k,
                filters=filters,
                expand_query=False,
            )

        # Very short queries (1–2 words) → expand before embedding
        expand = len(words) <= 2 and any(w in self.DOMAIN_EXPANSIONS for w in words)

        # Citation-aware: user asks about references or related work
        if any(token in q_lower for token in ("cited by", "references", "related work", "influenced")):
            return RetrievalStrategy(
                mode="citation_aware",
                query=query,
                top_k=min(top_k * 2, 20),
                filters={},
                expand_query=expand,
            )

        # Default to hybrid
        return RetrievalStrategy(
            mode="hybrid",
            query=query,
            top_k=top_k,
            filters={},
            expand_query=expand,
        )

    def _expand_query(self, strategy: RetrievalStrategy) -> str:
        if not strategy.expand_query:
            return strategy.query
        words = strategy.query.lower().split()
        expansions: list[str] = []
        for word in words:
            expansions.extend(self.DOMAIN_EXPANSIONS.get(word, []))
        if expansions:
            return f"{strategy.query} {' '.join(expansions[:8])}"
        return strategy.query
