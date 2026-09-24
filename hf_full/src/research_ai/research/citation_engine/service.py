"""Citation intelligence engine derived from arXiv metadata.

Real citation graph edges are not present in the standard arXiv metadata dump,
so this engine derives proxy citation signals from category co-occurrence,
temporal proximity, title/abstract keyword overlap, and shared author tokens.
The interface is designed so that a future graph with actual citation edges
(e.g. Semantic Scholar) can be swapped in without touching the orchestrator.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class CitationEngine:
    """Derives citation-like signals from paper metadata.

    Exposes three main operations:
    - ``proxy_citations``: for each paper, finds candidate papers that are
      likely to share a citation relationship based on metadata signals.
    - ``co_citation_clusters``: groups papers into clusters by shared category
      and keyword overlap — a proxy for topic-based citation communities.
    - ``influence_timeline``: orders papers by year to approximate intellectual
      lineage within a result set.
    """

    def __init__(self) -> None:
        self._paper_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def proxy_citations(
        self, papers: list[dict], candidate_pool: list[dict] | None = None
    ) -> dict:
        """Find candidate related papers for each paper via metadata signals."""
        pool = candidate_pool or papers
        self._cache_papers(pool)

        relations: list[dict] = []
        for paper in papers:
            pid = paper.get("paper_id", "")
            candidates = self._find_related(paper, pool, exclude_self=pid)
            relations.append({
                "paper_id": pid,
                "title": paper.get("title", "")[:120],
                "proxy_related": candidates[:5],
                "signals_used": ["category_overlap", "keyword_overlap", "temporal_proximity"],
            })

        return {
            "citation_graph_mode": "proxy_metadata",
            "paper_count": len(papers),
            "relations": relations,
            "note": (
                "Real citation edges unavailable. Signals derived from category, "
                "keyword, and temporal proximity."
            ),
        }

    def co_citation_clusters(self, papers: list[dict]) -> dict:
        """Group papers into topic clusters by shared metadata signals."""
        if not papers:
            return {"clusters": [], "paper_count": 0}

        category_map: dict[str, list[dict]] = defaultdict(list)
        for paper in papers:
            primary = self._primary_category(paper)
            category_map[primary].append(paper)

        clusters = []
        for category, members in sorted(category_map.items(), key=lambda item: -len(item[1])):
            clusters.append({
                "category": category,
                "size": len(members),
                "papers": [
                    {"paper_id": p.get("paper_id", ""), "title": p.get("title", "")[:80]}
                    for p in members[:8]
                ],
                "shared_keywords": self._shared_keywords(members),
            })

        return {"clusters": clusters, "paper_count": len(papers), "cluster_count": len(clusters)}

    def influence_timeline(self, papers: list[dict]) -> dict:
        """Order papers by year to approximate intellectual lineage."""
        dated = []
        undated = []
        for paper in papers:
            year_str = str(paper.get("year", "")).strip()
            try:
                year = int(year_str)
                dated.append((year, paper))
            except ValueError:
                undated.append(paper)

        sorted_papers = [
            {
                "year": year,
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", "")[:100],
                "category": p.get("category", ""),
            }
            for year, p in sorted(dated)
        ]
        return {
            "timeline": sorted_papers,
            "undated_count": len(undated),
            "year_range": (
                f"{sorted_papers[0]['year']}–{sorted_papers[-1]['year']}"
                if len(sorted_papers) >= 2
                else str(sorted_papers[0]["year"]) if sorted_papers else "unknown"
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_papers(self, papers: list[dict]) -> None:
        for paper in papers:
            pid = paper.get("paper_id", "")
            if pid:
                self._paper_cache[pid] = paper

    def _find_related(
        self, paper: dict, pool: list[dict], exclude_self: str
    ) -> list[dict]:
        paper_cats = set(str(paper.get("category", "")).lower().split())
        paper_tokens = self._keyword_tokens(paper)
        paper_year = self._safe_year(paper)

        scored: list[tuple[float, dict]] = []
        for candidate in pool:
            cid = candidate.get("paper_id", "")
            if cid == exclude_self:
                continue
            score = 0.0
            # Category overlap
            cand_cats = set(str(candidate.get("category", "")).lower().split())
            cat_overlap = len(paper_cats & cand_cats) / max(1, len(paper_cats | cand_cats))
            score += cat_overlap * 0.5
            # Keyword overlap
            cand_tokens = self._keyword_tokens(candidate)
            kw_overlap = len(paper_tokens & cand_tokens) / max(1, len(paper_tokens | cand_tokens))
            score += kw_overlap * 0.4
            # Temporal proximity (closer in time → higher score)
            cand_year = self._safe_year(candidate)
            if paper_year and cand_year:
                distance = abs(paper_year - cand_year)
                score += max(0.0, 0.1 - distance * 0.01)

            if score > 0.05:
                scored.append((score, candidate))

        return [
            {
                "paper_id": c.get("paper_id", ""),
                "title": c.get("title", "")[:100],
                "score": round(s, 4),
            }
            for s, c in sorted(scored, key=lambda item: -item[0])
        ]

    @staticmethod
    def _primary_category(paper: dict) -> str:
        cats = str(paper.get("category", "unknown")).split()
        return cats[0] if cats else "unknown"

    @staticmethod
    def _keyword_tokens(paper: dict) -> set[str]:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        tokens = re.findall(r"\b[a-z]{4,}\b", text)
        stopwords = {"that", "with", "this", "from", "have", "using", "also", "been", "their",
                     "which", "they", "paper", "show", "results", "method", "approach", "model"}
        return {t for t in tokens if t not in stopwords}

    @staticmethod
    def _shared_keywords(papers: list[dict]) -> list[str]:
        if not papers:
            return []
        stopwords = {"that", "with", "this", "from", "have", "using", "also", "been",
                     "paper", "show", "results", "method", "approach", "model", "propose"}
        token_counts: dict[str, int] = defaultdict(int)
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            for token in set(re.findall(r"\b[a-z]{5,}\b", text)):
                if token not in stopwords:
                    token_counts[token] += 1
        threshold = max(1, len(papers) // 2)
        return sorted(
            [token for token, count in token_counts.items() if count >= threshold],
            key=lambda t: -token_counts[t],
        )[:8]

    @staticmethod
    def _safe_year(paper: dict) -> int | None:
        try:
            return int(str(paper.get("year", "")).strip())
        except ValueError:
            return None
