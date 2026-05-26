"""Paper metadata analysis service.

Provides structural intelligence over arXiv paper metadata beyond what
the trend analysis and citation services expose — author disambiguation,
venue/category cross-referencing, abstract statistics, and metadata
quality scoring.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict


class MetadataService:
    """Analyse paper metadata to produce author, category, and quality signals.

    Designed to operate on the list-of-dicts format returned by retrieval
    so it can be wired directly into orchestrator tool calls.
    """

    # Minimum token count for an abstract to be considered informative
    MIN_ABSTRACT_WORDS = 30

    def analyse(self, papers: list[dict]) -> dict:
        """Full metadata analysis over a list of paper dicts."""
        if not papers:
            return {"error": "No papers provided for metadata analysis."}

        return {
            "paper_count": len(papers),
            "author_signals": self._author_signals(papers),
            "category_distribution": self._category_dist(papers),
            "year_distribution": self._year_dist(papers),
            "abstract_quality": self._abstract_quality(papers),
            "metadata_completeness": self._completeness(papers),
        }

    def author_network(self, papers: list[dict]) -> dict:
        """Derive author co-occurrence network from paper metadata."""
        coauthorships: dict[str, Counter] = defaultdict(Counter)
        author_paper_count: Counter = Counter()

        for paper in papers:
            authors = self._parse_authors(paper.get("authors", ""))
            for author in authors:
                author_paper_count[author] += 1
            for i, author_a in enumerate(authors):
                for author_b in authors[i + 1:]:
                    coauthorships[author_a][author_b] += 1
                    coauthorships[author_b][author_a] += 1

        top_authors = author_paper_count.most_common(10)
        return {
            "unique_authors": len(author_paper_count),
            "top_authors": [{"author": a, "papers": c} for a, c in top_authors],
            "coauthor_pairs": [
                {"author_a": a, "author_b": b, "shared_papers": count}
                for a, coauthors in coauthorships.items()
                for b, count in coauthors.most_common(3)
                if a < b
            ][:15],
        }

    def quality_score(self, paper: dict) -> float:
        """Compute a 0–1 metadata quality score for a single paper."""
        score = 0.0
        if paper.get("title", "").strip():
            score += 0.25
        abstract = paper.get("abstract", "")
        word_count = len(abstract.split())
        if word_count >= self.MIN_ABSTRACT_WORDS:
            score += 0.25
        if word_count >= 100:
            score += 0.15
        if paper.get("authors", "").strip():
            score += 0.15
        if paper.get("year", "").strip():
            score += 0.10
        if paper.get("category", "").strip():
            score += 0.10
        return round(min(score, 1.0), 3)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _author_signals(self, papers: list[dict]) -> dict:
        all_authors: list[str] = []
        for paper in papers:
            all_authors.extend(self._parse_authors(paper.get("authors", "")))
        counter = Counter(all_authors)
        return {
            "unique_authors": len(counter),
            "prolific_authors": [{"author": a, "papers": c} for a, c in counter.most_common(5)],
            "avg_authors_per_paper": round(len(all_authors) / max(1, len(papers)), 2),
        }

    @staticmethod
    def _category_dist(papers: list[dict]) -> list[dict]:
        counter: Counter = Counter()
        for paper in papers:
            for cat in str(paper.get("category", "")).split():
                counter[cat.lower()] += 1
        return [{"category": cat, "count": count} for cat, count in counter.most_common(10)]

    @staticmethod
    def _year_dist(papers: list[dict]) -> dict:
        years: Counter = Counter()
        for paper in papers:
            year = str(paper.get("year", "")).strip()
            if year.isdigit():
                years[year] += 1
        return dict(sorted(years.items()))

    def _abstract_quality(self, papers: list[dict]) -> dict:
        word_counts = [len(p.get("abstract", "").split()) for p in papers]
        if not word_counts:
            return {}
        below_min = sum(1 for wc in word_counts if wc < self.MIN_ABSTRACT_WORDS)
        return {
            "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts),
            "below_minimum_threshold": below_min,
            "quality_rate": round(1 - below_min / max(1, len(word_counts)), 3),
        }

    def _completeness(self, papers: list[dict]) -> dict:
        fields = ("title", "abstract", "authors", "year", "category", "paper_id")
        counts = {f: sum(1 for p in papers if p.get(f, "").strip()) for f in fields}
        total = len(papers)
        return {
            field: {"present": count, "rate": round(count / max(1, total), 3)}
            for field, count in counts.items()
        }

    @staticmethod
    def _parse_authors(raw: str) -> list[str]:
        """Split an author string into individual normalised names."""
        if not raw:
            return []
        # arXiv metadata uses commas or semicolons
        parts = re.split(r"[,;]", raw)
        return [p.strip() for p in parts if len(p.strip()) > 2]
