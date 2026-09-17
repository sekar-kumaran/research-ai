from __future__ import annotations


class RankingService:
    """Rank papers using retrieval score plus metadata freshness/category signals."""

    def rank(self, papers: list[dict], preferred_category: str | None = None) -> list[dict]:
        ranked = []
        for paper in papers:
            score = float(paper.get("score", 0.0))
            category = str(paper.get("category", ""))
            year = str(paper.get("year", ""))
            category_bonus = 0.05 if preferred_category and preferred_category in category else 0.0
            recency_bonus = 0.0
            if year.isdigit():
                recency_bonus = max(0.0, min(0.04, (int(year) - 2018) * 0.005))
            item = dict(paper)
            item["ranking_score"] = round(score + category_bonus + recency_bonus, 4)
            ranked.append(item)
        return sorted(ranked, key=lambda item: item.get("ranking_score", item.get("score", 0)), reverse=True)

