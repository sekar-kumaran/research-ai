from __future__ import annotations


class TrendAnalysisService:
    def analyze(self, papers: list[dict]) -> dict:
        by_year: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for paper in papers:
            year = str(paper.get("year", "")).strip() or "unknown"
            by_year[year] = by_year.get(year, 0) + 1
            for category in str(paper.get("category", "")).split():
                by_category[category] = by_category.get(category, 0) + 1
        return {
            "paper_count": len(papers),
            "by_year": dict(sorted(by_year.items())),
            "top_categories": dict(sorted(by_category.items(), key=lambda item: -item[1])[:10]),
        }

