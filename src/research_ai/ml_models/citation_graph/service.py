from __future__ import annotations


class CitationGraphService:
    """Citation intelligence facade.

    The current arXiv metadata artifacts do not include reference edges, so this
    service exposes citation-aware contracts and derives local related-paper
    signals from category/year/title metadata until a citation graph is trained.
    """

    def related_signals(self, papers: list[dict]) -> dict:
        categories: dict[str, int] = {}
        years: dict[str, int] = {}
        for paper in papers:
            for category in str(paper.get("category", "")).split():
                categories[category] = categories.get(category, 0) + 1
            year = str(paper.get("year", "")).strip()
            if year:
                years[year] = years.get(year, 0) + 1
        return {
            "citation_graph_available": False,
            "category_cooccurrence": sorted(categories.items(), key=lambda item: -item[1])[:10],
            "year_distribution": sorted(years.items()),
        }

