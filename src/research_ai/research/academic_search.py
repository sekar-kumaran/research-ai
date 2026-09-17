from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)


@dataclass
class AcademicSearchSettings:
    arxiv_enabled: bool = True
    crossref_enabled: bool = True
    openalex_enabled: bool = True
    timeout_seconds: int = 12


class AcademicSearchService:
    """Search public academic metadata APIs and normalize paper records.

    This service never fabricates citations. If all external APIs fail it returns
    an empty result set plus structured provider errors for debug/readiness use.
    """

    def __init__(self, settings: AcademicSearchSettings | None = None) -> None:
        self.settings = settings or AcademicSearchSettings()

    def search(self, query: str, top_k: int = 8) -> dict:
        started = time.perf_counter()
        providers: list[tuple[str, object]] = []
        if self.settings.arxiv_enabled:
            providers.append(("arxiv", self._search_arxiv))
        if self.settings.openalex_enabled:
            providers.append(("openalex", self._search_openalex))
        if self.settings.crossref_enabled:
            providers.append(("crossref", self._search_crossref))

        papers: list[dict] = []
        errors: list[dict] = []
        per_provider = max(top_k * 3, 10)
        for name, fn in providers:
            try:
                found = fn(query, per_provider)
                papers.extend(found)
            except Exception as exc:
                logger.warning("[ACADEMIC_SEARCH] provider=%s failed: %s", name, exc)
                errors.append({"provider": name, "error": str(exc)})

        ranked = self.rank(query, self.deduplicate(papers))[:top_k]
        return {
            "query": query,
            "retrieval_strategy": "academic_apis_arxiv_openalex_crossref",
            "results": ranked,
            "count": len(ranked),
            "candidate_count": len(papers),
            "provider_errors": errors,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def deduplicate(papers: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for paper in papers:
            key = (
                str(paper.get("doi") or paper.get("arxiv_id") or paper.get("paper_id") or "").lower().strip()
                or re.sub(r"\W+", "", str(paper.get("title", "")).lower())[:120]
            )
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(paper)
        return out

    @staticmethod
    def rank(query: str, papers: list[dict]) -> list[dict]:
        q_terms = set(_tokens(query))
        current_year = 2026
        ranked: list[dict] = []
        for paper in papers:
            title = str(paper.get("title", ""))
            abstract = str(paper.get("abstract", ""))
            haystack_terms = set(_tokens(f"{title} {abstract}"))
            overlap = len(q_terms & haystack_terms) / max(len(q_terms), 1)
            try:
                year = int(str(paper.get("year") or "0")[:4])
            except ValueError:
                year = 0
            recency = max(0.0, 1.0 - max(current_year - year, 0) / 12.0) if year else 0.0
            cited_by = float(paper.get("citation_count") or 0)
            citation_signal = min(cited_by / 500.0, 1.0)
            reliability = 1.0 if paper.get("source") in {"arxiv", "openalex", "crossref"} else 0.5
            quality = 1.0 if abstract else 0.65
            score = 0.45 * overlap + 0.20 * recency + 0.15 * citation_signal + 0.10 * reliability + 0.10 * quality
            item = dict(paper)
            item["score"] = round(score, 4)
            ranked.append(item)
        return sorted(ranked, key=lambda p: p.get("score", 0), reverse=True)

    def _search_arxiv(self, query: str, limit: int) -> list[dict]:
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query=all:{quote_plus(query)}&start=0&max_results={min(limit, 50)}"
            "&sortBy=relevance&sortOrder=descending"
        )
        text = self._get(url).text
        root = ET.fromstring(text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers: list[dict] = []
        for entry in root.findall("a:entry", ns):
            link = entry.findtext("a:id", default="", namespaces=ns).strip()
            arxiv_id = link.rstrip("/").split("/")[-1]
            title = _clean(entry.findtext("a:title", default="", namespaces=ns))
            abstract = _clean(entry.findtext("a:summary", default="", namespaces=ns))
            published = entry.findtext("a:published", default="", namespaces=ns)
            authors = ", ".join(
                _clean(author.findtext("a:name", default="", namespaces=ns))
                for author in entry.findall("a:author", ns)
            )
            categories = [c.attrib.get("term", "") for c in entry.findall("a:category", ns)]
            papers.append({
                "paper_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": published[:4],
                "category": categories[0] if categories else "",
                "categories": categories,
                "source": "arxiv",
                "url": link,
                "arxiv_url": link,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            })
        return papers

    def _search_openalex(self, query: str, limit: int) -> list[dict]:
        data = self._get_json(f"https://api.openalex.org/works?search={quote_plus(query)}&per-page={min(limit, 50)}")
        papers: list[dict] = []
        for work in data.get("results", []):
            title = _clean(work.get("title") or "")
            if not title:
                continue
            abstract = _openalex_abstract(work.get("abstract_inverted_index") or {})
            doi = str(work.get("doi") or "").replace("https://doi.org/", "")
            authorships = work.get("authorships") or []
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in authorships[:8]
                if (a.get("author") or {}).get("display_name")
            )
            papers.append({
                "paper_id": doi or str(work.get("id", "")).rsplit("/", 1)[-1],
                "doi": doi,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": str(work.get("publication_year") or ""),
                "citation_count": int(work.get("cited_by_count") or 0),
                "source": "openalex",
                "url": work.get("doi") or work.get("id") or "",
                "arxiv_url": "",
            })
        return papers

    def _search_crossref(self, query: str, limit: int) -> list[dict]:
        data = self._get_json(
            "https://api.crossref.org/works"
            f"?query={quote_plus(query)}&rows={min(limit, 50)}&select=DOI,title,author,abstract,published-print,published-online,URL,is-referenced-by-count"
        )
        papers: list[dict] = []
        for item in (data.get("message") or {}).get("items", []):
            title = _clean(" ".join(item.get("title") or []))
            if not title:
                continue
            authors = ", ".join(
                " ".join(part for part in [a.get("given", ""), a.get("family", "")] if part).strip()
                for a in (item.get("author") or [])[:8]
            )
            year = _crossref_year(item)
            doi = item.get("DOI") or ""
            papers.append({
                "paper_id": doi,
                "doi": doi,
                "title": title,
                "authors": authors,
                "abstract": _clean(item.get("abstract") or ""),
                "year": str(year or ""),
                "citation_count": int(item.get("is-referenced-by-count") or 0),
                "source": "crossref",
                "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                "arxiv_url": "",
            })
        return papers

    def _get(self, url: str) -> requests.Response:
        headers = {"User-Agent": "research-ai/3.2 (mailto:research-ai@example.com)"}
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.get(url, headers=headers, timeout=self.settings.timeout_seconds)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(str(last_exc))

    def _get_json(self, url: str) -> dict:
        return self._get(url).json()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9][a-z0-9-]{1,}\b", text.lower())


def _clean(text: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip())


def _openalex_abstract(index: dict) -> str:
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for pos in positions:
            words.append((int(pos), word))
    return " ".join(word for _, word in sorted(words))


def _crossref_year(item: dict) -> int | None:
    for key in ("published-online", "published-print"):
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return int(parts[0][0])
    created = item.get("created", {}).get("date-time")
    if created:
        return parsedate_to_datetime(created).year
    return None
