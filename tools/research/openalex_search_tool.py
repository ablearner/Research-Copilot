from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from domain.schemas.research import PaperCandidate


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    indexed_tokens: dict[int, str] = {}
    for token, positions in inverted_index.items():
        for position in positions:
            indexed_tokens[position] = token
    return " ".join(indexed_tokens[index] for index in sorted(indexed_tokens))


def _looks_like_pdf_url(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return lowered.endswith(".pdf") or "/pdf" in lowered or "pdf?" in lowered


class OpenAlexSearchTool:
    """Search OpenAlex works and normalize to paper candidates."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 20.0,
        mailto: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.mailto = mailto

    async def search(self, *, query: str, max_results: int, days_back: int) -> list[PaperCandidate]:
        cutoff = (datetime.now(UTC) - timedelta(days=days_back)).date().isoformat()
        params = {
            "search": query,
            "per-page": max(1, min(max_results, 50)),
            "sort": "publication_date:desc",
            "filter": f"from_publication_date:{cutoff}",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()
        payload = response.json()
        papers: list[PaperCandidate] = []
        for item in payload.get("results", []):
            work_id = str(item.get("id") or "")
            title = str(item.get("display_name") or "").strip()
            if not work_id or not title:
                continue
            authors = [
                str(authorship.get("author", {}).get("display_name") or "").strip()
                for authorship in item.get("authorships", [])
            ]
            open_access = item.get("open_access") or {}
            primary_location = item.get("primary_location") or {}
            best_oa_location = item.get("best_oa_location") or {}
            landing_page_url = primary_location.get("landing_page_url")
            oa_url = open_access.get("oa_url")
            pdf_url = best_oa_location.get("pdf_url") or primary_location.get("pdf_url")
            if not pdf_url and _looks_like_pdf_url(oa_url):
                pdf_url = oa_url
            venue = (
                (primary_location.get("source") or {}).get("display_name")
                or (item.get("primary_topic") or {}).get("display_name")
            )
            papers.append(
                PaperCandidate(
                    paper_id=work_id,
                    title=title,
                    authors=[author for author in authors if author],
                    abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                    year=item.get("publication_year"),
                    venue=str(venue) if venue else None,
                    source="openalex",
                    doi=item.get("doi"),
                    pdf_url=pdf_url,
                    url=landing_page_url or oa_url or work_id,
                    citations=item.get("cited_by_count"),
                    is_open_access=open_access.get("is_oa"),
                    published_at=item.get("publication_date"),
                    metadata={
                        "type": item.get("type"),
                        "host_venue": (primary_location.get("source") or {}).get("display_name"),
                    },
                )
            )
        return papers
