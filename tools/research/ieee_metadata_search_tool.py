from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from domain.schemas.research import PaperCandidate

_QUARTER_PATTERN = re.compile(r"q([1-4])\s+(\d{4})", re.IGNORECASE)


def _parse_publication_date(value: str) -> datetime | None:
    normalized = " ".join((value or "").strip().split())
    if not normalized:
        return None

    quarter_match = _QUARTER_PATTERN.fullmatch(normalized)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        month = 1 + (quarter - 1) * 3
        return datetime(year, month, 1, tzinfo=UTC)

    iso_candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass

    for pattern in (
        "%Y%m%d",
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    ):
        try:
            parsed = datetime.strptime(normalized, pattern)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _extract_authors(item: dict[str, Any]) -> list[str]:
    authors_payload = item.get("authors") or {}
    if isinstance(authors_payload, dict):
        author_items = authors_payload.get("authors") or []
    elif isinstance(authors_payload, list):
        author_items = authors_payload
    else:
        author_items = []

    authors: list[str] = []
    for author in author_items:
        if not isinstance(author, dict):
            continue
        name = str(author.get("full_name") or author.get("name") or "").strip()
        if name:
            authors.append(name)
    return authors


class IEEEMetadataSearchTool:
    """Search IEEE Xplore metadata records and normalize to paper candidates."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def search(self, *, query: str, max_results: int, days_back: int) -> list[PaperCandidate]:
        if not self.api_key:
            raise RuntimeError("IEEE API key is required for IEEE metadata search")

        now = datetime.now(UTC)
        cutoff = now - timedelta(days=days_back)
        params = {
            "apikey": self.api_key,
            "format": "json",
            "querytext": query,
            "max_records": max(1, min(max_results, 50)),
            "start_record": 1,
            "start_year": max(1900, cutoff.year),
            "end_year": now.year,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=True) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
        payload = response.json()
        return [
            paper
            for item in payload.get("articles", [])
            if (paper := self._item_to_paper_candidate(item, cutoff=cutoff)) is not None
        ]

    def _item_to_paper_candidate(self, item: dict[str, Any], *, cutoff: datetime) -> PaperCandidate | None:
        article_number = str(item.get("article_number") or "").strip()
        title = str(item.get("title") or "").strip()
        if not article_number or not title:
            return None

        publication_date = str(item.get("publication_date") or item.get("publication_year") or "").strip()
        published_at = _parse_publication_date(publication_date)
        if published_at is not None:
            if published_at < cutoff:
                return None
        else:
            year_value = item.get("publication_year")
            if year_value is not None:
                try:
                    if int(year_value) < cutoff.year:
                        return None
                except (TypeError, ValueError):
                    pass

        pdf_url = str(item.get("pdf_url") or "").strip() or None
        html_url = str(item.get("html_url") or "").strip() or None
        doi = str(item.get("doi") or "").strip() or None
        access_type = str(item.get("accessType") or "").strip()
        is_open_access = access_type.lower() in {"open access", "ephemera"} if access_type else None
        publication_year = item.get("publication_year")
        try:
            year = int(publication_year) if publication_year is not None else None
        except (TypeError, ValueError):
            year = None

        return PaperCandidate(
            paper_id=f"ieee:{article_number}",
            title=title,
            authors=_extract_authors(item),
            abstract=str(item.get("abstract") or "").strip(),
            year=year,
            venue=str(item.get("publication_title") or "").strip() or None,
            source="ieee",
            doi=doi,
            pdf_url=pdf_url,
            url=html_url or pdf_url,
            citations=item.get("citing_paper_count"),
            is_open_access=is_open_access,
            published_at=published_at.isoformat() if published_at is not None else publication_date or None,
            metadata={
                "provider": "ieee",
                "article_number": article_number,
                "content_type": item.get("content_type"),
                "access_type": access_type or None,
            },
        )
