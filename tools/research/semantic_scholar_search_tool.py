from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from domain.schemas.research import PaperCandidate

logger = logging.getLogger(__name__)


class SemanticScholarSearchTool:
    """Search Semantic Scholar paper metadata and normalize to paper candidates."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 20.0,
        api_key: str | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_retry_delay_seconds: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.max_retry_delay_seconds = max(0.0, max_retry_delay_seconds)
        self._cooldown_until_monotonic: float = 0.0

    async def search(self, *, query: str, max_results: int, days_back: int) -> list[PaperCandidate]:
        now = time.monotonic()
        if now < self._cooldown_until_monotonic:
            raise RuntimeError("Semantic Scholar is in cooldown after repeated rate limiting; skipped for this round.")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        params = {
            "query": query,
            "limit": max(1, min(max_results, 50)),
            "fields": ",".join(
                [
                    "title",
                    "abstract",
                    "year",
                    "venue",
                    "publicationDate",
                    "authors",
                    "externalIds",
                    "openAccessPdf",
                    "url",
                    "citationCount",
                    "isOpenAccess",
                ]
            ),
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers, trust_env=True) as client:
            payload = await self._fetch_payload(client=client, query=query, params=params)
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        papers: list[PaperCandidate] = []
        for item in payload.get("data", []):
            paper = self._item_to_paper_candidate(item, cutoff=cutoff)
            if paper is not None:
                papers.append(paper)
        return papers

    async def _fetch_payload(
        self,
        *,
        client: httpx.AsyncClient,
        query: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request_url = f"{self.base_url}/paper/search"
        for attempt_index in range(self.max_retries + 1):
            response = await client.get(request_url, params=params)
            try:
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt_index >= self.max_retries:
                    if exc.response.status_code == 429:
                        self._cooldown_until_monotonic = time.monotonic() + max(30.0, self.max_retry_delay_seconds)
                    raise
                delay_seconds = self._retry_delay_seconds(exc.response, attempt_index=attempt_index)
                logger.warning(
                    "Semantic Scholar rate limit hit; retrying search",
                    extra={
                        "query": query[:180],
                        "attempt": attempt_index + 1,
                        "max_retries": self.max_retries,
                        "delay_seconds": delay_seconds,
                    },
                )
                await asyncio.sleep(delay_seconds)
        self._cooldown_until_monotonic = time.monotonic() + max(30.0, self.max_retry_delay_seconds)
        raise RuntimeError("Semantic Scholar search exhausted retry loop without a response")

    def _retry_delay_seconds(self, response: httpx.Response, *, attempt_index: int) -> float:
        retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is not None:
            return min(retry_after, self.max_retry_delay_seconds)
        backoff = self.retry_backoff_seconds * (2**attempt_index)
        return min(backoff, self.max_retry_delay_seconds)

    def _retry_after_seconds(self, value: str | None) -> float | None:
        if not value:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return max(0.0, float(stripped))
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    def _item_to_paper_candidate(
        self,
        item: dict[str, Any],
        *,
        cutoff: datetime,
    ) -> PaperCandidate | None:
        paper_id = str(item.get("paperId") or "").strip()
        title = str(item.get("title") or "").strip()
        if not paper_id or not title:
            return None
        publication_date = str(item.get("publicationDate") or "").strip()
        if publication_date:
            try:
                published_at = datetime.fromisoformat(publication_date.replace("Z", "+00:00"))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=UTC)
                if published_at < cutoff:
                    return None
            except ValueError:
                published_at = None
        else:
            published_at = None
        external_ids = item.get("externalIds") or {}
        open_access_pdf = item.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url")
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        authors = [
            str(author.get("name") or "").strip()
            for author in item.get("authors", [])
            if str(author.get("name") or "").strip()
        ]
        return PaperCandidate(
            paper_id=f"semantic_scholar:{paper_id}",
            title=title,
            authors=authors,
            abstract=str(item.get("abstract") or "").strip(),
            year=item.get("year"),
            venue=str(item.get("venue") or "").strip() or None,
            source="semantic_scholar",
            doi=str(doi).strip() or None if doi else None,
            arxiv_id=str(arxiv_id).strip() or None if arxiv_id else None,
            pdf_url=str(pdf_url).strip() or None if pdf_url else None,
            url=str(item.get("url") or "").strip() or None,
            citations=item.get("citationCount"),
            is_open_access=item.get("isOpenAccess"),
            published_at=published_at.isoformat() if published_at is not None else publication_date or None,
            metadata={
                "provider": "semantic_scholar",
                "paper_id": paper_id,
            },
        )
