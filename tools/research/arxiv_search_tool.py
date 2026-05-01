from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Final

import httpx

from domain.schemas.research import PaperCandidate

_ATOM_NS: Final = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_NS: Final = {"arxiv": "http://arxiv.org/schemas/atom"}
_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9_-]+")
_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS: Final = 3.2
_DEFAULT_MAX_RETRIES: Final = 3

logger = logging.getLogger(__name__)


class ArxivSearchTool:
    """Search arXiv Atom feeds and normalize to paper candidates."""

    def __init__(
        self,
        *,
        base_url: str,
        app_name: str,
        timeout_seconds: float = 20.0,
        min_request_interval_seconds: float = _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.user_agent = f"{app_name} literature-research-agent"
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.max_retries = max(1, max_retries)
        self._request_lock = asyncio.Lock()
        self._last_request_started_at = 0.0

    async def search(self, *, query: str, max_results: int, days_back: int) -> list[PaperCandidate]:
        params = {
            "search_query": self._build_query(query),
            "start": 0,
            "max_results": max(1, min(max_results, 50)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
            trust_env=True,
        ) as client:
            response = await self._get_with_retry(client, params=params)
        papers = self._parse_feed(response.text, days_back=days_back)
        logger.info(
            "arXiv search result | query=%s | built_query=%s | hits=%s | days_back=%s",
            query[:180],
            str(params["search_query"])[:220],
            len(papers),
            days_back,
        )
        return papers

    async def _get_with_retry(self, client: httpx.AsyncClient, *, params: dict[str, object]) -> httpx.Response:
        last_error: httpx.HTTPStatusError | None = None
        for attempt in range(1, self.max_retries + 1):
            response = await self._rate_limited_get(client, params=params)
            try:
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429 or attempt >= self.max_retries:
                    raise
                retry_delay = self._retry_delay_seconds(exc.response, attempt)
                logger.warning(
                    "arXiv rate limited request; retrying in %.1fs (attempt %s/%s)",
                    retry_delay,
                    attempt,
                    self.max_retries,
                )
                await asyncio.sleep(retry_delay)
        assert last_error is not None
        raise last_error

    async def _rate_limited_get(
        self,
        client: httpx.AsyncClient,
        *,
        params: dict[str, object],
    ) -> httpx.Response:
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_started_at
            if elapsed < self.min_request_interval_seconds:
                await asyncio.sleep(self.min_request_interval_seconds - elapsed)
            self._last_request_started_at = time.monotonic()
            return await client.get(self.base_url, params=params)

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(self.min_request_interval_seconds, float(retry_after))
            except ValueError:
                pass
        return max(self.min_request_interval_seconds, min(20.0, 3.0 * attempt))

    def _build_query(self, query: str) -> str:
        tokens = [token for token in _TOKEN_PATTERN.findall(query) if len(token) > 1]
        if not tokens:
            return f'all:"{query}"'
        return " AND ".join(f"all:{token}" for token in tokens[:6])

    def _parse_feed(self, xml_text: str, *, days_back: int) -> list[PaperCandidate]:
        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        root = ET.fromstring(xml_text)
        papers: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
            published_text = (entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
            paper_url = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
            if not title or not paper_url:
                continue
            published_at = None
            year = None
            if published_text:
                try:
                    published_dt = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
                    if published_dt < cutoff:
                        continue
                    published_at = published_dt.isoformat()
                    year = published_dt.year
                except ValueError:
                    published_at = published_text
            pdf_url = None
            for link in entry.findall("atom:link", _ATOM_NS):
                href = link.attrib.get("href")
                title_attr = link.attrib.get("title", "")
                link_type = link.attrib.get("type", "")
                if href and (title_attr == "pdf" or link_type == "application/pdf"):
                    pdf_url = href
                    break
            authors = [
                (author.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
                for author in entry.findall("atom:author", _ATOM_NS)
            ]
            arxiv_id = paper_url.split("/")[-1]
            primary_category = entry.find("arxiv:primary_category", _ARXIV_NS)
            papers.append(
                PaperCandidate(
                    paper_id=f"arxiv:{arxiv_id}",
                    title=" ".join(title.split()),
                    authors=[author for author in authors if author],
                    abstract=" ".join(abstract.split()),
                    year=year,
                    venue="arXiv",
                    source="arxiv",
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                    url=paper_url,
                    is_open_access=True,
                    published_at=published_at,
                    metadata={
                        "primary_category": primary_category.attrib.get("term") if primary_category is not None else None
                    },
                )
            )
        return papers
