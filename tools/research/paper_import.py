from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx

from domain.schemas.research import PaperCandidate

_SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
_PDF_LINK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"""<meta[^>]+(?:name|property)=["']citation_pdf_url["'][^>]+content=["']([^"']+)["']""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""<meta[^>]+content=["']([^"']+)["'][^>]+(?:name|property)=["']citation_pdf_url["']""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""<a[^>]+href=["']([^"']+\.pdf(?:\?[^"']*)?)["']""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""<a[^>]+href=["']([^"']*?/pdf(?:[/?][^"']*)?)["']""",
        re.IGNORECASE,
    ),
)
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _safe_basename(title: str, fallback: str) -> str:
    normalized = _SAFE_FILENAME_PATTERN.sub("_", title.strip())[:80].strip("._")
    return normalized or fallback


def _dedupe_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _strip_url_query_and_fragment(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    if not parsed.query and not parsed.fragment:
        return None
    stripped = parsed._replace(query="", fragment="").geturl().strip()
    if not stripped or stripped == url.strip():
        return None
    return stripped


@dataclass(slots=True)
class DownloadedPaperArtifact:
    paper: PaperCandidate
    document_id: str
    storage_uri: str
    filename: str


class PaperImportService:
    """Download research paper PDFs into the existing upload directory."""

    def __init__(
        self,
        *,
        upload_dir: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.upload_dir = upload_dir
        self.timeout_seconds = timeout_seconds

    async def download_paper(self, paper: PaperCandidate) -> DownloadedPaperArtifact:
        existing_document_id = str(paper.metadata.get("document_id") or "").strip()
        existing_storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        if existing_document_id and existing_storage_uri and Path(existing_storage_uri).exists():
            return DownloadedPaperArtifact(
                paper=paper,
                document_id=existing_document_id,
                storage_uri=existing_storage_uri,
                filename=Path(existing_storage_uri).name,
            )

        pdf_url = self._resolve_pdf_url(paper)
        document_id = f"paper_{uuid4().hex}"
        filename = f"{document_id}_{_safe_basename(paper.title, 'paper')}.pdf"
        target_path = self.upload_dir / filename
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        local_pdf_path = self._resolve_local_pdf_path(paper)
        if local_pdf_path is not None and local_pdf_path.exists():
            shutil.copyfile(local_pdf_path, target_path)
            return DownloadedPaperArtifact(
                paper=paper,
                document_id=document_id,
                storage_uri=str(target_path.resolve()),
                filename=filename,
            )

        landing_page_url = self._resolve_landing_page_url(paper, pdf_url)
        if not pdf_url and not landing_page_url:
            raise ValueError(f"No PDF URL or landing page available for paper: {paper.title}")

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=self._browser_headers(),
            trust_env=True,
        ) as client:
            candidate_urls: list[str] = []
            if pdf_url:
                candidate_urls.append(pdf_url)
                alternate_pdf_url = _strip_url_query_and_fragment(pdf_url)
                if alternate_pdf_url:
                    candidate_urls.append(alternate_pdf_url)
            if landing_page_url:
                candidate_urls.extend(await self._discover_pdf_urls(client, landing_page_url))
            candidate_urls = _dedupe_urls(candidate_urls)
            if not candidate_urls:
                raise ValueError(f"No downloadable PDF discovered for paper: {paper.title}")

            content = b""
            content_type = ""
            last_error: Exception | None = None
            for candidate_url in candidate_urls:
                try:
                    content, content_type = await self._download_pdf_bytes(
                        client,
                        candidate_url,
                        referer=landing_page_url or paper.url,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise last_error or ValueError(f"Failed to download PDF for paper: {paper.title}")

        if "pdf" not in content_type and not content.startswith(b"%PDF"):
            raise ValueError(f"Downloaded content is not a PDF for paper: {paper.title}")

        target_path.write_bytes(content)
        return DownloadedPaperArtifact(
            paper=paper,
            document_id=document_id,
            storage_uri=str(target_path.resolve()),
            filename=filename,
        )

    def _resolve_pdf_url(self, paper: PaperCandidate) -> str | None:
        if paper.pdf_url:
            return paper.pdf_url
        if paper.source == "arxiv" and paper.url and "/abs/" in paper.url:
            return paper.url.replace("/abs/", "/pdf/") + ".pdf"
        return None

    def _resolve_landing_page_url(self, paper: PaperCandidate, pdf_url: str | None) -> str | None:
        if not paper.url:
            return None
        normalized_url = paper.url.strip()
        if not normalized_url:
            return None
        if pdf_url and normalized_url == pdf_url:
            return None
        return normalized_url

    def _resolve_local_pdf_path(self, paper: PaperCandidate) -> Path | None:
        raw_path = paper.metadata.get("zotero_local_path")
        if not isinstance(raw_path, str):
            return None
        normalized = raw_path.strip()
        if normalized.startswith("file://"):
            normalized = normalized[len("file://") :]
        if not normalized:
            return None
        candidate = Path(normalized)
        return candidate if candidate.suffix.lower() == ".pdf" else None

    def _browser_headers(
        self,
        *,
        referer: str | None = None,
        accept: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "User-Agent": _BROWSER_USER_AGENT,
            "Accept": accept or "application/pdf,application/octet-stream,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    async def _discover_pdf_urls(self, client: httpx.AsyncClient, landing_page_url: str) -> list[str]:
        try:
            response = await client.get(
                landing_page_url,
                headers=self._browser_headers(
                    referer=self._origin_url(landing_page_url),
                    accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                ),
            )
            response.raise_for_status()
        except Exception:
            return []

        html_text = response.text
        discovered_urls: list[str] = []
        for pattern in _PDF_LINK_PATTERNS:
            for match in pattern.findall(html_text):
                discovered_urls.append(urljoin(str(response.url), match))
        return _dedupe_urls(discovered_urls)

    async def _download_pdf_bytes(
        self,
        client: httpx.AsyncClient,
        pdf_url: str,
        *,
        referer: str | None,
    ) -> tuple[bytes, str]:
        response = await client.get(
            pdf_url,
            headers=self._browser_headers(referer=referer or self._origin_url(pdf_url)),
        )
        response.raise_for_status()
        return response.content, (response.headers.get("content-type") or "").lower()

    def _origin_url(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url
        return f"{parsed.scheme}://{parsed.netloc}/"
