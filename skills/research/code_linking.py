from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate


class CodeRepositoryCandidate(BaseModel):
    repo_name: str
    url: str
    stars: int | None = Field(default=None, ge=0)
    language: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeLinkingSkill:
    """Resolve likely code repositories for a paper using metadata or optional GitHub search."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        enable_remote_lookup: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.enable_remote_lookup = enable_remote_lookup

    async def enrich_papers(self, papers: list[PaperCandidate], *, top_k: int = 3) -> list[PaperCandidate]:
        enriched: list[PaperCandidate] = []
        for paper in papers:
            repositories = await self.find_repositories(paper=paper, top_k=top_k)
            enriched.append(
                paper.model_copy(
                    update={
                        "metadata": {
                            **paper.metadata,
                            "code_repository_candidates": [
                                repo.model_dump(mode="json") for repo in repositories
                            ],
                        }
                    }
                )
            )
        return enriched

    async def find_repositories(
        self,
        *,
        paper: PaperCandidate,
        top_k: int = 3,
    ) -> list[CodeRepositoryCandidate]:
        metadata_candidates = self._metadata_candidates(paper)
        if metadata_candidates or not self.enable_remote_lookup:
            return metadata_candidates[:top_k]
        return await self._github_candidates(paper=paper, top_k=top_k)

    def _metadata_candidates(self, paper: PaperCandidate) -> list[CodeRepositoryCandidate]:
        raw_candidates = paper.metadata.get("code_repository_candidates") or []
        candidates: list[CodeRepositoryCandidate] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    candidates.append(CodeRepositoryCandidate.model_validate(item))
                elif isinstance(item, str) and item.strip():
                    repo_name = item.rstrip("/").split("/")[-1]
                    candidates.append(CodeRepositoryCandidate(repo_name=repo_name, url=item.strip()))
        github_url = str(paper.metadata.get("github_url") or "").strip()
        if github_url:
            repo_name = github_url.rstrip("/").split("/")[-1]
            candidates.append(CodeRepositoryCandidate(repo_name=repo_name, url=github_url))
        deduped: list[CodeRepositoryCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.url in seen:
                continue
            seen.add(candidate.url)
            deduped.append(candidate)
        return deduped

    async def _github_candidates(
        self,
        *,
        paper: PaperCandidate,
        top_k: int,
    ) -> list[CodeRepositoryCandidate]:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            return []
        query = self._build_query(paper)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
            response = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": max(1, min(top_k, 5)), "sort": "stars", "order": "desc"},
            )
            response.raise_for_status()
        payload = response.json()
        repositories: list[CodeRepositoryCandidate] = []
        for item in payload.get("items", []):
            repositories.append(
                CodeRepositoryCandidate(
                    repo_name=str(item.get("full_name") or ""),
                    url=str(item.get("html_url") or ""),
                    stars=item.get("stargazers_count"),
                    language=item.get("language"),
                    updated_at=item.get("updated_at"),
                    metadata={"provider": "github_search"},
                )
            )
        return repositories

    def _build_query(self, paper: PaperCandidate) -> str:
        tokens = [token for token in paper.title.replace(":", " ").split() if len(token) > 2]
        return " ".join(tokens[:6])
