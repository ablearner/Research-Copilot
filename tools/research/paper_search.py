from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from domain.schemas.research import (
    PaperCandidate,
    PaperSource,
    ResearchReport,
    ResearchTopicPlan,
)
from tools.research import CodeLinker, PaperRanker, SurveyWriter, TopicPlanner
from tools.research.external_tool_gateway import ResearchExternalToolGateway
from tools.research import (
    ArxivSearchTool,
    IEEEMetadataSearchTool,
    OpenAlexSearchTool,
    SemanticScholarSearchTool,
)

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def format_search_warning(*, source: str, query: str, exc: Exception) -> str:
    prefix = f"{source}:{query}" if query else source
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            if source == "semantic_scholar":
                detail = "已被 Semantic Scholar 限流 (HTTP 429)，本轮已跳过该数据源。稍后重试通常会恢复。"
            else:
                detail = f"已被上游数据源限流 (HTTP {status_code})，本轮已跳过该数据源。"
            return f"{prefix} 查询失败: {detail}"
        if 500 <= status_code < 600:
            return f"{prefix} 查询失败: 上游数据源暂时不可用 (HTTP {status_code})，本轮已跳过该数据源。"
        return f"{prefix} 查询失败: 上游数据源返回 HTTP {status_code}。"
    if isinstance(exc, httpx.RequestError):
        return f"{prefix} 查询失败: 网络请求失败，{exc.__class__.__name__}。"
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{prefix} 查询失败: {detail}"


@dataclass(slots=True)
class SearchResultBundle:
    plan: ResearchTopicPlan
    papers: list[PaperCandidate]
    report: ResearchReport
    warnings: list[str]


class PaperSearchService:
    """Query multiple academic sources, deduplicate results, rank papers, and write a report."""

    def __init__(
        self,
        *,
        arxiv_tool: ArxivSearchTool,
        openalex_tool: OpenAlexSearchTool,
        semantic_scholar_tool: SemanticScholarSearchTool | None = None,
        ieee_tool: IEEEMetadataSearchTool | None = None,
        zotero_tool: Any | None = None,
        external_tool_gateway: ResearchExternalToolGateway | None = None,
        external_tool_registry: Any | None = None,
        topic_planner: TopicPlanner | None = None,
        paper_ranker: PaperRanker | None = None,
        survey_writer: SurveyWriter | None = None,
        code_linking_skill: CodeLinker | None = None,
        llm_adapter: Any | None = None,
        ranking_mode: str = "heuristic",
    ) -> None:
        self.arxiv_tool = arxiv_tool
        self.openalex_tool = openalex_tool
        self.semantic_scholar_tool = semantic_scholar_tool
        self.ieee_tool = ieee_tool
        self.zotero_tool = zotero_tool
        self.external_tool_gateway = external_tool_gateway or ResearchExternalToolGateway(
            registry=external_tool_registry
        )
        self.llm_adapter = llm_adapter
        self.topic_planner = topic_planner or TopicPlanner(llm_adapter=llm_adapter)
        self.paper_ranker = paper_ranker or PaperRanker(
            llm_adapter=llm_adapter,
            default_mode=ranking_mode,
        )
        self.survey_writer = survey_writer or SurveyWriter(llm_adapter=llm_adapter)
        self.code_linking_skill = code_linking_skill

    async def search(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[PaperSource],
        task_id: str | None = None,
    ) -> SearchResultBundle:
        # Use async plan if LLM is available
        if self.llm_adapter is not None:
            plan = await self.topic_planner.plan_async(
                topic=topic,
                days_back=days_back,
                max_papers=max_papers,
                sources=sources,
            )
        else:
            plan = self.topic_planner.plan(
                topic=topic,
                days_back=days_back,
                max_papers=max_papers,
                sources=sources,
            )
        provider_tasks: list[tuple[str, asyncio.Task[list[PaperCandidate]]]] = []
        per_query_limit = max(max_papers, 12)
        for source in sources:
            for query in self._queries_for_source(source=source, queries=plan.queries):
                tool = self._get_tool(source)
                if tool is None:
                    continue
                task_name = f"{source}:{query}"
                provider_tasks.append(
                    (
                        task_name,
                        asyncio.create_task(
                            self._search_source(
                                source=source,
                                tool=tool,
                                query=query,
                                max_results=per_query_limit,
                                days_back=days_back,
                            )
                        ),
                    )
                )

        warnings: list[str] = []
        raw_papers: list[PaperCandidate] = []
        for task_name, task in provider_tasks:
            try:
                papers = await task
                raw_papers.extend(papers)
                source, _, query = task_name.partition(":")
                logger.info(
                    "Paper search source result | source=%s | query=%s | hits=%s",
                    source,
                    query[:180],
                    len(papers),
                )
            except Exception as exc:  # pragma: no cover - network/provider failures are environment-specific
                source, _, query = task_name.partition(":")
                warnings.append(format_search_warning(source=source, query=query, exc=exc))

        deduped_papers = self._dedupe(raw_papers)
        logger.info(
            "Paper search aggregation | topic=%s | raw=%s | deduped=%s | warnings=%s | sources=%s",
            topic[:180],
            len(raw_papers),
            len(deduped_papers),
            len(warnings),
            ",".join(sources),
        )
        # Use async rank if LLM is available
        if self.llm_adapter is not None:
            ranked_papers = await self.paper_ranker.rank_async(topic=topic, papers=deduped_papers, max_papers=max_papers)
        else:
            ranked_papers = self.paper_ranker.rank(topic=topic, papers=deduped_papers, max_papers=max_papers)
        logger.info(
            "Paper search ranking | topic=%s | ranked=%s | top_titles=%s",
            topic[:180],
            len(ranked_papers),
            " | ".join(paper.title for paper in ranked_papers[:5]),
        )
        if self.code_linking_skill is not None:
            ranked_papers = await self.code_linking_skill.enrich_papers(ranked_papers)
        # Use async generate if LLM is available
        if self.llm_adapter is not None:
            report = await self.survey_writer.generate_async(topic=topic, task_id=task_id, papers=ranked_papers, warnings=warnings)
        else:
            report = self.survey_writer.generate(topic=topic, task_id=task_id, papers=ranked_papers, warnings=warnings)
        return SearchResultBundle(plan=plan, papers=ranked_papers, report=report, warnings=warnings)

    def _queries_for_source(self, *, source: PaperSource, queries: list[str]) -> list[str]:
        source_query_selector = getattr(self.topic_planner, "queries_for_source", None)
        if callable(source_query_selector):
            return source_query_selector(source=source, queries=queries)
        return queries

    def _get_tool(self, source: PaperSource):
        if source == "arxiv":
            return self.arxiv_tool
        if source == "openalex":
            return self.openalex_tool
        if source == "semantic_scholar":
            return self.semantic_scholar_tool
        if source == "ieee":
            return self.ieee_tool
        if source == "zotero":
            return self.zotero_tool
        return None

    async def _search_source(
        self,
        *,
        source: PaperSource,
        tool: Any,
        query: str,
        max_results: int,
        days_back: int,
    ) -> list[PaperCandidate]:
        mcp_tool_name = self._academic_mcp_tool_name(source)
        if mcp_tool_name and self.external_tool_gateway.is_configured():
            result = await self.external_tool_gateway.call_tool(
                tool_name=mcp_tool_name,
                arguments={
                    "query": query,
                    "max_results": max_results,
                    "days_back": days_back,
                },
                server_name="academic-search",
            )
            if result.status == "succeeded" and isinstance(result.output, dict):
                papers = result.output.get("papers")
                if isinstance(papers, list):
                    return [PaperCandidate.model_validate(paper) for paper in papers]
            if result.status not in {"not_found", "disabled"}:
                logger.warning(
                    "Academic MCP search failed; falling back to local tool | source=%s | query=%s | status=%s | error=%s",
                    source,
                    query[:180],
                    result.status,
                    result.error_message,
                )
        return await tool.search(query=query, max_results=max_results, days_back=days_back)

    def _academic_mcp_tool_name(self, source: PaperSource) -> str | None:
        if source == "arxiv":
            return "academic_search_arxiv"
        if source == "openalex":
            return "academic_search_openalex"
        if source == "semantic_scholar":
            return "academic_search_semantic_scholar"
        if source == "ieee":
            return "academic_search_ieee"
        return None

    def _dedupe(self, papers: list[PaperCandidate]) -> list[PaperCandidate]:
        deduped: dict[str, PaperCandidate] = {}
        key_aliases: dict[str, str] = {}
        for paper in papers:
            candidate_keys = [
                f"doi:{paper.doi.lower()}" if paper.doi else None,
                f"arxiv:{paper.arxiv_id.lower()}" if paper.arxiv_id else None,
                f"title:{_normalize_title(paper.title)}",
            ]
            key = next((key_aliases[item] for item in candidate_keys if item and item in key_aliases), None)
            if key is None:
                key = next((item for item in candidate_keys if item), f"title:{_normalize_title(paper.title)}")
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
                for item in candidate_keys:
                    if item:
                        key_aliases[item] = key
                continue

            merged = existing.model_copy(
                update={
                    "citations": max(existing.citations or 0, paper.citations or 0) or None,
                    "pdf_url": existing.pdf_url or paper.pdf_url,
                    "url": existing.url or paper.url,
                    "abstract": existing.abstract or paper.abstract,
                    "authors": existing.authors or paper.authors,
                    "is_open_access": existing.is_open_access if existing.is_open_access is not None else paper.is_open_access,
                    "metadata": {**paper.metadata, **existing.metadata},
                }
            )
            deduped[key] = merged
            for item in candidate_keys:
                if item:
                    key_aliases[item] = key
        return list(deduped.values())
