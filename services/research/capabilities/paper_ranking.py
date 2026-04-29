from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_DOMAIN_SYNONYM_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"(?:大模型|大语言模型|大型语言模型|基础模型|基座模型|\bllm(?:s)?\b|large language model|foundation model|generative ai)",
            re.IGNORECASE,
        ),
        (
            "large",
            "language",
            "model",
            "llm",
            "foundation",
            "generative",
            "ai",
            "reasoning",
            "multimodal",
            "transformer",
            "gpt",
            "chatgpt",
        ),
    ),
    (
        re.compile(r"(?:\buav\b|\bdrone\b|无人机)", re.IGNORECASE),
        ("uav", "drone", "unmanned", "aerial", "vehicle"),
    ),
    (
        re.compile(r"(?:path planning|路径规划)", re.IGNORECASE),
        ("path", "planning", "trajectory", "navigation"),
    ),
    (
        re.compile(r"(?:swarm|群体|协同)", re.IGNORECASE),
        ("swarm", "multi", "agent", "cooperative", "coordination"),
    ),
    (
        re.compile(r"(?:detection|检测)", re.IGNORECASE),
        ("detection", "detector", "object", "perception"),
    ),
)
_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_RANKING_PROMPT = (
    "你是一个学术论文相关性评估专家。请根据研究主题评估以下论文的相关性。\n\n"
    "研究主题：{topic}\n\n"
    "论文列表（JSON 格式）：\n{papers_json}\n\n"
    "请为每篇论文打分（0.0-1.0），并给出简短理由。分数越高表示与主题越相关。\n"
    "评估维度：方法创新性、与主题的直接相关性、实验完整性、发表时间新近性。"
)


class _PaperScore(BaseModel):
    paper_id: str = Field(description="论文 ID")
    score: float = Field(description="相关性分数 0.0-1.0", ge=0.0, le=1.0)
    reason: str = Field(default="", description="简短评分理由")


class _LLMRankingResponse(BaseModel):
    ranked_papers: list[_PaperScore] = Field(description="按相关性排序的论文评分列表")


def _tokenize(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in _EN_STOPWORDS
    }


def _normalize_token(token: str) -> str:
    normalized = token.lower()
    if len(normalized) > 3 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 3 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def _topic_terms(topic: str) -> set[str]:
    terms = set(_tokenize(topic))
    for pattern, synonyms in _DOMAIN_SYNONYM_RULES:
        if pattern.search(topic):
            terms.update(_normalize_token(token) for token in synonyms if len(token) > 2)
    for token in _CJK_TOKEN_PATTERN.findall(topic):
        if len(token) > 1:
            terms.add(token)
    return terms


def _cjk_overlap_score(topic_terms: set[str], text: str) -> int:
    return sum(1 for term in topic_terms if any("\u4e00" <= char <= "\u9fff" for char in term) and term in text)


def _has_ascii_term(terms: set[str]) -> bool:
    return any(re.search(r"[a-z0-9]", term, re.IGNORECASE) for term in terms)


def _source_quality_adjustment(paper: PaperCandidate) -> float:
    adjustment = 0.0
    if paper.source == "arxiv":
        adjustment += 0.08
    elif paper.source == "semantic_scholar":
        adjustment += 0.04

    locator = " ".join(str(value or "") for value in (paper.url, paper.pdf_url, paper.doi)).lower()
    if paper.source == "openalex" and any(host in locator for host in ("zenodo", "figshare", "github.com")):
        adjustment -= 0.12

    work_type = str(paper.metadata.get("type") or "").lower()
    if work_type in {"book", "book-chapter", "editorial", "letter"}:
        adjustment -= 0.12
    if not paper.abstract:
        adjustment -= 0.03
    return adjustment


class PaperRanker:
    """Rank paper candidates using LLM relevance assessment with deterministic fallback.
    
    When llm_adapter is provided, uses LLM for intelligent relevance scoring.
    Falls back to deterministic keyword-overlap scoring when LLM is unavailable.
    """

    def __init__(self, *, llm_adapter: Any | None = None, default_mode: str = "heuristic") -> None:
        self.llm_adapter = llm_adapter
        self.default_mode = (default_mode or "heuristic").strip().lower()

    def rank(self, *, topic: str, papers: list[PaperCandidate], max_papers: int) -> list[PaperCandidate]:
        """Synchronous rank — uses heuristic logic."""
        return self._heuristic_rank(topic=topic, papers=papers, max_papers=max_papers)

    async def rank_async(self, *, topic: str, papers: list[PaperCandidate], max_papers: int) -> list[PaperCandidate]:
        """Async rank — uses LLM if available, falls back to heuristic."""
        if self.default_mode != "llm":
            return self._heuristic_rank(topic=topic, papers=papers, max_papers=max_papers)
        if self.llm_adapter is not None and papers:
            try:
                return await self._llm_rank(topic=topic, papers=papers, max_papers=max_papers)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM ranking failed, falling back to heuristic: %s", exc)
        return self._heuristic_rank(topic=topic, papers=papers, max_papers=max_papers)

    async def _llm_rank(self, *, topic: str, papers: list[PaperCandidate], max_papers: int) -> list[PaperCandidate]:
        # Prepare paper summaries for LLM (limit to avoid token overflow)
        papers_for_llm = papers[:20]
        papers_json = "\n".join(
            f'  {{"paper_id": "{p.paper_id}", "title": "{p.title}", "abstract": "{(p.abstract or "")[:200]}"}}'
            for p in papers_for_llm
        )

        result = await self.llm_adapter.generate_structured(
            prompt=_RANKING_PROMPT,
            input_data={"topic": topic, "papers_json": papers_json},
            response_model=_LLMRankingResponse,
        )

        # Build score map from LLM response
        score_map = {item.paper_id: item.score for item in result.ranked_papers}

        # Apply LLM scores, keeping heuristic for papers not scored by LLM
        ranked: list[PaperCandidate] = []
        for paper in papers:
            llm_score = score_map.get(paper.paper_id)
            if llm_score is not None:
                score = round(llm_score, 4)
            else:
                # Fallback to heuristic for papers not in LLM response
                score = self._heuristic_score(topic, paper)
            ranked.append(
                paper.model_copy(
                    update={
                        "relevance_score": score,
                        "metadata": {
                            **paper.metadata,
                            "rank_method": "LLM" if llm_score is not None else "heuristic",
                        },
                    }
                )
            )

        ranked.sort(
            key=lambda p: (p.relevance_score or 0, p.year or 0, p.citations or 0),
            reverse=True,
        )
        return ranked[:max_papers]

    def _heuristic_rank(self, *, topic: str, papers: list[PaperCandidate], max_papers: int) -> list[PaperCandidate]:
        topic_terms = _topic_terms(topic)
        enforce_overlap = _has_ascii_term(topic_terms)
        ranked: list[PaperCandidate] = []
        now = datetime.now(UTC)
        for paper in papers:
            paper_text = f"{paper.title} {paper.abstract}"
            text_terms = _tokenize(paper_text)
            overlap = len(topic_terms & text_terms) + _cjk_overlap_score(topic_terms, paper_text)
            if enforce_overlap and overlap == 0:
                continue
            keyword_score = overlap / max(len(topic_terms), 1)

            citation_score = 0.0
            if paper.citations:
                citation_score = min(math.log10(max(paper.citations, 1) + 1) / 3, 1.0)

            recency_score = 0.0
            if paper.published_at:
                try:
                    published = datetime.fromisoformat(paper.published_at.replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=UTC)
                    delta_days = max((now - published).days, 0)
                    recency_score = max(0.0, 1.0 - min(delta_days / 365, 1.0))
                except ValueError:
                    recency_score = 0.0

            pdf_bonus = 0.1 if paper.pdf_url else 0.0
            oa_bonus = 0.05 if paper.is_open_access else 0.0

            quality_adjustment = _source_quality_adjustment(paper)
            score = round(
                max(0.0, (keyword_score * 0.65) + (recency_score * 0.15) + (citation_score * 0.1) + pdf_bonus + oa_bonus + quality_adjustment),
                4,
            )
            ranked.append(
                paper.model_copy(
                    update={
                        "relevance_score": score,
                        "metadata": {
                            **paper.metadata,
                            "rank_keyword_overlap": overlap,
                            "rank_topic_terms": min(len(topic_terms), 50),
                            "rank_source_quality_adjustment": round(quality_adjustment, 4),
                        },
                    }
                )
            )

        ranked.sort(
            key=lambda paper: (
                paper.relevance_score or 0,
                paper.year or 0,
                paper.citations or 0,
            ),
            reverse=True,
        )
        return ranked[:max_papers]

    def _heuristic_score(self, topic: str, paper: PaperCandidate) -> float:
        """Compute a single paper's heuristic score."""
        topic_terms = _topic_terms(topic)
        paper_text = f"{paper.title} {paper.abstract}"
        text_terms = _tokenize(paper_text)
        overlap = len(topic_terms & text_terms) + _cjk_overlap_score(topic_terms, paper_text)
        keyword_score = overlap / max(len(topic_terms), 1)
        quality_adjustment = _source_quality_adjustment(paper)
        pdf_bonus = 0.1 if paper.pdf_url else 0.0
        oa_bonus = 0.05 if paper.is_open_access else 0.0
        return round(max(0.0, (keyword_score * 0.65) + pdf_bonus + oa_bonus + quality_adjustment), 4)


# Compatibility alias for the previous agent-like name.
PaperRankerAgent = PaperRanker
