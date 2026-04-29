from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.paper_knowledge import PaperFigureInsight, PaperFormulaInsight, PaperKnowledgeCard
from domain.schemas.research import PaperCandidate

logger = logging.getLogger(__name__)

def _preferred_answer_language_from_text(text: str | None) -> str:
    value = str(text or "").strip()
    if not value:
        return "zh-CN"
    cjk_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in value if ("a" <= char.lower() <= "z"))
    if cjk_count > 0 and cjk_count >= max(1, latin_count // 2):
        return "zh-CN"
    return "en-US"


def resolve_answer_language(
    *,
    question: str | None = None,
    metadata: dict[str, Any] | None = None,
    fallback_text: str | None = None,
) -> str:
    metadata = metadata or {}
    explicit = str(metadata.get("answer_language") or "").strip()
    if explicit:
        return explicit
    question_text = str(question or metadata.get("question") or "").strip()
    if question_text:
        return _preferred_answer_language_from_text(question_text)
    return _preferred_answer_language_from_text(fallback_text)


def _is_chinese_language(answer_language: str | None) -> bool:
    return str(answer_language or "").strip().lower().startswith("zh")


def _reading_prompt(answer_language: str) -> str:
    if _is_chinese_language(answer_language):
        return (
            "你是一个学术论文深度阅读助手。请根据论文标题和摘要，提取结构化的知识卡片。\n"
            "用中文回答，专有名词可保留英文。\n\n"
            "论文标题：{title}\n"
            "摘要：{abstract}\n"
            "来源：{source}\n"
            "年份：{year}\n\n"
            "请提取以下信息：\n"
            "1. contribution: 该论文的核心贡献（1-2句话）\n"
            "2. method: 使用的方法/框架（1-2句话）\n"
            "3. experiment: 实验设置和结果概述（1-2句话）\n"
            "4. limitation: 当前局限性（1-2句话）\n"
            "5. summary: 一句话总结"
        )
    return (
        "You are an academic paper reading assistant. Extract a structured knowledge card from the title and abstract.\n"
        "Respond in English.\n\n"
        "Paper title: {title}\n"
        "Abstract: {abstract}\n"
        "Source: {source}\n"
        "Year: {year}\n\n"
        "Extract the following fields:\n"
        "1. contribution: the core contribution of the paper (1-2 sentences)\n"
        "2. method: the method or framework used (1-2 sentences)\n"
        "3. experiment: experimental setup and results summary (1-2 sentences)\n"
        "4. limitation: current limitation (1-2 sentences)\n"
        "5. summary: one-sentence summary"
    )


class _LLMReadingResponse(BaseModel):
    contribution: str = Field(description="核心贡献")
    method: str = Field(description="方法/框架")
    experiment: str = Field(description="实验设置和结果")
    limitation: str = Field(description="局限性")
    summary: str = Field(description="一句话总结")


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return []
    return [
        segment.strip()
        for segment in re.split(r"(?<=[。！？.!?])\s+", normalized)
        if segment.strip()
    ]


class PaperReader:
    """Extract structured paper knowledge cards using LLM with heuristic fallback.
    
    When llm_adapter is provided, uses LLM for intelligent extraction.
    Falls back to keyword-matching heuristics when LLM is unavailable.
    """

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    def extract(
        self,
        *,
        paper: PaperCandidate,
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PaperKnowledgeCard:
        """Synchronous extract — uses heuristic logic."""
        return self._heuristic_extract(paper=paper, full_text=full_text, metadata=metadata)

    async def extract_async(
        self,
        *,
        paper: PaperCandidate,
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PaperKnowledgeCard:
        """Async extract — uses LLM if available, falls back to heuristic."""
        if self.llm_adapter is not None and (paper.abstract or full_text):
            try:
                return await self._llm_extract(paper=paper, full_text=full_text, metadata=metadata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM paper reading failed for %s, falling back to heuristic: %s", paper.paper_id, exc)
        return self._heuristic_extract(paper=paper, full_text=full_text, metadata=metadata)

    async def _llm_extract(
        self,
        *,
        paper: PaperCandidate,
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PaperKnowledgeCard:
        source_text = full_text or paper.abstract or paper.title
        answer_language = resolve_answer_language(
            metadata=metadata,
            fallback_text=f"{paper.title}\n{source_text}",
        )
        result = await self.llm_adapter.generate_structured(
            prompt=_reading_prompt(answer_language),
            input_data={
                "title": paper.title,
                "abstract": source_text[:1500],
                "source": paper.source,
                "year": str(paper.year or "unknown"),
            },
            response_model=_LLMReadingResponse,
        )
        resolved_metadata = {**(paper.metadata or {}), **(metadata or {})}
        formulas = self._extract_formulas(paper=paper, metadata=resolved_metadata)
        figures = self._extract_figures(paper=paper, metadata=resolved_metadata)
        return PaperKnowledgeCard(
            paper_id=paper.paper_id,
            title=paper.title,
            contribution=result.contribution,
            method=result.method,
            experiment=result.experiment,
            limitation=result.limitation,
            key_formulas=formulas,
            figures=figures,
            summary=result.summary,
            metadata={
                "reader": "PaperReader+LLM",
                "source": paper.source,
                "year": paper.year,
                "open_access": paper.is_open_access,
            },
        )

    def _heuristic_extract(
        self,
        *,
        paper: PaperCandidate,
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PaperKnowledgeCard:
        resolved_metadata = {
            **(paper.metadata or {}),
            **(metadata or {}),
        }
        source_text = full_text or paper.summary or paper.abstract or paper.title
        answer_language = resolve_answer_language(
            metadata=resolved_metadata,
            fallback_text=f"{paper.title}\n{source_text}",
        )
        sentences = _split_sentences(source_text)
        contribution_fallback = (
            f"{paper.title} 针对该研究主题给出了一个具备代表性的工作切入点。"
            if _is_chinese_language(answer_language)
            else f"{paper.title} provides a representative entry point into this research problem."
        )
        contribution = self._first_matching_sentence(
            sentences,
            keywords=("contribution", "contribute", "propose", "introduce", "提出", "贡献"),
            fallback=sentences[0] if sentences else contribution_fallback,
        )
        method = self._first_matching_sentence(
            sentences,
            keywords=("method", "approach", "framework", "model", "pipeline", "方法", "框架"),
            fallback=sentences[1] if len(sentences) > 1 else contribution,
        )
        experiment = self._first_matching_sentence(
            sentences,
            keywords=("experiment", "benchmark", "dataset", "evaluation", "ablation", "实验", "评估", "数据集"),
            fallback=resolved_metadata.get("experiment_note")
            or (
                "当前摘要中未展开完整实验设置，需要结合全文进一步核对。"
                if _is_chinese_language(answer_language)
                else "The abstract does not fully spell out the experimental setup, so the full paper should be checked."
            ),
        )
        limitation = self._build_limitation(
            paper=paper,
            sentences=sentences,
            metadata=resolved_metadata,
            answer_language=answer_language,
        )
        formulas = self._extract_formulas(paper=paper, metadata=resolved_metadata)
        figures = self._extract_figures(paper=paper, metadata=resolved_metadata)
        return PaperKnowledgeCard(
            paper_id=paper.paper_id,
            title=paper.title,
            contribution=contribution,
            method=method,
            experiment=experiment,
            limitation=limitation,
            key_formulas=formulas,
            figures=figures,
            summary=paper.summary or (sentences[0] if sentences else paper.title),
            metadata={
                "reader": "PaperReader",
                "source": paper.source,
                "year": paper.year,
                "open_access": paper.is_open_access,
            },
        )

    def _first_matching_sentence(
        self,
        sentences: list[str],
        *,
        keywords: tuple[str, ...],
        fallback: str,
    ) -> str:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                return sentence
        return fallback

    def _build_limitation(
        self,
        *,
        paper: PaperCandidate,
        sentences: list[str],
        metadata: dict[str, Any],
        answer_language: str,
    ) -> str:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in ("limitation", "future work", "challenge", "局限", "挑战")):
                return sentence
        limitations: list[str] = []
        if not paper.pdf_url:
            limitations.append(
                "当前仅有摘要级信息，缺少全文细节。"
                if _is_chinese_language(answer_language)
                else "Only abstract-level information is available, so full-text details are missing."
            )
        if not any(
            keyword in f"{paper.title} {paper.abstract}".lower()
            for keyword in ("experiment", "benchmark", "dataset", "evaluation", "实验")
        ):
            limitations.append(
                "摘要中没有展开完整实验设计。"
                if _is_chinese_language(answer_language)
                else "The abstract does not lay out the full experimental design."
            )
        if metadata.get("code_repository_candidates") in (None, [], ()):
            limitations.append(
                "暂未识别到稳定的代码复现线索。"
                if _is_chinese_language(answer_language)
                else "No stable code reproduction signal has been identified yet."
            )
        return " ".join(limitations) or (
            "需要进一步结合全文核实方法细节、实验设置与失败案例。"
            if _is_chinese_language(answer_language)
            else "The method details, experimental setup, and failure cases still need to be verified against the full paper."
        )

    def _extract_formulas(
        self,
        *,
        paper: PaperCandidate,
        metadata: dict[str, Any],
    ) -> list[PaperFormulaInsight]:
        raw_formulas = metadata.get("formulas") or metadata.get("key_formulas") or []
        formulas: list[PaperFormulaInsight] = []
        if isinstance(raw_formulas, list):
            for index, item in enumerate(raw_formulas[:3], start=1):
                if isinstance(item, dict):
                    formulas.append(
                        PaperFormulaInsight(
                            name=str(item.get("name") or f"Formula {index}"),
                            formula=str(item.get("formula") or ""),
                            explanation=str(item.get("explanation") or "该公式用于表述论文中的关键优化目标或评分机制。"),
                            purpose=item.get("purpose"),
                            metadata={"paper_id": paper.paper_id},
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    formulas.append(
                        PaperFormulaInsight(
                            name=f"Formula {index}",
                            formula=item.strip(),
                            explanation="该公式来自论文元数据，需要在全文中进一步核对符号定义。",
                            metadata={"paper_id": paper.paper_id},
                        )
                    )
        return formulas

    def _extract_figures(
        self,
        *,
        paper: PaperCandidate,
        metadata: dict[str, Any],
    ) -> list[PaperFigureInsight]:
        raw_figures = metadata.get("figures") or []
        figures: list[PaperFigureInsight] = []
        if isinstance(raw_figures, list):
            for index, item in enumerate(raw_figures[:3], start=1):
                if isinstance(item, dict):
                    figures.append(
                        PaperFigureInsight(
                            figure_id=str(item.get("figure_id") or f"fig-{index}"),
                            title=str(item.get("title") or ""),
                            explanation=str(item.get("explanation") or "该图用于说明论文中的核心流程或实验结果。"),
                            purpose=item.get("purpose"),
                            metadata={"paper_id": paper.paper_id},
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    figures.append(
                        PaperFigureInsight(
                            figure_id=f"fig-{index}",
                            title="",
                            explanation=item.strip(),
                            metadata={"paper_id": paper.paper_id},
                        )
                    )
        return figures
