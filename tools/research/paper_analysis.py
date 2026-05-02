from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import AnalyzePapersFunctionOutput, PaperAnalysisNote
from tools.research.paper_reading import PaperReader, resolve_answer_language

logger = logging.getLogger(__name__)

def _is_chinese_language(answer_language: str | None) -> bool:
    return str(answer_language or "").strip().lower().startswith("zh")


def _analysis_prompt(answer_language: str) -> str:
    if _is_chinese_language(answer_language):
        return (
            "你是一个论文分析与讲解助手。请根据用户问题和选中的论文，直接输出结构化分析结果。\n\n"
            "问题：{question}\n"
            "研究主题：{topic}\n"
            "报告亮点：{highlights}\n\n"
            "论文信息（JSON）：\n{papers_json}\n\n"
            "补充证据（JSON）：\n{evidence_json}\n\n"
            "要求：\n"
            "- 用中文回答，专有名词可保留英文\n"
            "- 如果问题偏比较，请突出方法差异、实验差异、适用场景和局限\n"
            "- 如果问题偏推荐，请给出最值得优先读/精读的论文及原因\n"
            "- 请优先根据问题语义判断回答 focus，不要仅按关键词机械分类\n"
            "- 优先使用补充证据中的正文片段、图谱摘要和相关段落来支撑结论\n"
            "- 如果证据只覆盖部分论文或部分维度，要明确指出结论边界\n"
            "- 返回结构化字段，不要输出额外解释"
        )
    return (
        "You are a paper analysis assistant. Use the user question and the selected papers to produce a structured analysis.\n\n"
        "Question: {question}\n"
        "Research topic: {topic}\n"
        "Report highlights: {highlights}\n\n"
        "Paper metadata (JSON):\n{papers_json}\n\n"
        "Additional evidence (JSON):\n{evidence_json}\n\n"
        "Requirements:\n"
        "- Respond in English\n"
        "- If the question is comparative, emphasize method differences, experimental differences, use cases, and limitations\n"
        "- If the question is about recommendation, name the best papers to prioritize and explain why\n"
        "- Infer the answer focus from the question semantics instead of mechanically classifying by keywords\n"
        "- Prefer body snippets, graph summaries, and relevant passages from the evidence when supporting conclusions\n"
        "- If the evidence only covers part of the papers or dimensions, state the boundary clearly\n"
        "- Return structured fields only, without extra explanation"
    )


class _PaperAnalysisNoteLLM(BaseModel):
    paper_id: str
    summary: str = ""
    relevance_to_question: str = ""
    strengths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class _PaperAnalysisLLMResponse(BaseModel):
    answer: str
    focus: str = "analysis"
    key_points: list[str] = Field(default_factory=list)
    recommended_paper_ids: list[str] = Field(default_factory=list)
    paper_notes: list[_PaperAnalysisNoteLLM] = Field(default_factory=list)


class PaperAnalyzer:
    """Unified selected-paper analysis capability for comparison, recommendation, and explanation."""

    name = "PaperAnalyzer"

    def __init__(
        self,
        *,
        paper_reading_skill: PaperReader | None = None,
        llm_adapter: Any | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.paper_reading_skill = paper_reading_skill or PaperReader(llm_adapter=llm_adapter)

    async def analyze_async(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        task_topic: str = "",
        report_highlights: list[str] | None = None,
        evidence_hits: list[RetrievalHit] | None = None,
        supervisor_instruction: str | None = None,
    ) -> AnalyzePapersFunctionOutput:
        if self.llm_adapter is not None and papers:
            try:
                result = await self._llm_analyze(
                    question=question,
                    papers=papers,
                    task_topic=task_topic,
                    report_highlights=report_highlights or [],
                    evidence_hits=evidence_hits or [],
                    supervisor_instruction=supervisor_instruction,
                )
                if result.answer.strip():
                    return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM paper analysis failed, falling back to heuristic: %s", exc)
        return self._heuristic_analyze(
            question=question,
            papers=papers,
            task_topic=task_topic,
            evidence_hits=evidence_hits or [],
        )

    async def _llm_analyze(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        task_topic: str,
        report_highlights: list[str],
        evidence_hits: list[RetrievalHit],
        supervisor_instruction: str | None = None,
    ) -> AnalyzePapersFunctionOutput:
        papers_json = json.dumps(
            [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "citations": paper.citations or 0,
                    "has_pdf": bool(paper.pdf_url),
                    "abstract": (paper.abstract or paper.summary or "")[:1000],
                }
                for paper in papers[:8]
            ],
            ensure_ascii=False,
            indent=2,
        )
        evidence_json = json.dumps(
            self._serialize_evidence_hits(evidence_hits=evidence_hits, papers=papers),
            ensure_ascii=False,
            indent=2,
        )
        answer_language = resolve_answer_language(question=question, fallback_text=task_topic)
        result = await self.llm_adapter.generate_structured(
            prompt=_analysis_prompt(answer_language),
            input_data={
                "question": question,
                "topic": task_topic,
                "highlights": " | ".join(report_highlights[:4]),
                "papers_json": papers_json,
                "evidence_json": evidence_json,
                **(({"supervisor_instruction": supervisor_instruction}) if supervisor_instruction else {}),
            },
            response_model=_PaperAnalysisLLMResponse,
        )
        paper_map = {paper.paper_id: paper for paper in papers}
        notes = [
            PaperAnalysisNote(
                paper_id=item.paper_id,
                title=paper_map.get(item.paper_id).title if paper_map.get(item.paper_id) else item.paper_id,
                summary=item.summary,
                relevance_to_question=item.relevance_to_question,
                strengths=list(item.strengths),
                caveats=list(item.caveats),
            )
            for item in result.paper_notes
            if item.paper_id in paper_map
        ]
        return AnalyzePapersFunctionOutput(
            answer=result.answer.strip(),
            focus=self._normalize_focus(result.focus, question=question),
            key_points=[point.strip() for point in result.key_points if point.strip()][:6],
            recommended_paper_ids=[
                paper_id for paper_id in result.recommended_paper_ids if paper_id in paper_map
            ][:3],
            paper_notes=notes,
            metadata={
                "generation_method": "llm",
                "skill": self.name,
                **self._evidence_metadata(evidence_hits=evidence_hits, papers=papers),
            },
        )

    def _heuristic_analyze(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        task_topic: str,
        evidence_hits: list[RetrievalHit],
    ) -> AnalyzePapersFunctionOutput:
        answer_language = resolve_answer_language(question=question, fallback_text=task_topic)
        focus = self._normalize_focus("", question=question)
        notes: list[PaperAnalysisNote] = []
        evidence_by_paper = self._evidence_summary_by_paper(evidence_hits=evidence_hits, papers=papers)
        for paper in papers[:6]:
            card = self.paper_reading_skill.extract(paper=paper)
            evidence_summary = evidence_by_paper.get(paper.paper_id, [])
            summary = card.summary or paper.summary or paper.abstract or paper.title
            if evidence_summary:
                summary = (
                    f"{summary} 证据片段：{' '.join(evidence_summary[:2])}"
                    if _is_chinese_language(answer_language)
                    else f"{summary} Evidence snippets: {' '.join(evidence_summary[:2])}"
                )
            notes.append(
                PaperAnalysisNote(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    summary=summary,
                    relevance_to_question=card.contribution or paper.title,
                    strengths=[
                        item for item in [card.method or card.contribution, card.experiment, *evidence_summary[:2]] if item
                    ][:3],
                    caveats=[card.limitation] if card.limitation else [],
                )
            )

        ranked = sorted(
            papers,
            key=lambda paper: (
                float(paper.relevance_score or 0.0),
                int(bool(paper.pdf_url)),
                int(paper.citations or 0),
                int(paper.year or 0),
            ),
            reverse=True,
        )
        recommended_ids = [paper.paper_id for paper in ranked[: min(3, len(ranked))]]
        answer = self._build_answer(
            question=question,
            task_topic=task_topic,
            focus=focus,
            notes=notes,
            papers=papers,
            recommended_ids=recommended_ids,
            answer_language=answer_language,
        )
        key_points = [note.relevance_to_question for note in notes if note.relevance_to_question][:4]
        return AnalyzePapersFunctionOutput(
            answer=answer,
            focus=focus,
            key_points=key_points,
            recommended_paper_ids=recommended_ids if focus == "recommend" else [],
            paper_notes=notes,
            metadata={
                "generation_method": "heuristic",
                "skill": self.name,
                **self._evidence_metadata(evidence_hits=evidence_hits, papers=papers),
            },
        )

    def _build_answer(
        self,
        *,
        question: str,
        task_topic: str,
        focus: str,
        notes: list[PaperAnalysisNote],
        papers: list[PaperCandidate],
        recommended_ids: list[str],
        answer_language: str,
    ) -> str:
        title_by_id = {paper.paper_id: paper.title for paper in papers}
        if _is_chinese_language(answer_language):
            lines = [
                f"围绕“{question.strip()}”，我基于当前选中的 {len(papers)} 篇论文做了定向分析。"
            ]
            if task_topic.strip():
                lines.append(f"这些结论默认服务于研究主题“{task_topic.strip()}”。")
            if focus == "compare":
                lines.append("从方法、实验和局限来看，这组论文的主要差异如下：")
            elif focus == "recommend":
                lines.append("如果目标是先抓住最值得投入时间精读的论文，可以优先看以下工作：")
                for index, paper_id in enumerate(recommended_ids[:3], start=1):
                    lines.append(f"{index}. {title_by_id.get(paper_id, paper_id)}")
            else:
                lines.append("这组论文可以从各自贡献、方法特点和适用边界来理解：")
        else:
            lines = [
                f'I analyzed the {len(papers)} currently selected papers for "{question.strip()}".'
            ]
            if task_topic.strip():
                lines.append(f'These takeaways are grounded in the research topic "{task_topic.strip()}".')
            if focus == "compare":
                lines.append("Across methods, experiments, and limitations, the main differences are:")
            elif focus == "recommend":
                lines.append("If you want the papers most worth prioritizing, start with:")
                for index, paper_id in enumerate(recommended_ids[:3], start=1):
                    lines.append(f"{index}. {title_by_id.get(paper_id, paper_id)}")
            else:
                lines.append("The papers can be understood through their contributions, method choices, and scope:")
        for note in notes[:5]:
            strengths = "；".join(item for item in note.strengths if item)
            caveats = "；".join(item for item in note.caveats if item)
            if _is_chinese_language(answer_language):
                detail = note.summary or note.relevance_to_question or "该论文提供了与当前问题相关的线索。"
            else:
                detail = note.summary or note.relevance_to_question or "This paper provides a relevant clue for the current question."
            if strengths:
                detail = (
                    f"{detail} 方法/证据重点：{strengths}。"
                    if _is_chinese_language(answer_language)
                    else f"{detail} Method/evidence focus: {strengths}."
                )
            if caveats:
                detail = (
                    f"{detail} 需要注意：{caveats}"
                    if _is_chinese_language(answer_language)
                    else f"{detail} Caveat: {caveats}"
                )
            lines.append(f"- {note.title}：{detail}")
        lines.append(
            "当前分析主要基于标题、摘要和已有知识卡片；如果你继续追问实验设置、失败案例或适用场景，我可以继续针对这组论文细化。"
            if _is_chinese_language(answer_language)
            else "This analysis is mainly based on titles, abstracts, and available knowledge cards. If you want, I can drill further into experimental setup, failure cases, or applicability."
        )
        return "\n".join(lines)

    def _normalize_focus(self, focus: str, *, question: str) -> str:
        normalized = (focus or "").strip().lower()
        if normalized in {"analysis", "compare", "recommend", "explain"}:
            return normalized
        # Heuristic fallback only: if the LLM did not return a valid focus,
        # infer a coarse focus from explicit wording in the question.
        lowered_question = question.lower()
        if any(marker in question for marker in ("对比", "比较", "区别", "差异")) or any(
            marker in lowered_question for marker in ("compare", "comparison", "versus", "vs")
        ):
            return "compare"
        if any(marker in question for marker in ("推荐", "精读", "先读", "优先")) or any(
            marker in lowered_question for marker in ("recommend", "priority", "worth reading")
        ):
            return "recommend"
        if any(marker in question for marker in ("讲解", "解释", "怎么理解")) or any(
            marker in lowered_question for marker in ("explain", "interpret")
        ):
            return "explain"
        return "analysis"

    def _serialize_evidence_hits(
        self,
        *,
        evidence_hits: list[RetrievalHit],
        papers: list[PaperCandidate],
    ) -> list[dict[str, Any]]:
        paper_map = {paper.paper_id: paper for paper in papers}
        serialized: list[dict[str, Any]] = []
        for hit in evidence_hits[:12]:
            paper_id = self._paper_id_for_hit(hit=hit, papers=papers)
            serialized.append(
                {
                    "paper_id": paper_id,
                    "paper_title": paper_map.get(paper_id).title if paper_id in paper_map else None,
                    "document_id": hit.document_id,
                    "source_type": hit.source_type,
                    "source_id": hit.source_id,
                    "score": float(hit.merged_score or hit.vector_score or hit.graph_score or hit.sparse_score or 0.0),
                    "content": (hit.content or "")[:600],
                    "metadata": {
                        key: value
                        for key, value in hit.metadata.items()
                        if key in {"page_number", "heading", "provider", "manifest_kind", "paper_id", "title"}
                    },
                }
            )
        return serialized

    def _evidence_summary_by_paper(
        self,
        *,
        evidence_hits: list[RetrievalHit],
        papers: list[PaperCandidate],
    ) -> dict[str, list[str]]:
        summaries: dict[str, list[str]] = {}
        for hit in evidence_hits:
            paper_id = self._paper_id_for_hit(hit=hit, papers=papers)
            if not paper_id:
                continue
            snippet = " ".join((hit.content or "").strip().split())
            if not snippet:
                continue
            summaries.setdefault(paper_id, [])
            if snippet not in summaries[paper_id]:
                summaries[paper_id].append(snippet[:240])
        return summaries

    def _paper_id_for_hit(self, *, hit: RetrievalHit, papers: list[PaperCandidate]) -> str | None:
        paper_id = str(hit.metadata.get("paper_id") or "").strip()
        if paper_id:
            return paper_id
        document_id = str(hit.document_id or "").strip()
        if not document_id:
            return None
        for paper in papers:
            if str(paper.metadata.get("document_id") or "").strip() == document_id:
                return paper.paper_id
        return None

    def _evidence_metadata(
        self,
        *,
        evidence_hits: list[RetrievalHit],
        papers: list[PaperCandidate],
    ) -> dict[str, Any]:
        covered = sorted(
            {
                paper_id
                for paper_id in (
                    self._paper_id_for_hit(hit=hit, papers=papers)
                    for hit in evidence_hits
                )
                if paper_id
            }
        )
        return {
            "evidence_hit_count": len(evidence_hits),
            "evidence_backed_paper_ids": covered,
            "evidence_backed_paper_count": len(covered),
        }
