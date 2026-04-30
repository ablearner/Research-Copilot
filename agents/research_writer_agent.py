from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from core.utils import now_iso as _now_iso
from domain.schemas.api import QAResponse
from domain.schemas.research import ResearchReport, ResearchTodoItem
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalQuery
from retrieval.evidence_builder import build_evidence_bundle
from services.research.capabilities import PaperAnalyzer
from services.research.research_knowledge_access import ResearchKnowledgeAccess

if TYPE_CHECKING:
    from services.research.paper_search_service import PaperSearchService


def _preferred_answer_language_from_question(question: str) -> str:
    return "zh-CN" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en-US"


class ResearchWriterAgent:
    """Primary agent for report writing, TODO planning, and grounded answers."""

    name = "ResearchWriterAgent"

    def __init__(
        self,
        paper_search_service: PaperSearchService | None = None,
        *,
        llm_adapter: Any | None = None,
        paper_analysis_skill: PaperAnalyzer | None = None,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.llm_adapter = llm_adapter
        self.paper_analysis_skill = paper_analysis_skill or PaperAnalyzer(llm_adapter=self.llm_adapter)

    def synthesize(self, state: Any) -> ResearchReport:
        paper_search_service = self._require_search_service()
        language = self._resolve_answer_language(state)
        base_report = paper_search_service.survey_writer.generate(
            topic=state.topic,
            task_id=state.task_id,
            papers=state.curated_papers,
            warnings=state.warnings,
            language=language,
        )
        return self._finalize_report(base_report, state)

    async def synthesize_async(self, state: Any) -> ResearchReport:
        """Async synthesize — uses LLM-powered survey generation if available."""
        paper_search_service = self._require_search_service()
        language = self._resolve_answer_language(state)
        base_report = await paper_search_service.survey_writer.generate_async(
            topic=state.topic,
            task_id=state.task_id,
            papers=state.curated_papers,
            warnings=state.warnings,
            language=language,
        )
        return self._finalize_report(base_report, state)

    def _finalize_report(self, base_report: ResearchReport, state: Any) -> ResearchReport:
        language = self._resolve_answer_language(state)
        is_chinese = language.startswith("zh")
        must_read_titles = [
            paper.title
            for paper in state.curated_papers
            if paper.paper_id in set(state.must_read_ids)
        ]
        ingest_titles = [
            paper.title
            for paper in state.curated_papers
            if paper.paper_id in set(state.ingest_candidate_ids)
        ]
        trace_lines = [
            f"- step {item.step_index}: {item.agent} -> {item.decision} | {item.rationale}"
            for item in state.trace
        ]
        markdown_lines = [
            base_report.markdown.rstrip(),
            "",
            "## 自主多智能体决策摘要" if is_chinese else "## Autonomous Multi-Agent Decision Summary",
            *(trace_lines or (["- 当前未记录到额外决策轨迹。"] if is_chinese else ["- No additional decision trace was recorded."])),
            "",
            "## 建议优先阅读" if is_chinese else "## Recommended First Reads",
            *([f"- {title}" for title in must_read_titles] or (["- 当前候选池不足，建议继续扩展检索。"] if is_chinese else ["- The current candidate pool is still thin; broaden the search space."])),
            "",
            "## 建议优先入库" if is_chinese else "## Recommended For Import",
            *([f"- {title}" for title in ingest_titles] or (["- 当前缺少可直接入库的开放 PDF。"] if is_chinese else ["- Open-access PDFs are still limited for direct import."])),
        ]
        markdown = "\n".join(markdown_lines)
        highlights = [*base_report.highlights]
        if must_read_titles:
            highlights.insert(0, f"{'优先阅读' if is_chinese else 'Recommended first read'}：{must_read_titles[0]}")
        gaps = [*base_report.gaps]
        if state.refinement_used and len(state.curated_papers) < min(state.max_papers, 6):
            gaps.insert(0, "系统已经自动补做一轮检索，但当前主题的高质量候选论文仍然偏少。" if is_chinese else "The system already ran one refinement pass, but the high-quality candidate pool is still limited.")
        return base_report.model_copy(
            update={
                "markdown": markdown,
                "highlights": highlights[:10],
                "gaps": list(dict.fromkeys(gaps))[:10],
                "metadata": {
                    **base_report.metadata,
                    "writer": self.name,
                    "manager_agent": "ResearchSupervisorAgent",
                    "agent_architecture": "main_agents_plus_skills",
                    "autonomy_mode": "lead_agent_loop",
                    "autonomy_rounds": state.round_index + 1,
                    "autonomy_trace_steps": len(state.trace),
                    "answer_language": language,
                },
            }
        )

    def plan_todos(self, state: Any) -> list[ResearchTodoItem]:
        created_at = _now_iso()
        todo_items: list[ResearchTodoItem] = []
        if state.ingest_candidate_ids:
            titles = [
                paper.title
                for paper in state.curated_papers
                if paper.paper_id in set(state.ingest_candidate_ids[:3])
            ]
            if titles:
                todo_items.append(
                    ResearchTodoItem(
                        todo_id=f"todo_{uuid4().hex}",
                        content=f"优先导入并精读这些开放论文：{'；'.join(titles)}。",
                        rationale="这些论文同时具备较高相关性和可直接入库的 PDF，适合作为研究集合的核心证据。",
                        status="open",
                        priority="high",
                        created_at=created_at,
                        source="qa_follow_up",
                        metadata={
                            "todo_kind": "ingest_priority",
                            "writer_agent": self.name,
                            "paper_ids": list(state.ingest_candidate_ids[:3]),
                        },
                    )
                )
        if state.report and state.report.gaps:
            todo_items.append(
                ResearchTodoItem(
                    todo_id=f"todo_{uuid4().hex}",
                    content=state.report.gaps[0],
                    rationale="这是当前自动综述中识别出的首要证据缺口。",
                    status="open",
                    priority="high",
                    created_at=created_at,
                    source="evidence_gap",
                    metadata={"todo_kind": "gap_follow_up", "writer_agent": self.name},
                )
            )
        if state.report and state.report.clusters:
            top_cluster = state.report.clusters[0]
            todo_items.append(
                ResearchTodoItem(
                    todo_id=f"todo_{uuid4().hex}",
                    content=f"围绕“{top_cluster.name}”整理方法、数据集和实验指标对比表。",
                    rationale="研究助手已经识别出主簇，下一步适合沉淀成结构化综述素材。",
                    status="open",
                    priority="medium",
                    created_at=created_at,
                    source="qa_follow_up",
                    metadata={"todo_kind": "cluster_synthesis", "writer_agent": self.name},
                )
            )
        deduped: list[ResearchTodoItem] = []
        seen_contents: set[str] = set()
        for item in todo_items:
            if item.content in seen_contents:
                continue
            seen_contents.add(item.content)
            deduped.append(item)
        return deduped[:3]

    async def plan_todos_async(self, state: Any) -> list[ResearchTodoItem]:
        """LLM-powered TODO planning."""
        if self.llm_adapter is None:
            return self.plan_todos(state)
        try:
            from pydantic import BaseModel, Field

            class _TodoItem(BaseModel):
                content: str = Field(description="TODO 内容（中文）")
                rationale: str = Field(description="理由（中文）")
                priority: str = Field(description="优先级: high/medium/low")

            class _TodoPlanResponse(BaseModel):
                todos: list[_TodoItem] = Field(description="TODO 列表")

            paper_titles = [p.title for p in state.curated_papers[:6]]
            ingest_titles = [
                p.title for p in state.curated_papers
                if p.paper_id in set(state.ingest_candidate_ids[:3])
            ]
            gaps = state.report.gaps[:3] if state.report else []
            clusters = [c.name for c in (state.report.clusters[:3] if state.report else [])]

            prompt = (
                "你是一个研究助手。请根据当前文献调研结果，生成 2-3 个后续行动建议（TODO）。\n\n"
                "研究主题：{topic}\n"
                "已检索论文：{paper_titles}\n"
                "可导入论文：{ingest_titles}\n"
                "研究空白：{gaps}\n"
                "论文分组：{clusters}\n\n"
                "要求：\n"
                "- 每个 TODO 应具体可执行\n"
                "- 包含理由说明\n"
                "- 按优先级排序"
            )
            result = await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data={
                    "topic": state.topic,
                    "paper_titles": "；".join(paper_titles),
                    "ingest_titles": "；".join(ingest_titles),
                    "gaps": "；".join(gaps),
                    "clusters": "、".join(clusters),
                },
                response_model=_TodoPlanResponse,
            )
            created_at = _now_iso()
            return [
                ResearchTodoItem(
                    todo_id=f"todo_{uuid4().hex}",
                    content=item.content,
                    rationale=item.rationale,
                    status="open",
                    priority=item.priority,
                    created_at=created_at,
                    source="llm_plan",
                    metadata={"todo_kind": "llm_generated", "writer_agent": self.name},
                )
                for item in result.todos[:3]
            ]
        except Exception:  # noqa: BLE001
            return self.plan_todos(state)

    async def answer_collection_question(
        self,
        *,
        graph_runtime: Any,
        state: Any,
        primary_agents: list[str],
    ) -> QAResponse:
        from agents.research_knowledge_agent import merge_retrieval_hits

        all_hits = merge_retrieval_hits(state.retrieval_hits, state.summary_hits, state.manifest_hits)
        evidence_bundle = build_evidence_bundle(all_hits[: max(state.top_k * 2, 12)])
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(
                query=state.question,
                document_ids=state.document_ids,
                mode="hybrid",
                top_k=state.top_k,
                filters={
                    "research_task_id": state.task.task_id,
                    "research_topic": state.task.topic,
                    "qa_mode": "research_collection",
                },
            ),
            hits=all_hits[: max(state.top_k * 2, 12)],
            evidence_bundle=evidence_bundle,
            metadata={
                "autonomy_mode": "lead_agent_loop",
                "agent_architecture": "main_agents_only",
                "primary_agents": primary_agents,
                "collection_hit_mix": {
                    "retrieval_hits": len(state.retrieval_hits),
                    "graph_summary_hits": len(state.summary_hits),
                    "manifest_hits": len(state.manifest_hits),
                },
            },
        )

        original_question = str(getattr(state, "original_question", state.question))
        resolved_question = state.question
        answer_metadata = {
            "research_task_id": state.task.task_id,
            "research_topic": state.task.topic,
            "qa_mode": "research_collection",
            "autonomy_mode": "lead_agent_loop",
            "agent_architecture": "main_agents_only",
            "primary_agents": primary_agents,
            "writer_agent": self.name,
            "original_question": original_question,
            "resolved_question": resolved_question,
        }
        task_context = {
            "task_id": state.task.task_id,
            "research_topic": state.task.topic,
            "paper_count": len(state.papers),
            "report_id": state.report.report_id if state.report else None,
        }
        preference_context = {
            "reasoning_style": state.request.reasoning_style or "cot",
            "skill_name": state.request.skill_name,
            "min_length": getattr(state.request, "min_length", 400),
            "return_citations": getattr(state.request, "return_citations", True),
            "answer_language": self._preferred_answer_language(state.question),
            "follow_user_language": True,
            "preserve_paper_title_language": True,
        }
        execution_context = getattr(state, "execution_context", None)
        session_context = getattr(execution_context, "session_context", None) or {}
        task_context = {
            **(getattr(execution_context, "task_context", None) or {}),
            **task_context,
            "selected_paper_ids": list(getattr(state.request, "paper_ids", []) or []),
            "selected_paper_titles": [paper.title for paper in state.papers[:8]],
            "qa_scope_mode": str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported"),
            "question_scope_document_count": len(state.document_ids),
        }
        preference_context = {
            **(getattr(execution_context, "preference_context", None) or {}),
            **preference_context,
        }
        memory_hints = getattr(execution_context, "memory_hints", None) or {}
        selected_paper_analysis = await self._analyze_selected_papers(state)
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        qa = await knowledge_access.answer_with_evidence(
            question=state.question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=answer_metadata,
            session_context=session_context,
            task_context=task_context,
            preference_context=preference_context,
            memory_hints=memory_hints,
            available_tool_names=["answer_with_evidence"],
        )
        citations = self._build_citations(state=state, retrieval_result=retrieval_result)
        extended_analysis = await self._build_extended_analysis_async(
            state=state,
            citations=citations,
            evidence_bundle=evidence_bundle,
        )
        scope_mode = str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported")
        use_selected_paper_analysis_answer = (
            selected_paper_analysis is not None
            and (
                scope_mode == "metadata_only"
                or not (qa.answer or "").strip()
                or len(evidence_bundle.evidences) == 0
            )
        )
        if selected_paper_analysis is not None and selected_paper_analysis.key_points:
            analysis_lines = ["", "## 选中论文分析要点"]
            analysis_lines.extend(f"- {point}" for point in selected_paper_analysis.key_points[:4])
            extended_analysis = f"{extended_analysis.rstrip()}\n" + "\n".join(analysis_lines)
        structured_answer = self._compose_structured_answer(
            state=state,
            raw_answer=(
                selected_paper_analysis.answer
                if use_selected_paper_analysis_answer and selected_paper_analysis is not None
                else qa.answer
            ),
            citations=citations,
            extended_analysis=extended_analysis,
            evidence_count=len(evidence_bundle.evidences),
        )
        related_sections = self._related_sections(citations)

        return qa.model_copy(
            update={
                "question": original_question,
                "answer": structured_answer,
                "evidence_bundle": evidence_bundle,
                "retrieval_result": retrieval_result,
                "metadata": {
                    **qa.metadata,
                    "answer_format": "direct_research_collection",
                    "original_question": original_question,
                    "resolved_question": resolved_question,
                    "citations": citations,
                    "related_sections": related_sections,
                    "extended_analysis": extended_analysis,
                    "paper_scope": {
                        "paper_ids": [paper.paper_id for paper in state.papers],
                        "paper_titles": [paper.title for paper in state.papers],
                        "document_ids": list(state.document_ids),
                        "scope_mode": str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported"),
                    },
                    "selection_warnings": list(getattr(state.request, "metadata", {}).get("selection_warnings") or []),
                    "selection_summary": getattr(state.request, "metadata", {}).get("selection_summary"),
                    "scope_statistics": {
                        "paper_count": len(state.papers),
                        "document_count": len(state.document_ids),
                        "evidence_count": len(evidence_bundle.evidences),
                    },
                    **(
                        {
                            "selected_paper_analysis": selected_paper_analysis.model_dump(mode="json"),
                            "recommended_paper_ids": list(selected_paper_analysis.recommended_paper_ids),
                        }
                        if selected_paper_analysis is not None
                        else {}
                    ),
                },
            }
        )

    async def answer(self, *, graph_runtime: Any, state: Any) -> QAResponse:
        return await self.answer_collection_question(
            graph_runtime=graph_runtime,
            state=state,
            primary_agents=[
                "ResearchSupervisorAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
            ],
        )

    def plan(self, state: Any) -> list[ResearchTodoItem]:
        return self.plan_todos(state)

    async def _analyze_selected_papers(self, state: Any):
        paper_ids = list(getattr(state.request, "paper_ids", []) or [])
        if not paper_ids or not getattr(state, "papers", None):
            return None
        return await self.paper_analysis_skill.analyze_async(
            question=str(getattr(state, "original_question", state.question)),
            papers=list(state.papers),
            task_topic=getattr(state.task, "topic", ""),
            report_highlights=list(getattr(getattr(state, "report", None), "highlights", [])[:4]),
        )

    def _preferred_answer_language(self, question: str) -> str:
        return _preferred_answer_language_from_question(question)

    def _resolve_answer_language(self, state: Any) -> str:
        execution_context = getattr(state, "execution_context", None)
        preference_context = getattr(execution_context, "preference_context", None) or {}
        language = str(preference_context.get("answer_language") or "").strip()
        if language:
            return language
        question = getattr(state, "question", None) or getattr(state, "topic", "")
        return _preferred_answer_language_from_question(question)

    def _build_citations(
        self,
        *,
        state: Any,
        retrieval_result: HybridRetrievalResult,
    ) -> list[dict[str, Any]]:
        paper_labels = {
            paper.paper_id: f"P{index}"
            for index, paper in enumerate(state.papers, start=1)
        }
        paper_by_id = {paper.paper_id: paper for paper in state.papers}
        document_to_paper_id = {
            str(paper.metadata.get("document_id")): paper.paper_id
            for paper in state.papers
            if str(paper.metadata.get("document_id") or "").strip()
        }
        fallback_doc_labels: dict[str, str] = {}
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None, str | None]] = set()
        for hit in retrieval_result.hits[: max(state.top_k, 8)]:
            paper_id = self._resolve_paper_id(
                hit=hit,
                document_to_paper_id=document_to_paper_id,
                paper_by_id=paper_by_id,
            )
            document_id = hit.document_id
            label: str
            title: str
            if paper_id:
                label = paper_labels.setdefault(paper_id, f"P{len(paper_labels) + 1}")
                paper = paper_by_id.get(paper_id)
                title = paper.title if paper is not None else str(hit.metadata.get("title") or paper_id)
            else:
                fallback_key = document_id or hit.source_id or hit.id
                if fallback_key not in fallback_doc_labels:
                    fallback_doc_labels[fallback_key] = f"D{len(fallback_doc_labels) + 1}"
                label = fallback_doc_labels[fallback_key]
                title = str(hit.metadata.get("title") or document_id or hit.source_id)
            page_number = hit.metadata.get("page_number") if isinstance(hit.metadata.get("page_number"), int) else None
            section = next(
                (
                    str(hit.metadata.get(key)).strip()
                    for key in ("section", "heading", "caption", "title")
                    if str(hit.metadata.get(key) or "").strip()
                ),
                None,
            )
            marker = (paper_id, document_id, hit.source_id)
            if marker in seen:
                continue
            seen.add(marker)
            citations.append(
                {
                    "label": label,
                    "paper_id": paper_id,
                    "title": title,
                    "document_id": document_id,
                    "page_number": page_number,
                    "section": section,
                    "source_type": hit.source_type,
                    "source_id": hit.source_id,
                    "snippet": (hit.content or "")[:280],
                    "score": hit.merged_score if hit.merged_score is not None else hit.vector_score or hit.graph_score,
                }
            )
        return citations[:8]

    def _compose_structured_answer(
        self,
        *,
        state: Any,
        raw_answer: str,
        citations: list[dict[str, Any]],
        extended_analysis: str,
        evidence_count: int,
    ) -> str:
        del citations, extended_analysis, evidence_count
        language = self._preferred_answer_language(state.question)
        direct_answer = (raw_answer or "").strip()
        if not direct_answer:
            return "证据不足" if language == "zh-CN" else "Insufficient evidence."
        if language == "zh-CN" and direct_answer.strip().lower() == "insufficient evidence.":
            return "证据不足"
        if language != "zh-CN" and direct_answer.strip() == "证据不足":
            return "Insufficient evidence."
        return direct_answer

    def _extend_answer_with_evidence_digest(
        self,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        min_length: int,
    ) -> str:
        if len(answer) >= min_length:
            return answer
        lines = [answer, "", "## 补充证据摘记"]
        for citation in citations:
            if len("\n".join(lines)) >= min_length:
                break
            if not citation.get("snippet"):
                continue
            lines.append(
                f"- [{citation['label']}] {citation['title']}：{citation['snippet']}"
            )
        return "\n".join(lines)

    def _paper_level_evidence_lines(self, *, citations: list[dict[str, Any]]) -> list[str]:
        if not citations:
            return ["- 当前没有可回溯的论文级证据映射，答案主要来自研究集合级摘要。"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for citation in citations:
            grouped.setdefault(str(citation.get("label") or "P?"), []).append(citation)
        lines: list[str] = []
        for label, group in grouped.items():
            head = group[0]
            descriptor: list[str] = [f"- [{label}] {head['title']}"]
            if head.get("paper_id"):
                descriptor.append(f"paper_id={head['paper_id']}")
            if head.get("document_id"):
                descriptor.append(f"doc={head['document_id']}")
            lines.append(" | ".join(descriptor))
            snippets = [
                str(item.get("snippet") or "").strip()
                for item in group
                if str(item.get("snippet") or "").strip()
            ]
            section_names = [
                str(item.get("section") or "").strip()
                for item in group
                if str(item.get("section") or "").strip()
            ]
            summary_bits = [f"命中证据 {len(group)} 条"]
            if section_names:
                summary_bits.append(f"主要来自 {' / '.join(dict.fromkeys(section_names))}")
            if snippets:
                summary_bits.append(f"可支持的核心内容是：{snippets[0]}")
            lines.append(f"  证据概括：{'；'.join(summary_bits)}。")
        return lines

    def _judgement_text(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_count: int,
    ) -> str:
        scope_mode = str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported")
        unique_labels = [item["label"] for item in citations if item.get("label")]
        unique_papers = list(dict.fromkeys(unique_labels))
        if scope_mode == "metadata_only":
            return (
                "当前结论主要依据论文元数据和已有摘要，适合做方向判断或候选论文筛选，"
                "不适合替代全文级的严格方法比较。"
            )
        if len(unique_papers) <= 1:
            return (
                f"当前回答基本由单篇论文或单个文档范围支撑，共覆盖 {evidence_count} 条证据。"
                "这有助于给出更聚焦的结论，但也意味着跨论文比较能力有限。"
            )
        return (
            f"当前回答由 {len(unique_papers)} 个论文级引用和 {evidence_count} 条证据共同支撑，"
            "可以形成相对稳定的综合判断；若需要进一步比较方法优劣，仍建议继续查看实验设置与数据集细节。"
        )

    async def _judgement_text_async(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_count: int,
    ) -> str:
        """LLM-powered judgement text generation."""
        if self.llm_adapter is None:
            return self._judgement_text(state=state, citations=citations, evidence_count=evidence_count)
        try:
            from pydantic import BaseModel, Field

            class _JudgementResponse(BaseModel):
                judgement: str = Field(description="可靠性判断（中文，2-3句话）")

            scope_mode = str(getattr(state.request, "metadata", {}).get("qa_scope_mode") or "all_imported")
            unique_papers = list(dict.fromkeys([item["label"] for item in citations if item.get("label")]))
            prompt = (
                "你是一个学术证据可靠性评估助手。请根据以下信息生成一段简洁的可靠性判断。\n\n"
                "范围模式：{scope_mode}\n"
                "论文引用数：{paper_count}\n"
                "证据条数：{evidence_count}\n"
                "问题：{question}\n\n"
                "要求：评估当前回答的可靠性和局限性（2-3句话）"
            )
            result = await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data={
                    "scope_mode": scope_mode,
                    "paper_count": str(len(unique_papers)),
                    "evidence_count": str(evidence_count),
                    "question": state.question,
                },
                response_model=_JudgementResponse,
            )
            return result.judgement
        except Exception:  # noqa: BLE001
            return self._judgement_text(state=state, citations=citations, evidence_count=evidence_count)

    def _limitation_text(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_count: int,
    ) -> str:
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        warnings = list(metadata.get("selection_warnings") or [])
        scope_mode = str(metadata.get("qa_scope_mode") or "all_imported")
        limitation_bits: list[str] = []
        if scope_mode == "metadata_only":
            limitation_bits.append("当前选中论文尚未全部导入全文，因此回答可能缺少段落级直接证据。")
        if evidence_count < 3:
            limitation_bits.append("当前证据条数偏少，结论更适合视为初步判断。")
        if len({item.get('label') for item in citations if item.get('label')}) <= 1:
            limitation_bits.append("当前证据集中在单篇论文，跨论文对比结论仍然偏弱。")
        if warnings:
            limitation_bits.append("范围选择提示：" + "；".join(warnings[:3]))
        return " ".join(limitation_bits) or "当前没有额外的范围告警，但仍建议结合全文和实验细节复核最终判断。"

    def _scope_text(self, *, state: Any, scope_mode: str, evidence_count: int) -> str:
        scope_names = {
            "selected_papers": "用户指定的论文范围",
            "selected_documents": "用户指定的文档范围",
            "metadata_only": "仅元数据范围（论文尚未全部导入全文）",
            "all_imported": "当前任务下的全部已导入论文",
        }
        metadata = dict(getattr(state.request, "metadata", {}) or {})
        selected_titles = [paper.title for paper in state.papers[:6]]
        suffix = f"涉及论文：{'；'.join(selected_titles)}。" if selected_titles else "当前没有可解析的论文标题。"
        selection_summary = str(metadata.get("selection_summary") or "").strip()
        warning_text = "；".join(list(metadata.get("selection_warnings") or [])[:3])
        return (
            f"本轮问答基于“{scope_names.get(scope_mode, scope_mode)}”展开，"
            f"当前纳入 {len(state.papers)} 篇论文、{len(state.document_ids)} 个文档对象和 "
            f"{evidence_count} 条候选证据。{suffix}"
            f"{(' 选择摘要：' + selection_summary + '。') if selection_summary else ''}"
            f"{(' 范围告警：' + warning_text + '。') if warning_text else ''}"
        )

    def _build_extended_analysis(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_bundle,
    ) -> str:
        citation_counter = Counter(
            citation["label"] for citation in citations if citation.get("label")
        )
        most_supported = ", ".join(
            f"[{label}]×{count}" for label, count in citation_counter.most_common(3)
        ) or "当前没有形成稳定的论文支持分布"
        warnings = list(getattr(state.request, "metadata", {}).get("selection_warnings") or [])
        reliability = "当前结论的可靠性偏高。" if len(evidence_bundle.evidences) >= 3 else "当前结论仍需补充更多直接证据。"
        warning_text = (
            " 选择范围提示：" + "；".join(warnings[:3])
            if warnings
            else ""
        )
        return (
            f"从证据覆盖度看，本轮回答主要依赖 {len(evidence_bundle.evidences)} 条证据与 "
            f"{len(citations)} 个可回溯引用条目，最主要的支持来源为 {most_supported}。"
            f"{reliability}{warning_text}"
        )

    async def _build_extended_analysis_async(
        self,
        *,
        state: Any,
        citations: list[dict[str, Any]],
        evidence_bundle,
    ) -> str:
        """LLM-powered extended analysis generation."""
        if self.llm_adapter is None:
            return self._build_extended_analysis(state=state, citations=citations, evidence_bundle=evidence_bundle)
        try:
            from pydantic import BaseModel, Field

            class _AnalysisResponse(BaseModel):
                analysis: str = Field(description="中文扩展分析（2-4句话）")

            citation_counter = Counter(
                citation["label"] for citation in citations if citation.get("label")
            )
            prompt = (
                "你是一个学术证据分析助手。请根据以下证据信息生成一段简洁的中文扩展分析。\n\n"
                "证据条数：{evidence_count}\n"
                "引用条目数：{citation_count}\n"
                "主要支持来源：{top_sources}\n"
                "问题：{question}\n\n"
                "要求：分析证据覆盖度、可靠性和局限性（2-4句话）"
            )
            result = await self.llm_adapter.generate_structured(
                prompt=prompt,
                input_data={
                    "evidence_count": str(len(evidence_bundle.evidences)),
                    "citation_count": str(len(citations)),
                    "top_sources": ", ".join(
                        f"[{label}]×{count}" for label, count in citation_counter.most_common(3)
                    ),
                    "question": state.question,
                },
                response_model=_AnalysisResponse,
            )
            return result.analysis
        except Exception:  # noqa: BLE001
            return self._build_extended_analysis(state=state, citations=citations, evidence_bundle=evidence_bundle)

    def _related_sections(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        related: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None]] = set()
        for citation in citations:
            marker = (citation.get("paper_id"), citation.get("source_id"))
            if marker in seen:
                continue
            seen.add(marker)
            related.append(
                {
                    "paper_id": citation.get("paper_id"),
                    "section_id": citation.get("source_id"),
                    "heading": citation.get("section"),
                    "relevance_score": citation.get("score"),
                }
            )
        return related

    def _unique_citation_map(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for citation in citations:
            marker = str(citation.get("label"))
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(citation)
        return unique

    def _resolve_paper_id(
        self,
        *,
        hit: Any,
        document_to_paper_id: dict[str, str],
        paper_by_id: dict[str, Any],
    ) -> str | None:
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        if isinstance(metadata.get("paper_id"), str) and metadata.get("paper_id") in paper_by_id:
            return metadata["paper_id"]
        if isinstance(metadata.get("research_paper_id"), str) and metadata.get("research_paper_id") in paper_by_id:
            return metadata["research_paper_id"]
        if hit.source_id in paper_by_id:
            return hit.source_id
        if hit.document_id and hit.document_id in document_to_paper_id:
            return document_to_paper_id[hit.document_id]
        return None

    def _require_search_service(self) -> PaperSearchService:
        if self.paper_search_service is None:
            raise RuntimeError("ResearchWriterAgent requires PaperSearchService for report synthesis")
        return self.paper_search_service
