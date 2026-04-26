from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from agents.paper_analysis_agent import PaperAnalysisAgent
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.paper_knowledge import PaperKnowledgeRecord
from domain.schemas.research import PaperCandidate
from domain.schemas.research import ImportPapersRequest
from domain.schemas.research_context import ResearchContext, ResearchContextPaperMeta
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import (
    AnalyzePapersFunctionOutput,
    AnswerCitation,
    AskPaperFunctionOutput,
    ComparePapersFunctionOutput,
    ComparisonTableRow,
    DecomposeTaskFunctionOutput,
    ExtractPaperStructureFunctionOutput,
    ExecuteResearchPlanFunctionOutput,
    GenerateReviewFunctionOutput,
    RecommendPapersFunctionOutput,
    RecommendedPaper,
    RelatedSection,
    ResearchPlanStepResult,
    SearchPaperResult,
    SearchPapersFunctionOutput,
)
from domain.schemas.sub_manager import TaskEvaluation, TaskStep
from skills.research import (
    PaperAnalysisSkill,
    PaperReadingSkill,
    ResearchEvaluationSkill,
    ReviewWritingSkill,
)
from agents.research_knowledge_agent import merge_retrieval_hits
from tooling.research_runtime_schemas import (
    CodeExecutionToolOutput,
    LibrarySyncToolOutput,
    LocalFileEntry,
    LocalFileToolOutput,
    NotificationItem,
    NotificationToolOutput,
    SearchOrImportPaperToolOutput,
    WebSearchResultItem,
    WebSearchToolOutput,
)


class ResearchFunctionService:
    """Provide callable research functions and local runtime tool handlers."""

    def __init__(
        self,
        *,
        research_service,
        graph_runtime: Any | None = None,
        allowed_file_roots: list[str | Path] | None = None,
        code_execution_enabled: bool = False,
        zotero_api_base_url: str = "https://api.zotero.org",
        zotero_api_key: str | None = None,
        zotero_library_type: str | None = None,
        zotero_library_id: str | None = None,
    ) -> None:
        self.research_service = research_service
        self.graph_runtime = graph_runtime
        self.paper_search_service = research_service.paper_search_service
        self.report_service = research_service.report_service
        self.memory_manager = research_service.memory_manager
        self.paper_reading_skill = getattr(research_service, "paper_reading_skill", None) or PaperReadingSkill()
        self.paper_analysis_skill = PaperAnalysisSkill(
            paper_reading_skill=self.paper_reading_skill,
            llm_adapter=self._get_llm_adapter() if graph_runtime is not None else None,
        )
        self.paper_analysis_agent = PaperAnalysisAgent(
            paper_analysis_skill=self.paper_analysis_skill,
        )
        self.review_writing_skill = getattr(research_service, "review_writing_skill", None) or ReviewWritingSkill()
        self.evaluation_skill = getattr(research_service, "evaluation_skill", None) or ResearchEvaluationSkill()
        _llm = None
        if graph_runtime is not None:
            _agent = getattr(graph_runtime, "plan_and_solve_reasoning_agent", None)
            _llm = getattr(_agent, "llm_adapter", None) if _agent else None
        self.manager_agent = ResearchSupervisorAgent(llm_adapter=_llm)
        self.code_execution_enabled = code_execution_enabled
        self.zotero_api_base_url = zotero_api_base_url.rstrip("/")
        self.zotero_api_key = zotero_api_key
        self.zotero_library_type = zotero_library_type
        self.zotero_library_id = zotero_library_id
        roots = allowed_file_roots or [self.report_service.storage_root]
        self.allowed_file_roots = [Path(root).resolve() for root in roots]
        self.notification_queue_path = self.report_service.storage_root / "notifications" / "queue.json"
        self.library_sync_root = self.report_service.storage_root / "library_sync"
        self.execution_root = self.report_service.storage_root / "code_execution"

    def _get_llm_adapter(self):
        """Get LLM adapter from graph_runtime if available."""
        if self.graph_runtime is None:
            return None
        plan_and_solve_agent = getattr(self.graph_runtime, "plan_and_solve_reasoning_agent", None)
        if plan_and_solve_agent is None:
            return None
        return getattr(plan_and_solve_agent, "llm_adapter", None)

    def _current_topic(self) -> str:
        """Get current research topic from context if available."""
        try:
            context = self.memory_manager.get_current_context()
            return context.topic if context else ""
        except Exception:  # noqa: BLE001
            return ""

    def _get_external_tool_registry(self):
        if self.graph_runtime is None:
            return None
        registry = getattr(self.graph_runtime, "external_tool_registry", None)
        if registry is not None:
            return registry
        return getattr(self.graph_runtime, "mcp_client_registry", None)

    def build_function_handlers(self) -> dict[str, Any]:
        return {
            "search_papers": self.search_papers,
            "extract_paper_structure": self.extract_paper_structure,
            "analyze_papers": self.analyze_papers,
            "compare_papers": self.compare_papers,
            "generate_review": self.generate_review,
            "ask_paper": self.ask_paper,
            "recommend_papers": self.recommend_papers,
            "update_research_context": self.update_research_context,
            "decompose_task": self.decompose_task,
            "evaluate_result": self.evaluate_result,
            "execute_research_plan": self.execute_research_plan,
        }

    def build_runtime_tool_handlers(self) -> dict[str, Any]:
        handlers = {
            "academic_search": self.academic_search,
            "search_or_import_paper": self.search_or_import_paper,
            "local_file": self.local_file,
            "web_search": self.web_search,
            "notification": self.notification,
            "library_sync": self.library_sync,
        }
        if self.code_execution_enabled:
            handlers["code_execution"] = self.code_execution
        return handlers

    async def search_papers(
        self,
        *,
        query: str,
        source: list[str],
        date_range=None,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> SearchPapersFunctionOutput:
        days_back = self._days_back_from_range(date_range)
        raw_sources = list(source or ["arxiv", "openalex"])
        normalized_sources = [
            item for item in raw_sources if item in {"arxiv", "openalex", "semantic_scholar", "ieee", "zotero"}
        ] or ["arxiv", "openalex"]
        bundle = await self.paper_search_service.search(
            topic=query,
            days_back=days_back,
            max_papers=max_results,
            sources=normalized_sources,  # type: ignore[arg-type]
            task_id=None,
        )
        papers = self._sort_papers(bundle.papers, sort_by=sort_by)[:max_results]
        return SearchPapersFunctionOutput(
            papers=[
                SearchPaperResult(
                    id=paper.paper_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    year=paper.year,
                    url=paper.url or paper.pdf_url,
                    source=paper.source,
                )
                for paper in papers
            ]
        )

    async def extract_paper_structure(self, *, paper_id: str):
        paper = self._locate_paper(paper_id)
        if paper is None:
            knowledge_card = self.paper_reading_skill.extract(
                paper=PaperCandidate(paper_id=paper_id, title=paper_id, source="arxiv"),
            )
            return ExtractPaperStructureFunctionOutput(
                contribution=knowledge_card.contribution,
                method=knowledge_card.method,
                experiment=knowledge_card.experiment,
                limitation=knowledge_card.limitation,
                key_formulas=knowledge_card.key_formulas,
                figures=knowledge_card.figures,
                knowledge_card=knowledge_card,
            )
        knowledge_card = self.paper_reading_skill.extract(paper=paper)
        record = PaperKnowledgeRecord(
            paper_id=paper.paper_id,
            document_id=str(paper.metadata.get("document_id") or "") or None,
            title=paper.title,
            core_contribution=knowledge_card.contribution,
            related_paper_ids=[],
            citation_count=paper.citations,
            knowledge_card=knowledge_card,
            metadata={"source": paper.source},
        )
        try:
            self.memory_manager.update_paper_knowledge(record)
        except Exception:
            pass
        return ExtractPaperStructureFunctionOutput(
            contribution=knowledge_card.contribution,
            method=knowledge_card.method,
            experiment=knowledge_card.experiment,
            limitation=knowledge_card.limitation,
            key_formulas=knowledge_card.key_formulas,
            figures=knowledge_card.figures,
            knowledge_card=knowledge_card,
        )

    async def compare_papers(
        self,
        *,
        paper_ids: list[str],
        dimensions: list[str],
    ) -> ComparePapersFunctionOutput:
        papers = self._locate_papers(paper_ids)
        analysis = await self.analyze_papers(
            question=f"请对比这几篇论文的{'、'.join(dimensions) if dimensions else '贡献、方法、实验与局限'}",
            paper_ids=paper_ids,
        )
        rows: list[ComparisonTableRow] = []
        cards = {paper.paper_id: self.paper_reading_skill.extract(paper=paper) for paper in papers}
        if not dimensions:
            dimensions = ["contribution", "method", "experiment", "limitation"]
        for dimension in dimensions:
            values: dict[str, str] = {}
            for paper in papers:
                card = cards[paper.paper_id]
                values[paper.paper_id] = str(getattr(card, dimension, "") or paper.summary or paper.title)
            rows.append(ComparisonTableRow(dimension=dimension, values=values))
        return ComparePapersFunctionOutput(
            table=rows,
            summary=analysis.answer,
            metadata={"delegated_to": "analyze_papers", **analysis.metadata},
        )

    async def generate_review(
        self,
        *,
        paper_ids: list[str],
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
    ) -> GenerateReviewFunctionOutput:
        papers = self._locate_papers(paper_ids)
        # Use async LLM-powered generation if available
        llm_adapter = self._get_llm_adapter()
        if llm_adapter is not None:
            report = await self.review_writing_skill.generate_async(
                topic=self._topic_for_papers(papers),
                task_id=None,
                papers=papers,
                warnings=[],
                style=style,
                min_length=min_length,
                include_citations=include_citations,
            )
        else:
            report = self.review_writing_skill.generate(
                topic=self._topic_for_papers(papers),
                task_id=None,
                papers=papers,
                warnings=[],
                style=style,
                min_length=min_length,
                include_citations=include_citations,
            )
        citations = sorted({token.strip("[]") for token in report.markdown.split() if token.startswith("[P")})
        return GenerateReviewFunctionOutput(
            review_text=report.markdown,
            citations=[f"[{token}]" for token in citations],
            word_count=len([token for token in report.markdown.split() if token.strip()]),
        )

    async def ask_paper(
        self,
        *,
        question: str,
        paper_ids: list[str],
        return_citations: bool = True,
        min_length: int = 400,
    ) -> AskPaperFunctionOutput:
        papers = self._locate_papers(paper_ids)
        analysis = await self.analyze_papers(question=question, paper_ids=paper_ids)
        # Use LLM-powered answer if available
        llm_adapter = self._get_llm_adapter()
        if llm_adapter is not None:
            try:
                result = await self._llm_ask_paper(
                    question=question, papers=papers,
                    return_citations=return_citations, min_length=min_length,
                    llm_adapter=llm_adapter,
                )
                if analysis.answer.strip():
                    return result.model_copy(update={"answer": analysis.answer})
                return result
            except Exception:  # noqa: BLE001
                pass  # Fall through to heuristic
        # Heuristic fallback
        cards = [self.paper_reading_skill.extract(paper=paper) for paper in papers]
        lines = ["## 直接回答", f"问题：{question}"]
        citations: list[AnswerCitation] = []
        related_sections: list[RelatedSection] = []
        for index, (paper, card) in enumerate(zip(papers, cards, strict=False), start=1):
            label = f"P{index}"
            answer_line = (
                f"- [{label}] {paper.title}：核心贡献是 {card.contribution}；方法上强调 {card.method}；"
                f"实验证据显示 {card.experiment}。"
            )
            lines.append(answer_line)
            if return_citations:
                citations.append(
                    AnswerCitation(
                        paper_id=paper.paper_id,
                        section_id=f"{paper.paper_id}:summary",
                        evidence_text=card.summary or card.contribution,
                        rationale=f"[{label}] 直接支撑该问题的论文级概括。",
                    )
                )
                related_sections.append(
                    RelatedSection(
                        paper_id=paper.paper_id,
                        section_id=f"{paper.paper_id}:summary",
                        heading="summary",
                        relevance_score=0.8,
                    )
                )
        lines.extend(
            [
                "",
                "## 综合分析",
                "这些论文的共同点在于都围绕研究问题给出了明确的方法路径，但在实验覆盖度和可复现性上仍存在差异。",
                "",
                "## 局限",
                "当前回答基于论文标题、摘要和已有知识卡片，若需要严格结论，仍应回到全文证据。",
            ]
        )
        answer = "\n".join(lines)
        if len(answer) < min_length:
            answer += "\n\n## 扩展说明\n" + "\n".join(
                f"- {paper.title}：{card.limitation}" for paper, card in zip(papers, cards, strict=False)
            )
        return AskPaperFunctionOutput(
            answer=analysis.answer or answer,
            citations=citations if return_citations else [],
            related_sections=related_sections,
            extended_analysis="该回答优先覆盖论文级贡献、方法与实验线索，适合做后续精读的入口。",
        )

    async def _llm_ask_paper(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        return_citations: bool,
        min_length: int,
        llm_adapter,
    ) -> AskPaperFunctionOutput:
        from pydantic import BaseModel, Field

        class _AskPaperResponse(BaseModel):
            answer: str = Field(description="完整的中文回答（包含引用标记如[P1][P2]）")
            extended_analysis: str = Field(default="", description="扩展分析")

        papers_json = "\n".join(
            f'  [P{i}] title="{p.title}", abstract="{(p.abstract or "")[:400]}"'
            for i, p in enumerate(papers, start=1)
        )
        prompt = (
            "你是一个学术论文问答助手。请根据以下论文信息回答用户问题。\n\n"
            "问题：{question}\n\n"
            "论文信息：\n{papers_json}\n\n"
            "要求：\n"
            "- 用中文回答（专有名词可保留英文）\n"
            "- 使用 [P1][P2] 等标记引用论文\n"
            "- 包含直接回答、综合分析和局限性说明\n"
            "- 回答不少于 {min_length} 字"
        )
        result = await llm_adapter.generate_structured(
            prompt=prompt,
            input_data={"question": question, "papers_json": papers_json, "min_length": str(min_length)},
            response_model=_AskPaperResponse,
        )
        citations: list[AnswerCitation] = []
        related_sections: list[RelatedSection] = []
        if return_citations:
            for index, paper in enumerate(papers, start=1):
                citations.append(
                    AnswerCitation(
                        paper_id=paper.paper_id,
                        section_id=f"{paper.paper_id}:summary",
                        evidence_text=paper.summary or paper.abstract or paper.title,
                        rationale=f"[P{index}] 支撑该问题的论文级证据。",
                    )
                )
                related_sections.append(
                    RelatedSection(
                        paper_id=paper.paper_id,
                        section_id=f"{paper.paper_id}:summary",
                        heading="summary",
                        relevance_score=0.8,
                    )
                )
        return AskPaperFunctionOutput(
            answer=result.answer,
            citations=citations,
            related_sections=related_sections,
            extended_analysis=result.extended_analysis or "该回答基于 LLM 对论文摘要的综合分析。",
        )

    async def recommend_papers(
        self,
        *,
        based_on_context: str = "",
        based_on_history: list[str],
        top_k: int = 5,
    ) -> RecommendPapersFunctionOutput:
        corpus = self._list_all_papers()
        analysis = await self.paper_analysis_agent.analyze(
            question=based_on_context or self._current_topic() or "推荐最值得阅读的论文",
            papers=corpus[: min(len(corpus), 12)],
            task_topic=self._current_topic(),
        )
        paper_map = {paper.paper_id: paper for paper in corpus}
        recommendations = [
            RecommendedPaper(
                paper_id=paper.paper_id,
                title=paper.title,
                reason=next(
                    (
                        note.relevance_to_question or note.summary
                        for note in analysis.paper_notes
                        if note.paper_id == paper.paper_id
                    ),
                    "与当前问题和研究上下文高度相关，适合作为下一步重点阅读对象。",
                ),
                source=paper.source,
                year=paper.year,
                url=paper.url or paper.pdf_url,
            )
            for paper in [
                paper_map[paper_id]
                for paper_id in analysis.recommended_paper_ids[:top_k]
                if paper_id in paper_map
            ]
        ]
        return RecommendPapersFunctionOutput(
            recommendations=recommendations,
            metadata={"delegated_to": "analyze_papers", **analysis.metadata},
        )

    async def analyze_papers(
        self,
        *,
        question: str,
        paper_ids: list[str],
    ) -> AnalyzePapersFunctionOutput:
        papers = self._locate_papers(paper_ids)
        evidence_hits = await self._collect_analysis_evidence(question=question, papers=papers)
        return await self.paper_analysis_agent.analyze(
            question=question,
            papers=papers,
            task_topic=self._current_topic(),
            evidence_hits=evidence_hits,
        )

    async def update_research_context(
        self,
        *,
        topic: str = "",
        keywords: list[str],
        goals: list[str],
        known_conclusions: list[str],
        selected_papers: list[str],
        imported_papers: list[ResearchContextPaperMeta],
        open_questions: list[str],
        session_history,
        user_preferences=None,
        metadata: dict[str, Any],
    ) -> ResearchContext:
        return ResearchContext(
            research_topic=topic,
            research_goals=list(dict.fromkeys([*keywords, *goals])),
            selected_papers=selected_papers,
            imported_papers=imported_papers,
            known_conclusions=known_conclusions,
            open_questions=open_questions,
            session_history=session_history,
            user_preferences=user_preferences or ResearchContext().user_preferences,
            metadata=metadata,
        )

    async def decompose_task(
        self,
        *,
        user_request: str,
        context: ResearchContext,
    ) -> DecomposeTaskFunctionOutput:
        state = ResearchSupervisorState(
            goal=user_request,
            mode="qa" if "?" in user_request or "？" in user_request else "research",
            task_id=context.metadata.get("task_id") if isinstance(context.metadata, dict) else None,
            has_task=bool(context.metadata.get("task_id")) if isinstance(context.metadata, dict) else False,
            has_report=bool(context.imported_papers or context.known_conclusions),
            paper_count=len(context.imported_papers),
            imported_document_count=len(context.imported_papers),
            selected_paper_count=len(context.selected_papers),
        )
        decision = await self.manager_agent.decide_next_action_async(state, context_slice=None)
        active_message = decision.metadata.get("active_message") if isinstance(decision.metadata, dict) else None
        if active_message is not None:
            task_plan = [
                TaskStep(
                    task_id=active_message.task_id,
                    assigned_to=active_message.agent_to,
                    instruction=active_message.instruction,
                    task_type=active_message.task_type,
                    depends_on=list(active_message.depends_on),
                    context_slice=(
                        active_message.context_slice.model_dump(mode="json")
                        if hasattr(active_message.context_slice, "model_dump")
                        else dict(active_message.context_slice)
                    ),
                    expected_output_schema=dict(active_message.expected_output_schema),
                    priority="high" if active_message.priority in {"high", "critical"} else "normal",
                    retry_count=active_message.retry_count,
                    metadata=dict(active_message.metadata),
                )
            ]
            return DecomposeTaskFunctionOutput(
                task_plan=task_plan,
                assigned_sub_manager=task_plan[0].assigned_to,
                parallel_allowed=False,
                clarification_needed=False,
                clarification_question=None,
                metadata={
                    "decision_source": decision.metadata.get("decision_source") if isinstance(decision.metadata, dict) else None,
                    "action_name": decision.action_name,
                    "worker_agent": decision.metadata.get("worker_agent") if isinstance(decision.metadata, dict) else None,
                },
            )
        fallback_task = TaskStep(
            task_id=f"task_{uuid4().hex[:8]}",
            assigned_to="ResearchSupervisorAgent",
            instruction=decision.stop_reason or user_request,
            task_type="finalize",
            expected_output_schema={"required_fields": ["stop_reason"]},
        )
        return DecomposeTaskFunctionOutput(
            task_plan=[fallback_task],
            assigned_sub_manager="ResearchSupervisorAgent",
            parallel_allowed=False,
            clarification_needed=bool(decision.stop_reason),
            clarification_question=decision.stop_reason,
            metadata={"action_name": decision.action_name},
        )

    async def evaluate_result(
        self,
        *,
        result: dict[str, Any],
        task_instruction: str,
        expected_schema: dict[str, Any],
    ) -> TaskEvaluation:
        return self.evaluation_skill.evaluate_result(
            task_type=str(result.get("task_type") or "research"),
            result_status=str(result.get("status") or "succeeded"),
            payload=dict(result),
            task_instruction=task_instruction,
            expected_schema=expected_schema,
        )

    async def execute_research_plan(
        self,
        *,
        plan_steps,
        parallel: bool = False,
    ) -> ExecuteResearchPlanFunctionOutput:
        del parallel
        handlers = self.build_function_handlers()
        results: list[ResearchPlanStepResult] = []
        completed_ids: set[str] = set()
        for step in plan_steps:
            if any(dep not in completed_ids for dep in step.depends_on):
                results.append(
                    ResearchPlanStepResult(
                        step_id=step.step_id,
                        function_name=step.function_name,
                        status="skipped",
                        error_message="dependencies_not_satisfied",
                    )
                )
                continue
            handler = handlers.get(step.function_name)
            if handler is None:
                results.append(
                    ResearchPlanStepResult(
                        step_id=step.step_id,
                        function_name=step.function_name,
                        status="failed",
                        error_message="unknown_function",
                    )
                )
                continue
            try:
                output = await handler(**step.arguments)
                payload = output.model_dump(mode="json") if hasattr(output, "model_dump") else output
                results.append(
                    ResearchPlanStepResult(
                        step_id=step.step_id,
                        function_name=step.function_name,
                        status="succeeded",
                        output=payload,
                    )
                )
                completed_ids.add(step.step_id)
            except Exception as exc:
                results.append(
                    ResearchPlanStepResult(
                        step_id=step.step_id,
                        function_name=step.function_name,
                        status="failed",
                        error_message=str(exc),
                    )
                )
        succeeded = sum(1 for item in results if item.status == "succeeded")
        return ExecuteResearchPlanFunctionOutput(
            step_results=results,
            summary_report=f"Executed {len(results)} plan steps; succeeded={succeeded}.",
        )

    async def academic_search(self, **kwargs) -> SearchPapersFunctionOutput:
        return await self.search_papers(**kwargs)

    async def search_or_import_paper(
        self,
        *,
        query: str,
        source: list[str],
        date_range=None,
        max_results: int = 5,
        sort_by: str = "relevance",
        candidate_index: int = 0,
        collection_name: str | None = None,
        ingest_to_workspace: bool = False,
    ) -> SearchOrImportPaperToolOutput:
        registry = self._get_external_tool_registry()
        if registry is None:
            return SearchOrImportPaperToolOutput(
                status="not_configured",
                action="none",
                warnings=["External MCP registry is not configured."],
            )

        days_back = self._days_back_from_range(date_range)
        raw_sources = list(source or ["arxiv", "openalex"])
        normalized_sources = [
            item for item in raw_sources if item in {"arxiv", "openalex", "semantic_scholar", "ieee", "zotero"}
        ] or ["arxiv", "openalex"]
        bundle = await self.paper_search_service.search(
            topic=query,
            days_back=days_back,
            max_papers=max_results,
            sources=normalized_sources,  # type: ignore[arg-type]
            task_id=None,
        )
        papers = self._sort_papers(bundle.papers, sort_by=sort_by)[:max_results]
        candidates = [
            SearchPaperResult(
                id=item.paper_id,
                title=item.title,
                abstract=item.abstract,
                authors=item.authors,
                year=item.year,
                url=item.url or item.pdf_url,
                source=item.source,
            )
            for item in papers
        ]
        if not papers:
            return SearchOrImportPaperToolOutput(
                status="not_found",
                action="none",
                candidate_index=None,
                candidates=[],
                warnings=["No online paper candidate matched the query."],
            )
        if candidate_index >= len(papers):
            return SearchOrImportPaperToolOutput(
                status="failed",
                action="none",
                candidate_index=None,
                candidates=candidates,
                warnings=[f"candidate_index {candidate_index} is out of range for {len(papers)} candidates."],
            )

        paper = papers[candidate_index]
        zotero_sync = await self.sync_paper_to_zotero(paper, collection_name=collection_name)
        workspace_document_id, workspace_status, workspace_warnings = await self._ingest_paper_to_workspace(
            paper,
            enabled=ingest_to_workspace,
        )
        return SearchOrImportPaperToolOutput(
            status=zotero_sync["status"],
            action=zotero_sync["action"],
            selected_paper_id=paper.paper_id,
            selected_paper_title=paper.title,
            candidate_index=candidate_index,
            candidates=candidates,
            zotero_item_key=zotero_sync["zotero_item_key"],
            workspace_document_id=workspace_document_id,
            workspace_status=workspace_status,
            matched_by=zotero_sync["matched_by"],
            collection_name=zotero_sync["collection_name"],
            attachment_count=zotero_sync["attachment_count"],
            warnings=[
                *zotero_sync["warnings"],
                *workspace_warnings,
            ],
        )

    async def local_file(
        self,
        *,
        operation: str,
        path: str,
        content: str | None = None,
        encoding: str = "utf-8",
    ) -> LocalFileToolOutput:
        resolved = self._resolve_local_path(path)
        if operation == "read":
            return LocalFileToolOutput(
                operation=operation,
                path=str(resolved),
                success=True,
                existed=resolved.exists(),
                content=resolved.read_text(encoding=encoding) if resolved.exists() else "",
            )
        if operation == "write":
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content or "", encoding=encoding)
            return LocalFileToolOutput(operation=operation, path=str(resolved), success=True, existed=True)
        if operation == "append":
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with resolved.open("a", encoding=encoding) as handle:
                handle.write(content or "")
            return LocalFileToolOutput(operation=operation, path=str(resolved), success=True, existed=True)
        if operation == "delete":
            existed = resolved.exists()
            if resolved.is_dir():
                raise ValueError("local_file delete does not support directories")
            if existed:
                resolved.unlink()
            return LocalFileToolOutput(operation=operation, path=str(resolved), success=True, existed=existed)
        if operation == "list":
            if not resolved.exists():
                return LocalFileToolOutput(operation=operation, path=str(resolved), success=True, entries=[])
            entries = [
                LocalFileEntry(
                    path=str(item),
                    entry_type="directory" if item.is_dir() else "file",
                    size_bytes=item.stat().st_size if item.is_file() else None,
                )
                for item in sorted(resolved.iterdir())
            ]
            return LocalFileToolOutput(operation=operation, path=str(resolved), success=True, entries=entries)
        raise ValueError(f"unsupported local file operation: {operation}")

    async def code_execution(
        self,
        *,
        code: str,
        timeout_seconds: int = 10,
        working_directory: str | None = None,
    ) -> CodeExecutionToolOutput:
        if not self.code_execution_enabled:
            return CodeExecutionToolOutput(
                success=False,
                return_code=126,
                stdout="",
                stderr=(
                    "code_execution is disabled. Set MCP_CODE_EXECUTION_ENABLED=true "
                    "only in a trusted local environment to enable it."
                ),
                executed_with=sys.executable,
            )
        root = Path(working_directory).resolve() if working_directory else self.execution_root.resolve()
        if not any(root == base or base in root.parents for base in self.allowed_file_roots + [self.execution_root.resolve()]):
            root = self.execution_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CodeExecutionToolOutput(
            success=completed.returncode == 0,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            executed_with=sys.executable,
        )

    async def web_search(
        self,
        *,
        query: str,
        provider: str = "auto",
        max_results: int = 5,
    ) -> WebSearchToolOutput:
        selected_provider = provider
        if provider == "auto":
            selected_provider = "tavily" if os.getenv("TAVILY_API_KEY") else "brave"
        if selected_provider == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return WebSearchToolOutput(provider="tavily", warnings=["TAVILY_API_KEY is not configured"])
            async with httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {api_key}"}) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": max_results},
                )
                response.raise_for_status()
            payload = response.json()
            results = [
                WebSearchResultItem(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("content") or ""),
                    published_at=item.get("published_date"),
                )
                for item in payload.get("results", [])
            ]
            return WebSearchToolOutput(provider="tavily", results=results)
        if selected_provider == "brave":
            api_key = os.getenv("BRAVE_SEARCH_API_KEY")
            if not api_key:
                return WebSearchToolOutput(provider="brave", warnings=["BRAVE_SEARCH_API_KEY is not configured"])
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"X-Subscription-Token": api_key},
            ) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                )
                response.raise_for_status()
            payload = response.json()
            web_items = ((payload.get("web") or {}).get("results") or [])
            results = [
                WebSearchResultItem(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("description") or ""),
                    published_at=item.get("age"),
                )
                for item in web_items
            ]
            return WebSearchToolOutput(provider="brave", results=results)
        return WebSearchToolOutput(provider=selected_provider, warnings=["unsupported_provider"])

    async def notification(
        self,
        *,
        operation: str = "enqueue",
        message: str = "",
        channel: str = "queue",
        trigger_at: datetime | None = None,
        notification_id: str | None = None,
        metadata: dict[str, Any],
    ) -> NotificationToolOutput:
        items = self._load_notifications()
        if operation == "enqueue":
            item = NotificationItem(
                notification_id=notification_id or f"notify_{uuid4().hex}",
                message=message,
                channel=channel,  # type: ignore[arg-type]
                trigger_at=trigger_at,
                metadata=metadata,
            )
            items.append(item)
        elif operation == "dismiss" and notification_id:
            items = [
                item.model_copy(update={"status": "dismissed"})
                if item.notification_id == notification_id
                else item
                for item in items
            ]
        self._save_notifications(items)
        visible_items = items if operation == "list" else items[-5:]
        return NotificationToolOutput(status=operation, queue_size=len(items), items=visible_items)

    async def library_sync(
        self,
        *,
        provider: str = "filesystem",
        operation: str = "export",
        paper_ids: list[str],
        target_collection: str | None = None,
    ) -> LibrarySyncToolOutput:
        papers = self._locate_papers(paper_ids) if paper_ids else self._list_all_papers()[:20]
        if provider == "zotero":
            return await self._library_sync_to_zotero(
                papers=papers,
                operation=operation,
                target_collection=target_collection,
            )
        if provider != "filesystem":
            return LibrarySyncToolOutput(
                provider=provider,
                status="not_configured",
                exported_count=0,
                warnings=[f"{provider} sync is not configured in the current environment"],
            )
        self.library_sync_root.mkdir(parents=True, exist_ok=True)
        collection_name = target_collection or "default"
        output_path = self.library_sync_root / f"{collection_name}.json"
        payload = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "url": paper.url or paper.pdf_url,
                "source": paper.source,
            }
            for paper in papers
        ]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return LibrarySyncToolOutput(
            provider=provider,
            status=operation,
            exported_count=len(payload),
            output_path=str(output_path),
        )

    async def _library_sync_to_zotero(
        self,
        *,
        papers: list[PaperCandidate],
        operation: str,
        target_collection: str | None,
    ) -> LibrarySyncToolOutput:
        if not self.zotero_api_key or not self.zotero_library_type or not self.zotero_library_id:
            return LibrarySyncToolOutput(
                provider="zotero",
                status="not_configured",
                exported_count=0,
                warnings=[
                    "Zotero sync requires ZOTERO_API_KEY, ZOTERO_LIBRARY_TYPE, and ZOTERO_LIBRARY_ID.",
                ],
            )
        if self.zotero_library_type not in {"users", "groups"}:
            return LibrarySyncToolOutput(
                provider="zotero",
                status="not_configured",
                exported_count=0,
                warnings=["ZOTERO_LIBRARY_TYPE must be 'users' or 'groups'."],
            )

        warnings: list[str] = []
        collection_key: str | None = None
        try:
            if target_collection:
                collection_key = await self._ensure_zotero_collection(target_collection)
            item_payload = [
                self._paper_to_zotero_item(paper, collection_key=collection_key)
                for paper in papers
            ]
            if not item_payload:
                return LibrarySyncToolOutput(
                    provider="zotero",
                    status=operation,
                    exported_count=0,
                    output_path=self._zotero_collection_url(collection_key),
                )
            response = await self._zotero_post(
                path="/items",
                payload=item_payload,
            )
        except httpx.HTTPError as exc:
            return LibrarySyncToolOutput(
                provider="zotero",
                status="failed",
                exported_count=0,
                warnings=[f"Zotero sync failed: {exc.__class__.__name__}"],
            )

        payload = response.json()
        exported_count = len(payload.get("successful", {}))
        failed = payload.get("failed", {})
        for failure in failed.values():
            if isinstance(failure, dict):
                message = str(failure.get("message") or "unknown error")
                warnings.append(message)
        return LibrarySyncToolOutput(
            provider="zotero",
            status=operation,
            exported_count=exported_count,
            output_path=self._zotero_collection_url(collection_key),
            warnings=warnings,
        )

    async def _ensure_zotero_collection(self, name: str) -> str:
        response = await self._zotero_get(path="/collections", params={"format": "json", "limit": 100})
        payload = response.json()
        for item in payload:
            data = item.get("data", item) if isinstance(item, dict) else {}
            if isinstance(data, dict) and data.get("name") == name:
                key = item.get("key") or data.get("key")
                if isinstance(key, str) and key:
                    return key

        create_response = await self._zotero_post(
            path="/collections",
            payload=[{"name": name}],
        )
        create_payload = create_response.json()
        successful = create_payload.get("successful", {})
        created = successful.get("0")
        if isinstance(created, dict):
            data = created.get("data", created)
            key = created.get("key") or (data.get("key") if isinstance(data, dict) else None)
            if isinstance(key, str) and key:
                return key
        raise httpx.HTTPError("Zotero collection creation did not return a collection key")

    def _paper_to_zotero_item(self, paper: PaperCandidate, *, collection_key: str | None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "itemType": "journalArticle",
            "title": paper.title,
            "creators": [
                {"creatorType": "author", "name": author}
                for author in paper.authors
            ],
            "abstractNote": paper.abstract,
            "url": paper.url or paper.pdf_url or "",
            "tags": [
                {"tag": "Research-Copilot"},
                {"tag": f"source:{paper.source}"},
            ],
            "extra": self._paper_to_zotero_extra(paper),
        }
        if collection_key:
            item["collections"] = [collection_key]
        if paper.year is not None:
            item["date"] = str(paper.year)
        if paper.venue:
            item["publicationTitle"] = paper.venue
        if paper.doi:
            item["DOI"] = paper.doi
        return item

    def _paper_to_zotero_extra(self, paper: PaperCandidate) -> str:
        extra_lines = [
            f"Paper ID: {paper.paper_id}",
            f"Source: {paper.source}",
        ]
        if paper.citations is not None:
            extra_lines.append(f"Citations: {paper.citations}")
        if paper.arxiv_id:
            extra_lines.append(f"arXiv: {paper.arxiv_id}")
        return "\n".join(extra_lines)

    def _zotero_library_prefix(self) -> str:
        return f"/{self.zotero_library_type}/{self.zotero_library_id}"

    def _zotero_headers(self) -> dict[str, str]:
        headers = {"Zotero-API-Version": "3"}
        if self.zotero_api_key:
            headers["Zotero-API-Key"] = self.zotero_api_key
        return headers

    async def _zotero_get(self, *, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        async with self._build_async_client(timeout=15.0, headers=self._zotero_headers()) as client:
            response = await client.get(
                f"{self.zotero_api_base_url}{self._zotero_library_prefix()}{path}",
                params=params,
            )
            response.raise_for_status()
            return response

    async def _zotero_post(self, *, path: str, payload: list[dict[str, Any]]) -> httpx.Response:
        headers = {
            **self._zotero_headers(),
            "Content-Type": "application/json",
            "Zotero-Write-Token": uuid4().hex,
        }
        async with self._build_async_client(timeout=15.0, headers=headers) as client:
            response = await client.post(
                f"{self.zotero_api_base_url}{self._zotero_library_prefix()}{path}",
                json=payload,
            )
            response.raise_for_status()
            return response

    def _zotero_collection_url(self, collection_key: str | None) -> str | None:
        if not collection_key:
            return None
        return f"{self.zotero_api_base_url}{self._zotero_library_prefix()}/collections/{collection_key}"

    def _build_async_client(self, **kwargs) -> httpx.AsyncClient:
        return httpx.AsyncClient(**kwargs)

    def _days_back_from_range(self, date_range) -> int:
        if date_range is None or date_range.start_date is None:
            return 365
        today = datetime.now(UTC).date()
        return max(1, (today - date_range.start_date).days)

    def _sort_papers(self, papers: list[PaperCandidate], *, sort_by: str) -> list[PaperCandidate]:
        if sort_by == "date":
            return sorted(papers, key=lambda item: (item.year or 0, item.published_at or ""), reverse=True)
        if sort_by == "citations":
            return sorted(papers, key=lambda item: (item.citations or 0, item.year or 0), reverse=True)
        return list(papers)

    def _topic_for_papers(self, papers: list[PaperCandidate]) -> str:
        if not papers:
            return "research review"
        return papers[0].title.split(":")[0]

    def _list_all_papers(self) -> list[PaperCandidate]:
        papers: list[PaperCandidate] = []
        for path in sorted(self.report_service.papers_root.glob("*.json")):
            payload = self.report_service._read_json(path) or []
            for item in payload:
                papers.append(PaperCandidate.model_validate(item))
        return papers

    def _locate_paper(self, paper_id: str) -> PaperCandidate | None:
        for paper in self._list_all_papers():
            if paper.paper_id == paper_id:
                return paper
        return None

    def _locate_papers(self, paper_ids: list[str]) -> list[PaperCandidate]:
        paper_map = {paper.paper_id: paper for paper in self._list_all_papers()}
        return [paper_map[paper_id] for paper_id in paper_ids if paper_id in paper_map]

    async def _collect_analysis_evidence(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
    ) -> list[RetrievalHit]:
        if self.graph_runtime is None:
            return []
        retrieval_tools = getattr(self.graph_runtime, "retrieval_tools", None)
        if retrieval_tools is None:
            return []
        document_ids = list(
            dict.fromkeys(
                str(paper.metadata.get("document_id") or "").strip()
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip()
            )
        )
        if not document_ids:
            return []
        skill_context = (
            self.graph_runtime.resolve_skill_context(task_type="analyze_papers")
            if hasattr(self.graph_runtime, "resolve_skill_context")
            else None
        )
        retrieval_output = await retrieval_tools.retrieve(
            question=question,
            document_ids=document_ids,
            top_k=max(8, min(16, len(document_ids) * 4)),
            filters={
                "analysis_mode": "paper_analysis",
                "selected_paper_ids": [paper.paper_id for paper in papers],
                "selected_document_ids": document_ids,
            },
            session_id=None,
            task_id=None,
            memory_hints={},
            skill_context=skill_context,
        )
        retrieval_hits = [
            self._attach_paper_id_to_hit(hit=hit, papers=papers)
            for hit in list(retrieval_output.retrieval_result.hits or [])
        ]
        summary_hits: list[RetrievalHit] = []
        if hasattr(self.graph_runtime, "query_graph_summary"):
            summary_output = await self.graph_runtime.query_graph_summary(
                question=question,
                document_ids=document_ids,
                top_k=max(3, min(6, len(document_ids) * 2)),
                filters={
                    "analysis_mode": "paper_analysis",
                    "selected_paper_ids": [paper.paper_id for paper in papers],
                    "selected_document_ids": document_ids,
                },
                session_id=None,
                task_id=None,
                memory_hints={},
                skill_context=skill_context,
            )
            summary_hits = [
                self._attach_paper_id_to_hit(hit=hit, papers=papers)
                for hit in list(getattr(summary_output, "hits", []) or [])
            ]
        return merge_retrieval_hits(retrieval_hits, summary_hits)[:12]

    def _attach_paper_id_to_hit(
        self,
        *,
        hit: RetrievalHit,
        papers: list[PaperCandidate],
    ) -> RetrievalHit:
        document_id = str(hit.document_id or "").strip()
        matched_paper = next(
            (
                paper
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip() == document_id
            ),
            None,
        )
        if matched_paper is None:
            return hit
        metadata = dict(hit.metadata)
        metadata.setdefault("paper_id", matched_paper.paper_id)
        metadata.setdefault("title", matched_paper.title)
        return hit.model_copy(update={"metadata": metadata})

    def _resolve_local_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.report_service.storage_root / candidate
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.allowed_file_roots):
            raise ValueError(f"path is outside allowed roots: {raw_path}")
        return resolved

    def _load_notifications(self) -> list[NotificationItem]:
        if not self.notification_queue_path.exists():
            return []
        payload = json.loads(self.notification_queue_path.read_text(encoding="utf-8"))
        return [NotificationItem.model_validate(item) for item in payload]

    async def _find_matching_zotero_item(
        self,
        paper: PaperCandidate,
    ) -> tuple[dict[str, Any] | None, str | None]:
        registry = self._get_external_tool_registry()
        if registry is None:
            return None, None
        queries = []
        if paper.doi:
            queries.append(("doi", paper.doi))
        if paper.title:
            queries.append(("title", paper.title))
        if paper.url:
            queries.append(("url", paper.url))
        if paper.pdf_url:
            queries.append(("pdf_url", paper.pdf_url))

        seen_queries: set[str] = set()
        for _kind, query in queries:
            normalized_query = query.strip()
            if not normalized_query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            result = await registry.call_tool(
                tool_name="zotero_search_items",
                arguments={
                    "query": normalized_query,
                    "limit": 10,
                    "include_attachments": True,
                },
            )
            if result.status != "succeeded" or not isinstance(result.output, dict):
                continue
            items = result.output.get("items")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                matched_by = self._match_zotero_item(paper, item)
                if matched_by is not None:
                    return item, matched_by
        return None, None

    async def sync_paper_to_zotero(
        self,
        paper: PaperCandidate,
        *,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        registry = self._get_external_tool_registry()
        if registry is None:
            return {
                "status": "not_configured",
                "action": "none",
                "zotero_item_key": None,
                "matched_by": None,
                "collection_name": None,
                "attachment_count": 0,
                "warnings": ["External MCP registry is not configured."],
            }
        resolved_pdf_url = self._resolve_pdf_url_for_paper(paper)
        matched_item, matched_by = await self._find_matching_zotero_item(paper)
        if matched_item is not None:
            attachments = matched_item.get("attachments") or []
            attachment_count = len(attachments) if isinstance(attachments, list) else 0
            attach_warnings: list[str] = []
            if attachment_count == 0 and resolved_pdf_url:
                attachment_count, attach_warnings = await self._attach_pdf_to_existing_zotero_item(
                    item_key=self._coerce_string(matched_item.get("key")),
                    paper=paper,
                    pdf_url=resolved_pdf_url,
                )
            return {
                "status": "reused",
                "action": "reused",
                "zotero_item_key": self._coerce_string(matched_item.get("key")),
                "matched_by": matched_by,
                "collection_name": collection_name,
                "attachment_count": attachment_count,
                "warnings": [
                    "Matching Zotero item already exists; skipped duplicate import.",
                    *attach_warnings,
                ],
            }

        import_result = await registry.call_tool(
            tool_name="zotero_import_paper",
            arguments={
                "title": paper.title,
                "authors": list(paper.authors),
                "abstract": paper.abstract,
                "year": paper.year,
                "url": paper.url or paper.pdf_url,
                "doi": paper.doi,
                "publication_title": paper.venue,
                "pdf_url": resolved_pdf_url,
                "collection_name": collection_name,
            },
        )
        if import_result.status != "succeeded":
            return {
                "status": "failed",
                "action": "none",
                "zotero_item_key": None,
                "matched_by": None,
                "collection_name": None,
                "attachment_count": 0,
                "warnings": [import_result.error_message or "Zotero import failed."],
            }
        payload = import_result.output if isinstance(import_result.output, dict) else {}
        selected_collection = payload.get("selected_collection") if isinstance(payload, dict) else {}
        warnings = payload.get("warnings") if isinstance(payload, dict) else []
        return {
            "status": "imported",
            "action": "imported",
            "zotero_item_key": self._coerce_string(payload.get("imported_item_key")) if isinstance(payload, dict) else None,
            "matched_by": None,
            "collection_name": self._coerce_string(selected_collection.get("collection_name"))
            if isinstance(selected_collection, dict)
            else None,
            "attachment_count": 1 if self._coerce_string(payload.get("attachment_title")) else 0,
            "warnings": [str(item) for item in warnings] if isinstance(warnings, list) else [],
        }

    def _match_zotero_item(self, paper: PaperCandidate, item: dict[str, Any]) -> str | None:
        item_doi = self._coerce_string(item.get("doi"))
        if paper.doi and item_doi and paper.doi.strip().casefold() == item_doi.casefold():
            return "doi"

        item_title = self._coerce_string(item.get("title"))
        if paper.title and item_title and self._normalize_match_text(paper.title) == self._normalize_match_text(item_title):
            return "title"

        item_url = self._coerce_string(item.get("url"))
        if paper.url and item_url and paper.url.strip().rstrip("/") == item_url.rstrip("/"):
            return "url"

        attachments = item.get("attachments")
        if paper.pdf_url and isinstance(attachments, list):
            target_pdf = paper.pdf_url.strip().rstrip("/")
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                attachment_url = self._coerce_string(attachment.get("url"))
                if attachment_url and attachment_url.rstrip("/") == target_pdf:
                    return "pdf_url"
        return None

    def _normalize_match_text(self, value: str) -> str:
        return "".join(char.lower() for char in value if char.isalnum())

    def _resolve_pdf_url_for_paper(self, paper: PaperCandidate) -> str | None:
        if paper.pdf_url:
            return paper.pdf_url
        if paper.source == "arxiv" and paper.url and "/abs/" in paper.url:
            return f"{paper.url.replace('/abs/', '/pdf/')}.pdf"
        return None

    def _coerce_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    async def _ingest_paper_to_workspace(
        self,
        paper: PaperCandidate,
        *,
        enabled: bool,
    ) -> tuple[str | None, str | None, list[str]]:
        if not enabled:
            return None, None, []
        if self.graph_runtime is None or not hasattr(self.research_service, "import_papers"):
            return None, None, ["Workspace ingest is unavailable in the current runtime."]
        try:
            response = await self.research_service.import_papers(
                ImportPapersRequest(
                    papers=[paper],
                    paper_ids=[paper.paper_id],
                    include_graph=True,
                    include_embeddings=True,
                    fast_mode=True,
                ),
                graph_runtime=self.graph_runtime,
            )
        except Exception as exc:  # noqa: BLE001
            return None, "failed", [f"Workspace ingest failed: {exc.__class__.__name__}"]
        if not response.results:
            return None, None, ["Workspace ingest returned no results."]
        result = response.results[0]
        warnings = [result.error_message] if result.error_message else []
        return result.document_id, result.status, warnings

    async def _attach_pdf_to_existing_zotero_item(
        self,
        *,
        item_key: str | None,
        paper: PaperCandidate,
        pdf_url: str,
    ) -> tuple[int, list[str]]:
        registry = self._get_external_tool_registry()
        if registry is None or not item_key or not pdf_url:
            return 0, []
        result = await registry.call_tool(
            tool_name="zotero_attach_pdf_to_item",
            arguments={
                "item_key": item_key,
                "pdf_url": pdf_url,
                "title": f"{paper.title} PDF",
                "source_url": paper.url or pdf_url,
            },
        )
        if result.status != "succeeded" or not isinstance(result.output, dict):
            return 0, [result.error_message or "Existing Zotero item has no PDF attachment, and auto-attach failed."]
        warnings = result.output.get("warnings")
        attachment_count = result.output.get("attachment_count")
        status = self._coerce_string(result.output.get("status"))
        if status != "attached":
            return 0, [
                *([str(item) for item in warnings] if isinstance(warnings, list) else []),
                "Existing Zotero item has no PDF attachment, and the attach attempt could not be verified.",
            ]
        return (
            int(attachment_count) if isinstance(attachment_count, int) else 1,
            [str(item) for item in warnings] if isinstance(warnings, list) else [],
        )

    def _save_notifications(self, items: list[NotificationItem]) -> None:
        self.notification_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.notification_queue_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
