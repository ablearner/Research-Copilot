from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from core.utils import now_iso as _now_iso
from domain.schemas.api import QAResponse
from domain.schemas.research import ResearchReport, ResearchTodoItem
from tools.research import PaperAnalysisTool
from tools.research.collection_qa_capability import ResearchCollectionAnswerCapability

if TYPE_CHECKING:
    from tools.research.paper_search import PaperSearchService
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

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
        paper_analysis_tool: PaperAnalysisTool | None = None,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.llm_adapter = llm_adapter
        self.paper_analysis_tool = paper_analysis_tool or PaperAnalysisTool(llm_adapter=self.llm_adapter)
        self._collection_answer = ResearchCollectionAnswerCapability(
            llm_adapter=llm_adapter, paper_analysis_tool=self.paper_analysis_tool
        )

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import (
            build_review_draft_input,
            build_review_draft_output,
        )

        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for review drafting",
                metadata={"reason": "missing_task"},
            )

        review_input = build_review_draft_input(context=context, decision=decision)
        if "task_context" in review_input.expert_context and not review_input.curated_papers and task_response.papers:
            return ResearchToolResult(
                status="skipped",
                observation="no papers are available for the review task scope",
                metadata={"reason": "no_papers"},
            )
        candidate_report = await self._write_report(review_input)
        retry_count = 0
        quality = self._quality_metrics(candidate_report)
        if not quality["passed"]:
            retry_count += 1
            candidate_report = await self._write_report(review_input)
            quality = self._quality_metrics(candidate_report)

        if task_response.report is not None:
            candidate_report = self._preserve_report_history(candidate_report, task_response.report)

        generated_at = _now_iso()
        task = task_response.task
        saved_report = candidate_report.model_copy(
            update={
                "generated_at": generated_at,
                "metadata": {
                    **candidate_report.metadata,
                    "search_plan": task.metadata.get("search_plan", {}),
                    "decision_model": "supervisor_direct_execution",
                    "worker_agent": "ResearchWriterAgent",
                    "write_review_retry_count": retry_count,
                    "write_review_quality_passed": quality["passed"],
                    "write_review_issue_count": len(quality["issues"]),
                },
            }
        )
        todo_state = review_input.model_copy(update={"report": saved_report})
        generated_todos = await self._plan_report_todos(todo_state)
        existing_contents = {item.content for item in task.todo_items}
        todo_items = [*task.todo_items, *(item for item in generated_todos if item.content not in existing_contents)]
        from domain.research_workspace import build_workspace_state

        workspace = build_workspace_state(
            objective=task.topic,
            stage="qa" if task.imported_document_ids else "complete",
            papers=list(task_response.papers),
            imported_document_ids=list(task.imported_document_ids),
            report=saved_report,
            todo_items=todo_items,
            must_read_ids=list(task.workspace.must_read_paper_ids),
            ingest_candidate_ids=list(task.workspace.ingest_candidate_ids),
            extra_questions=list(task.workspace.research_questions),
            extra_findings=list(task.workspace.key_findings),
            metadata={
                **task.workspace.metadata,
                "write_review_retry_count": retry_count,
                "open_todo_count": sum(item.status == "open" for item in todo_items),
                "gap_count": len(saved_report.gaps[:4]),
            },
        )
        if task.workspace.hypotheses:
            workspace = workspace.model_copy(update={"hypotheses": list(task.workspace.hypotheses)})
        updated_task = task.model_copy(
            update={
                "updated_at": generated_at,
                "report_id": saved_report.report_id,
                "todo_items": todo_items,
                "workspace": workspace,
            }
        )
        saved_report = saved_report.model_copy(update={"workspace": updated_task.workspace})
        request = context.request
        updated_response = task_response.model_copy(update={"task": updated_task, "report": saved_report})

        # --- P1: Build state delta (Runtime will apply + persist) ---
        delta = ResearchStateDelta(
            task=updated_task,
            report=saved_report,
            task_response=updated_response,
            save_task_conversation_id=request.conversation_id,
            save_task_event_type="memory_updated",
            save_task_event_payload={"tool_name": "write_review", "report_id": saved_report.report_id},
            rebuild_execution_context=True,
            rebuild_execution_context_params={
                "graph_runtime": context.graph_runtime,
                "conversation_id": request.conversation_id,
                "task": updated_task,
                "report": saved_report,
                "papers": task_response.papers,
                "document_ids": updated_task.imported_document_ids,
                "selected_paper_ids": request.selected_paper_ids,
                "skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style,
                "metadata": request.metadata,
            },
        )
        output = build_review_draft_output(
            task_id=updated_task.task_id,
            report_id=saved_report.report_id,
            quality=quality,
            retry_count=retry_count,
        )
        return ResearchToolResult(
            status="succeeded" if quality["passed"] else "failed",
            observation=(
                f"review drafted; words={quality['word_count']}; citations={quality['has_citations']}; retries={retry_count}"
            ),
            metadata=output.to_metadata(),
            state_delta=delta,
        )

    @staticmethod
    def _preserve_report_history(new_report: ResearchReport, existing_report: ResearchReport) -> ResearchReport:
        markdown = new_report.markdown.rstrip()
        lines = existing_report.markdown.splitlines()
        for heading in ("## 研究集合问答补充", "## TODO 执行记录"):
            start = next((index for index, line in enumerate(lines) if line.strip() == heading), None)
            if start is None:
                continue
            end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
            section = "\n".join(lines[start:end]).strip()
            if section not in markdown:
                markdown = f"{markdown}\n\n{section}"
        carry_highlights = [item for item in existing_report.highlights if item.startswith(("问答补充：", "TODO执行："))]
        return new_report.model_copy(update={
            "report_id": existing_report.report_id,
            "task_id": existing_report.task_id or new_report.task_id,
            "markdown": markdown,
            "highlights": list(dict.fromkeys([*new_report.highlights, *carry_highlights]))[:12],
            "gaps": list(dict.fromkeys([*new_report.gaps, *existing_report.gaps]))[:12],
            "metadata": {**existing_report.metadata, **new_report.metadata},
        })

    async def _write_report(self, state: Any) -> ResearchReport:
        from runtime.research.agent_protocol.base import (
            _SURVEY_WRITING_TIMEOUT_SECONDS, _llm_stage_timeout_seconds, _should_fallback_llm_stage,
        )
        try:
            return await asyncio.wait_for(
                self.synthesize_async(state),
                timeout=_llm_stage_timeout_seconds(self.llm_adapter, fallback_seconds=_SURVEY_WRITING_TIMEOUT_SECONDS, slack_seconds=15.0),
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            return self.synthesize(state)

    async def _plan_report_todos(self, state: Any) -> list[ResearchTodoItem]:
        from runtime.research.agent_protocol.base import (
            _TODO_PLANNING_TIMEOUT_SECONDS, _llm_stage_timeout_seconds, _should_fallback_llm_stage,
        )
        try:
            return await asyncio.wait_for(
                self.plan_todos_async(state),
                timeout=_llm_stage_timeout_seconds(self.llm_adapter, fallback_seconds=_TODO_PLANNING_TIMEOUT_SECONDS, slack_seconds=8.0),
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            return self.plan_todos(state)

    def _quality_metrics(self, report: ResearchReport) -> dict[str, Any]:
        text = report.markdown
        word_count = len([token for token in text.replace("\n", " ").split(" ") if token.strip()])
        cjk_char_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        effective_word_count = max(word_count, cjk_char_count // 2)
        has_citations = "[P" in text or "引用" in text
        has_key_sections = any(
            all(section in text for section in required_sections)
            for required_sections in (
                ("## 研究背景", "## 方法对比", "## 关键发现"),
                ("## Background", "## Method Comparison", "## Key Findings"),
            )
        )
        issues: list[str] = []
        if effective_word_count < 250:
            issues.append("review_too_short")
        if not has_citations:
            issues.append("missing_citations")
        if not has_key_sections:
            issues.append("missing_required_sections")
        return {
            "passed": not issues,
            "word_count": effective_word_count,
            "raw_word_count": word_count,
            "cjk_char_count": cjk_char_count,
            "has_citations": has_citations,
            "has_key_sections": has_key_sections,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Legacy unified runtime entry point (will be removed in Step 6)
    # ------------------------------------------------------------------

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
        kwargs: dict[str, Any] = dict(
            topic=state.topic,
            task_id=state.task_id,
            papers=state.curated_papers,
            warnings=state.warnings,
            language=language,
            supervisor_instruction=getattr(state, "supervisor_instruction", None),
            skill_context=getattr(state, "skill_context", None),
        )
        expert_context = getattr(state, "expert_context", None)
        if expert_context:
            parameters = inspect.signature(paper_search_service.survey_writer.generate_async).parameters
            if "expert_context" in parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
                kwargs["expert_context"] = expert_context
        base_report = await paper_search_service.survey_writer.generate_async(**kwargs)
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
                    "agent_architecture": "main_agents_plus_tools",
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
        self, *, graph_runtime: Any, state: Any, primary_agents: list[str]
    ) -> QAResponse:
        return await self._collection_answer.answer_collection_question(
            graph_runtime=graph_runtime,
            state=state,
            primary_agents=primary_agents,
            writer_agent=self.name,
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

    def _resolve_answer_language(self, state: Any) -> str:
        explicit_language = str(getattr(state, "answer_language", None) or "").strip()
        if explicit_language:
            return explicit_language
        execution_context = getattr(state, "execution_context", None)
        preference_context = getattr(execution_context, "preference_context", None) or {}
        language = str(preference_context.get("answer_language") or "").strip()
        if language:
            return language
        question = getattr(state, "question", None) or getattr(state, "topic", "")
        return _preferred_answer_language_from_question(question)

    def _require_search_service(self) -> PaperSearchService:
        if self.paper_search_service is None:
            raise RuntimeError("ResearchWriterAgent requires PaperSearchService for report synthesis")
        return self.paper_search_service
