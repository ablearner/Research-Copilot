"""Search literature / create research task supervisor tool."""

from __future__ import annotations

from agents.research_supervisor_agent import ResearchSupervisorDecision
from core.utils import now_iso as _now_iso
from domain.schemas.research import ResearchReport, ResearchTask, ResearchTaskResponse
from services.research.research_workspace import build_workspace_from_task
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
    _update_runtime_progress,
)
from services.research.unified_action_adapters import (
    build_literature_search_input,
    build_literature_search_output,
)


class CreateResearchTaskTool:
    name = "create_research_task"

    def __init__(
        self,
        *,
        literature_scout_agent=None,
        research_writer_agent=None,
        curation_skill=None,
    ) -> None:
        self.literature_scout_agent = literature_scout_agent
        self.research_writer_agent = research_writer_agent
        self.curation_skill = curation_skill

    def _search_metadata(self, *, topic: str, bundle) -> dict[str, object]:
        return {
            "last_search_query": topic,
            "last_search_discovered_paper_ids": [paper.paper_id for paper in bundle.papers],
            "last_search_discovered_count": len(bundle.papers),
            "search_plan": bundle.plan.model_dump(mode="json"),
        }

    def _build_discovery_execution_context(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        report: ResearchReport | None,
        papers,
    ):
        request = context.request
        return context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=task,
            report=report,
            papers=list(papers),
            document_ids=list(task.imported_document_ids),
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )

    def _persist_new_task_response(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        bundle,
        topic: str,
    ) -> ResearchTaskResponse:
        search_metadata = self._search_metadata(topic=topic, bundle=bundle)
        saved_report = bundle.report.model_copy(
            update={"metadata": {**bundle.report.metadata, **search_metadata}}
        )
        completed_task = task.model_copy(
            update={
                "status": "completed",
                "paper_count": len(bundle.papers),
                "report_id": saved_report.report_id,
                "todo_items": bundle.todo_items,
                "workspace": bundle.workspace,
                "updated_at": _now_iso(),
                "metadata": {**task.metadata, **search_metadata},
            }
        )
        context.research_service.report_service.save_papers(completed_task.task_id, bundle.papers)
        context.research_service.report_service.save_report(saved_report)
        context.research_service.save_task_state(completed_task, conversation_id=context.request.conversation_id)
        return ResearchTaskResponse(
            task=completed_task,
            papers=bundle.papers,
            report=saved_report,
            warnings=bundle.warnings,
        )

    def _persist_existing_task_response(
        self,
        *,
        context: ResearchAgentToolContext,
        task: ResearchTask,
        bundle,
        topic: str,
    ) -> ResearchTaskResponse:
        existing_papers = context.research_service.report_service.load_papers(task.task_id)
        merged_papers = context.research_service._refresh_existing_pool(
            existing_papers=existing_papers,
            incoming_papers=bundle.papers,
            ranking_topic=topic,
        )
        existing_report = context.research_service.report_service.load_report(task.task_id, task.report_id)
        search_metadata = self._search_metadata(topic=topic, bundle=bundle)
        refreshed_report = bundle.report.model_copy(
            update={
                "task_id": task.task_id,
                "report_id": existing_report.report_id if existing_report is not None else bundle.report.report_id,
                "generated_at": _now_iso(),
                "metadata": {
                    **(existing_report.metadata if existing_report is not None else {}),
                    **bundle.report.metadata,
                    **search_metadata,
                },
            }
        )
        if existing_report is not None:
            markdown = refreshed_report.markdown.rstrip()
            qa_section = context.research_service._extract_markdown_section(existing_report.markdown, "## 研究集合问答补充")
            todo_section = context.research_service._extract_markdown_section(existing_report.markdown, "## TODO 执行记录")
            if qa_section:
                markdown = f"{markdown}\n\n{qa_section.strip()}"
            if todo_section:
                markdown = f"{markdown}\n\n{todo_section.strip()}"
            carry_highlights = [
                item for item in existing_report.highlights
                if item.startswith("问答补充：") or item.startswith("TODO执行：")
            ]
            refreshed_report = refreshed_report.model_copy(
                update={
                    "markdown": markdown,
                    "highlights": context.research_service._merge_text_entries(
                        refreshed_report.highlights,
                        carry_highlights,
                        limit=12,
                    ),
                    "gaps": context.research_service._merge_text_entries(
                        refreshed_report.gaps,
                        list(existing_report.gaps),
                        limit=12,
                    ),
                }
            )
        updated_task = task.model_copy(
            update={
                "status": "completed",
                "paper_count": len(merged_papers),
                "report_id": refreshed_report.report_id,
                "updated_at": _now_iso(),
                "metadata": {**task.metadata, **search_metadata},
            }
        )
        updated_workspace = build_workspace_from_task(
            task=updated_task,
            report=refreshed_report,
            papers=merged_papers,
            plan=bundle.plan,
            stop_reason="Literature discovery refreshed the current research workspace.",
            metadata={
                **dict(updated_task.workspace.metadata),
                "last_search_query": topic,
                "last_search_discovered_count": len(bundle.papers),
            },
        )
        refreshed_report = refreshed_report.model_copy(update={"workspace": updated_workspace})
        updated_task = updated_task.model_copy(update={"workspace": updated_workspace})
        context.research_service.report_service.save_papers(updated_task.task_id, merged_papers)
        context.research_service.report_service.save_report(refreshed_report)
        context.research_service.save_task_state(updated_task, conversation_id=context.request.conversation_id)
        return ResearchTaskResponse(
            task=updated_task,
            papers=merged_papers,
            report=refreshed_report,
            warnings=bundle.warnings,
        )

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        request = context.request
        search_input = build_literature_search_input(context=context, decision=decision)
        search_request = search_input.to_create_research_task_request().model_copy(
            update={"run_immediately": False}
        )
        task_response = context.task_response
        active_task = task_response.task if task_response is not None else None
        if active_task is None:
            task_response = await context.research_service.create_task(
                search_request,
                graph_runtime=context.graph_runtime,
            )
            active_task = task_response.task
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:planning",
            status="running",
            summary="Planning literature discovery queries.",
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:source_search",
            status="running",
            summary="Searching literature sources.",
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:curation",
            status="running",
            summary="Curating candidate papers.",
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:survey_writing",
            status="running",
            summary="Writing literature survey report.",
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:todo_planning",
            status="running",
            summary="Planning follow-up research todos.",
        )
        bundle = await context.research_service.research_discovery_capability.discover(
            topic=search_request.topic,
            days_back=search_request.days_back,
            max_papers=search_request.max_papers,
            sources=list(search_request.sources),
            task_id=active_task.task_id,
            execution_context=self._build_discovery_execution_context(
                context=context,
                task=active_task,
                report=task_response.report if task_response is not None else None,
                papers=task_response.papers if task_response is not None else [],
            ),
            literature_scout_agent=self.literature_scout_agent,
            research_writer_agent=self.research_writer_agent,
            curation_skill=self.curation_skill,
        )
        response = (
            self._persist_existing_task_response(
                context=context,
                task=active_task,
                bundle=bundle,
                topic=search_request.topic,
            )
            if context.task is not None
            else self._persist_new_task_response(
                context=context,
                task=active_task,
                bundle=bundle,
                topic=search_request.topic,
            )
        )
        context.task_response = response
        context.execution_context = self._build_discovery_execution_context(
            context=context,
            task=response.task,
            report=response.report,
            papers=response.papers,
        )
        if request.conversation_id:
            context.research_service.record_task_turn(request.conversation_id, response=response)
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:completed",
            status="completed",
            summary="Literature discovery completed.",
            extra={"paper_count": len(response.papers), "query_count": len(bundle.plan.queries)},
        )
        output = build_literature_search_output(task_response=response)
        action_label = "updated task" if task_response is not None else "created task"
        return ResearchToolResult(
            status="succeeded",
            observation=f"{action_label} {response.task.task_id}; papers={len(response.papers)}; report={bool(response.report)}",
            metadata=output.to_metadata(),
        )


class SearchLiteratureTool(CreateResearchTaskTool):
    name = "search_literature"
