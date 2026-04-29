"""Search literature / create research task supervisor tool."""

from __future__ import annotations

import asyncio
import logging

from agents.literature_scout_agent import LiteratureScoutAgent
from agents.research_supervisor_agent import ResearchSupervisorDecision
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.research import ResearchTaskResponse
from core.utils import now_iso as _now_iso
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
    _llm_stage_timeout_seconds,
    _should_fallback_llm_stage,
    _update_runtime_progress,
    _DISCOVERY_PLAN_TIMEOUT_SECONDS,
    _SURVEY_WRITING_TIMEOUT_SECONDS,
    _TODO_PLANNING_TIMEOUT_SECONDS,
)
from services.research.unified_action_adapters import (
    build_literature_search_input,
    build_literature_search_output,
)
from services.research.research_workspace import build_workspace_state
from services.research.capabilities import PaperCurator

logger = logging.getLogger(__name__)


class CreateResearchTaskTool:
    name = "create_research_task"

    def __init__(
        self,
        *,
        literature_scout_agent: LiteratureScoutAgent,
        research_writer_agent: ResearchWriterAgent,
        curation_skill: PaperCurator,
    ) -> None:
        self.literature_scout_agent = literature_scout_agent
        self.research_writer_agent = research_writer_agent
        self.curation_skill = curation_skill

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        request = context.request
        search_input = build_literature_search_input(context=context, decision=decision)
        create_request = search_input.to_create_research_task_request().model_copy(
            update={"run_immediately": False}
        )
        response = await context.research_service.create_task(
            create_request,
            graph_runtime=context.graph_runtime,
        )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:planning",
            status="running",
            summary="Planning literature discovery queries.",
        )
        planning_timeout_seconds = _llm_stage_timeout_seconds(
            self.literature_scout_agent.reasoning_strategies.llm_adapter,
            fallback_seconds=_DISCOVERY_PLAN_TIMEOUT_SECONDS,
            slack_seconds=10.0,
        )
        try:
            plan = await asyncio.wait_for(
                self.literature_scout_agent.plan(create_request),
                timeout=planning_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning("Discovery planning timed out or hit transport failure; falling back to heuristic planner: %s", exc)
            plan = self.literature_scout_agent._require_search_service().topic_planner.plan(
                topic=create_request.topic,
                days_back=create_request.days_back,
                max_papers=create_request.max_papers,
                sources=create_request.sources,
            )
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:source_search",
            status="running",
            summary="Searching literature sources.",
        )
        scout_state = type(
            "SupervisorDiscoveryState",
            (),
            {
                "topic": create_request.topic,
                "days_back": create_request.days_back,
                "max_papers": create_request.max_papers,
                "sources": create_request.sources,
                "task_id": response.task.task_id,
                "execution_context": context.research_service.build_execution_context(
                    graph_runtime=context.graph_runtime,
                    conversation_id=request.conversation_id,
                    task=response.task,
                    report=None,
                    papers=[],
                    document_ids=[],
                    selected_paper_ids=request.selected_paper_ids,
                    skill_name=request.skill_name,
                    reasoning_style=request.reasoning_style,
                    metadata=request.metadata,
                ),
                "initial_plan": plan,
                "active_queries": list(plan.queries),
                "round_index": 0,
                "max_rounds": 2,
                "queried_pairs": set(),
                "search_completed": False,
                "curation_completed": False,
                "raw_papers": [],
                "trace": [],
                "warnings": [],
                "curated_papers": [],
                "must_read_ids": [],
                "ingest_candidate_ids": [],
                "report": None,
                "todo_items": [],
                "refinement_used": False,
            },
        )()
        raw_papers, warnings = await self.literature_scout_agent.search(scout_state)
        scout_state.warnings = list(warnings)
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:curation",
            status="running",
            summary="Curating candidate papers.",
            extra={"raw_paper_count": len(raw_papers)},
        )
        curated_papers, must_read_ids, ingest_candidate_ids = self.curation_skill.curate(
            topic=create_request.topic,
            raw_papers=raw_papers,
            max_papers=create_request.max_papers,
        )
        scout_state.curated_papers = curated_papers
        scout_state.must_read_ids = must_read_ids
        scout_state.ingest_candidate_ids = ingest_candidate_ids
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:survey_writing",
            status="running",
            summary="Writing literature survey report.",
            extra={"paper_count": len(curated_papers)},
        )
        survey_timeout_seconds = _llm_stage_timeout_seconds(
            self.research_writer_agent.llm_adapter,
            fallback_seconds=_SURVEY_WRITING_TIMEOUT_SECONDS,
            slack_seconds=15.0,
        )
        try:
            report = await asyncio.wait_for(
                self.research_writer_agent.synthesize_async(scout_state),
                timeout=survey_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning("Survey writing timed out or hit transport failure; falling back to heuristic report generation: %s", exc)
            report = self.research_writer_agent.synthesize(scout_state)
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:todo_planning",
            status="running",
            summary="Planning follow-up research todos.",
        )
        todo_timeout_seconds = _llm_stage_timeout_seconds(
            self.research_writer_agent.llm_adapter,
            fallback_seconds=_TODO_PLANNING_TIMEOUT_SECONDS,
            slack_seconds=8.0,
        )
        try:
            todo_items = await asyncio.wait_for(
                self.research_writer_agent.plan_todos_async(scout_state),
                timeout=todo_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning("TODO planning timed out or hit transport failure; falling back to heuristic TODO generation: %s", exc)
            todo_items = self.research_writer_agent.plan_todos(scout_state)
        scout_state.report = report
        scout_state.todo_items = todo_items
        workspace = build_workspace_state(
            objective=create_request.topic,
            stage="complete",
            papers=curated_papers,
            imported_document_ids=[],
            report=report,
            plan=plan,
            todo_items=todo_items,
            must_read_ids=must_read_ids,
            ingest_candidate_ids=ingest_candidate_ids,
            stop_reason="ResearchSupervisorAgent completed direct literature discovery.",
            metadata={
                "decision_model": "supervisor_direct_execution",
                "autonomy_rounds": 1,
                "trace_steps": 0,
            },
        )
        saved_report = report.model_copy(update={"workspace": workspace})
        completed_task = response.task.model_copy(
            update={
                "status": "completed",
                "paper_count": len(curated_papers),
                "report_id": saved_report.report_id,
                "todo_items": todo_items,
                "workspace": workspace,
                "updated_at": _now_iso(),
            }
        )
        context.research_service.report_service.save_papers(completed_task.task_id, curated_papers)
        context.research_service.report_service.save_report(saved_report)
        context.research_service.save_task_state(completed_task, conversation_id=request.conversation_id)
        response = ResearchTaskResponse(
            task=completed_task,
            papers=curated_papers,
            report=saved_report,
            warnings=warnings,
        )
        context.task_response = response
        context.execution_context = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=response.task,
            report=response.report,
            papers=response.papers,
            document_ids=response.task.imported_document_ids,
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )
        if request.conversation_id:
            context.research_service.record_task_turn(request.conversation_id, response=response)
        _update_runtime_progress(
            context,
            stage="search_literature",
            node="search_literature:completed",
            status="completed",
            summary="Literature discovery completed.",
            extra={"paper_count": len(response.papers)},
        )
        output = build_literature_search_output(task_response=response)
        return ResearchToolResult(
            status="succeeded",
            observation=f"created task {response.task.task_id}; papers={len(response.papers)}; report={bool(response.report)}",
            metadata=output.to_metadata(),
        )


class SearchLiteratureTool(CreateResearchTaskTool):
    name = "search_literature"
