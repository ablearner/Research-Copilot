from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTodoItem,
    ResearchTopicPlan,
    ResearchWorkspaceState,
)
from services.research.capabilities import PaperCurator
from services.research.research_context import ResearchExecutionContext
from services.research.research_workspace import build_workspace_state
from services.research.supervisor_tools.base import (
    _DISCOVERY_PLAN_TIMEOUT_SECONDS,
    _SURVEY_WRITING_TIMEOUT_SECONDS,
    _TODO_PLANNING_TIMEOUT_SECONDS,
    _llm_stage_timeout_seconds,
    _should_fallback_llm_stage,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchDiscoveryBundle:
    plan: ResearchTopicPlan
    papers: list[PaperCandidate]
    report: ResearchReport
    workspace: ResearchWorkspaceState
    warnings: list[str]
    todo_items: list[ResearchTodoItem]
    must_read_ids: list[str]
    ingest_candidate_ids: list[str]


class ResearchDiscoveryCapability:
    """Shared research discovery capability used by search and supervisor flows."""

    def __init__(
        self,
        *,
        literature_scout_agent: Any,
        research_writer_agent: Any,
        curation_skill: PaperCurator,
    ) -> None:
        self.literature_scout_agent = literature_scout_agent
        self.research_writer_agent = research_writer_agent
        self.curation_skill = curation_skill

    async def discover(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[str],
        task_id: str | None = None,
        execution_context: ResearchExecutionContext | None = None,
        literature_scout_agent: Any | None = None,
        research_writer_agent: Any | None = None,
        curation_skill: PaperCurator | None = None,
    ) -> ResearchDiscoveryBundle:
        scout_agent = literature_scout_agent or self.literature_scout_agent
        writer_agent = research_writer_agent or self.research_writer_agent
        curator = curation_skill or self.curation_skill
        state = SimpleNamespace(
            topic=topic,
            days_back=days_back,
            max_papers=max_papers,
            sources=sources,
            task_id=task_id,
            execution_context=execution_context,
            max_rounds=2,
            round_index=0,
            queried_pairs=set(),
            search_completed=False,
            curation_completed=False,
            raw_papers=[],
            trace=[],
            warnings=[],
            curated_papers=[],
            must_read_ids=[],
            ingest_candidate_ids=[],
            report=None,
            todo_items=[],
            refinement_used=False,
        )
        state.initial_plan = await self._plan_discovery(state, literature_scout_agent=scout_agent)
        state.active_queries = list(state.initial_plan.queries)
        raw_papers, warnings = await scout_agent.search(state)
        state.warnings = list(warnings)
        curated_papers, must_read_ids, ingest_candidate_ids = curator.curate(
            topic=topic,
            raw_papers=raw_papers,
            max_papers=max_papers,
        )
        state.curated_papers = curated_papers
        state.must_read_ids = must_read_ids
        state.ingest_candidate_ids = ingest_candidate_ids
        state.report = await self._write_report(state, research_writer_agent=writer_agent)
        state.todo_items = await self._plan_todos(state, research_writer_agent=writer_agent)
        workspace = build_workspace_state(
            objective=topic,
            stage="complete",
            papers=curated_papers,
            imported_document_ids=[],
            report=state.report,
            plan=state.initial_plan,
            todo_items=state.todo_items,
            must_read_ids=must_read_ids,
            ingest_candidate_ids=ingest_candidate_ids,
            stop_reason="Research discovery capability completed.",
            metadata={
                "decision_model": "supervisor_direct_execution",
                "autonomy_rounds": 1,
                "trace_steps": 0,
            },
        )
        saved_report = state.report.model_copy(
            update={
                "workspace": workspace,
                "metadata": {
                    **state.report.metadata,
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": "main_agents_plus_skills",
                    "decision_model": "supervisor_direct_execution",
                    "primary_agents": [
                        "ResearchSupervisorAgent",
                        "LiteratureScoutAgent",
                        "ResearchWriterAgent",
                    ],
                    "primary_skills": ["PaperCurator"],
                    "supervisor_agent_architecture": "supervisor_direct_execution",
                    "supervisor_decision_model": "supervisor_direct_execution",
                    "autonomy_rounds": 1,
                    "search_plan": state.initial_plan.model_dump(mode="json"),
                },
            }
        )
        return ResearchDiscoveryBundle(
            plan=state.initial_plan,
            papers=curated_papers,
            report=saved_report,
            workspace=workspace,
            warnings=list(warnings),
            todo_items=list(state.todo_items),
            must_read_ids=list(must_read_ids),
            ingest_candidate_ids=list(ingest_candidate_ids),
        )

    async def _plan_discovery(self, state: Any, *, literature_scout_agent: Any) -> ResearchTopicPlan:
        planning_timeout_seconds = _llm_stage_timeout_seconds(
            literature_scout_agent.llm_adapter,
            fallback_seconds=_DISCOVERY_PLAN_TIMEOUT_SECONDS,
            slack_seconds=10.0,
        )
        try:
            return await asyncio.wait_for(
                literature_scout_agent.plan(state),
                timeout=planning_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning(
                "Discovery planning timed out or hit transport failure; falling back to heuristic planner: %s",
                exc,
            )
            return literature_scout_agent._require_search_service().topic_planner.plan(
                topic=state.topic,
                days_back=state.days_back,
                max_papers=state.max_papers,
                sources=state.sources,
            )

    async def _write_report(self, state: Any, *, research_writer_agent: Any) -> ResearchReport:
        survey_timeout_seconds = _llm_stage_timeout_seconds(
            research_writer_agent.llm_adapter,
            fallback_seconds=_SURVEY_WRITING_TIMEOUT_SECONDS,
            slack_seconds=15.0,
        )
        try:
            return await asyncio.wait_for(
                research_writer_agent.synthesize_async(state),
                timeout=survey_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning(
                "Survey writing timed out or hit transport failure; falling back to heuristic report generation: %s",
                exc,
            )
            return research_writer_agent.synthesize(state)

    async def _plan_todos(self, state: Any, *, research_writer_agent: Any) -> list[ResearchTodoItem]:
        todo_timeout_seconds = _llm_stage_timeout_seconds(
            research_writer_agent.llm_adapter,
            fallback_seconds=_TODO_PLANNING_TIMEOUT_SECONDS,
            slack_seconds=8.0,
        )
        try:
            return await asyncio.wait_for(
                research_writer_agent.plan_todos_async(state),
                timeout=todo_timeout_seconds,
            )
        except Exception as exc:
            if not _should_fallback_llm_stage(exc):
                raise
            logger.warning(
                "TODO planning timed out or hit transport failure; falling back to heuristic TODO generation: %s",
                exc,
            )
            return research_writer_agent.plan_todos(state)
