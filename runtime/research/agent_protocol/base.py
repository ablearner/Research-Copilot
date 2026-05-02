"""Shared types, protocols and helper functions used by all supervisor tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Awaitable, Callable, Protocol, TypedDict
from uuid import uuid4

import httpx

from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import (
    ImportPapersResponse,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchAgentTraceStep,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskResponse,
    ResearchTaskResponse,
)
from domain.schemas.research_functions import AnalyzePapersFunctionOutput, RecommendPapersFunctionOutput
from core.utils import now_iso as _now_iso
from domain.schemas.research_context import ResearchExecutionContext

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.research.literature_research_service import LiteratureResearchService

logger = logging.getLogger(__name__)

_DISCOVERY_PLAN_TIMEOUT_SECONDS = 12.0
_SURVEY_WRITING_TIMEOUT_SECONDS = 45.0
_TODO_PLANNING_TIMEOUT_SECONDS = 20.0
_MAX_LLM_STAGE_TIMEOUT_SECONDS = 180.0


def _llm_stage_timeout_seconds(
    adapter: Any | None,
    *,
    fallback_seconds: float,
    slack_seconds: float,
) -> float:
    configured_timeout = getattr(adapter, "timeout_seconds", None)
    provider_binding = getattr(adapter, "provider_binding", None)
    if configured_timeout is None and provider_binding is not None:
        configured_timeout = getattr(provider_binding, "timeout_seconds", None)
    if isinstance(configured_timeout, (int, float)) and configured_timeout > 0:
        return min(max(float(fallback_seconds), float(configured_timeout) + float(slack_seconds)), _MAX_LLM_STAGE_TIMEOUT_SECONDS)
    return float(fallback_seconds)


def _should_fallback_llm_stage(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | asyncio.TimeoutError | httpx.TimeoutException | httpx.TransportError):
        return True
    cause = exc.__cause__
    if isinstance(cause, Exception):
        return _should_fallback_llm_stage(cause)
    return False


def _message(
    *,
    role: str,
    kind: str,
    title: str,
    content: str = "",
    meta: str | None = None,
    payload: dict[str, Any] | None = None,
) -> ResearchMessage:
    return ResearchMessage(
        message_id=f"msg_{uuid4().hex}",
        role=role,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        title=title,
        content=content,
        meta=meta,
        created_at=_now_iso(),
        payload=payload or {},
    )


@dataclass(slots=True)
class ResearchAgentToolContext:
    request: ResearchAgentRunRequest
    research_service: LiteratureResearchService
    graph_runtime: Any
    execution_context: ResearchExecutionContext | None = None
    task_response: ResearchTaskResponse | None = None
    import_result: ImportPapersResponse | None = None
    qa_result: ResearchTaskAskResponse | None = None
    paper_analysis_result: AnalyzePapersFunctionOutput | None = None
    preference_recommendation_result: RecommendPapersFunctionOutput | None = None
    compressed_context_summary: dict[str, Any] | None = None
    parsed_document: Any | None = None
    document_index_result: dict[str, Any] | None = None
    chart_result: Any | None = None
    general_answer: str | None = None
    general_answer_metadata: dict[str, Any] | None = None
    import_attempted: bool = False
    zotero_sync_results: list[dict[str, Any]] | None = None
    document_attempted: bool = False
    chart_attempted: bool = False
    warnings: list[str] | None = None
    unified_blueprint: dict[str, Any] | None = None
    unified_agent_registry: Any | None = None
    unified_runtime_context: Any | None = None
    runtime_progress: dict[str, Any] | None = None
    progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    skill_context: str | None = None
    skill_selection: Any | None = None
    knowledge_access: Any | None = None

    @property
    def task(self) -> ResearchTask | None:
        return self.task_response.task if self.task_response else None

    @property
    def papers(self) -> list[PaperCandidate]:
        return self.task_response.papers if self.task_response else []

    @property
    def report(self) -> ResearchReport | None:
        return self.task_response.report if self.task_response else None

    @property
    def workspace(self):
        if self.task is not None:
            return self.task.workspace
        if self.report is not None:
            return self.report.workspace
        return None


STATE_DELTA_METADATA_KEY = "__state_delta__"


@dataclass
class MemoryOp:
    """A single deferred memory operation to be executed by the Runtime."""

    op_type: str  # e.g. "set_active_papers", "persist_research_update", "promote_conclusion", "update_paper_knowledge", "record_turn"
    params: dict[str, Any]


@dataclass
class ResearchStateDelta:
    """State mutations proposed by a specialist, applied and persisted by the Runtime.

    Specialists MUST NOT call save/persist methods directly. Instead they populate
    this delta and return it as part of ResearchToolResult. The Runtime extracts the
    delta and calls the unified _apply_state_delta() → _persist_state_delta() path.
    """

    # -- Persistence targets (saved to storage by Runtime) --
    task: ResearchTask | None = None
    papers: list[PaperCandidate] | None = None
    report: ResearchReport | None = None
    save_task_conversation_id: str | None = None
    save_task_event_type: str | None = None
    save_task_event_payload: dict[str, Any] | None = None

    # -- Context mutations (applied to ResearchAgentToolContext by Runtime) --
    task_response: ResearchTaskResponse | None = None
    qa_result: ResearchTaskAskResponse | None = None
    paper_analysis_result: AnalyzePapersFunctionOutput | None = None
    preference_recommendation_result: RecommendPapersFunctionOutput | None = None
    import_result: ImportPapersResponse | None = None
    compressed_context_summary: dict[str, Any] | None = None
    rebuild_execution_context: bool = False
    rebuild_execution_context_params: dict[str, Any] | None = None

    # -- Memory operations (executed by Runtime via memory_gateway) --
    memory_ops: list[MemoryOp] | None = None

    # -- Conversation recording --
    record_task_turn: bool = False


@dataclass(slots=True)
class ResearchToolResult:
    status: str
    observation: str
    metadata: dict[str, Any]
    state_delta: ResearchStateDelta | None = None


def _update_runtime_progress(
    context: ResearchAgentToolContext,
    *,
    stage: str,
    node: str,
    status: str,
    summary: str,
    extra: dict[str, Any] | None = None,
) -> None:
    progress = {
        "stage": stage,
        "node": node,
        "status": status,
        "summary": summary,
        "updated_at": _now_iso(),
        **dict(extra or {}),
    }
    context.runtime_progress = progress
    if context.progress_callback is not None:
        try:
            asyncio.get_event_loop().create_task(context.progress_callback(progress))
        except RuntimeError:
            pass
    context.research_service.append_runtime_event(
        conversation_id=context.request.conversation_id,
        event_type="memory_updated",
        task_id=context.task.task_id if context.task is not None else context.request.task_id,
        correlation_id=(
            context.task.status_metadata.correlation_id
            if context.task is not None
            else None
        ),
        payload={
            "runtime_event": "supervisor_progress",
            **progress,
        },
    )


def _observation_envelope(
    *,
    progress_made: bool,
    confidence: float | None = None,
    missing_inputs: list[str] | None = None,
    suggested_next_actions: list[str] | None = None,
    state_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "progress_made": progress_made,
        "confidence": confidence,
        "missing_inputs": list(missing_inputs or []),
        "suggested_next_actions": list(suggested_next_actions or []),
        "state_delta": dict(state_delta or {}),
    }


class ResearchAgentGraphState(TypedDict, total=False):
    context: ResearchAgentToolContext
    trace: list[ResearchAgentTraceStep]
    current_decision: ResearchSupervisorDecision | None
    current_step_index: int
    failed: bool
    exhausted: bool
    pending_agent_messages: list[AgentMessage]
    agent_messages: list[AgentMessage]
    agent_results: list[AgentResultMessage]
    completed_agent_task_ids: list[str]
    failed_agent_task_ids: list[str]
    replanned_failure_task_ids: list[str]
    planner_runs: int
    replan_count: int
    clarification_request: str | None
    active_plan_id: str | None
    progress_signature: str
    stagnant_decision_count: int
    repeated_action_count: int
    new_topic_detected: bool


class ResearchAgentTool(Protocol):
    name: str

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        ...


class SpecialistAgent(Protocol):
    """Unified interface for specialist agents that execute supervisor actions directly."""

    name: str

    async def run_action(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        ...


# Re-export for backward-compatible imports from this module
from agents.research_supervisor_agent import ResearchSupervisorDecision  # noqa: E402
