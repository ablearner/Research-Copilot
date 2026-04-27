from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
import re
from datetime import UTC, datetime
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.literature_scout_agent import LiteratureScoutAgent
from agents.preference_memory_agent import PreferenceMemoryAgent
from agents.research_knowledge_agent import ResearchKnowledgeAgent, merge_retrieval_hits
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    ResearchAdvancedStrategy,
    CreateResearchConversationRequest,
    CreateResearchTaskRequest,
    ImportPapersRequest,
    ImportPapersResponse,
    ImportedPaperResult,
    PaperCandidate,
    ResearchPaperFigureListResponse,
    ResearchPaperFigurePreview,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversation,
    ResearchConversationResponse,
    ResearchConversationSnapshot,
    ResearchRouteMode,
    ResearchContextSummary,
    ResearchJob,
    ResearchReport,
    ResearchMessage,
    ResearchRuntimeEvent,
    ResearchStatusMetadata,
    ResearchThreadSnapshot,
    ResearchTaskAskRequest,
    ResearchTaskAskResponse,
    ResearchTodoActionRequest,
    ResearchTodoActionResponse,
    ResearchTask,
    ResearchTaskResponse,
    ResearchTodoItem,
    ResearchWorkspaceState,
    ResearchLifecycleStatus,
    SearchPapersRequest,
    SearchPapersResponse,
)
from memory.memory_manager import MemoryManager
from memory.long_term_memory import JsonLongTermMemoryStore, LongTermMemory
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from memory.session_memory import JsonSessionMemoryStore, SessionMemory
from skills.research import (
    PaperReadingSkill,
    ResearchQARoutingSkill,
    ResearchEvaluationSkill,
    ResearchUserIntentResolverSkill,
    ReviewWritingSkill,
    WritingPolishSkill,
)
from skills.research.paper_curation import PaperCurationSkill
from services.research.research_context import ResearchExecutionContext
from services.research.research_context_manager import ResearchContextManager
from services.research.paper_selector_service import PaperSelectionScope, PaperSelectorService
from services.research.paper_import_service import PaperImportService
from services.research.observability_service import ResearchObservabilityService
from services.research.paper_search_service import PaperSearchService
from services.research.research_report_service import ResearchReportService
from services.research.research_workspace import build_workspace_from_task, build_workspace_state
from tools.paper_figure_toolkit import PaperFigureAnalyzeTarget

logger = logging.getLogger(__name__)


def _normalize_paper_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def _preferred_answer_language_from_text(text: str | None) -> str:
    value = str(text or "").strip()
    if not value:
        return "zh-CN"
    cjk_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in value if ("a" <= char.lower() <= "z"))
    if cjk_count > 0 and cjk_count >= max(1, latin_count // 2):
        return "zh-CN"
    return "en-US"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_topic_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text or "").lower())).strip()


ImportProgressCallback = Callable[[int, int, ImportedPaperResult], Awaitable[None] | None]
INTERNAL_CONVERSATION_MESSAGE_TITLES = {
    "Manager 任务分解",
    "Research Workspace",
    "Agent 决策轨迹",
    "TODO 动作回执",
    "Research-Copilot",
}
@dataclass(slots=True)
class ResearchQARouteDecision:
    route: str
    confidence: float
    rationale: str
    visual_anchor: dict[str, Any] | None = None
    recovery_count: int = 0


class LiteratureResearchService:
    """MVP coordination layer for literature search, persistence, and report retrieval."""

    def __init__(
        self,
        *,
        paper_search_service: PaperSearchService,
        report_service: ResearchReportService,
        paper_import_service: PaperImportService,
        research_runtime: Any | None = None,
        research_qa_runtime: Any | None = None,
        research_context_manager: ResearchContextManager | None = None,
        memory_manager: MemoryManager | None = None,
        paper_selector_service: PaperSelectorService | None = None,
        paper_reading_skill: PaperReadingSkill | None = None,
        evaluation_skill: ResearchEvaluationSkill | None = None,
        review_writing_skill: ReviewWritingSkill | None = None,
        writing_polish_skill: WritingPolishSkill | None = None,
        import_concurrency: int = 2,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.report_service = report_service
        self.paper_import_service = paper_import_service
        # Legacy compatibility only. Main business paths no longer depend on
        # the old discovery/QA manager runtimes by default.
        self.research_runtime = research_runtime
        self.research_qa_runtime = research_qa_runtime
        self._agent_runtime: Any | None = None
        self.research_context_manager = research_context_manager or ResearchContextManager()
        self.memory_manager = memory_manager or MemoryManager(
            session_memory=SessionMemory(
                JsonSessionMemoryStore(report_service.storage_root / "memory" / "sessions")
            ),
            long_term_memory=LongTermMemory(
                JsonLongTermMemoryStore(report_service.storage_root / "memory" / "long_term")
            ),
            paper_knowledge_memory=PaperKnowledgeMemory(
                JsonPaperKnowledgeStore(report_service.storage_root / "memory" / "paper_knowledge")
            ),
        )
        self.paper_selector_service = paper_selector_service or PaperSelectorService()
        self.paper_reading_skill = paper_reading_skill or PaperReadingSkill()
        llm_adapter = getattr(paper_search_service, "llm_adapter", None)
        self.llm_adapter = llm_adapter
        self.literature_scout_agent = LiteratureScoutAgent(
            paper_search_service,
        )
        self.paper_curation_skill = PaperCurationSkill(paper_search_service)
        self.research_knowledge_agent = ResearchKnowledgeAgent()
        self.research_writer_agent = ResearchWriterAgent(
            paper_search_service,
            llm_adapter=llm_adapter,
        )
        self.qa_routing_skill = ResearchQARoutingSkill(llm_adapter=llm_adapter)
        self.user_intent_resolver = ResearchUserIntentResolverSkill(llm_adapter=llm_adapter)
        self.chart_analysis_agent = ChartAnalysisAgent(
            llm_adapter=llm_adapter,
            storage_root=report_service.storage_root / "assets",
        )
        self.preference_memory_agent = PreferenceMemoryAgent(
            memory_manager=self.memory_manager,
            paper_search_service=self.paper_search_service,
            storage_root=report_service.storage_root,
        )
        self.evaluation_skill = evaluation_skill or ResearchEvaluationSkill()
        self.review_writing_skill = review_writing_skill or ReviewWritingSkill()
        self.writing_polish_skill = writing_polish_skill or WritingPolishSkill()
        self.import_concurrency = max(1, import_concurrency)
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self.observability_service = ResearchObservabilityService(
            report_service.storage_root / "observability"
        )

    def _find_conversation_id_for_task(self, task_id: str | None) -> str | None:
        if not task_id:
            return None
        for conversation in self.report_service.list_conversations():
            if conversation.task_id == task_id:
                return conversation.conversation_id
        return None

    def _resolve_research_session_id(
        self,
        *,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        resolved_conversation_id = conversation_id or self._find_conversation_id_for_task(task_id)
        return resolved_conversation_id or task_id, resolved_conversation_id

    def build_execution_context(
        self,
        *,
        graph_runtime: Any,
        conversation_id: str | None = None,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        papers: list[PaperCandidate] | None = None,
        document_ids: list[str] | None = None,
        selected_paper_ids: list[str] | None = None,
        skill_name: str | None = None,
        reasoning_style: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchExecutionContext:
        session_id, resolved_conversation_id = self._resolve_research_session_id(
            conversation_id=conversation_id,
            task_id=task.task_id if task else None,
        )
        conversation = (
            self.report_service.load_conversation(resolved_conversation_id)
            if resolved_conversation_id
            else None
        )
        snapshot = conversation.snapshot if conversation else None
        if task is None and snapshot and snapshot.task_result is not None:
            task = snapshot.task_result.task
        if report is None and snapshot and snapshot.task_result is not None:
            report = snapshot.task_result.report
        if papers is None and snapshot and snapshot.task_result is not None:
            papers = snapshot.task_result.papers

        session_memory = getattr(graph_runtime, "session_memory", None)
        memory_snapshot = (
            session_memory.load(session_id)
            if session_id and session_memory is not None and hasattr(session_memory, "load")
            else None
        )
        session_context = (
            session_memory.as_prompt_context(memory_snapshot)
            if session_memory is not None and hasattr(session_memory, "as_prompt_context")
            else {"memory_enabled": False}
        )
        research_history = (
            session_memory.research_history(session_id)
            if session_id and session_memory is not None and hasattr(session_memory, "research_history")
            else []
        )
        recent_messages = (
            self.report_service.load_messages(resolved_conversation_id)[-6:]
            if resolved_conversation_id
            else []
        )
        resolved_selected_paper_ids = list(
            dict.fromkeys(selected_paper_ids or (snapshot.selected_paper_ids if snapshot else []))
        )
        resolved_document_ids = list(
            document_ids or (task.imported_document_ids if task else [])
        )
        resolved_papers = list(papers or [])
        task_context = {
            "task_id": task.task_id if task else None,
            "research_topic": (task.topic if task else None) or (snapshot.topic if snapshot else ""),
            "days_back": task.days_back if task else (snapshot.days_back if snapshot else None),
            "max_papers": task.max_papers if task else (snapshot.max_papers if snapshot else None),
            "sources": list(task.sources if task else (snapshot.sources if snapshot else [])),
            "paper_count": len(resolved_papers) if resolved_papers else (task.paper_count if task else 0),
            "report_id": report.report_id if report else (task.report_id if task else None),
            "document_ids": resolved_document_ids[:12],
            "selected_paper_ids": resolved_selected_paper_ids[:8],
            "report_highlights": report.highlights[:3] if report else [],
            "report_gaps": report.gaps[:2] if report else [],
            "paper_titles": [paper.title for paper in resolved_papers[:5]],
            "workspace_stage": task.workspace.current_stage if task else snapshot.workspace.current_stage if snapshot else "discover",
            "workspace_summary": task.workspace.status_summary if task else snapshot.workspace.status_summary if snapshot else "",
            "workspace_next_actions": (
                task.workspace.next_actions[:4]
                if task
                else snapshot.workspace.next_actions[:4]
                if snapshot
                else []
            ),
        }
        conversation_context = {
            "conversation_id": resolved_conversation_id,
            "conversation_title": conversation.title if conversation else None,
            "composer_mode": snapshot.composer_mode if snapshot else None,
            "active_route_mode": snapshot.active_route_mode if snapshot else None,
            "active_thread_id": snapshot.active_thread_id if snapshot else None,
            "selected_paper_ids": resolved_selected_paper_ids[:8],
            "last_notice": snapshot.last_notice if snapshot else None,
            "active_job_id": snapshot.active_job_id if snapshot else None,
            "thread_history": [
                {
                    "thread_id": thread.thread_id,
                    "route_mode": thread.route_mode,
                    "topic": thread.topic,
                    "task_id": thread.task_id,
                    "last_user_message": thread.last_user_message,
                    "workspace_summary": thread.metadata.get("workspace_summary"),
                    "next_actions": thread.metadata.get("next_actions", []),
                    "stop_reason": thread.metadata.get("stop_reason"),
                }
                for thread in (snapshot.thread_history[-4:] if snapshot else [])
            ],
            "recent_messages": [
                {
                    "role": message.role,
                    "kind": message.kind,
                    "title": message.title,
                    "content": self._compact_text(message.content or message.title, limit=160),
                }
                for message in recent_messages
                if (message.content or message.title)
            ],
        }
        user_profile = self.memory_manager.load_user_profile()
        metadata_context = {}
        if isinstance(metadata, dict) and isinstance(metadata.get("context"), dict):
            metadata_context = dict(metadata["context"])
        latest_user_message = (
            str(metadata.get("user_message")).strip()
            if isinstance(metadata, dict) and str(metadata.get("user_message") or "").strip()
            else next(
                (
                    str(item.get("content") or "").strip()
                    for item in reversed(conversation_context["recent_messages"])
                    if item.get("role") == "user" and str(item.get("content") or "").strip()
                ),
                "",
            )
        )
        answer_language = (
            str(metadata_context.get("answer_language") or "").strip()
            or _preferred_answer_language_from_text(latest_user_message)
            or str(getattr(user_profile, "preferred_answer_language", "") or "").strip()
        )
        preference_context = {
            "skill_name": skill_name,
            "reasoning_style": reasoning_style or "cot",
            "composer_mode": snapshot.composer_mode if snapshot else None,
            "answer_language": answer_language,
            "follow_user_language": True,
            "preserve_paper_title_language": True,
        }
        compressed_summaries = self.research_context_manager.compress_papers(
            papers=resolved_papers,
            selected_paper_ids=resolved_selected_paper_ids,
            paper_reading_skill=self.paper_reading_skill,
        )
        research_context = self.research_context_manager.build_from_artifacts(
            task=task,
            report=report,
            papers=resolved_papers,
            selected_paper_ids=resolved_selected_paper_ids,
            history_entries=research_history,
            paper_summaries=compressed_summaries,
            metadata={
                "conversation_id": resolved_conversation_id,
                "workspace_stage": task_context["workspace_stage"],
            },
        )
        if session_id:
            research_context = self.memory_manager.hydrate_context(
                session_id,
                base_context=research_context,
            )
        memory_hints = {
            **session_context,
            "session_id": session_id,
            "research_history": research_history,
            "recalled_memories": list(research_context.metadata.get("recalled_memories") or []),
            "conversation": conversation_context,
            "task": task_context,
            "user_profile": {
                "research_interests": list(user_profile.research_interests[:6]),
                "interest_topics": [item.model_dump(mode="json") for item in user_profile.interest_topics[:6]],
                "preferred_sources": list(user_profile.preferred_sources[:6]),
                "preferred_keywords": list(user_profile.preferred_keywords[:10]),
            },
            "answer_language": answer_language,
            **metadata_context,
        }
        return ResearchExecutionContext(
            session_id=session_id,
            session_context=session_context,
            memory_hints=memory_hints,
            task_context=task_context,
            preference_context=preference_context,
            conversation_context=conversation_context,
            research_context=research_context,
            context_slices=self.build_context_slices(
                research_context,
                selected_paper_ids=resolved_selected_paper_ids,
            ),
        )

    def build_context_slices(
        self,
        research_context,
        *,
        selected_paper_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "default": self.research_context_manager.slice_for_agent(
                research_context,
                paper_ids=selected_paper_ids or None,
            ),
            "manager": self.research_context_manager.slice_for_agent(
                research_context,
                paper_ids=selected_paper_ids or None,
                agent_scope="manager",
                summary_level="section",
            ),
            "research": self.research_context_manager.slice_for_agent(
                research_context,
                paper_ids=selected_paper_ids or None,
                agent_scope="sub_manager",
                sub_manager_key="research",
                summary_level="document",
            ),
            "writing": self.research_context_manager.slice_for_agent(
                research_context,
                paper_ids=selected_paper_ids or None,
                agent_scope="sub_manager",
                sub_manager_key="writing",
                summary_level="section",
            ),
        }

    def _update_research_memory(
        self,
        *,
        graph_runtime: Any | None,
        conversation_id: str | None = None,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        papers: list[PaperCandidate] | None = None,
        document_ids: list[str] | None = None,
        selected_paper_ids: list[str] | None = None,
        task_intent: str,
        question: str | None = None,
        answer: str | None = None,
        retrieval_summary: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> None:
        session_id, resolved_conversation_id = self._resolve_research_session_id(
            conversation_id=conversation_id,
            task_id=task.task_id if task else None,
        )
        if not session_id:
            return
        session_memory = getattr(graph_runtime, "session_memory", None) if graph_runtime is not None else None
        resolved_document_ids = list(document_ids or (task.imported_document_ids if task else []))
        resolved_selected_paper_ids = list(dict.fromkeys(selected_paper_ids or []))
        resolved_papers = list(papers or [])
        report_summary = ""
        if report is not None and report.highlights:
            report_summary = self._compact_text("；".join(report.highlights[:2]), limit=280)
        elif report is not None:
            report_summary = self._compact_text(report.markdown, limit=280)
        cleaned_metadata = {
            "conversation_id": resolved_conversation_id,
            "task_id": task.task_id if task else None,
            "research_topic": task.topic if task else None,
            "paper_count": len(resolved_papers) if papers is not None else (task.paper_count if task else None),
            "document_ids": resolved_document_ids[:12],
            "selected_paper_ids": resolved_selected_paper_ids[:8] or None,
            "report_highlights": report.highlights[:3] if report else None,
            "report_gaps": report.gaps[:2] if report else None,
            **(metadata_update or {}),
        }
        research_context = self.research_context_manager.build_from_artifacts(
            task=task,
            report=report,
            papers=resolved_papers,
            selected_paper_ids=resolved_selected_paper_ids,
            paper_summaries=self.research_context_manager.compress_papers(
                papers=resolved_papers,
                selected_paper_ids=resolved_selected_paper_ids,
                paper_reading_skill=self.paper_reading_skill,
            ),
            metadata=cleaned_metadata,
        )
        hydrated_context = self.memory_manager.hydrate_context(
            session_id,
            base_context=research_context,
        )
        merged_context = self.research_context_manager.update_context(
            current_context=hydrated_context,
            topic=research_context.research_topic,
            goals=research_context.research_goals,
            selected_papers=resolved_selected_paper_ids,
            imported_papers=research_context.imported_papers,
            known_conclusions=research_context.known_conclusions,
            open_questions=research_context.open_questions,
            paper_summaries=research_context.paper_summaries,
            current_task_plan=hydrated_context.current_task_plan,
            sub_manager_states=hydrated_context.sub_manager_states,
            metadata=cleaned_metadata,
        )
        self.memory_manager.save_context(session_id, merged_context)
        if retrieval_summary:
            self.memory_manager.working_memory.append_intermediate_step(
                session_id=session_id,
                content=retrieval_summary,
                step_type="retrieve",
                metadata={
                    "task_intent": task_intent,
                    "document_ids": resolved_document_ids[:8],
                },
            )
        if question and answer:
            self.memory_manager.record_turn(
                session_id,
                question=question,
                answer=answer,
                selected_paper_ids=resolved_selected_paper_ids,
                metadata={
                    "task_id": task.task_id if task else None,
                    "conversation_id": resolved_conversation_id,
                    "document_ids": resolved_document_ids,
                    "task_intent": task_intent,
                    "paper_count": len(resolved_papers) if papers is not None else (task.paper_count if task else 0),
                },
            )
        if session_memory is not None and hasattr(session_memory, "update_research_context"):
            session_memory.update_research_context(
                session_id=session_id,
                current_document_id=(resolved_document_ids or [None])[0],
                last_retrieval_summary=retrieval_summary,
                last_answer_summary=(
                    self._compact_text(answer, limit=280) if answer else report_summary or None
                ),
                current_task_intent=task_intent,
                metadata_update=cleaned_metadata,
            )
        if question and answer and session_memory is not None and hasattr(session_memory, "append_research_turn"):
            session_memory.append_research_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                task_id=task.task_id if task else None,
                conversation_id=resolved_conversation_id,
                document_ids=resolved_document_ids,
                metadata={
                    "task_intent": task_intent,
                    "paper_count": len(resolved_papers) if papers is not None else (task.paper_count if task else 0),
                },
            )

    def list_conversations(self) -> list[ResearchConversation]:
        return self.report_service.list_conversations()

    def _topic_continuity_score(self, current: str | None, previous: str | None) -> float:
        current_tokens = set(_normalize_topic_text(current).split())
        previous_tokens = set(_normalize_topic_text(previous).split())
        if not current_tokens or not previous_tokens:
            return 0.0
        overlap = len(current_tokens & previous_tokens)
        union = len(current_tokens | previous_tokens)
        return overlap / max(1, union)

    def _looks_like_general_chat(self, message: str | None) -> bool:
        normalized = _normalize_topic_text(message)
        if not normalized:
            return False
        general_markers = {
            "你好",
            "您好",
            "hello",
            "hi",
            "天气",
            "翻译",
            "讲个笑话",
            "what is",
            "how are you",
        }
        research_markers = {
            "论文",
            "paper",
            "文献",
            "调研",
            "research",
            "survey",
            "compare",
            "chart",
            "document",
            "pdf",
        }
        return any(marker in normalized for marker in general_markers) and not any(
            marker in normalized for marker in research_markers
        )

    def _infer_route_mode(
        self,
        *,
        snapshot_update: dict[str, Any],
        last_user_message: str | None,
    ) -> ResearchRouteMode:
        ask_result = snapshot_update.get("ask_result")
        response_metadata = {}
        if ask_result is not None and ask_result.qa is not None and isinstance(ask_result.qa.metadata, dict):
            response_metadata = dict(ask_result.qa.metadata)
        qa_route = str(response_metadata.get("qa_route") or "").strip()
        if qa_route == "chart_drilldown":
            return "chart_drilldown"
        if qa_route == "document_drilldown":
            return "document_drilldown"
        active_paper_ids = [
            str(item).strip()
            for item in snapshot_update.get("active_paper_ids", [])
            if str(item).strip()
        ]
        selected_paper_ids = [
            str(item).strip()
            for item in snapshot_update.get("selected_paper_ids", [])
            if str(item).strip()
        ]
        if snapshot_update.get("task_result") is not None:
            return "research_discovery"
        if ask_result is not None:
            return "paper_follow_up" if (selected_paper_ids or active_paper_ids) else "research_follow_up"
        if self._looks_like_general_chat(last_user_message):
            return "general_chat"
        return "paper_follow_up" if (selected_paper_ids or active_paper_ids) else "research_follow_up"

    def _thread_metadata_update(
        self,
        *,
        conversation: ResearchConversation,
        snapshot_update: dict[str, Any],
        last_user_message: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        route_mode = self._infer_route_mode(snapshot_update=snapshot_update, last_user_message=last_user_message)
        topic = str(snapshot_update.get("topic") or conversation.snapshot.topic or "").strip()
        workspace = snapshot_update.get("workspace", conversation.snapshot.workspace)
        workspace_summary = ""
        next_actions: list[str] = []
        stop_reason = None
        if isinstance(workspace, ResearchWorkspaceState):
            workspace_summary = workspace.status_summary
            next_actions = list(workspace.next_actions[:4])
            stop_reason = workspace.stop_reason
        continuity = self._topic_continuity_score(topic or last_user_message, conversation.snapshot.topic)
        existing_threads = list(conversation.snapshot.thread_history or [])
        active_thread_id = conversation.snapshot.active_thread_id
        start_new_thread = False
        if route_mode == "general_chat":
            active_thread_id = conversation.snapshot.active_thread_id
        elif route_mode == "research_discovery":
            start_new_thread = continuity < 0.42 or not conversation.snapshot.active_thread_id
        elif conversation.snapshot.active_route_mode == "general_chat":
            start_new_thread = True
        if start_new_thread:
            active_thread_id = f"thread_{uuid4().hex[:12]}"
        elif not active_thread_id and route_mode != "general_chat":
            active_thread_id = f"thread_{uuid4().hex[:12]}"

        next_threads: list[ResearchThreadSnapshot] = []
        updated_current = False
        for thread in existing_threads[-7:]:
            if thread.thread_id == active_thread_id and active_thread_id is not None:
                next_threads.append(
                    thread.model_copy(
                        update={
                            "route_mode": route_mode,
                            "topic": topic or thread.topic,
                            "task_id": task_id or thread.task_id,
                            "selected_paper_ids": list(snapshot_update.get("selected_paper_ids", thread.selected_paper_ids)),
                            "active_paper_ids": list(snapshot_update.get("active_paper_ids", thread.active_paper_ids)),
                            "last_user_message": last_user_message or thread.last_user_message,
                            "last_updated_at": _now_iso(),
                            "metadata": {
                                **thread.metadata,
                                "topic_continuity_score": continuity,
                                "workspace_summary": workspace_summary,
                                "next_actions": next_actions,
                                "stop_reason": stop_reason,
                            },
                        }
                    )
                )
                updated_current = True
            else:
                next_threads.append(thread)
        if active_thread_id is not None and not updated_current and route_mode != "general_chat":
            next_threads.append(
                ResearchThreadSnapshot(
                    thread_id=active_thread_id,
                    route_mode=route_mode,
                    topic=topic or last_user_message or "",
                    task_id=task_id,
                    selected_paper_ids=list(snapshot_update.get("selected_paper_ids", [])),
                    active_paper_ids=list(snapshot_update.get("active_paper_ids", [])),
                    last_user_message=last_user_message,
                    last_updated_at=_now_iso(),
                    metadata={
                        "topic_continuity_score": continuity,
                        "workspace_summary": workspace_summary,
                        "next_actions": next_actions,
                        "stop_reason": stop_reason,
                    },
                )
            )
        return {
            "active_route_mode": route_mode,
            "active_thread_id": active_thread_id,
            "thread_history": next_threads[-8:],
        }

    def create_conversation(self, request: CreateResearchConversationRequest) -> ResearchConversationResponse:
        now = _now_iso()
        title = (request.title or request.topic or "未命名研究会话").strip() or "未命名研究会话"
        conversation_id = f"conv_{uuid4().hex}"
        workspace = build_workspace_state(
            objective=request.topic or title,
            stage="discover",
            stop_reason="Conversation created; waiting for the first autonomous research action.",
            metadata={"source": "create_conversation"},
        )
        correlation_id = f"conv_init_{uuid4().hex}"
        initial_thread_id = f"thread_{uuid4().hex[:12]}"
        conversation = ResearchConversation(
            conversation_id=conversation_id,
            title=title,
            created_at=now,
            updated_at=now,
            snapshot=ResearchConversationSnapshot(
                topic=request.topic or "",
                days_back=request.days_back,
                max_papers=request.max_papers,
                sources=request.sources,
                active_route_mode="research_discovery",
                active_thread_id=initial_thread_id,
                thread_history=[
                    ResearchThreadSnapshot(
                        thread_id=initial_thread_id,
                        route_mode="research_discovery",
                        topic=request.topic or title,
                        last_updated_at=now,
                        metadata={"source": "create_conversation"},
                    )
                ],
                workspace=workspace,
                context_summary=self._build_context_summary(
                    workspace=workspace,
                    topic=request.topic or title,
                    days_back=request.days_back,
                    max_papers=request.max_papers,
                    sources=request.sources,
                    correlation_id=correlation_id,
                ),
                recent_events=[
                    self._build_runtime_event(
                        event_type="agent_started",
                        conversation_id=conversation_id,
                        correlation_id=correlation_id,
                        payload={"source": "create_conversation", "topic": request.topic or title},
                    )
                ],
            ),
            status_metadata=ResearchStatusMetadata(
                lifecycle_status="waiting_input",
                updated_at=now,
                correlation_id=correlation_id,
            ),
        )
        welcome = ResearchMessage(
            message_id=f"msg_{uuid4().hex}",
            role="assistant",
            kind="welcome",
            title="Research-Copilot",
            content="这里会持续记录你的研究检索、导入、问答和 TODO 执行历史。",
            meta="多源检索、入库、研究集合问答、TODO 闭环",
            created_at=now,
        )
        self.report_service.save_conversation(conversation)
        self.report_service.save_messages(conversation.conversation_id, [welcome])
        self.memory_manager.update_user_profile(
            topic=request.topic,
            answer_language=_preferred_answer_language_from_text(request.topic),
            note="conversation_created",
        )
        return ResearchConversationResponse(conversation=conversation, messages=[welcome])

    def get_conversation(self, conversation_id: str) -> ResearchConversationResponse:
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        return ResearchConversationResponse(
            conversation=conversation,
            messages=self.report_service.load_messages(conversation_id),
        )

    def rename_conversation(self, conversation_id: str, title: str) -> ResearchConversationResponse:
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        updated = conversation.model_copy(update={"title": title.strip(), "updated_at": _now_iso()})
        self.report_service.save_conversation(updated)
        return self.get_conversation(conversation_id)

    def _conversation_task_id(self, conversation: ResearchConversation) -> str | None:
        if conversation.task_id:
            return conversation.task_id
        task_result = getattr(conversation.snapshot, "task_result", None)
        task = getattr(task_result, "task", None)
        if task is not None and getattr(task, "task_id", None):
            return task.task_id
        search_result = getattr(conversation.snapshot, "search_result", None)
        report = getattr(search_result, "report", None)
        if report is not None and getattr(report, "task_id", None):
            return report.task_id
        return None

    def delete_conversation(self, conversation_id: str) -> None:
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        task_id = self._conversation_task_id(conversation)
        self.memory_manager.clear_session(conversation_id)
        self.report_service.delete_jobs(conversation_id=conversation_id)
        if task_id:
            self.report_service.delete_task_artifacts(task_id)
            self.report_service.delete_jobs(task_id=task_id)
        self.report_service.delete_conversation(conversation_id)

    async def reset_state(self) -> None:
        running_tasks = list(self._job_tasks.values())
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)
        self._job_tasks.clear()
        self.report_service.clear_all()

    def record_search_turn(
        self,
        conversation_id: str,
        *,
        topic: str,
        response: SearchPapersResponse,
        days_back: int,
        max_papers: int,
        sources: list[str],
    ) -> ResearchConversationResponse:
        return self._record_conversation_messages(
            conversation_id,
            messages=[
                self._build_message(role="user", kind="topic", title="当前研究主题", content=topic),
                self._build_message(
                    role="assistant",
                    kind="report",
                    title="文献综述结果",
                    meta=f"候选论文 {response.report.paper_count} 篇",
                    content=response.report.markdown,
                    payload={"report": response.report.model_dump(mode="json")},
                ),
                self._build_message(
                    role="assistant",
                    kind="candidates",
                    title="候选论文池",
                    meta=f"当前共 {len(response.papers)} 篇，可勾选后导入",
                    payload={"papers": [paper.model_dump(mode="json") for paper in response.papers]},
                ),
            ],
            snapshot_update={
                "topic": topic,
                "days_back": days_back,
                "max_papers": max_papers,
                "sources": sources,
                "composer_mode": "research",
                "workspace": response.report.workspace,
                "search_result": response,
                "task_result": None,
                "import_result": None,
                "ask_result": None,
                "last_error": None,
            },
            title_hint=topic,
        )

    def record_task_turn(
        self,
        conversation_id: str,
        *,
        response: ResearchTaskResponse,
    ) -> ResearchConversationResponse:
        report = response.report
        messages = [
            self._build_message(
                role="user",
                kind="topic",
                title="当前研究主题",
                meta="研究任务已创建",
                content=response.task.topic,
            )
        ]
        if report is not None:
            messages.append(
                self._build_message(
                    role="assistant",
                    kind="report",
                    title="文献综述结果",
                    meta=f"候选论文 {report.paper_count} 篇",
                    content=report.markdown,
                    payload={"report": report.model_dump(mode="json")},
                )
            )
        if response.papers:
            messages.append(
                self._build_message(
                    role="assistant",
                    kind="candidates",
                    title="候选论文池",
                    meta=f"当前共 {len(response.papers)} 篇，可勾选后导入",
                    payload={"papers": [paper.model_dump(mode='json') for paper in response.papers]},
                )
            )
        return self._record_conversation_messages(
            conversation_id,
            messages=messages,
            snapshot_update={
                "topic": response.task.topic,
                "days_back": response.task.days_back,
                "max_papers": response.task.max_papers,
                "sources": response.task.sources,
                "composer_mode": "qa",
                "workspace": response.task.workspace,
                "search_result": None,
                "task_result": response,
                "import_result": None,
                "ask_result": None,
                "last_error": None,
            },
            task_id=response.task.task_id,
            title_hint=response.task.topic,
        )

    def record_import_turn(
        self,
        conversation_id: str,
        *,
        task_response: ResearchTaskResponse | None,
        import_response: ImportPapersResponse,
        selected_paper_ids: list[str] | None = None,
        notice: str | None = None,
    ) -> ResearchConversationResponse:
        correlation_id = (
            task_response.task.status_metadata.correlation_id
            if task_response is not None
            else None
        )
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="tool_succeeded" if import_response.failed_count == 0 else "tool_failed",
            task_id=task_response.task.task_id if task_response else None,
            correlation_id=correlation_id,
            payload={
                "tool_name": "paper_import",
                "imported_count": import_response.imported_count,
                "skipped_count": import_response.skipped_count,
                "failed_count": import_response.failed_count,
            },
        )
        preview_lines = [
            f"imported={import_response.imported_count} · skipped={import_response.skipped_count} · failed={import_response.failed_count}"
        ]
        for result in import_response.results[:5]:
            suffix = f" · doc={result.document_id}" if result.document_id else ""
            error = f" · {result.error_message}" if result.error_message else ""
            preview_lines.append(f"- {result.title} · {result.status}{suffix}{error}")
        messages = [
            self._build_message(
                role="assistant",
                kind="import_result",
                title="导入结果",
                meta="候选论文已进入文档链路",
                content="\n".join(preview_lines),
                payload={"import_result": import_response.model_dump(mode="json")},
            )
        ]
        if notice:
            messages.append(self._build_message(role="assistant", kind="notice", title="系统通知", content=notice))
        return self._record_conversation_messages(
            conversation_id,
            messages=messages,
            snapshot_update={
                "composer_mode": "qa",
                "workspace": task_response.task.workspace
                if task_response
                else build_workspace_state(
                    objective="",
                    stage="ingest",
                    stop_reason=notice or "Paper import finished.",
                    metadata={"source": "record_import_turn"},
                ),
                "import_result": import_response,
                "task_result": task_response,
                "ask_result": None,
                "selected_paper_ids": selected_paper_ids or [],
                "active_paper_ids": selected_paper_ids or [],
                "last_notice": notice,
                "last_error": None,
                "active_job_id": None,
            },
            task_id=task_response.task.task_id if task_response else None,
        )

    def record_qa_turn(
        self,
        conversation_id: str,
        *,
        task_response: ResearchTaskResponse,
        ask_response: ResearchTaskAskResponse,
    ) -> ResearchConversationResponse:
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="tool_succeeded",
            task_id=task_response.task.task_id,
            correlation_id=task_response.task.status_metadata.correlation_id,
            payload={
                "tool_name": "collection_qa",
                "qa_route": ask_response.qa.metadata.get("qa_route")
                if isinstance(ask_response.qa.metadata, dict)
                else None,
                "evidence_count": len(ask_response.qa.evidence_bundle.evidences),
                "confidence": ask_response.qa.confidence,
            },
        )
        citations = [
            self._format_evidence_citation(evidence.document_id, evidence.page_number, evidence.source_type)
            for evidence in ask_response.qa.evidence_bundle.evidences
        ]
        qa_metadata = ask_response.qa.metadata if isinstance(ask_response.qa.metadata, dict) else {}
        qa_route = qa_metadata.get("qa_route") if isinstance(qa_metadata.get("qa_route"), str) else "empty"
        drilldown_runtime = (
            qa_metadata.get("drilldown_runtime")
            if isinstance(qa_metadata.get("drilldown_runtime"), str)
            else "collection"
        )
        return self._record_conversation_messages(
            conversation_id,
            messages=[
                self._build_message(
                    role="user",
                    kind="question",
                    title="研究集合提问",
                    content=ask_response.qa.question,
                ),
                self._build_message(
                    role="assistant",
                    kind="answer",
                    title="研究集合回答",
                    meta=(
                        f"route={qa_route} · runtime={drilldown_runtime} · "
                        f"evidence={len(ask_response.qa.evidence_bundle.evidences)} · "
                        f"confidence={ask_response.qa.confidence if ask_response.qa.confidence is not None else 'empty'} · "
                        f"docs={len(ask_response.document_ids)}"
                    ),
                    content=ask_response.qa.answer,
                    citations=[citation for citation in citations if citation],
                    payload={"ask_result": ask_response.model_dump(mode="json")},
                ),
            ],
            snapshot_update={
                "composer_mode": "qa",
                "workspace": task_response.task.workspace,
                "task_result": task_response,
                "import_result": None,
                "ask_result": ask_response,
                "last_error": None,
            },
            task_id=task_response.task.task_id,
        )

    def record_agent_turn(
        self,
        conversation_id: str,
        *,
        request: ResearchAgentRunRequest,
        response: ResearchAgentRunResponse,
    ) -> ResearchConversationResponse:
        task_response = self._task_response_from_agent_response(response)
        ask_response = self._ask_response_from_agent_response(response)
        advanced_strategy_payload = (
            response.metadata.get("advanced_strategy")
            if isinstance(response.metadata, dict)
            else None
        )
        response_route_mode = (
            str(response.metadata.get("route_mode") or "").strip()
            if isinstance(response.metadata, dict)
            else ""
        )
        advanced_strategy = (
            ResearchAdvancedStrategy.model_validate(advanced_strategy_payload)
            if isinstance(advanced_strategy_payload, dict)
            else ResearchAdvancedStrategy()
        )
        active_paper_ids = self._active_paper_ids_from_agent_turn(request=request, response=response)
        topic = (
            response.task.topic
            if response.task is not None
            else request.message
            if request.mode != "qa"
            else ""
        )
        has_general_answer = bool(
            isinstance(response.metadata, dict) and response.metadata.get("has_general_answer")
        )
        composer_mode = "research"
        if request.mode == "qa" or response.qa is not None:
            composer_mode = "qa"
        elif response.task is not None:
            composer_mode = "research"
        elif has_general_answer:
            composer_mode = "research"
        last_error = None
        if response.status == "failed" and response.warnings:
            last_error = response.warnings[-1]
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="agent_routed" if response.status != "failed" else "task_failed",
            task_id=response.task.task_id if response.task is not None else request.task_id,
            correlation_id=response.task.status_metadata.correlation_id if response.task is not None else None,
            payload={
                "mode": request.mode,
                "advanced_action": request.advanced_action,
                "status": response.status,
                "warning_count": len(response.warnings),
            },
        )
        messages = self._filter_persisted_conversation_messages(
            list(response.messages or [])
        )
        if not messages:
            messages = [
                self._build_message(
                    role="user",
                    kind="question" if request.mode == "qa" else "topic",
                    title="用户研究目标" if request.mode != "qa" else "研究集合提问",
                    content=request.message,
                )
            ]
        return self._record_conversation_messages(
            conversation_id,
            messages=messages,
            snapshot_update={
                "topic": topic,
                "days_back": response.task.days_back if response.task is not None else request.days_back,
                "max_papers": response.task.max_papers if response.task is not None else request.max_papers,
                "sources": response.task.sources if response.task is not None else request.sources,
                "composer_mode": composer_mode,
                "active_route_mode": (
                    response_route_mode
                    or "general_chat"
                    if has_general_answer
                    else "paper_follow_up"
                    if active_paper_ids
                    else "research_follow_up"
                    if response.qa is not None
                    else "research_discovery"
                    if response.task is not None
                    else "research_follow_up"
                ),
                "advanced_strategy": advanced_strategy,
                "selected_paper_ids": request.selected_paper_ids,
                "active_paper_ids": active_paper_ids,
                "workspace": response.workspace,
                "search_result": None,
                "task_result": task_response,
                "import_result": response.import_result,
                "ask_result": ask_response,
                "last_notice": response.workspace.stop_reason or response.workspace.status_summary or None,
                "last_error": last_error,
                "active_job_id": None,
            },
            task_id=response.task.task_id if response.task is not None else request.task_id,
            title_hint=topic or None,
        )

    def _active_paper_ids_from_agent_turn(
        self,
        *,
        request: ResearchAgentRunRequest,
        response: ResearchAgentRunResponse,
    ) -> list[str]:
        ids: list[str] = []
        if response.qa is not None:
            qa_metadata = response.qa.metadata if isinstance(response.qa.metadata, dict) else {}
            ids.extend(str(item).strip() for item in qa_metadata.get("paper_ids", []) if str(item).strip())
            ids.extend(str(item).strip() for item in qa_metadata.get("selected_paper_ids", []) if str(item).strip())
        response_metadata = response.metadata if isinstance(response.metadata, dict) else {}
        ids.extend(str(item).strip() for item in response_metadata.get("active_paper_ids", []) if str(item).strip())
        if not ids:
            ids.extend(str(item).strip() for item in request.selected_paper_ids if str(item).strip())
        return list(dict.fromkeys(ids))

    def record_notice(
        self,
        conversation_id: str,
        *,
        task_response: ResearchTaskResponse | None = None,
        notice: str,
        kind: str = "notice",
        active_job_id: str | None = None,
        last_error: str | None = None,
    ) -> ResearchConversationResponse:
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="tool_failed" if kind == "error" else "memory_updated",
            task_id=task_response.task.task_id if task_response else None,
            correlation_id=task_response.task.status_metadata.correlation_id if task_response else None,
            payload={"notice": notice, "kind": kind},
        )
        return self._record_conversation_messages(
            conversation_id,
            messages=[self._build_message(role="assistant", kind=kind, title="系统通知", content=notice)],
            snapshot_update={
                "task_result": task_response,
                "last_notice": notice if kind != "error" else None,
                "last_error": last_error if kind == "error" else None,
                "active_job_id": active_job_id,
            },
            task_id=task_response.task.task_id if task_response else None,
        )

    def _task_response_from_agent_response(
        self,
        response: ResearchAgentRunResponse,
    ) -> ResearchTaskResponse | None:
        if response.task is None:
            return None
        return ResearchTaskResponse(
            task=response.task,
            papers=list(response.papers),
            report=response.report,
            warnings=list(response.warnings),
        )

    def _ask_response_from_agent_response(
        self,
        response: ResearchAgentRunResponse,
    ) -> ResearchTaskAskResponse | None:
        if response.task is None or response.qa is None:
            return None
        qa_metadata = response.qa.metadata if isinstance(response.qa.metadata, dict) else {}
        paper_scope = (
            qa_metadata.get("paper_scope")
            if isinstance(qa_metadata.get("paper_scope"), dict)
            else {}
        )
        paper_ids = [
            value
            for value in qa_metadata.get("selected_paper_ids", [])
            if isinstance(value, str)
        ]
        if not paper_ids:
            paper_ids = [
                value
                for value in paper_scope.get("paper_ids", [])
                if isinstance(value, str)
            ]
        scope_mode = (
            qa_metadata.get("qa_scope_mode")
            if isinstance(qa_metadata.get("qa_scope_mode"), str)
            else (
                paper_scope.get("scope_mode")
                if isinstance(paper_scope.get("scope_mode"), str)
                else "all_imported"
            )
        )
        document_ids = [
            value
            for value in qa_metadata.get("selected_document_ids", [])
            if isinstance(value, str)
        ]
        if scope_mode == "all_imported":
            if not paper_ids:
                imported_document_ids = {
                    str(document_id).strip()
                    for document_id in response.task.imported_document_ids
                    if str(document_id).strip()
                }
                paper_ids = [
                    paper.paper_id
                    for paper in response.papers
                    if str(paper.metadata.get("document_id") or "").strip() in imported_document_ids
                ]
            if not document_ids:
                document_ids = list(response.task.imported_document_ids)
        warnings = [
            value
            for value in qa_metadata.get("selection_warnings", [])
            if isinstance(value, str)
        ]
        return ResearchTaskAskResponse(
            task_id=response.task.task_id,
            paper_ids=paper_ids,
            document_ids=document_ids,
            scope_mode=scope_mode,
            qa=response.qa,
            report=response.report,
            todo_items=list(response.task.todo_items),
            warnings=warnings,
        )

    async def search_papers(
        self,
        request: SearchPapersRequest,
        *,
        graph_runtime: Any | None = None,
    ) -> SearchPapersResponse:
        execution_context = (
            self.build_execution_context(
                graph_runtime=graph_runtime,
                conversation_id=request.conversation_id,
                skill_name="research_report",
                metadata={},
            )
            if graph_runtime is not None
            else None
        )
        bundle = await self._run_direct_discovery(
            topic=request.topic,
            days_back=request.days_back,
            max_papers=request.max_papers,
            sources=request.sources,
            execution_context=execution_context,
        )
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            report=bundle.report,
            papers=bundle.papers,
            task_intent="research_search",
            metadata_update={
                "days_back": request.days_back,
                "max_papers": request.max_papers,
                "sources": list(request.sources),
                "report_id": bundle.report.report_id,
            },
        )
        return SearchPapersResponse(
            plan=bundle.plan,
            papers=bundle.papers,
            report=bundle.report,
            warnings=bundle.warnings,
        )

    async def create_task(
        self,
        request: CreateResearchTaskRequest,
        *,
        graph_runtime: Any | None = None,
    ) -> ResearchTaskResponse:
        now = _now_iso()
        correlation_id = f"task_{uuid4().hex}"
        task = ResearchTask(
            task_id=f"research_{uuid4().hex}",
            topic=request.topic,
            status="created",
            created_at=now,
            updated_at=now,
            days_back=request.days_back,
            max_papers=request.max_papers,
            sources=request.sources,
            workspace=build_workspace_state(
                objective=request.topic,
                stage="discover",
                extra_questions=[request.topic],
                stop_reason="Task created; waiting for autonomous discovery.",
                metadata={"source": "create_task"},
            ),
            status_metadata=self._build_status_metadata(
                lifecycle_status="queued",
                correlation_id=correlation_id,
            ),
            metadata={"correlation_id": correlation_id},
        )
        self.save_task_state(task)
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=task,
            task_intent="research_task_created",
            metadata_update={
                "days_back": request.days_back,
                "max_papers": request.max_papers,
                "sources": list(request.sources),
            },
        )
        self.memory_manager.update_user_profile(
            topic=request.topic,
            answer_language=_preferred_answer_language_from_text(request.topic),
        )
        self.observability_service.record_metric(
            metric_type="task_created",
            payload={"task_id": task.task_id, "topic": request.topic},
        )
        if not request.run_immediately:
            return ResearchTaskResponse(task=task, papers=[], report=None, warnings=[])
        return await self.run_task(
            task.task_id,
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
        )

    async def run_agent(
        self,
        request: ResearchAgentRunRequest,
        *,
        graph_runtime: Any,
    ) -> ResearchAgentRunResponse:
        from services.research.research_supervisor_graph_runtime import ResearchSupervisorGraphRuntime

        if self._agent_runtime is None:
            self._agent_runtime = ResearchSupervisorGraphRuntime(research_service=self)
        self.preference_memory_agent.observe_user_message(
            message=request.message,
            sources=list(request.sources),
            answer_language=_preferred_answer_language_from_text(request.message),
            metadata={
                "conversation_id": request.conversation_id,
                "task_id": request.task_id,
                "mode": request.mode,
            },
        )
        return await self._agent_runtime.run(
            request=request,
            graph_runtime=graph_runtime,
        )

    async def run_task(
        self,
        task_id: str,
        *,
        graph_runtime: Any | None = None,
        conversation_id: str | None = None,
    ) -> ResearchTaskResponse:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        running_task = self._transition_task(
            task,
            status="running",
            correlation_id=task.metadata.get("correlation_id") if isinstance(task.metadata, dict) else None,
        )
        self.save_task_state(running_task)
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="agent_started",
            task_id=running_task.task_id,
            correlation_id=running_task.status_metadata.correlation_id,
            payload={
                "phase": "research_discovery",
                "topic": running_task.topic,
                "days_back": running_task.days_back,
                "max_papers": running_task.max_papers,
            },
        )
        self.append_runtime_event(
            conversation_id=conversation_id,
            event_type="tool_called",
            task_id=running_task.task_id,
            correlation_id=running_task.status_metadata.correlation_id,
            payload={"tool_name": "research_discovery_runtime"},
        )
        try:
            execution_context = (
                self.build_execution_context(
                    graph_runtime=graph_runtime,
                    conversation_id=conversation_id,
                    task=running_task,
                    skill_name="research_report",
                )
                if graph_runtime is not None
                else None
            )
            bundle = await self._run_direct_discovery(
                topic=task.topic,
                days_back=task.days_back,
                max_papers=task.max_papers,
                sources=task.sources,
                task_id=task.task_id,
                execution_context=execution_context,
            )
            runtime_report = bundle.report.model_copy(
                update={
                    "metadata": {
                        **bundle.report.metadata,
                        "decision_model": "llm_dynamic_single_manager",
                    }
                }
            )
            completed_task = self._transition_task(
                running_task,
                status="completed",
                correlation_id=running_task.metadata.get("correlation_id") if isinstance(running_task.metadata, dict) else None,
                paper_count=len(bundle.papers),
                report_id=runtime_report.report_id,
                todo_items=bundle.todo_items,
                workspace=bundle.workspace,
                metadata={
                    **running_task.metadata,
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": runtime_report.metadata.get("agent_architecture"),
                    "decision_model": runtime_report.metadata.get("decision_model"),
                    "primary_agents": runtime_report.metadata.get("primary_agents"),
                    "primary_skills": runtime_report.metadata.get("primary_skills"),
                    "autonomy_rounds": runtime_report.metadata.get("autonomy_rounds"),
                    "autonomy_trace_steps": runtime_report.metadata.get("autonomy_trace_steps"),
                },
            )
            self.save_task_state(completed_task)
            self.report_service.save_papers(task.task_id, bundle.papers)
            self.report_service.save_report(runtime_report)
            self._update_research_memory(
                graph_runtime=graph_runtime,
                conversation_id=conversation_id,
                task=completed_task,
                report=runtime_report,
                papers=bundle.papers,
                document_ids=completed_task.imported_document_ids,
                task_intent="research_discovery",
                metadata_update={
                    "report_id": runtime_report.report_id,
                    "autonomy_rounds": runtime_report.metadata.get("autonomy_rounds"),
                    "autonomy_trace_steps": runtime_report.metadata.get("autonomy_trace_steps"),
                },
            )
            self.observability_service.record_metric(
                metric_type="task_completed",
                payload={
                    "task_id": completed_task.task_id,
                    "paper_count": len(bundle.papers),
                    "warning_count": len(bundle.warnings),
                },
            )
            self.append_runtime_event(
                conversation_id=conversation_id,
                event_type="task_completed",
                task_id=completed_task.task_id,
                correlation_id=completed_task.status_metadata.correlation_id,
                payload={
                    "paper_count": len(bundle.papers),
                    "report_id": runtime_report.report_id,
                    "warning_count": len(bundle.warnings),
                },
            )
            return ResearchTaskResponse(
                task=completed_task,
                papers=bundle.papers,
                report=runtime_report,
                warnings=bundle.warnings,
            )
        except Exception:
            failed_task = self._transition_task(
                running_task,
                status="failed",
                correlation_id=running_task.metadata.get("correlation_id") if isinstance(running_task.metadata, dict) else None,
            )
            self.save_task_state(failed_task)
            self.append_runtime_event(
                conversation_id=conversation_id,
                event_type="task_failed",
                task_id=failed_task.task_id,
                correlation_id=failed_task.status_metadata.correlation_id,
                payload={"phase": "research_discovery"},
            )
            self.observability_service.archive_failure(
                failure_type="task_failed",
                payload={"task_id": failed_task.task_id, "topic": failed_task.topic},
            )
            raise

    def get_task(self, task_id: str) -> ResearchTaskResponse:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        papers = self.report_service.load_papers(task_id)
        report = self.report_service.load_report(task_id, task.report_id)
        return ResearchTaskResponse(task=task, papers=papers, report=report, warnings=[])

    async def list_paper_figures(
        self,
        task_id: str,
        paper_id: str,
        *,
        graph_runtime: Any,
    ) -> ResearchPaperFigureListResponse:
        task, paper = self._resolve_imported_paper(task_id=task_id, paper_id=paper_id)
        return await self.chart_analysis_agent.list_paper_figures(
            task_id=task.task_id,
            paper=paper,
            graph_runtime=graph_runtime,
            load_cached_figure_payload=self._load_cached_figure_payload,
            persist_paper_figure_cache=self._persist_paper_figure_cache,
            parse_imported_paper_document=self._parse_imported_paper_document,
        )

    async def analyze_paper_figure(
        self,
        task_id: str,
        paper_id: str,
        request: AnalyzeResearchPaperFigureRequest,
        *,
        graph_runtime: Any,
    ) -> AnalyzeResearchPaperFigureResponse:
        _, paper = self._resolve_imported_paper(task_id=task_id, paper_id=paper_id)
        return await self.chart_analysis_agent.analyze_paper_figure(
            task_id=task_id,
            paper=paper,
            request=request,
            graph_runtime=graph_runtime,
            load_cached_figure_target=self._load_cached_figure_target,
            parse_imported_paper_document=self._parse_imported_paper_document,
        )

    def _load_cached_figure_payload(self, *, paper: PaperCandidate) -> dict[str, Any] | None:
        cache = paper.metadata.get("paper_figure_cache")
        if not isinstance(cache, dict):
            return None
        cached_document_id = str(cache.get("document_id") or "").strip()
        cached_storage_uri = str(cache.get("storage_uri") or "").strip()
        current_document_id = str(paper.metadata.get("document_id") or "").strip()
        current_storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        if not cached_document_id or cached_document_id != current_document_id:
            return None
        if cached_storage_uri and current_storage_uri and cached_storage_uri != current_storage_uri:
            return None
        figures = cache.get("figures")
        if not isinstance(figures, list):
            return None
        return cache

    def _load_cached_figure_target(self, *, paper: PaperCandidate, figure_id: str | None) -> PaperFigureAnalyzeTarget | None:
        resolved_figure_id = str(figure_id or "").strip()
        if not resolved_figure_id:
            return None
        cache = self._load_cached_figure_payload(paper=paper)
        if cache is None:
            return None
        analyze_targets = cache.get("analyze_targets")
        if not isinstance(analyze_targets, dict):
            return None
        payload = analyze_targets.get(resolved_figure_id)
        if not isinstance(payload, dict):
            return None
        try:
            return PaperFigureAnalyzeTarget.model_validate(payload)
        except Exception:
            return None

    def _persist_paper_figure_cache(
        self,
        *,
        task_id: str,
        paper_id: str,
        document_id: str,
        storage_uri: str,
        figures: list[ResearchPaperFigurePreview],
        targets: list[PaperFigureAnalyzeTarget],
        warnings: list[str],
    ) -> None:
        papers = self.report_service.load_papers(task_id)
        updated_papers: list[PaperCandidate] = []
        for item in papers:
            if item.paper_id != paper_id:
                updated_papers.append(item)
                continue
            metadata = {
                **item.metadata,
                "paper_figure_cache": {
                    "document_id": document_id,
                    "storage_uri": storage_uri,
                    "figures": [figure.model_dump(mode="json") for figure in figures],
                    "analyze_targets": {
                        target.figure_id: target.model_dump(mode="json")
                        for target in targets
                    },
                    "warnings": list(warnings),
                },
            }
            updated_papers.append(item.model_copy(update={"metadata": metadata}))
        self.report_service.save_papers(task_id, updated_papers)

    def _resolve_imported_paper(self, *, task_id: str, paper_id: str) -> tuple[ResearchTask, PaperCandidate]:
        task_response = self.get_task(task_id)
        paper = next((item for item in task_response.papers if item.paper_id == paper_id), None)
        if paper is None:
            raise KeyError(f"Paper not found in task {task_id}: {paper_id}")
        document_id = str(paper.metadata.get("document_id") or "").strip()
        storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        if paper.ingest_status != "ingested" or not document_id or not storage_uri:
            raise ValueError("Paper must be imported before chart analysis is available.")
        return task_response.task, paper

    async def _parse_imported_paper_document(self, *, paper: PaperCandidate, graph_runtime: Any):
        storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
        document_id = str(paper.metadata.get("document_id") or "").strip()
        if not storage_uri or not document_id:
            raise ValueError("Imported paper is missing storage metadata.")
        return await graph_runtime.handle_parse_document(
            file_path=storage_uri,
            document_id=document_id,
            metadata={
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "source": "paper_figure_analysis",
            },
            skill_name="paper_chart_analysis",
        )

    def persist_runtime_state(
        self,
        *,
        task_response: ResearchTaskResponse | None,
        workspace: ResearchWorkspaceState,
        conversation_id: str | None = None,
        advanced_strategy: ResearchAdvancedStrategy | None = None,
    ) -> ResearchTaskResponse | None:
        updated_response = task_response
        if task_response is not None:
            updated_task = task_response.task.model_copy(
                update={
                    "workspace": workspace,
                    "updated_at": _now_iso(),
                }
            )
            self.save_task_state(updated_task, conversation_id=conversation_id)
            updated_report = task_response.report
            if updated_report is None and updated_task.report_id:
                updated_report = self.report_service.load_report(updated_task.task_id, updated_task.report_id)
            if updated_report is not None:
                updated_report = updated_report.model_copy(update={"workspace": workspace})
                self.report_service.save_report(updated_report)
            updated_response = task_response.model_copy(
                update={
                    "task": updated_task,
                    "report": updated_report,
                }
            )
        self._persist_runtime_conversation_snapshot(
            conversation_id=conversation_id,
            task_response=updated_response,
            workspace=workspace,
            advanced_strategy=advanced_strategy,
        )
        return updated_response

    def get_job(self, job_id: str) -> ResearchJob:
        job = self.report_service.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def start_import_job(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        now = _now_iso()
        correlation_id = f"job_{uuid4().hex}"
        job = ResearchJob(
            job_id=f"job_{uuid4().hex}",
            kind="paper_import",
            status="queued",
            created_at=now,
            updated_at=now,
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            progress_message="导入任务已创建，等待后台执行。",
            metadata={
                "paper_ids": request.paper_ids,
                "include_graph": request.include_graph,
                "include_embeddings": request.include_embeddings,
                "fast_mode": request.fast_mode,
                "question": request.question,
                "top_k": request.top_k,
                "reasoning_style": request.reasoning_style,
                "correlation_id": correlation_id,
            },
            status_metadata=self._build_status_metadata(
                lifecycle_status="queued",
                correlation_id=correlation_id,
            ),
        )
        self.save_job_state(
            job,
            event_type="tool_called",
            payload={"tool_name": "paper_import_job", "job_kind": job.kind},
        )
        self.observability_service.record_metric(
            metric_type="job_created",
            payload={"job_id": job.job_id, "job_kind": job.kind, "task_id": request.task_id},
        )
        self._set_conversation_active_job(request.conversation_id, job.job_id)
        task = asyncio.create_task(self._run_import_job(job.job_id, request, graph_runtime=graph_runtime))
        self._job_tasks[job.job_id] = task
        task.add_done_callback(lambda _: self._job_tasks.pop(job.job_id, None))
        return job

    async def start_todo_import_job(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        now = _now_iso()
        correlation_id = f"job_{uuid4().hex}"
        job = ResearchJob(
            job_id=f"job_{uuid4().hex}",
            kind="todo_import",
            status="queued",
            created_at=now,
            updated_at=now,
            task_id=task_id,
            conversation_id=request.conversation_id,
            progress_message="TODO 补充导入任务已创建，等待后台执行。",
            metadata={
                "todo_id": todo_id,
                "max_papers": request.max_papers,
                "include_graph": request.include_graph,
                "include_embeddings": request.include_embeddings,
                "correlation_id": correlation_id,
            },
            status_metadata=self._build_status_metadata(
                lifecycle_status="queued",
                correlation_id=correlation_id,
            ),
        )
        self.save_job_state(
            job,
            event_type="tool_called",
            payload={"tool_name": "todo_import_job", "job_kind": job.kind},
        )
        self.observability_service.record_metric(
            metric_type="job_created",
            payload={"job_id": job.job_id, "job_kind": job.kind, "task_id": task_id},
        )
        self._set_conversation_active_job(request.conversation_id, job.job_id)
        task = asyncio.create_task(
            self._run_todo_import_job(job.job_id, task_id, todo_id, request, graph_runtime=graph_runtime)
        )
        self._job_tasks[job.job_id] = task
        task.add_done_callback(lambda _: self._job_tasks.pop(job.job_id, None))
        return job

    def update_todo_status(self, task_id: str, todo_id: str, status: str) -> ResearchTaskResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        now = datetime.now(UTC).isoformat()
        updated_todo = todo.model_copy(
            update={
                "status": status,
                "metadata": {
                    **todo.metadata,
                    "last_status_change_at": now,
                },
            }
        )
        updated_task = self._replace_task_todo(task, updated_todo, updated_at=now)
        self.save_task_state(updated_task)
        return self.get_task(task_id)

    async def rerun_todo_search(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
    ) -> ResearchTodoActionResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        existing_report = self.report_service.load_report(task.task_id, task.report_id)
        query, discovered_papers, merged_papers, warnings = await self._search_follow_up_papers(
            task=task,
            todo=todo,
            max_papers=request.max_papers,
        )
        now = datetime.now(UTC).isoformat()
        updated_todo = todo.model_copy(
            update={
                "metadata": {
                    **todo.metadata,
                    "last_action_at": now,
                    "last_action_type": "search",
                    "last_search_query": query,
                    "last_search_found": len(discovered_papers),
                },
            }
        )
        updated_report = self._rebuild_task_report(
            task=task,
            papers=merged_papers,
            existing_report=existing_report,
            warnings=warnings,
            action_title="重新检索",
            action_lines=[
                f"TODO：{updated_todo.content}",
                f"查询：{query}",
                f"新增/刷新候选论文：{len(discovered_papers)} 篇",
            ],
            generated_at=now,
        )
        updated_task = self._replace_task_todo(
            task,
            updated_todo,
            updated_at=now,
            paper_count=len(merged_papers),
            report_id=updated_report.report_id,
        )
        self.save_task_state(updated_task)
        self.report_service.save_papers(task.task_id, merged_papers)
        self.report_service.save_report(updated_report)
        return ResearchTodoActionResponse(
            task=updated_task,
            todo=updated_todo,
            papers=merged_papers,
            report=updated_report,
            warnings=warnings,
        )

    async def import_from_todo(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchTodoActionResponse:
        task, todo = self._load_task_and_todo(task_id, todo_id)
        existing_report = self.report_service.load_report(task.task_id, task.report_id)
        current_papers = self.report_service.load_papers(task.task_id)
        warnings: list[str] = []
        search_query: str | None = None

        candidate_papers = self._select_todo_import_candidates(
            task=task,
            todo=todo,
            papers=current_papers,
            limit=request.max_papers,
        )
        if not candidate_papers:
            search_query, discovered_papers, merged_papers, warnings = await self._search_follow_up_papers(
                task=task,
                todo=todo,
                max_papers=request.max_papers,
            )
            current_papers = merged_papers
            candidate_papers = self._select_todo_import_candidates(
                task=task,
                todo=todo,
                papers=current_papers,
                limit=request.max_papers,
            )
            if current_papers:
                self.report_service.save_papers(task.task_id, current_papers)

        if not candidate_papers:
            raise ValueError(f"No candidate papers with PDF available for TODO import: {todo_id}")

        import_result = await self.import_papers(
            ImportPapersRequest(
                task_id=task_id,
                paper_ids=[paper.paper_id for paper in candidate_papers],
                include_graph=request.include_graph,
                include_embeddings=request.include_embeddings,
                skill_name=request.skill_name,
            ),
            graph_runtime=graph_runtime,
        )

        refreshed_state = self.get_task(task_id)
        now = datetime.now(UTC).isoformat()
        updated_todo = self._find_todo(refreshed_state.task, todo_id).model_copy(
            update={
                "metadata": {
                    **self._find_todo(refreshed_state.task, todo_id).metadata,
                    "last_action_at": now,
                    "last_action_type": "import",
                    "last_import_count": import_result.imported_count,
                    "last_import_failed_count": import_result.failed_count,
                    "last_import_paper_ids": [paper.paper_id for paper in candidate_papers],
                    **({"last_search_query": search_query} if search_query else {}),
                },
            }
        )
        updated_report = self._rebuild_task_report(
            task=refreshed_state.task,
            papers=refreshed_state.papers,
            existing_report=refreshed_state.report or existing_report,
            warnings=warnings,
            action_title="补充导入",
            action_lines=[
                f"TODO：{updated_todo.content}",
                *([f"补充检索查询：{search_query}"] if search_query else []),
                f"尝试导入：{len(candidate_papers)} 篇",
                f"成功导入：{import_result.imported_count} 篇",
                f"跳过：{import_result.skipped_count} 篇",
                f"失败：{import_result.failed_count} 篇",
            ],
            generated_at=now,
        )
        updated_task = self._replace_task_todo(
            refreshed_state.task,
            updated_todo,
            updated_at=now,
            paper_count=len(refreshed_state.papers),
            report_id=updated_report.report_id,
        )
        self.save_task_state(updated_task)
        self.report_service.save_report(updated_report)
        return ResearchTodoActionResponse(
            task=updated_task,
            todo=updated_todo,
            papers=refreshed_state.papers,
            report=updated_report,
            warnings=warnings,
            import_result=import_result,
        )

    async def import_papers(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
        progress_callback: ImportProgressCallback | None = None,
    ) -> ImportPapersResponse:
        candidate_papers, persisted_papers = self._resolve_import_candidates(request)
        persisted_by_id = {paper.paper_id: paper for paper in persisted_papers}
        results: list[ImportedPaperResult] = []
        session_id, _ = self._resolve_research_session_id(
            conversation_id=request.conversation_id,
            task_id=request.task_id,
        )

        if candidate_papers:
            semaphore = asyncio.Semaphore(self.import_concurrency)
            progress_lock = asyncio.Lock()
            completed_count = 0

            async def process_candidate(index: int, paper: PaperCandidate) -> tuple[int, ImportedPaperResult, PaperCandidate]:
                async with semaphore:
                    result, updated_paper = await self._import_single_paper(
                        paper,
                        request=request,
                        graph_runtime=graph_runtime,
                        session_id=session_id,
                    )
                if progress_callback is not None:
                    async with progress_lock:
                        nonlocal completed_count
                        completed_count += 1
                        try:
                            maybe_awaitable = progress_callback(completed_count, len(candidate_papers), result)
                            if maybe_awaitable is not None:
                                await maybe_awaitable
                        except Exception:
                            pass
                return index, result, updated_paper

            completed_results = await asyncio.gather(
                *(process_candidate(index, paper) for index, paper in enumerate(candidate_papers))
            )
            for _, result, updated_paper in sorted(completed_results, key=lambda item: item[0]):
                results.append(result)
                persisted_by_id[updated_paper.paper_id] = updated_paper

        if request.task_id and persisted_papers:
            task = self.report_service.load_task(request.task_id)
            if task is not None:
                imported_document_ids = list(task.imported_document_ids)
                for result in results:
                    if result.status in {"imported", "skipped"} and result.document_id and result.document_id not in imported_document_ids:
                        imported_document_ids.append(result.document_id)
                current_report = self.report_service.load_report(task.task_id, task.report_id)
                updated_todo_items = self._resolve_todos_after_import(
                    task=task,
                    papers_by_id=persisted_by_id,
                )
                task_for_workspace = task.model_copy(
                    update={
                        "imported_document_ids": imported_document_ids,
                        "todo_items": updated_todo_items,
                    }
                )
                updated_task = task.model_copy(
                    update={
                        "updated_at": datetime.now(UTC).isoformat(),
                        "imported_document_ids": imported_document_ids,
                        "todo_items": updated_todo_items,
                        "workspace": build_workspace_from_task(
                            task=task_for_workspace,
                            report=current_report,
                            papers=list(persisted_by_id.values()),
                            stage="qa" if imported_document_ids else "ingest",
                            stop_reason="Paper import finished; the workspace is ready for grounded collection QA.",
                            metadata={
                                "imported_count": sum(1 for result in results if result.status == "imported"),
                                "skipped_count": sum(1 for result in results if result.status == "skipped"),
                                "failed_count": sum(1 for result in results if result.status == "failed"),
                                "fast_import_mode": request.fast_mode,
                                "pending_graph_backfill_ids": [
                                    result.document_id
                                    for result in results
                                    if result.graph_pending and result.document_id
                                ][:12],
                            },
                        ),
                    }
                )
                self.save_task_state(updated_task, conversation_id=request.conversation_id)
                if current_report is not None:
                    self.report_service.save_report(
                        current_report.model_copy(update={"workspace": updated_task.workspace})
                    )
            self.report_service.save_papers(request.task_id, list(persisted_by_id.values()))

        imported_count = sum(1 for result in results if result.status == "imported")
        skipped_count = sum(1 for result in results if result.status == "skipped")
        failed_count = sum(1 for result in results if result.status == "failed")
        pending_graph_backfill_ids = [
            result.document_id
            for result in results
            if result.graph_pending and result.document_id
        ]
        updated_task = self.report_service.load_task(request.task_id) if request.task_id else None
        updated_papers = list(persisted_by_id.values()) if persisted_by_id else []
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            papers=updated_papers,
            document_ids=updated_task.imported_document_ids if updated_task else [],
            selected_paper_ids=request.paper_ids,
            task_intent="research_import",
            metadata_update={
                "imported_count": imported_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "last_imported_document_ids": [
                    result.document_id
                    for result in results
                    if result.status in {"imported", "skipped"} and result.document_id
                ][:12],
                "pending_graph_backfill_ids": pending_graph_backfill_ids[:12],
            },
        )
        return ImportPapersResponse(
            results=results,
            imported_count=imported_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
        )

    async def _import_single_paper(
        self,
        paper: PaperCandidate,
        *,
        request: ImportPapersRequest,
        graph_runtime: Any,
        session_id: str | None,
    ) -> tuple[ImportedPaperResult, PaperCandidate]:
        try:
            existing_document_id = str(paper.metadata.get("document_id") or "").strip()
            existing_storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
            if paper.ingest_status == "ingested" and existing_document_id and existing_storage_uri:
                return (
                    ImportedPaperResult(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        status="skipped",
                        document_id=existing_document_id,
                        storage_uri=existing_storage_uri,
                        parsed=True,
                        indexed=True,
                        metadata={"reason": "already_ingested"},
                    ),
                    paper,
                )

            artifact = await self.paper_import_service.download_paper(paper)
            parsed_document = await graph_runtime.handle_parse_document(
                file_path=artifact.storage_uri,
                document_id=artifact.document_id,
                session_id=session_id,
                metadata={
                    "research_paper_id": paper.paper_id,
                    "research_title": paper.title,
                    "research_source": paper.source,
                    "research_pdf_url": paper.pdf_url,
                },
                skill_name=request.skill_name,
            )
            index_result = await graph_runtime.handle_index_document(
                parsed_document=parsed_document,
                charts=[],
                include_graph=(request.include_graph and not request.fast_mode),
                include_embeddings=request.include_embeddings,
                session_id=session_id,
                metadata={
                    "research_paper_id": paper.paper_id,
                    "research_title": paper.title,
                    "research_source": paper.source,
                    "fast_mode": request.fast_mode,
                },
                skill_name=request.skill_name,
            )
            graph_pending = bool(request.include_graph and request.fast_mode)
            import_status = "imported" if index_result.status == "succeeded" else "failed"
            zotero_sync = await self._sync_imported_paper_to_zotero(
                paper=paper,
                request=request,
                graph_runtime=graph_runtime,
            )
            result_metadata = {
                "index_status": index_result.status,
                "filename": artifact.filename,
                "index_mode": "fast_embeddings_first" if request.fast_mode else "full_sync",
                "graph_backfill_pending": graph_pending,
            }
            if zotero_sync is not None:
                result_metadata["zotero_sync"] = zotero_sync

            updated_paper_metadata = {
                **paper.metadata,
                "document_id": parsed_document.id,
                "storage_uri": artifact.storage_uri,
                "filename": artifact.filename,
                "index_status": index_result.status,
                "index_mode": "fast_embeddings_first" if request.fast_mode else "full_sync",
                "graph_backfill_pending": graph_pending,
            }
            if zotero_sync is not None:
                updated_paper_metadata["zotero_sync"] = zotero_sync
            return (
                ImportedPaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    status=import_status,
                    document_id=parsed_document.id,
                    storage_uri=artifact.storage_uri,
                    parsed=parsed_document.status == "parsed",
                    indexed=index_result.status == "succeeded",
                    graph_pending=graph_pending,
                    error_message=None if import_status == "imported" else "Indexing failed",
                    metadata=result_metadata,
                ),
                paper.model_copy(
                    update={
                        "ingest_status": "ingested" if import_status == "imported" else "selected",
                        "metadata": updated_paper_metadata,
                    }
                ),
            )
        except Exception as exc:
            return (
                ImportedPaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    status="failed",
                    error_message=str(exc),
                ),
                paper.model_copy(
                    update={
                        "ingest_status": "unavailable" if "No PDF URL available" in str(exc) else paper.ingest_status,
                        "metadata": {
                            **paper.metadata,
                            "last_import_error": str(exc),
                        },
                    }
                ),
            )

    async def _sync_imported_paper_to_zotero(
        self,
        *,
        paper: PaperCandidate,
        request: ImportPapersRequest,
        graph_runtime: Any,
    ) -> dict[str, Any] | None:
        del request
        research_function_service = getattr(graph_runtime, "research_function_service", None)
        if research_function_service is None or not hasattr(research_function_service, "sync_paper_to_zotero"):
            return None
        try:
            return await research_function_service.sync_paper_to_zotero(paper)
        except Exception as exc:
            logger.warning("Failed to sync imported paper to Zotero: %s", exc)
            return {
                "status": "failed",
                "action": "none",
                "zotero_item_key": None,
                "matched_by": None,
                "collection_name": None,
                "attachment_count": 0,
                "warnings": [f"Zotero sync failed after workspace import: {exc.__class__.__name__}"],
            }

    def _resolve_import_candidates(self, request: ImportPapersRequest) -> tuple[list[PaperCandidate], list[PaperCandidate]]:
        persisted_papers: list[PaperCandidate] = []
        if request.task_id:
            persisted_papers = self.report_service.load_papers(request.task_id)
        if request.papers:
            if request.paper_ids:
                allowed_ids = set(request.paper_ids)
                candidates = [paper for paper in request.papers if paper.paper_id in allowed_ids]
            else:
                candidates = request.papers
        elif persisted_papers:
            if request.paper_ids:
                allowed_ids = set(request.paper_ids)
                candidates = [paper for paper in persisted_papers if paper.paper_id in allowed_ids]
            else:
                candidates = persisted_papers
        else:
            candidates = []
        return candidates, persisted_papers

    async def _search_follow_up_papers(
        self,
        *,
        task: ResearchTask,
        todo: ResearchTodoItem,
        max_papers: int,
    ) -> tuple[str, list[PaperCandidate], list[PaperCandidate], list[str]]:
        query = self._build_todo_query(task, todo)
        bundle = await self._run_direct_discovery(
            topic=query,
            days_back=task.days_back,
            max_papers=max_papers,
            sources=task.sources,
            task_id=task.task_id,
        )
        merged_papers = self._refresh_existing_pool(
            existing_papers=self.report_service.load_papers(task.task_id),
            incoming_papers=bundle.papers,
            ranking_topic=query,
        )
        return query, bundle.papers, merged_papers, bundle.warnings

    async def _run_direct_discovery(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[str],
        task_id: str | None = None,
        execution_context: ResearchExecutionContext | None = None,
    ):
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
        plan = await self.literature_scout_agent.plan(state)
        state.initial_plan = plan
        state.active_queries = list(plan.queries)
        raw_papers, warnings = await self.literature_scout_agent.search(state)
        state.warnings = list(warnings)
        curated_papers, must_read_ids, ingest_candidate_ids = self.paper_curation_skill.curate(
            topic=topic,
            raw_papers=raw_papers,
            max_papers=max_papers,
        )
        state.curated_papers = curated_papers
        state.must_read_ids = must_read_ids
        state.ingest_candidate_ids = ingest_candidate_ids
        report = await self.research_writer_agent.synthesize_async(state)
        state.report = report
        todo_items = await self.research_writer_agent.plan_todos_async(state)
        state.todo_items = todo_items
        workspace = build_workspace_state(
            objective=topic,
            stage="complete",
            papers=curated_papers,
            imported_document_ids=[],
            report=report,
            plan=plan,
            todo_items=todo_items,
            must_read_ids=must_read_ids,
            ingest_candidate_ids=ingest_candidate_ids,
            stop_reason="Direct supervisor-aligned discovery completed.",
            metadata={
                "decision_model": "supervisor_direct_execution",
                "autonomy_rounds": 1,
                "trace_steps": 0,
            },
        )
        saved_report = report.model_copy(
            update={
                "workspace": workspace,
                "metadata": {
                    **report.metadata,
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": "main_agents_plus_skills",
                    "decision_model": "supervisor_direct_execution",
                    "primary_agents": [
                        "ResearchSupervisorAgent",
                        "LiteratureScoutAgent",
                        "ResearchWriterAgent",
                    ],
                    "primary_skills": [
                        "PaperCurationSkill",
                    ],
                    "supervisor_agent_architecture": "supervisor_direct_execution",
                    "supervisor_decision_model": "supervisor_direct_execution",
                    "autonomy_rounds": 1,
                },
            }
        )
        return SimpleNamespace(
            plan=plan,
            papers=curated_papers,
            report=saved_report,
            workspace=workspace,
            warnings=list(warnings),
            todo_items=todo_items,
            trace=[],
            must_read_ids=must_read_ids,
            ingest_candidate_ids=ingest_candidate_ids,
        )

    async def ask_task_collection(
        self,
        task_id: str,
        request: ResearchTaskAskRequest,
        *,
        graph_runtime,
    ) -> ResearchTaskAskResponse:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        report = self.report_service.load_report(task.task_id, task.report_id)
        papers = self.report_service.load_papers(task.task_id)
        scope = self.paper_selector_service.resolve_qa_scope(
            task=task,
            papers=papers,
            requested_paper_ids=request.paper_ids,
            requested_document_ids=request.document_ids,
        )
        if scope.explicit_scope and not scope.paper_ids and not scope.document_ids:
            raise ValueError("The requested paper/document scope did not match the current research task.")
        document_ids = list(scope.document_ids)
        scoped_papers = list(scope.papers or papers)
        if not document_ids and not scoped_papers and report is None:
            raise ValueError(f"Research task has no imported documents or persisted research artifacts available for QA: {task_id}")
        routing_authority = str(request.metadata.get("routing_authority") or "").strip()
        preferred_qa_route = str(request.metadata.get("preferred_qa_route") or "").strip()
        user_intent = None
        if routing_authority != "supervisor_llm":
            user_intent = await self.user_intent_resolver.resolve_async(
                message=request.question,
                has_task=True,
                candidate_paper_count=len(papers),
                candidate_papers=[
                    {
                        "paper_id": paper.paper_id,
                        "index": index,
                        "title": paper.title,
                        "source": paper.source,
                        "year": paper.year,
                    }
                    for index, paper in enumerate(papers, start=1)
                ],
                active_paper_ids=[
                    str(item).strip()
                    for item in ((request.metadata.get("context") or {}).get("active_paper_ids", []) if isinstance(request.metadata.get("context"), dict) else [])
                    if str(item).strip()
                ],
                selected_paper_ids=list(scope.paper_ids),
                has_visual_anchor=bool(request.image_path or request.chart_id or request.metadata.get("image_path")),
                has_document_input=False,
            )
            resolved_intent_paper_ids = [
                paper_id
                for paper_id in user_intent.resolved_paper_ids
                if paper_id in {paper.paper_id for paper in papers}
            ]
            if resolved_intent_paper_ids and not scope.paper_ids:
                scope = self.paper_selector_service.resolve_qa_scope(
                    task=task,
                    papers=papers,
                    requested_paper_ids=resolved_intent_paper_ids,
                    requested_document_ids=request.document_ids,
                )
                document_ids = list(scope.document_ids)
                scoped_papers = list(scope.papers or papers)
            if user_intent.needs_clarification:
                raise ValueError(user_intent.clarification_question or "当前问题指向不明确，请补充具体论文或图表。")
        if preferred_qa_route in {"collection_qa", "document_drilldown", "chart_drilldown"}:
            qa_route_decision = ResearchQARouteDecision(
                route=preferred_qa_route,  # type: ignore[arg-type]
                confidence=0.99,
                rationale="Supervisor selected the QA route explicitly.",
                visual_anchor=self._extract_visual_anchor(request=request, metadata=request.metadata),
            )
        else:
            qa_route_decision = await self._select_qa_route(
                question=request.question,
                scope_mode=scope.scope_mode,
                paper_ids=scope.paper_ids,
                document_ids=document_ids,
                request=request,
                metadata=request.metadata,
            )
        logger.info(
            "Research QA route selected: route=%s confidence=%.2f has_visual_anchor=%s request_image_path=%s request_chart_id=%s request_page_id=%s request_page_number=%s",
            qa_route_decision.route,
            qa_route_decision.confidence,
            qa_route_decision.visual_anchor is not None,
            bool(str(request.image_path or "").strip()),
            str(request.chart_id or ""),
            str(request.page_id or ""),
            request.page_number,
        )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None:
            inferred_visual_anchor = await self._infer_or_discover_visual_anchor(
                task_id=task.task_id,
                papers=scoped_papers,
                document_ids=document_ids,
                question=request.question,
                graph_runtime=graph_runtime,
            )
            if inferred_visual_anchor is not None:
                logger.info(
                    "Research QA inferred visual anchor for chart drilldown: image_path=%s chart_id=%s page_id=%s page_number=%s",
                    str(inferred_visual_anchor.get("image_path") or ""),
                    str(inferred_visual_anchor.get("chart_id") or ""),
                    str(inferred_visual_anchor.get("page_id") or ""),
                    inferred_visual_anchor.get("page_number"),
                )
                qa_route_decision = ResearchQARouteDecision(
                    route=qa_route_decision.route,
                    confidence=max(qa_route_decision.confidence, 0.9),
                    rationale=(
                        f"{qa_route_decision.rationale} Auto-selected a paper figure preview "
                        "to ground chart-focused QA."
                    ),
                    visual_anchor=inferred_visual_anchor,
                )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None:
            restored_visual_anchor = self._restore_visual_anchor_from_workspace(task=task, report=report)
            if restored_visual_anchor is not None:
                logger.info(
                    "Research QA restored visual anchor from workspace: image_path=%s chart_id=%s page_id=%s page_number=%s",
                    str(restored_visual_anchor.get("image_path") or ""),
                    str(restored_visual_anchor.get("chart_id") or ""),
                    str(restored_visual_anchor.get("page_id") or ""),
                    restored_visual_anchor.get("page_number"),
                )
                qa_route_decision = ResearchQARouteDecision(
                    route=qa_route_decision.route,
                    confidence=max(qa_route_decision.confidence, 0.9),
                    rationale=(
                        f"{qa_route_decision.rationale} Restored the latest visual anchor from the saved workspace "
                        "to ground chart-focused QA."
                    ),
                    visual_anchor=restored_visual_anchor,
                )
        scoped_request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "qa_route": qa_route_decision.route,
                    "qa_route_confidence": qa_route_decision.confidence,
                    "qa_route_rationale": qa_route_decision.rationale,
                    "visual_anchor": qa_route_decision.visual_anchor,
                    "qa_scope_mode": scope.scope_mode,
                    "selected_paper_ids": scope.paper_ids,
                    "selected_document_ids": document_ids,
                    "selected_paper_titles": scope.selected_titles(),
                    "selection_warnings": scope.warnings,
                    "selection_summary": scope.metadata.get("selection_summary"),
                    "scope_metadata": scope.metadata,
                    "user_intent": user_intent.model_dump(mode="json") if user_intent is not None else None,
                    "routing_authority": routing_authority or None,
                    "preferred_qa_route": preferred_qa_route or None,
                }
            }
        )
        execution_context = self.build_execution_context(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=task,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            selected_paper_ids=scope.paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=scoped_request.metadata,
        )
        qa = await self._run_scoped_task_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=qa_route_decision,
        )
        quality_check = self._build_answer_quality_check(
            qa=qa,
            route=qa_route_decision.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        qa, qa_route_decision, quality_check = await self._maybe_recover_qa_route(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
            scope=scope,
        )
        qa = qa.model_copy(
            update={
                "metadata": {
                    **qa.metadata,
                    **request.metadata,
                    "qa_route": qa_route_decision.route,
                    "qa_route_confidence": qa_route_decision.confidence,
                    "qa_route_rationale": qa_route_decision.rationale,
                    "visual_anchor": qa_route_decision.visual_anchor,
                    "research_task_id": task_id,
                    "research_topic": task.topic,
                    "qa_scope_mode": scope.scope_mode,
                    "selected_paper_ids": scope.paper_ids,
                    "selected_document_ids": document_ids,
                    "selected_paper_titles": scope.selected_titles(),
                    "selection_warnings": scope.warnings,
                    "selection_summary": scope.metadata.get("selection_summary"),
                    "scope_metadata": scope.metadata,
                    "qa_route_recovery_count": qa_route_decision.recovery_count,
                }
            }
        )
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=scoped_papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )
        if visual_anchor_figure is not None:
            qa = qa.model_copy(
                update={
                    "metadata": {
                        **qa_metadata,
                        "visual_anchor_figure": visual_anchor_figure.model_dump(mode="json"),
                    }
                }
            )
        qa = qa.model_copy(
            update={
                "metadata": {
                    **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                    "answer_quality_check": quality_check,
                }
            }
        )
        updated_task, updated_report = self._apply_qa_follow_up(
            task=task,
            request=scoped_request,
            qa=qa,
            papers=papers,
            document_ids=document_ids,
            scope=scope,
        )
        self.save_task_state(updated_task, conversation_id=request.conversation_id)
        self.report_service.save_report(updated_report)
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            report=updated_report,
            papers=scoped_papers,
            document_ids=document_ids,
            selected_paper_ids=scope.paper_ids,
            task_intent=f"research_{qa_route_decision.route}:{scope.scope_mode}",
            question=request.question,
            answer=qa.answer,
            retrieval_summary=(
                f"documents={len(document_ids)}, evidences={len(qa.evidence_bundle.evidences)}, "
                f"confidence={qa.confidence if qa.confidence is not None else 'empty'}"
            ),
            metadata_update={
                "last_skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style or "cot",
                "evidence_count": len(qa.evidence_bundle.evidences),
                "qa_scope_mode": scope.scope_mode,
                "selected_paper_ids": scope.paper_ids[:8],
                "answer_quality_check": quality_check,
            },
        )
        if not quality_check["needs_recovery"] and execution_context.session_id:
            self.memory_manager.promote_conclusion_to_long_term(
                execution_context.session_id,
                conclusion=self._compact_text(qa.answer, limit=700),
                topic=task.topic,
                keywords=[qa_route_decision.route, scope.scope_mode],
                related_paper_ids=scope.paper_ids[:8],
                metadata={
                    "question": request.question,
                    "confidence": qa.confidence,
                    "evidence_count": len(qa.evidence_bundle.evidences),
                },
            )
        return ResearchTaskAskResponse(
            task_id=task_id,
            paper_ids=scope.paper_ids,
            document_ids=document_ids,
            scope_mode=scope.scope_mode,
            qa=qa,
            report=updated_report,
            todo_items=updated_task.todo_items,
            warnings=scope.warnings,
        )

    async def _select_qa_route(
        self,
        *,
        question: str,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchQARouteDecision:
        visual_anchor = self._extract_visual_anchor(request=request, metadata=metadata)
        route_result = await self.qa_routing_skill.classify_async(
            question=question,
            scope_mode=scope_mode,
            paper_ids=paper_ids,
            document_ids=document_ids,
            has_visual_anchor=visual_anchor is not None,
        )
        return ResearchQARouteDecision(
            route=route_result.route,
            confidence=route_result.confidence,
            rationale=route_result.rationale,
            visual_anchor=visual_anchor if route_result.route == "chart_drilldown" else None,
        )

    def _restore_visual_anchor_from_workspace(
        self,
        *,
        task: ResearchTask,
        report: ResearchReport | None,
    ) -> dict[str, Any] | None:
        workspace_candidates = [
            task.workspace.metadata if isinstance(task.workspace.metadata, dict) else {},
            report.workspace.metadata if report is not None and isinstance(report.workspace.metadata, dict) else {},
        ]
        for metadata in workspace_candidates:
            anchor = metadata.get("last_visual_anchor")
            if not isinstance(anchor, dict):
                continue
            image_path = str(anchor.get("image_path") or "").strip()
            if not image_path:
                continue
            restored: dict[str, Any] = {"image_path": image_path}
            page_id = str(anchor.get("page_id") or "").strip()
            if page_id:
                restored["page_id"] = page_id
            chart_id = str(anchor.get("chart_id") or "").strip()
            if chart_id:
                restored["chart_id"] = chart_id
            raw_page_number = anchor.get("page_number")
            try:
                page_number = int(raw_page_number) if raw_page_number is not None else None
            except (TypeError, ValueError):
                page_number = None
            if page_number is not None and page_number >= 1:
                restored["page_number"] = page_number
            for key in ("figure_id", "anchor_source", "anchor_selection", "anchor_rationale"):
                value = anchor.get(key)
                if isinstance(value, str) and value.strip():
                    restored[key] = value.strip()
            return restored
        return None

    def _extract_visual_anchor(
        self,
        *,
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        image_path = str(request.image_path or (metadata or {}).get("image_path") or "").strip()
        if not image_path:
            return None
        anchor = {"image_path": image_path}
        page_id = request.page_id or (metadata or {}).get("page_id")
        if page_id:
            anchor["page_id"] = str(page_id)
        try:
            raw_page_number = (
                request.page_number
                if request.page_number is not None
                else (metadata or {}).get("page_number")
            )
            page_number = int(raw_page_number) if raw_page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        if page_number is not None and page_number >= 1:
            anchor["page_number"] = page_number
        chart_id = request.chart_id or (metadata or {}).get("chart_id")
        if chart_id:
            anchor["chart_id"] = str(chart_id)
        return anchor

    async def _infer_cached_visual_anchor(
        self,
        *,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
    ) -> dict[str, Any] | None:
        return await self.chart_analysis_agent.infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )

    async def _infer_or_discover_visual_anchor(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
        graph_runtime: Any,
    ) -> dict[str, Any] | None:
        inferred = await self._infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
        )
        if inferred is not None:
            return inferred
        discovered = await self._discover_figures_for_scope(
            task_id=task_id,
            papers=papers,
            document_ids=document_ids,
            graph_runtime=graph_runtime,
        )
        if not discovered:
            return None
        refreshed_papers = self.report_service.load_papers(task_id)
        scoped_papers = [
            paper for paper in refreshed_papers
            if not papers or paper.paper_id in {item.paper_id for item in papers}
        ] or refreshed_papers
        return await self._infer_cached_visual_anchor(
            papers=scoped_papers,
            document_ids=document_ids,
            question=question,
        )

    async def _discover_figures_for_scope(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        graph_runtime: Any,
    ) -> bool:
        allowed_document_ids = {item for item in document_ids if item}
        discovered = False
        for paper in papers:
            document_id = str(paper.metadata.get("document_id") or "").strip()
            storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
            if paper.ingest_status != "ingested" or not document_id or not storage_uri:
                continue
            if allowed_document_ids and document_id not in allowed_document_ids:
                continue
            if self._load_cached_figure_payload(paper=paper) is not None:
                discovered = True
                continue
            try:
                await self.list_paper_figures(
                    task_id,
                    paper.paper_id,
                    graph_runtime=graph_runtime,
                )
                discovered = True
            except Exception:
                continue
        return discovered

    async def _run_scoped_task_qa(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None and not document_ids:
            selected_titles = [
                str(item).strip()
                for item in (request.metadata.get("selected_paper_titles", []) if isinstance(request.metadata, dict) else [])
                if str(item).strip()
            ]
            scoped_target = selected_titles[0] if len(selected_titles) == 1 else "当前目标论文"
            return QAResponse(
                answer=(
                    f"我知道你是在问 {scoped_target} 里的图表/系统框图，但当前工作区还没有这篇论文的已导入正文或图表锚点，"
                    "所以我不能可靠解释具体图示内容。请先导入这篇论文，或明确指定图号/上传图像后再问。"
                ),
                question=request.question,
                evidence_bundle=EvidenceBundle(
                    summary="chart_question_without_document_scope",
                    metadata={
                        "reason": "chart_question_without_document_scope",
                        "qa_route": qa_route_decision.route,
                    },
                ),
                confidence=0.18,
                metadata={
                    **(request.metadata if isinstance(request.metadata, dict) else {}),
                    "qa_route": qa_route_decision.route,
                    "qa_route_source": "literature_research_service",
                    "chart_drilldown_blocked_reason": "missing_document_scope_and_visual_anchor",
                },
            )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is not None:
            handle_ask_fused = getattr(graph_runtime, "handle_ask_fused", None)
            visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
                papers=papers,
                qa_metadata={"visual_anchor": qa_route_decision.visual_anchor},
                load_cached_figure_payload=self._load_cached_figure_payload,
            )
            figure_context = (
                {
                    "figure": visual_anchor_figure.model_dump(mode="json"),
                    "paper_id": visual_anchor_figure.paper_id,
                    "figure_id": visual_anchor_figure.figure_id,
                    "chart_id": visual_anchor_figure.chart_id,
                    "page_id": visual_anchor_figure.page_id,
                    "page_number": visual_anchor_figure.page_number,
                    "title": visual_anchor_figure.title,
                    "caption": visual_anchor_figure.caption,
                    "source": visual_anchor_figure.source,
                }
                if visual_anchor_figure is not None
                else {}
            )
            logger.info(
                "Research QA attempting fused chart drilldown: has_handle_ask_fused=%s image_path=%s chart_id=%s page_id=%s page_number=%s document_count=%s",
                callable(handle_ask_fused),
                str(qa_route_decision.visual_anchor.get("image_path") or ""),
                str(qa_route_decision.visual_anchor.get("chart_id") or ""),
                str(qa_route_decision.visual_anchor.get("page_id") or ""),
                qa_route_decision.visual_anchor.get("page_number"),
                len(document_ids),
            )
            if callable(handle_ask_fused):
                fused_result = await handle_ask_fused(
                    question=request.question,
                    doc_id=document_ids[0] if len(document_ids) == 1 else None,
                    document_ids=document_ids,
                    top_k=request.top_k,
                    session_id=execution_context.session_id,
                    filters={
                        "research_task_id": task.task_id,
                        "research_topic": task.topic,
                        "qa_mode": "research_collection",
                        "qa_route": qa_route_decision.route,
                        "qa_scope_mode": request.metadata.get("qa_scope_mode"),
                        "selected_paper_ids": request.metadata.get("selected_paper_ids", []),
                        "selected_document_ids": request.metadata.get("selected_document_ids", []),
                    },
                    metadata={
                        **request.metadata,
                        "qa_route": qa_route_decision.route,
                        "qa_route_source": "literature_research_service",
                        "visual_anchor": qa_route_decision.visual_anchor,
                        "visual_anchor_figure": (
                            visual_anchor_figure.model_dump(mode="json")
                            if visual_anchor_figure is not None
                            else None
                        ),
                        "figure_context": figure_context,
                    },
                    skill_name=request.skill_name,
                    reasoning_style=request.reasoning_style,
                    **qa_route_decision.visual_anchor,
                )
                qa = fused_result.qa
                return qa.model_copy(
                    update={
                        "metadata": {
                            **qa.metadata,
                            "autonomy_mode": "task_scoped_drilldown",
                            "agent_architecture": "research_service_to_graph_runtime",
                            "primary_agents": ["RagRuntime"],
                            "selected_skill": request.skill_name,
                            "memory_enabled": execution_context.memory_enabled,
                            "session_id": execution_context.session_id,
                            "drilldown_runtime": "fused_chart",
                        }
                    }
                )
            logger.warning("Research QA chart drilldown fell back because graph_runtime.handle_ask_fused is not callable")

        if qa_route_decision.route in {"document_drilldown", "chart_drilldown"} and document_ids:
            handle_ask_document = getattr(graph_runtime, "handle_ask_document", None)
            if callable(handle_ask_document):
                logger.info(
                    "Research QA using document drilldown fallback: route=%s document_count=%s has_visual_anchor=%s",
                    qa_route_decision.route,
                    len(document_ids),
                    qa_route_decision.visual_anchor is not None,
                )
                qa = await handle_ask_document(
                    question=request.question,
                    document_ids=document_ids,
                    top_k=request.top_k,
                    filters={
                        "research_task_id": task.task_id,
                        "research_topic": task.topic,
                        "qa_mode": "research_collection",
                        "qa_route": qa_route_decision.route,
                        "qa_scope_mode": request.metadata.get("qa_scope_mode"),
                        "selected_paper_ids": request.metadata.get("selected_paper_ids", []),
                        "selected_document_ids": request.metadata.get("selected_document_ids", []),
                    },
                    session_id=execution_context.session_id,
                    task_intent=f"research_{qa_route_decision.route}",
                    metadata={
                        **request.metadata,
                        "qa_route": qa_route_decision.route,
                        "qa_route_source": "literature_research_service",
                    },
                    skill_name=request.skill_name,
                    reasoning_style=request.reasoning_style,
                )
                return qa.model_copy(
                    update={
                        "metadata": {
                            **qa.metadata,
                            "autonomy_mode": "task_scoped_drilldown",
                            "agent_architecture": "research_service_to_graph_runtime",
                            "primary_agents": ["RagRuntime"],
                            "selected_skill": request.skill_name,
                            "memory_enabled": execution_context.memory_enabled,
                            "session_id": execution_context.session_id,
                            "drilldown_runtime": "document",
                        }
                    }
                )

        return await self._run_direct_collection_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
        )

    async def _run_direct_collection_qa(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
    ) -> QAResponse:
        if getattr(graph_runtime, "answer_tools", None) is None and self.research_qa_runtime is not None:
            legacy_result = await self.research_qa_runtime.run(
                graph_runtime=graph_runtime,
                task=task,
                request=request,
                report=report,
                papers=papers,
                document_ids=document_ids,
                execution_context=execution_context,
            )
            qa = legacy_result.qa
            return qa.model_copy(
                update={
                    "metadata": {
                        **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                        "autonomy_mode": "lead_agent_loop",
                        "agent_architecture": "main_agents_only",
                        "qa_execution_path": "research_supervisor_legacy_fallback",
                        "memory_enabled": execution_context.memory_enabled,
                        "session_id": execution_context.session_id,
                    }
                }
            )

        resolver = getattr(graph_runtime, "resolve_skill_context", None)
        skill_context = (
            resolver(
                task_type="ask_document",
                preferred_skill_name=request.skill_name or "research_report",
            )
            if callable(resolver)
            else None
        )
        resolved_question = self._rewrite_collection_question(
            question=request.question,
            task=task,
            papers=papers,
            scope_mode=str((request.metadata or {}).get("qa_scope_mode") or "all_imported"),
        )
        request_with_resolution = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "resolved_question": resolved_question,
                }
            }
        )
        runtime_state = SimpleNamespace(
            task=task,
            request=request_with_resolution,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            skill_context=skill_context,
            queries=[],
            completed_queries=set(),
            refinement_used=False,
            summary_checked=False,
            manifest_built=False,
            retrieval_hits=[],
            summary_hits=[],
            manifest_hits=[],
            evidence_bundle=EvidenceBundle(),
            retrieval_result=None,
            qa=None,
            warnings=[],
            trace=[],
            question=resolved_question,
            original_question=request.question,
            top_k=request.top_k,
        )
        runtime_state.queries = await self.research_knowledge_agent.plan_collection_queries(runtime_state)
        for query in list(runtime_state.queries):
            try:
                hits = await self.research_knowledge_agent.retrieve_collection_evidence(
                    graph_runtime=graph_runtime,
                    state=runtime_state,
                    query=query,
                )
                runtime_state.retrieval_hits = merge_retrieval_hits(
                    runtime_state.retrieval_hits,
                    hits,
                )
            except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
                runtime_state.warnings.append(f"collection_retrieval:{query} failed: {exc}")
            runtime_state.completed_queries.add(query)
        runtime_state.summary_checked = True
        try:
            summary_hits = await self.research_knowledge_agent.retrieve_graph_summary(
                graph_runtime=graph_runtime,
                state=runtime_state,
            )
            runtime_state.summary_hits = merge_retrieval_hits(
                runtime_state.summary_hits,
                summary_hits,
            )
        except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
            runtime_state.warnings.append(f"graph_summary:{runtime_state.question} failed: {exc}")
        runtime_state.manifest_hits = self.research_knowledge_agent.build_collection_manifest(runtime_state)
        runtime_state.manifest_built = True
        qa = await self.research_writer_agent.answer_collection_question(
            graph_runtime=graph_runtime,
            state=runtime_state,
            primary_agents=[
                "ResearchSupervisorAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
            ],
        )
        return qa.model_copy(
            update={
                "metadata": {
                    **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": "main_agents_only",
                    "primary_agents": [
                        "ResearchSupervisorAgent",
                        "ResearchKnowledgeAgent",
                        "ResearchWriterAgent",
                    ],
                    "supervisor_execution_mode": "single_supervisor_action",
                    "supervisor_agent_architecture": "supervisor_direct_execution",
                    "qa_execution_path": "research_supervisor_direct",
                    "qa_warnings": list(runtime_state.warnings),
                    "planned_queries": list(runtime_state.queries),
                    "completed_queries": list(runtime_state.completed_queries),
                    "memory_enabled": execution_context.memory_enabled,
                    "session_id": execution_context.session_id,
                }
            }
        )

    def _rewrite_collection_question(
        self,
        *,
        question: str,
        task: ResearchTask,
        papers: list[PaperCandidate],
        scope_mode: str,
    ) -> str:
        normalized = str(question or "").strip()
        if not normalized:
            return normalized
        compact = re.sub(r"\s+", "", normalized.lower())
        if compact in {"效果怎么样", "效果如何", "表现怎么样", "表现如何"}:
            return (
                f"请结合研究主题“{task.topic}”对当前研究集合做综合评价，"
                "说明整体效果、证据强弱与主要边界，不要只回答单篇论文。"
            )
        if scope_mode == "all_imported":
            return normalized
        return normalized

    def _refresh_existing_pool(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        merged = self.paper_search_service._dedupe([*existing_papers, *incoming_papers])
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged,
            max_papers=max(len(merged), 1),
        )

    async def _maybe_recover_qa_route(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa: QAResponse,
        qa_route_decision: ResearchQARouteDecision,
        quality_check: dict[str, Any],
        scope: PaperSelectionScope,
    ) -> tuple[QAResponse, ResearchQARouteDecision, dict[str, Any]]:
        rerouted = self._select_recovery_qa_route(
            request=request,
            scope=scope,
            document_ids=document_ids,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
        )
        if rerouted is None:
            return qa, qa_route_decision, quality_check
        recovered_qa = await self._run_scoped_task_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=rerouted,
        )
        recovered_quality = self._build_answer_quality_check(
            qa=recovered_qa,
            route=rerouted.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        recovered_qa = recovered_qa.model_copy(
            update={
                "metadata": {
                    **(recovered_qa.metadata if isinstance(recovered_qa.metadata, dict) else {}),
                    "qa_route_recovered_from": qa_route_decision.route,
                    "qa_route_recovery_reason": rerouted.rationale,
                }
            }
        )
        return recovered_qa, rerouted, recovered_quality

    def _select_recovery_qa_route(
        self,
        *,
        request: ResearchTaskAskRequest,
        scope: PaperSelectionScope,
        document_ids: list[str],
        qa: QAResponse,
        qa_route_decision: ResearchQARouteDecision,
        quality_check: dict[str, Any],
    ) -> ResearchQARouteDecision | None:
        if qa_route_decision.recovery_count >= 1:
            return None
        if qa_route_decision.visual_anchor is not None:
            return None
        if not quality_check.get("needs_recovery"):
            return None
        if document_ids and qa_route_decision.route == "collection_qa" and scope.scope_mode in {"selected_documents", "selected_papers"}:
            return ResearchQARouteDecision(
                route="document_drilldown",
                confidence=max(qa_route_decision.confidence, 0.72),
                rationale=(
                    "The initial collection QA answer was under-supported for a narrowed paper/document scope, "
                    "so a single conservative retry uses document drilldown."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count + 1,
            )
        if qa_route_decision.route == "document_drilldown" and not document_ids:
            return ResearchQARouteDecision(
                route="collection_qa",
                confidence=max(qa_route_decision.confidence, 0.7),
                rationale=(
                    "The initial document drilldown route had no usable document scope, "
                    "so a single conservative retry broadens to collection QA."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count + 1,
            )
        return None

    def _apply_qa_follow_up(
        self,
        *,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        qa,
        papers: list[PaperCandidate],
        document_ids: list[str],
        scope: PaperSelectionScope,
    ) -> tuple[ResearchTask, ResearchReport]:
        now = datetime.now(UTC).isoformat()
        report = self.report_service.load_report(task.task_id, task.report_id)
        if report is None:
            report = ResearchReport(
                report_id=task.report_id or f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                task_id=task.task_id,
                topic=task.topic,
                generated_at=now,
                markdown=f"# 文献调研报告：{task.topic}",
                paper_count=task.paper_count,
                source_counts={},
                metadata={"writer": "qa_follow_up"},
            )

        evidence_count = len(qa.evidence_bundle.evidences)
        confidence_value = qa.confidence if qa.confidence is not None else 0.0
        insufficient = self._is_insufficient_answer(
            answer=qa.answer,
            confidence=confidence_value,
            evidence_count=evidence_count,
        )
        question_summary = self._compact_text(request.question, limit=120)
        answer_summary = self._compact_text(qa.answer, limit=220)

        todo_items = self._upsert_follow_up_todo(
            existing_items=task.todo_items,
            question=request.question,
            question_summary=question_summary,
            answer_summary=answer_summary,
            insufficient=insufficient,
            evidence_count=evidence_count,
            confidence=confidence_value,
            created_at=now,
        )
        latest_todo = todo_items[0] if todo_items else None
        updated_markdown = self._append_qa_report_entry(
            markdown=report.markdown,
            asked_at=now,
            question=request.question,
            answer=qa.answer,
            document_count=len(document_ids),
            evidence_count=evidence_count,
            confidence=qa.confidence,
            todo_item=latest_todo,
            paper_titles=scope.selected_titles(),
            scope_mode=scope.scope_mode,
        )

        updated_highlights = list(report.highlights)
        updated_gaps = list(report.gaps)
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor = qa_metadata.get("visual_anchor") if isinstance(qa_metadata.get("visual_anchor"), dict) else None
        visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )
        if insufficient:
            gap = (
                f"关于“{question_summary}”的研究集合证据仍不足，当前仅有 {evidence_count} 条证据，"
                "建议补充更直接的论文或扩大检索范围。"
            )
            updated_gaps = self._prepend_unique_text(updated_gaps, gap)
        else:
            highlight = f"问答补充：关于“{question_summary}”，当前结论是 {answer_summary}"
            updated_highlights = self._prepend_unique_text(updated_highlights, highlight)

        report_metadata = dict(report.metadata)
        report_metadata.update(
            {
                "last_qa_at": now,
                "last_qa_question": question_summary,
                "qa_update_count": int(report.metadata.get("qa_update_count") or 0) + 1,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            }
        )
        updated_report = report.model_copy(
            update={
                "generated_at": now,
                "markdown": updated_markdown,
                "highlights": updated_highlights[:10],
                "gaps": updated_gaps[:10],
                "metadata": report_metadata,
            }
        )
        updated_task = task.model_copy(
            update={
                "updated_at": now,
                "report_id": updated_report.report_id,
                "todo_items": todo_items[:20],
            }
        )
        workspace = build_workspace_from_task(
            task=updated_task,
            report=updated_report,
            papers=papers,
            stage="qa",
            extra_questions=[request.question],
            extra_findings=[qa.answer],
            stop_reason=(
                "Collection QA found an evidence gap that should drive the next retrieval cycle."
                if insufficient
                else "Collection QA completed and committed its answer back into the research workspace."
            ),
            metadata={
                "last_qa_question": question_summary,
                "last_qa_confidence": round(confidence_value, 4),
                "last_qa_evidence_count": evidence_count,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure": (
                    visual_anchor_figure.model_dump(mode="json") if visual_anchor_figure is not None else None
                ),
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            },
        )
        updated_report = updated_report.model_copy(update={"workspace": workspace})
        updated_task = updated_task.model_copy(update={"workspace": workspace})
        return updated_task, updated_report

    def _upsert_follow_up_todo(
        self,
        *,
        existing_items: list[ResearchTodoItem],
        question: str,
        question_summary: str,
        answer_summary: str,
        insufficient: bool,
        evidence_count: int,
        confidence: float,
        created_at: str,
    ) -> list[ResearchTodoItem]:
        if insufficient:
            content = f"补充与“{question_summary}”直接相关的论文，并在扩展关键词或时间窗口后重新运行研究任务。"
            rationale = f"当前仅检索到 {evidence_count} 条可用证据，置信度 {confidence:.2f}，现有研究集合无法稳定回答该问题。"
            source = "evidence_gap"
            priority = "high"
        else:
            content = f"围绕“{question_summary}”整理一个更细粒度的对比表，并持续核验新论文是否改变当前结论。"
            rationale = f"当前已有可用答案，可继续沉淀成综述结论与实验对比材料。摘要：{answer_summary}"
            source = "qa_follow_up"
            priority = "medium"

        next_item = ResearchTodoItem(
            todo_id=f"todo_{uuid4().hex}",
            content=content,
            rationale=rationale,
            status="open",
            priority=priority,
            created_at=created_at,
            question=question,
            source=source,
            metadata={
                "evidence_count": evidence_count,
                "confidence": round(confidence, 4),
            },
        )
        updated_items: list[ResearchTodoItem] = []
        replaced = False
        for item in existing_items:
            if item.question == question and item.status == "open":
                updated_items.append(next_item)
                replaced = True
            else:
                updated_items.append(item)
        if not replaced:
            updated_items.insert(0, next_item)
        return updated_items

    def _append_qa_report_entry(
        self,
        *,
        markdown: str,
        asked_at: str,
        question: str,
        answer: str,
        document_count: int,
        evidence_count: int,
        confidence: float | None,
        todo_item: ResearchTodoItem | None,
        paper_titles: list[str] | None = None,
        scope_mode: str | None = None,
    ) -> str:
        section_heading = "## 研究集合问答补充"
        entry_lines = [
            f"### {asked_at}",
            f"问题：{question}",
        ]
        entry_lines.extend(
            [
                "",
                "回答：",
                answer.strip() or "（空）",
            ]
        )
        if section_heading in markdown:
            return f"{markdown.rstrip()}\n\n" + "\n".join(entry_lines)
        prefix = markdown.rstrip()
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{section_heading}\n\n" + "\n".join(entry_lines)

    def _prepend_unique_text(self, items: list[str], entry: str) -> list[str]:
        deduped = [item for item in items if item != entry]
        return [entry, *deduped]

    def _compact_text(self, text: str, *, limit: int) -> str:
        compacted = " ".join(text.strip().split())
        if len(compacted) <= limit:
            return compacted
        return f"{compacted[: max(limit - 1, 1)].rstrip()}…"

    def _is_insufficient_answer(self, *, answer: str, confidence: float, evidence_count: int) -> bool:
        lowered = answer.lower()
        insufficient_markers = (
            "证据不足",
            "无法确认",
            "不能确认",
            "信息不足",
            "insufficient evidence",
            "not enough evidence",
        )
        return confidence < 0.45 or evidence_count < 2 or any(marker in lowered for marker in insufficient_markers)

    def _build_answer_quality_check(
        self,
        *,
        qa: QAResponse,
        route: str,
        scope_mode: str,
        document_ids: list[str],
    ) -> dict[str, Any]:
        evidence_count = len(qa.evidence_bundle.evidences)
        confidence = qa.confidence if qa.confidence is not None else 0.0
        insufficient = self._is_insufficient_answer(
            answer=qa.answer,
            confidence=confidence,
            evidence_count=evidence_count,
        )
        warnings: list[str] = []
        if evidence_count < 2:
            warnings.append("low_evidence_count")
        if confidence < 0.45:
            warnings.append("low_confidence")
        if route in {"document_drilldown", "chart_drilldown"} and not document_ids:
            warnings.append("drilldown_without_document_scope")
        if "无法" in qa.answer or "不能确认" in qa.answer:
            warnings.append("answer_contains_uncertainty_marker")
        return {
            "evidence_count": evidence_count,
            "confidence": round(confidence, 4),
            "route": route,
            "scope_mode": scope_mode,
            "needs_recovery": insufficient,
            "recommended_recovery": (
                "import_or_expand_evidence"
                if insufficient
                else "none"
            ),
            "warnings": warnings,
        }

    def _build_message(
        self,
        *,
        role: str,
        kind: str,
        title: str,
        content: str = "",
        meta: str | None = None,
        citations: list[str] | None = None,
        payload: dict | None = None,
    ) -> ResearchMessage:
        return ResearchMessage(
            message_id=f"msg_{uuid4().hex}",
            role=role,
            kind=kind,
            title=title,
            content=content,
            meta=meta,
            created_at=_now_iso(),
            citations=citations or [],
            payload=payload or {},
        )

    def _build_runtime_event(
        self,
        *,
        event_type: str,
        conversation_id: str | None,
        task_id: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ResearchRuntimeEvent:
        return ResearchRuntimeEvent(
            event_id=f"evt_{uuid4().hex}",
            event_type=event_type,
            task_id=task_id,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            timestamp=_now_iso(),
            payload=payload or {},
        )

    def _build_context_summary(
        self,
        *,
        workspace: ResearchWorkspaceState,
        topic: str | None,
        days_back: int | None = None,
        max_papers: int | None = None,
        sources: list[str] | None = None,
        selected_paper_ids: list[str] | None = None,
        paper_count: int | None = None,
        imported_document_ids: list[str] | None = None,
        last_user_message: str | None = None,
        correlation_id: str | None = None,
        messages: list[ResearchMessage] | None = None,
    ) -> ResearchContextSummary:
        metadata = dict(workspace.metadata or {})
        if days_back is not None:
            metadata["days_back"] = days_back
        if max_papers is not None:
            metadata["max_papers"] = max_papers
        if sources is not None:
            metadata["sources"] = list(sources)
        if correlation_id:
            metadata["correlation_id"] = correlation_id
        compression_payload = metadata.get("context_compression")
        recent_messages = list(messages or [])
        compressed_history = self._build_compressed_history_summary(
            recent_messages,
            compression_payload=compression_payload if isinstance(compression_payload, dict) else None,
        )
        derived_findings = compressed_history.get("derived_findings", [])
        derived_actions = compressed_history.get("derived_next_actions", [])
        summary_version = 2 if compressed_history.get("compressed") else 1
        return ResearchContextSummary(
            summary_version=summary_version,
            objective=workspace.objective,
            current_stage=workspace.current_stage,
            topic=topic,
            paper_count=max(0, paper_count if paper_count is not None else metadata.get("paper_count", 0)),
            imported_document_count=len(imported_document_ids or workspace.document_ids),
            selected_paper_count=len(selected_paper_ids or []),
            key_findings=list(dict.fromkeys([*workspace.key_findings[:5], *derived_findings]))[:5],
            evidence_gaps=list(workspace.evidence_gaps[:5]),
            next_actions=list(dict.fromkeys([*workspace.next_actions[:5], *derived_actions]))[:5],
            status_summary=compressed_history.get("status_summary") or workspace.status_summary,
            last_user_message=last_user_message or compressed_history.get("last_user_message"),
            last_updated_at=_now_iso(),
        )

    def _build_compressed_history_summary(
        self,
        messages: list[ResearchMessage],
        *,
        compression_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not messages and not compression_payload:
            return {"compressed": False}
        recent_meaningful = [
            message
            for message in messages
            if (
                ((message.content or "").strip() or (message.title or "").strip())
                and message.title not in INTERNAL_CONVERSATION_MESSAGE_TITLES
                and message.kind != "welcome"
            )
        ]
        compressed = len(recent_meaningful) >= 8 or compression_payload is not None
        last_user_message = next(
            (
                self._compact_text(message.content or message.title, limit=120)
                for message in reversed(recent_meaningful)
                if message.role == "user"
            ),
            None,
        )
        assistant_summaries = [
            self._compact_text(
                message.content or (
                    message.title if message.kind not in {"candidates", "report"} else ""
                ),
                limit=160,
            )
            for message in recent_meaningful
            if message.role == "assistant"
            and (
                (message.content or "").strip()
                or (message.kind not in {"candidates", "report"} and (message.title or "").strip())
            )
        ]
        derived_findings = assistant_summaries[:2] if compressed else []
        derived_next_actions: list[str] = []
        if compression_payload is not None:
            paper_count = int(compression_payload.get("paper_count", 0))
            summary_count = int(compression_payload.get("summary_count", 0))
            derived_next_actions.append(
                f"当前已压缩上下文：覆盖 {paper_count} 篇论文、{summary_count} 条摘要，可继续基于压缩视图追问。"
            )
        status_summary = None
        if compressed:
            fragments: list[str] = []
            if last_user_message:
                fragments.append(f"最近关注：{last_user_message}")
            if compression_payload is not None:
                fragments.append(
                    f"已启用上下文压缩（papers={compression_payload.get('paper_count', 0)}，summaries={compression_payload.get('summary_count', 0)}）"
                )
            if assistant_summaries:
                fragments.append(f"最近结论：{assistant_summaries[0]}")
            status_summary = "；".join(fragments)[:280] if fragments else None
        return {
            "compressed": compressed,
            "last_user_message": last_user_message,
            "derived_findings": derived_findings,
            "derived_next_actions": derived_next_actions,
            "status_summary": status_summary,
        }

    def _build_status_metadata(
        self,
        *,
        lifecycle_status: str,
        correlation_id: str | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
        existing: ResearchStatusMetadata | None = None,
    ) -> ResearchStatusMetadata:
        now = _now_iso()
        started_at = existing.started_at if existing and existing.started_at else now
        finished_at = now if lifecycle_status in {"completed", "failed", "cancelled"} else None
        return ResearchStatusMetadata(
            lifecycle_status=lifecycle_status,
            started_at=started_at if lifecycle_status != "queued" else existing.started_at if existing else None,
            updated_at=now,
            finished_at=finished_at or (existing.finished_at if existing and existing.finished_at else None),
            error_message=error_message,
            retry_count=retry_count if retry_count else (existing.retry_count if existing else 0),
            correlation_id=correlation_id or (existing.correlation_id if existing else None),
        )

    def _map_task_status_to_lifecycle(self, status: str) -> ResearchLifecycleStatus:
        mapping: dict[str, ResearchLifecycleStatus] = {
            "created": "queued",
            "running": "running",
            "completed": "completed",
            "failed": "failed",
        }
        return mapping.get(status, "queued")

    def _map_job_status_to_lifecycle(self, status: str) -> ResearchLifecycleStatus:
        mapping: dict[str, ResearchLifecycleStatus] = {
            "queued": "queued",
            "running": "running",
            "completed": "completed",
            "failed": "failed",
        }
        return mapping.get(status, "queued")

    def _transition_task(
        self,
        task: ResearchTask,
        *,
        status: str | None = None,
        correlation_id: str | None = None,
        error_message: str | None = None,
        retry_count: int | None = None,
        **updates: Any,
    ) -> ResearchTask:
        next_status = status or task.status
        resolved_retry_count = (
            retry_count if retry_count is not None else task.status_metadata.retry_count
        )
        next_metadata = dict(task.metadata or {})
        metadata_update = updates.pop("metadata", None)
        if isinstance(metadata_update, dict):
            next_metadata.update(metadata_update)
        if correlation_id:
            next_metadata["correlation_id"] = correlation_id
        return task.model_copy(
            update={
                **updates,
                "status": next_status,
                "updated_at": _now_iso(),
                "metadata": next_metadata,
                "status_metadata": self._build_status_metadata(
                    lifecycle_status=self._map_task_status_to_lifecycle(next_status),
                    correlation_id=correlation_id,
                    error_message=error_message,
                    retry_count=resolved_retry_count,
                    existing=task.status_metadata,
                ),
            }
        )

    def _transition_job(
        self,
        job: ResearchJob,
        *,
        status: str | None = None,
        correlation_id: str | None = None,
        error_message: str | None = None,
        retry_count: int | None = None,
        **updates: Any,
    ) -> ResearchJob:
        next_status = status or job.status
        resolved_retry_count = (
            retry_count if retry_count is not None else job.status_metadata.retry_count
        )
        next_metadata = dict(job.metadata or {})
        metadata_update = updates.pop("metadata", None)
        if isinstance(metadata_update, dict):
            next_metadata.update(metadata_update)
        if correlation_id:
            next_metadata["correlation_id"] = correlation_id
        return job.model_copy(
            update={
                **updates,
                "status": next_status,
                "updated_at": _now_iso(),
                "metadata": next_metadata,
                "status_metadata": self._build_status_metadata(
                    lifecycle_status=self._map_job_status_to_lifecycle(next_status),
                    correlation_id=correlation_id,
                    error_message=error_message,
                    retry_count=resolved_retry_count,
                    existing=job.status_metadata,
                ),
            }
        )

    def _persist_runtime_conversation_snapshot(
        self,
        *,
        conversation_id: str | None,
        task_response: ResearchTaskResponse | None,
        workspace: ResearchWorkspaceState,
        advanced_strategy: ResearchAdvancedStrategy | None,
    ) -> None:
        if not conversation_id:
            return
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            return
        task = task_response.task if task_response is not None else None
        correlation_id = (
            task.metadata.get("correlation_id")
            if task is not None and isinstance(task.metadata, dict)
            else conversation.status_metadata.correlation_id
        )
        task_id = task.task_id if task is not None else conversation.task_id
        existing_messages = self.report_service.load_messages(conversation_id)
        summary = self._build_context_summary(
            workspace=workspace,
            topic=task.topic if task is not None else conversation.snapshot.topic,
            days_back=task.days_back if task is not None else conversation.snapshot.days_back,
            max_papers=task.max_papers if task is not None else conversation.snapshot.max_papers,
            sources=task.sources if task is not None else conversation.snapshot.sources,
            selected_paper_ids=conversation.snapshot.selected_paper_ids,
            paper_count=task.paper_count if task is not None else None,
            imported_document_ids=task.imported_document_ids if task is not None else workspace.document_ids,
            correlation_id=correlation_id,
            messages=existing_messages,
        )
        event_type = "task_completed" if task_response is not None else "memory_updated"
        next_events = [
            *conversation.snapshot.recent_events[-11:],
            self._build_runtime_event(
                event_type=event_type,
                conversation_id=conversation_id,
                task_id=task_id,
                correlation_id=correlation_id,
                payload={
                    "workspace_stage": workspace.current_stage,
                    "status_summary": workspace.status_summary,
                    "task_status": task.status if task is not None else None,
                },
            ),
        ]
        snapshot_update: dict[str, Any] = {
            "workspace": workspace,
            "context_summary": summary,
            "recent_events": next_events,
        }
        if advanced_strategy is not None:
            snapshot_update["advanced_strategy"] = advanced_strategy
        if task_response is not None:
            snapshot_update.update(
                {
                    "topic": task_response.task.topic,
                    "days_back": task_response.task.days_back,
                    "max_papers": task_response.task.max_papers,
                    "sources": task_response.task.sources,
                    "task_result": task_response,
                }
            )
        updated = conversation.model_copy(
            update={
                "updated_at": _now_iso(),
                "task_id": (
                    task_response.task.task_id
                    if task_response is not None
                    else conversation.task_id
                ),
                "snapshot": conversation.snapshot.model_copy(update=snapshot_update),
                "status_metadata": self._build_status_metadata(
                    lifecycle_status="completed" if task_response is not None else "running",
                    correlation_id=correlation_id,
                    existing=conversation.status_metadata,
                ),
            }
        )
        self.report_service.save_conversation(updated)

    def append_runtime_event(
        self,
        *,
        conversation_id: str | None,
        event_type: str,
        task_id: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not conversation_id:
            return
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            return
        next_events = [
            *conversation.snapshot.recent_events[-11:],
            self._build_runtime_event(
                event_type=event_type,
                conversation_id=conversation_id,
                task_id=task_id or conversation.task_id,
                correlation_id=correlation_id or conversation.status_metadata.correlation_id,
                payload=payload,
            ),
        ]
        updated = conversation.model_copy(
            update={
                "updated_at": _now_iso(),
                "snapshot": conversation.snapshot.model_copy(update={"recent_events": next_events}),
            }
        )
        self.report_service.save_conversation(updated)

    def _record_conversation_messages(
        self,
        conversation_id: str,
        *,
        messages: list[ResearchMessage],
        snapshot_update: dict,
        task_id: str | None = None,
        title_hint: str | None = None,
    ) -> ResearchConversationResponse:
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        existing_messages = self.report_service.load_messages(conversation_id)
        next_messages = [*existing_messages, *messages]
        last_user_message = next(
            (message.content for message in reversed(next_messages) if message.role == "user" and message.content),
            None,
        )
        workspace = snapshot_update.get("workspace", conversation.snapshot.workspace)
        task_result = snapshot_update.get("task_result", conversation.snapshot.task_result)
        topic = snapshot_update.get("topic", conversation.snapshot.topic)
        task_correlation_id = None
        if task_result is not None and isinstance(task_result.task.metadata, dict):
            task_correlation_id = task_result.task.metadata.get("correlation_id")
        correlation_id = (
            snapshot_update.get("correlation_id")
            or task_correlation_id
            or conversation.status_metadata.correlation_id
        )
        effective_snapshot_update = dict(snapshot_update)
        if effective_snapshot_update.get("active_route_mode") == "general_chat":
            effective_snapshot_update.setdefault("selected_paper_ids", [])
            effective_snapshot_update.setdefault("active_paper_ids", [])
        effective_snapshot_update.update(
            self._thread_metadata_update(
                conversation=conversation,
                snapshot_update=effective_snapshot_update,
                last_user_message=last_user_message,
                task_id=task_id if task_id is not None else conversation.task_id,
            )
        )
        context_summary = self._build_context_summary(
            workspace=workspace,
            topic=topic,
            days_back=effective_snapshot_update.get("days_back", conversation.snapshot.days_back),
            max_papers=effective_snapshot_update.get("max_papers", conversation.snapshot.max_papers),
            sources=effective_snapshot_update.get("sources", conversation.snapshot.sources),
            selected_paper_ids=effective_snapshot_update.get(
                "selected_paper_ids",
                conversation.snapshot.selected_paper_ids,
            ),
            paper_count=len(task_result.papers) if task_result is not None else None,
            imported_document_ids=task_result.task.imported_document_ids if task_result is not None else workspace.document_ids,
            last_user_message=last_user_message,
            correlation_id=correlation_id,
            messages=next_messages,
        )
        last_message = messages[-1] if messages else None
        event_type = "tool_failed" if snapshot_update.get("last_error") else "memory_updated"
        next_snapshot = conversation.snapshot.model_copy(
            update={
                **effective_snapshot_update,
                "context_summary": context_summary,
                "recent_events": [
                    *conversation.snapshot.recent_events[-11:],
                    self._build_runtime_event(
                        event_type=event_type,
                        conversation_id=conversation_id,
                        task_id=task_id if task_id is not None else conversation.task_id,
                        correlation_id=correlation_id,
                        payload={
                            "message_kind": last_message.kind if last_message is not None else None,
                            "message_title": last_message.title if last_message is not None else None,
                            "last_error": effective_snapshot_update.get("last_error"),
                            "active_route_mode": effective_snapshot_update.get("active_route_mode"),
                            "active_thread_id": effective_snapshot_update.get("active_thread_id"),
                        },
                    ),
                ],
            }
        )
        last_preview = ""
        if next_messages:
            last_preview = self._compact_text(next_messages[-1].content or next_messages[-1].title, limit=90)
        updated = conversation.model_copy(
            update={
                "title": (title_hint or conversation.title or "未命名研究会话").strip() or "未命名研究会话",
                "updated_at": _now_iso(),
                "task_id": task_id if task_id is not None else conversation.task_id,
                "message_count": len(next_messages),
                "last_message_preview": last_preview or conversation.last_message_preview,
                "snapshot": next_snapshot,
                "status_metadata": self._build_status_metadata(
                    lifecycle_status="failed" if effective_snapshot_update.get("last_error") else "running",
                    correlation_id=correlation_id,
                    error_message=effective_snapshot_update.get("last_error"),
                    existing=conversation.status_metadata,
                ),
            }
        )
        self.report_service.save_conversation(updated)
        self.report_service.save_messages(conversation_id, next_messages)
        return ResearchConversationResponse(conversation=updated, messages=next_messages)

    def _filter_persisted_conversation_messages(
        self,
        messages: list[ResearchMessage],
    ) -> list[ResearchMessage]:
        return [
            message
            for message in messages
            if message.title not in INTERNAL_CONVERSATION_MESSAGE_TITLES
        ]

    def _set_conversation_active_job(self, conversation_id: str | None, job_id: str | None) -> None:
        if not conversation_id:
            return
        conversation = self.report_service.load_conversation(conversation_id)
        if conversation is None:
            return
        updated = conversation.model_copy(
            update={
                "updated_at": _now_iso(),
                "snapshot": conversation.snapshot.model_copy(update={"active_job_id": job_id}),
            }
        )
        self.report_service.save_conversation(updated)

    def _update_job(self, job_id: str, **updates) -> ResearchJob:
        job = self.get_job(job_id)
        status = updates.pop("status", None)
        correlation_id = updates.pop("correlation_id", None) or (
            job.metadata.get("correlation_id") if isinstance(job.metadata, dict) else None
        )
        error_message = updates.pop("error_message", None)
        updated = self._transition_job(
            job,
            status=status,
            correlation_id=correlation_id,
            error_message=error_message,
            **updates,
        )
        self.save_job_state(updated)
        return updated

    def save_task_state(
        self,
        task: ResearchTask,
        *,
        conversation_id: str | None = None,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ResearchTask:
        self.report_service.save_task(task)
        if conversation_id:
            conversation = self.report_service.load_conversation(conversation_id)
            if conversation is not None:
                report = self.report_service.load_report(task.task_id, task.report_id)
                papers = self.report_service.load_papers(task.task_id)
                self._persist_runtime_conversation_snapshot(
                    conversation_id=conversation_id,
                    task_response=ResearchTaskResponse(
                        task=task,
                        papers=papers,
                        report=report,
                        warnings=[],
                    ),
                    workspace=task.workspace,
                    advanced_strategy=conversation.snapshot.advanced_strategy,
                )
        if event_type:
            self.append_runtime_event(
                conversation_id=conversation_id,
                event_type=event_type,
                task_id=task.task_id,
                correlation_id=task.status_metadata.correlation_id,
                payload=payload,
            )
        return task

    def save_job_state(
        self,
        job: ResearchJob,
        *,
        event_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ResearchJob:
        self.report_service.save_job(job)
        if event_type:
            self.append_runtime_event(
                conversation_id=job.conversation_id,
                event_type=event_type,
                task_id=job.task_id,
                correlation_id=job.status_metadata.correlation_id,
                payload=payload,
            )
        return job

    async def _run_import_job(
        self,
        job_id: str,
        request: ImportPapersRequest,
        *,
        graph_runtime,
    ) -> None:
        paper_total = max(len(request.paper_ids) or len(request.papers), 0)
        should_run_qa = bool(request.task_id and (request.question or "").strip())
        progress_total = max(paper_total + (1 if should_run_qa else 0), 1)
        self._update_job(
            job_id,
            status="running",
            progress_message="后台正在下载论文并执行 parse/index。",
            progress_current=0,
            progress_total=progress_total,
        )
        job = self.get_job(job_id)
        self.append_runtime_event(
            conversation_id=request.conversation_id,
            event_type="tool_called",
            task_id=request.task_id,
            correlation_id=job.status_metadata.correlation_id,
            payload={"tool_name": "paper_import", "paper_count": paper_total},
        )
        try:
            async def update_progress(current: int, total: int, result: ImportedPaperResult) -> None:
                self._update_job(
                    job_id,
                    progress_message=f"后台正在导入论文 {current}/{total}：{result.title} · {result.status}",
                    progress_current=current,
                )

            response = await self.import_papers(
                request,
                graph_runtime=graph_runtime,
                progress_callback=update_progress,
            )
            if request.fast_mode and request.include_graph:
                self._update_job(
                    job_id,
                    progress_message="主导入已完成，后台正在补齐图谱索引。",
                    progress_current=max(paper_total, 1),
                )
                await self._run_graph_backfill_for_import_results(
                    response=response,
                    request=request,
                    graph_runtime=graph_runtime,
                )
            task_response = self.get_task(request.task_id) if request.task_id else None
            ask_response = None
            qa_error_message: str | None = None
            output: dict[str, Any] = {
                "import_result": response.model_dump(mode="json"),
                "task_result": task_response.model_dump(mode="json") if task_response else None,
            }

            if should_run_qa and task_response and task_response.task.imported_document_ids:
                self._update_job(
                    job_id,
                    progress_message="论文导入完成，正在执行研究集合问答。",
                    progress_current=max(paper_total, 1),
                )
                self.append_runtime_event(
                    conversation_id=request.conversation_id,
                    event_type="tool_called",
                    task_id=request.task_id,
                    correlation_id=job.status_metadata.correlation_id,
                    payload={"tool_name": "collection_qa", "question": (request.question or "").strip()},
                )
                try:
                    ask_response = await self.ask_task_collection(
                        request.task_id,
                        ResearchTaskAskRequest(
                            question=(request.question or "").strip(),
                            top_k=request.top_k,
                            paper_ids=request.paper_ids,
                            skill_name=request.skill_name,
                            reasoning_style=request.reasoning_style,
                            conversation_id=request.conversation_id,
                        ),
                        graph_runtime=graph_runtime,
                    )
                    task_response = self.get_task(request.task_id)
                    output["task_result"] = task_response.model_dump(mode="json") if task_response else None
                    output["ask_result"] = ask_response.model_dump(mode="json")
                except Exception as exc:
                    qa_error_message = str(exc)
                    output["qa_error_message"] = qa_error_message

            notice = (
                f"后台导入完成：imported={response.imported_count} · skipped={response.skipped_count} · failed={response.failed_count}"
            )
            if request.fast_mode and request.include_graph:
                notice = f"{notice} · graph_backfill=completed"
            if ask_response is not None:
                notice = f"{notice} · qa=completed"
            elif qa_error_message:
                notice = f"{notice} · qa=failed"
            self._update_job(
                job_id,
                status="completed" if qa_error_message is None else "failed",
                progress_message=notice if qa_error_message is None else f"{notice}：{qa_error_message}",
                progress_current=progress_total if qa_error_message is None else max(paper_total, 1),
                error_message=qa_error_message,
                output=output,
            )
            self.observability_service.record_metric(
                metric_type="job_finished",
                payload={
                    "job_id": job_id,
                    "job_kind": "paper_import",
                    "status": "completed" if qa_error_message is None else "failed",
                },
            )
            if request.conversation_id:
                self.record_import_turn(
                    request.conversation_id,
                    task_response=task_response,
                    import_response=response,
                    selected_paper_ids=request.paper_ids,
                    notice=notice,
                )
                if ask_response is not None and task_response is not None:
                    self.record_qa_turn(
                        request.conversation_id,
                        task_response=task_response,
                        ask_response=ask_response,
                    )
                elif qa_error_message:
                    self.record_notice(
                        request.conversation_id,
                        task_response=task_response,
                        notice=f"研究集合问答失败：{qa_error_message}",
                        kind="error",
                        active_job_id=None,
                        last_error=qa_error_message,
                    )
        except Exception as exc:
            message = f"后台导入失败：{exc}"
            self._update_job(job_id, status="failed", progress_message=message, error_message=str(exc))
            self.observability_service.archive_failure(
                failure_type="paper_import_job_failed",
                payload={"job_id": job_id, "task_id": request.task_id, "error_message": str(exc)},
            )
            if request.conversation_id:
                task_response = self.get_task(request.task_id) if request.task_id else None
                self.record_notice(
                    request.conversation_id,
                    task_response=task_response,
                    notice=message,
                    kind="error",
                    active_job_id=None,
                    last_error=str(exc),
                )
            raise
        finally:
            self._set_conversation_active_job(request.conversation_id, None)

    async def _run_todo_import_job(
        self,
        job_id: str,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> None:
        self._update_job(
            job_id,
            status="running",
            progress_message="后台正在从 TODO 补充导入论文。",
            progress_current=0,
            progress_total=request.max_papers,
        )
        job = self.get_job(job_id)
        self.append_runtime_event(
            conversation_id=request.conversation_id,
            event_type="tool_called",
            task_id=task_id,
            correlation_id=job.status_metadata.correlation_id,
            payload={"tool_name": "todo_import", "todo_id": todo_id, "max_papers": request.max_papers},
        )
        try:
            response = await self.import_from_todo(task_id, todo_id, request, graph_runtime=graph_runtime)
            notice = "后台 TODO 补充导入已完成。"
            if response.import_result is not None:
                notice = (
                    f"后台 TODO 补充导入完成：imported={response.import_result.imported_count} · "
                    f"skipped={response.import_result.skipped_count} · failed={response.import_result.failed_count}"
                )
            self._update_job(
                job_id,
                status="completed",
                progress_message=notice,
                progress_current=response.import_result.imported_count if response.import_result else 0,
                output={
                    "imported_count": response.import_result.imported_count if response.import_result else 0,
                    "skipped_count": response.import_result.skipped_count if response.import_result else 0,
                    "failed_count": response.import_result.failed_count if response.import_result else 0,
                },
            )
            self.observability_service.record_metric(
                metric_type="job_finished",
                payload={"job_id": job_id, "job_kind": "todo_import", "status": "completed"},
            )
            if request.conversation_id:
                self.record_import_turn(
                    request.conversation_id,
                    task_response=ResearchTaskResponse(
                        task=response.task,
                        papers=response.papers,
                        report=response.report,
                        warnings=response.warnings,
                    ),
                    import_response=response.import_result or ImportPapersResponse(),
                    notice=notice,
                )
        except Exception as exc:
            message = f"后台 TODO 补充导入失败：{exc}"
            self._update_job(job_id, status="failed", progress_message=message, error_message=str(exc))
            self.observability_service.archive_failure(
                failure_type="todo_import_job_failed",
                payload={"job_id": job_id, "task_id": task_id, "error_message": str(exc)},
            )
            if request.conversation_id:
                self.record_notice(
                    request.conversation_id,
                    task_response=self.get_task(task_id),
                    notice=message,
                    kind="error",
                    active_job_id=None,
                    last_error=str(exc),
                )
            raise
        finally:
            self._set_conversation_active_job(request.conversation_id, None)

    async def _run_graph_backfill_for_import_results(
        self,
        *,
        response: ImportPapersResponse,
        request: ImportPapersRequest,
        graph_runtime: Any,
    ) -> None:
        if not request.task_id:
            return
        papers = self.report_service.load_papers(request.task_id)
        papers_by_document_id = {
            str(paper.metadata.get("document_id") or ""): paper
            for paper in papers
            if str(paper.metadata.get("document_id") or "")
        }
        for result in response.results:
            if not result.graph_pending or not result.document_id or not result.storage_uri:
                continue
            paper = papers_by_document_id.get(result.document_id)
            if paper is None:
                continue
            try:
                parsed_document = await graph_runtime.handle_parse_document(
                    file_path=result.storage_uri,
                    document_id=result.document_id,
                    session_id=request.conversation_id,
                    metadata={
                        "research_paper_id": paper.paper_id,
                        "research_title": paper.title,
                        "research_source": paper.source,
                        "graph_backfill": True,
                    },
                    skill_name=request.skill_name,
                )
                backfill_result = await graph_runtime.handle_graph_backfill_document(
                    parsed_document=parsed_document,
                    charts=[],
                    session_id=request.conversation_id,
                    metadata={
                        "research_paper_id": paper.paper_id,
                        "research_title": paper.title,
                        "research_source": paper.source,
                    },
                )
                result.graph_pending = False
                result.metadata["graph_backfill_status"] = backfill_result.status
                result.metadata["graph_backfill_pending"] = False
                paper.metadata["graph_backfill_pending"] = False
                paper.metadata["graph_backfill_status"] = backfill_result.status
            except Exception as exc:  # noqa: BLE001
                result.metadata["graph_backfill_status"] = "failed"
                result.metadata["graph_backfill_error"] = f"{exc.__class__.__name__}: {exc}"
        self.report_service.save_papers(request.task_id, papers)

    def _format_evidence_citation(self, document_id: str | None, page_number: int | None, source_type: str) -> str:
        parts = [part for part in [document_id or "unknown-doc", f"p.{page_number}" if page_number else None, source_type] if part]
        return " · ".join(parts)

    def _load_task_and_todo(self, task_id: str, todo_id: str) -> tuple[ResearchTask, ResearchTodoItem]:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task, self._find_todo(task, todo_id)

    def _find_todo(self, task: ResearchTask, todo_id: str) -> ResearchTodoItem:
        for item in task.todo_items:
            if item.todo_id == todo_id:
                return item
        raise KeyError(todo_id)

    def _replace_task_todo(
        self,
        task: ResearchTask,
        updated_todo: ResearchTodoItem,
        *,
        updated_at: str,
        **task_updates,
    ) -> ResearchTask:
        next_items = [
            updated_todo if item.todo_id == updated_todo.todo_id else item
            for item in task.todo_items
        ]
        return task.model_copy(
            update={
                "updated_at": updated_at,
                "todo_items": next_items,
                **task_updates,
            }
        )

    def _resolve_todos_after_import(
        self,
        *,
        task: ResearchTask,
        papers_by_id: dict[str, PaperCandidate],
    ) -> list[ResearchTodoItem]:
        now = datetime.now(UTC).isoformat()
        resolved_todos: list[ResearchTodoItem] = []
        for item in task.todo_items:
            todo_paper_ids = [
                paper_id
                for paper_id in item.metadata.get("paper_ids", [])
                if isinstance(paper_id, str) and paper_id
            ] or list(task.workspace.ingest_candidate_ids)
            todo_completed = bool(todo_paper_ids) and all(
                (
                    paper := papers_by_id.get(paper_id)
                ) is not None
                and paper.ingest_status == "ingested"
                and str(paper.metadata.get("document_id") or "").strip()
                for paper_id in todo_paper_ids
            )
            if item.status == "open" and item.metadata.get("todo_kind") == "ingest_priority" and todo_completed:
                resolved_todos.append(
                    item.model_copy(
                        update={
                            "status": "done",
                            "metadata": {
                                **item.metadata,
                                "last_status_change_at": now,
                                "auto_completed_by": "import_papers",
                                "completed_paper_ids": todo_paper_ids[:12],
                            },
                        }
                    )
                )
            else:
                resolved_todos.append(item)
        return resolved_todos

    def _build_todo_query(self, task: ResearchTask, todo: ResearchTodoItem) -> str:
        focus = (todo.question or todo.content).strip()
        if not focus:
            return task.topic
        normalized_topic = task.topic.lower()
        normalized_focus = focus.lower()
        if normalized_topic in normalized_focus:
            return focus
        return f"{task.topic} {focus}"

    def _select_todo_import_candidates(
        self,
        *,
        task: ResearchTask,
        todo: ResearchTodoItem,
        papers: list[PaperCandidate],
        limit: int,
    ) -> list[PaperCandidate]:
        available = [
            paper
            for paper in papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        ]
        if not available:
            return []
        ranked = self.paper_search_service.paper_ranker.rank(
            topic=self._build_todo_query(task, todo),
            papers=available,
            max_papers=max(limit, len(available)),
        )
        return ranked[:limit]

    def _merge_papers(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        deduped: dict[str, PaperCandidate] = {}
        key_aliases: dict[str, str] = {}
        for paper in [*existing_papers, *incoming_papers]:
            candidate_keys = self._paper_identity_keys(paper)
            key = next((key_aliases[item] for item in candidate_keys if item in key_aliases), None)
            if key is None:
                key = candidate_keys[0]
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
                for item in candidate_keys:
                    key_aliases[item] = key
                continue
            merged = existing.model_copy(
                update={
                    "authors": existing.authors or paper.authors,
                    "abstract": existing.abstract or paper.abstract,
                    "year": existing.year or paper.year,
                    "venue": existing.venue or paper.venue,
                    "pdf_url": existing.pdf_url or paper.pdf_url,
                    "url": existing.url or paper.url,
                    "citations": max(existing.citations or 0, paper.citations or 0) or None,
                    "published_at": existing.published_at or paper.published_at,
                    "relevance_score": max(existing.relevance_score or 0, paper.relevance_score or 0) or None,
                    "summary": existing.summary or paper.summary,
                    "ingest_status": existing.ingest_status
                    if existing.ingest_status != "not_selected"
                    else paper.ingest_status,
                    "metadata": {**paper.metadata, **existing.metadata},
                }
            )
            deduped[key] = merged
            for item in candidate_keys:
                key_aliases[item] = key
        merged_papers = list(deduped.values())
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged_papers,
            max_papers=max(len(merged_papers), 1),
        )

    def _paper_identity_keys(self, paper: PaperCandidate) -> list[str]:
        keys: list[str] = []
        if paper.doi:
            keys.append(f"doi:{paper.doi.lower()}")
        if paper.arxiv_id:
            keys.append(f"arxiv:{paper.arxiv_id.lower()}")
        keys.append(f"title:{_normalize_paper_title(paper.title)}")
        return keys

    def _rebuild_task_report(
        self,
        *,
        task: ResearchTask,
        papers: list[PaperCandidate],
        existing_report: ResearchReport | None,
        warnings: list[str],
        action_title: str,
        action_lines: list[str],
        generated_at: str,
    ) -> ResearchReport:
        base_report = self.paper_search_service.survey_writer.generate(
            topic=task.topic,
            task_id=task.task_id,
            papers=papers,
            warnings=warnings,
        )
        markdown = base_report.markdown.rstrip()
        if existing_report:
            qa_section = self._extract_markdown_section(existing_report.markdown, "## 研究集合问答补充")
            todo_section = self._extract_markdown_section(existing_report.markdown, "## TODO 执行记录")
            if qa_section:
                markdown = f"{markdown}\n\n{qa_section.strip()}"
            if todo_section:
                markdown = f"{markdown}\n\n{todo_section.strip()}"
        markdown = self._append_todo_action_entry(
            markdown=markdown,
            executed_at=generated_at,
            action_title=action_title,
            action_lines=action_lines,
        )
        carry_highlights = []
        carry_gaps = []
        metadata = {}
        if existing_report:
            carry_highlights = [
                item for item in existing_report.highlights
                if item.startswith("问答补充：") or item.startswith("TODO执行：")
            ]
            carry_gaps = list(existing_report.gaps)
            metadata.update(existing_report.metadata)
        action_highlight = f"TODO执行：{action_title} -> {action_lines[-1]}" if action_lines else f"TODO执行：{action_title}"
        metadata.update(base_report.metadata)
        metadata.update(
            {
                "last_todo_action_at": generated_at,
                "last_todo_action": action_title,
                "todo_action_count": int(metadata.get("todo_action_count") or 0) + 1,
            }
        )
        return base_report.model_copy(
            update={
                "report_id": existing_report.report_id if existing_report else base_report.report_id,
                "generated_at": generated_at,
                "markdown": markdown,
                "highlights": self._merge_text_entries(
                    [*base_report.highlights, action_highlight],
                    carry_highlights,
                    limit=12,
                ),
                "gaps": self._merge_text_entries(base_report.gaps, carry_gaps, limit=12),
                "metadata": metadata,
            }
        )

    def _extract_markdown_section(self, markdown: str, heading: str) -> str | None:
        lines = markdown.splitlines()
        start_index: int | None = None
        for index, line in enumerate(lines):
            if line.strip() == heading:
                start_index = index
                break
        if start_index is None:
            return None
        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            if lines[index].startswith("## ") and lines[index].strip() != heading:
                end_index = index
                break
        return "\n".join(lines[start_index:end_index]).strip()

    def _append_todo_action_entry(
        self,
        *,
        markdown: str,
        executed_at: str,
        action_title: str,
        action_lines: list[str],
    ) -> str:
        section_heading = "## TODO 执行记录"
        entry_lines = [f"### {executed_at} · {action_title}", *[f"- {line}" for line in action_lines]]
        if section_heading in markdown:
            return f"{markdown.rstrip()}\n\n" + "\n".join(entry_lines)
        prefix = markdown.rstrip()
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{section_heading}\n\n" + "\n".join(entry_lines)

    def _merge_text_entries(self, primary: list[str], secondary: list[str], *, limit: int) -> list[str]:
        merged: list[str] = []
        for item in [*primary, *secondary]:
            if item and item not in merged:
                merged.append(item)
            if len(merged) >= limit:
                break
        return merged
