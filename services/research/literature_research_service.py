from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from core.utils import now_iso as _now_iso
from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.preference_memory_agent import PreferenceMemoryAgent
from domain.schemas.research import (
    ResearchConversation,
    CreateResearchConversationRequest,
    ImportPapersRequest,
    ImportPapersResponse,
    ResearchTaskAskRequest,
    ResearchTaskAskResponse,
    ResearchPaperFigureListResponse,
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    ResearchTodoActionRequest,
    ResearchTodoActionResponse,
    ResearchAdvancedStrategy,
    CreateResearchTaskRequest,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversationResponse,
    ResearchJob,
    ResearchReport,
    ResearchStatusMetadata,
    ResearchTask,
    ResearchTaskResponse,
    ResearchTopicPlan,
    ResearchWorkspaceState,
    ResearchLifecycleStatus,
    SearchPapersRequest,
    SearchPapersResponse,
)
from memory.memory_manager import MemoryManager
from memory.long_term_memory import JsonLongTermMemoryStore, LongTermMemory
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from memory.session_memory import JsonSessionMemoryStore, SessionMemory
from tools.research import (
    PaperAnalysisTool,
    PaperReadingTool,
    QARoutingTool,
    ResearchEvaluationTool,
    IntentResolutionTool,
    VisualIntentRoutingTool,
    ReviewWritingTool,
    WritingPolishTool,
)
from tools.research.collection_qa_capability import ResearchCollectionQACapability
from domain.schemas.research_context import ResearchExecutionContext
from memory.research_context_manager import ResearchContextManager
from memory.research_memory_gateway import ResearchMemoryGateway
from tools.research.paper_selector import PaperSelectorService
from tools.research.paper_import import PaperImportService
from core.observability import ResearchObservabilityService
from tools.research.paper_search import PaperSearchService
from adapters.storage.research_report_service import ResearchReportService
from domain.research_workspace import build_workspace_state
from services.research.conversation_service import ConversationService, _preferred_answer_language_from_text
from services.research.paper_operations import PaperOperationsService, ImportProgressCallback
from services.research.qa_router import QAApplicationService

logger = logging.getLogger(__name__)


class LiteratureResearchService:
    """MVP coordination layer for literature search, persistence, and report retrieval."""

    def __init__(
        self,
        *,
        paper_search_service: PaperSearchService,
        report_service: ResearchReportService,
        paper_import_service: PaperImportService,
        research_runtime: Any | None = None,
        research_context_manager: ResearchContextManager | None = None,
        memory_manager: MemoryManager | None = None,
        paper_selector_service: PaperSelectorService | None = None,
        paper_reading_tool: PaperReadingTool | None = None,
        evaluation_tool: ResearchEvaluationTool | None = None,
        review_writing_tool: ReviewWritingTool | None = None,
        writing_polish_tool: WritingPolishTool | None = None,
        import_concurrency: int = 2,
        import_index_timeout_seconds: float | None = 60.0,
    ) -> None:
        self.paper_search_service = paper_search_service
        self.report_service = report_service
        self.paper_import_service = paper_import_service
        # Legacy compatibility only. Main business paths no longer depend on
        # the old discovery manager runtime by default.
        self.research_runtime = research_runtime
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
        self.paper_reading_tool = paper_reading_tool or PaperReadingTool()
        llm_adapter = getattr(paper_search_service, "llm_adapter", None)
        self.llm_adapter = llm_adapter
        self.research_collection_qa_capability = ResearchCollectionQACapability(
            llm_adapter=llm_adapter,
            paper_analysis_tool=PaperAnalysisTool(llm_adapter=llm_adapter),
        )
        self.qa_routing_tool = QARoutingTool(llm_adapter=llm_adapter)
        self.user_intent_resolver = IntentResolutionTool(llm_adapter=llm_adapter)
        self.visual_intent_router = VisualIntentRoutingTool(llm_adapter=llm_adapter)
        self.chart_analysis_agent = ChartAnalysisAgent(
            llm_adapter=llm_adapter,
            storage_root=report_service.storage_root / "assets",
        )
        self.memory_gateway = ResearchMemoryGateway(
            memory_manager=self.memory_manager,
            research_context_manager=self.research_context_manager,
            paper_reading_tool=self.paper_reading_tool,
            compact_text=lambda value: self._compact_text(value, limit=280),
        )
        self.preference_memory_agent = PreferenceMemoryAgent(
            memory_manager=self.memory_manager,
            memory_gateway=self.memory_gateway,
            paper_search_service=self.paper_search_service,
            storage_root=report_service.storage_root,
        )
        self.evaluation_tool = evaluation_tool or ResearchEvaluationTool()
        self.review_writing_tool = review_writing_tool or ReviewWritingTool()
        self.writing_polish_tool = writing_polish_tool or WritingPolishTool()
        self.import_concurrency = max(1, import_concurrency)
        self.import_index_timeout_seconds = import_index_timeout_seconds
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self.observability_service = ResearchObservabilityService(
            report_service.storage_root / "observability"
        )
        self.external_tool_gateway = getattr(self.paper_search_service, "external_tool_gateway", None)

        self.conversation_service = ConversationService(
            _build_status_metadata=self._build_status_metadata,
            _compact_text=self._compact_text,
            _format_evidence_citation=self._format_evidence_citation,
            _job_tasks=self._job_tasks,
            memory_gateway=self.memory_gateway,
            report_service=self.report_service,
            research_context_manager=self.research_context_manager,
        )
        self.paper_service = PaperOperationsService(
            _build_discovery_only_request=self._build_discovery_only_request,
            _build_status_metadata=self._build_status_metadata,
            _job_tasks=self._job_tasks,
            _persist_runtime_conversation_snapshot=self._persist_runtime_conversation_snapshot,
            _resolve_research_session_id=self._resolve_research_session_id,
            _set_conversation_active_job=self._set_conversation_active_job,
            _update_job=self._update_job,
            _update_research_memory=self._update_research_memory,
            append_runtime_event=lambda *args, **kwargs: self.append_runtime_event(*args, **kwargs),
            ask_task_collection=lambda *args, **kwargs: self.ask_task_collection(*args, **kwargs),
            chart_analysis_agent=self.chart_analysis_agent,
            get_task=lambda *args, **kwargs: self.get_task(*args, **kwargs),
            import_concurrency=self.import_concurrency,
            import_index_timeout_seconds=self.import_index_timeout_seconds,
            observability_service=self.observability_service,
            paper_import_service=self.paper_import_service,
            paper_search_service=self.paper_search_service,
            record_import_turn=lambda *args, **kwargs: self.record_import_turn(*args, **kwargs),
            record_notice=lambda *args, **kwargs: self.record_notice(*args, **kwargs),
            record_qa_turn=lambda *args, **kwargs: self.record_qa_turn(*args, **kwargs),
            report_service=self.report_service,
            run_agent=lambda *args, **kwargs: self.run_agent(*args, **kwargs),
            save_job_state=lambda *args, **kwargs: self.save_job_state(*args, **kwargs),
            save_task_state=lambda *args, **kwargs: self.save_task_state(*args, **kwargs),
        )
        self.qa_service = QAApplicationService(
            _load_cached_figure_payload=self._load_cached_figure_payload,
            build_execution_context=lambda *args, **kwargs: self.build_execution_context(*args, **kwargs),
            chart_analysis_agent=self.chart_analysis_agent,
            get_collection_qa_capability=lambda: self.research_collection_qa_capability,
            list_paper_figures=lambda *args, **kwargs: self.list_paper_figures(*args, **kwargs),
            llm_adapter=self.llm_adapter,
            memory_gateway=self.memory_gateway,
            paper_search_service=self.paper_search_service,
            paper_selector_service=self.paper_selector_service,
            qa_routing_tool=self.qa_routing_tool,
            report_service=self.report_service,
            save_task_state=lambda *args, **kwargs: self.save_task_state(*args, **kwargs),
            user_intent_resolver=self.user_intent_resolver,
            visual_intent_router=self.visual_intent_router,
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
        user_profile = self.memory_gateway.load_user_profile()
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
            paper_reading_tool=self.paper_reading_tool,
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
            research_context = self.memory_gateway.hydrate_context(
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
        resolved_document_ids = list(document_ids or (task.imported_document_ids if task else []))
        resolved_selected_paper_ids = list(dict.fromkeys(selected_paper_ids or []))
        resolved_papers = list(papers or [])
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
        self.memory_gateway.persist_research_update(
            session_id=session_id,
            conversation_id=resolved_conversation_id,
            graph_runtime=graph_runtime,
            task=task,
            report=report,
            papers=resolved_papers,
            document_ids=resolved_document_ids,
            selected_paper_ids=resolved_selected_paper_ids,
            task_intent=task_intent,
            question=question,
            answer=answer,
            retrieval_summary=retrieval_summary,
            metadata_update=cleaned_metadata,
        )

    def _build_discovery_only_request(
        self,
        *,
        message: str,
        days_back: int,
        max_papers: int,
        sources: list[str],
        conversation_id: str | None,
        trigger: str,
        task_id: str | None = None,
        selected_paper_ids: list[str] | None = None,
        selected_document_ids: list[str] | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> ResearchAgentRunRequest:
        return ResearchAgentRunRequest(
            message=message,
            mode="research",
            task_id=task_id,
            conversation_id=conversation_id,
            days_back=days_back,
            max_papers=max_papers,
            sources=list(sources),
            selected_paper_ids=list(selected_paper_ids or []),
            selected_document_ids=list(selected_document_ids or []),
            advanced_action="discover",
            auto_import=False,
            import_top_k=0,
            include_graph=True,
            include_embeddings=True,
            skill_name="research_report",
            reasoning_style="cot",
            metadata={
                "workflow_constraint": "discovery_only",
                "trigger": trigger,
                "routing_authority": "supervisor_llm",
                "context": {
                    "route_mode": "research_discovery",
                },
                **(metadata_update or {}),
            },
        )

    def _recover_search_plan(
        self,
        *,
        report: ResearchReport,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[str],
    ) -> ResearchTopicPlan:
        raw_plan = report.metadata.get("search_plan")
        if isinstance(raw_plan, dict):
            try:
                return ResearchTopicPlan.model_validate(raw_plan)
            except Exception:
                logger.warning("Failed to validate persisted search_plan metadata; rebuilding topic plan.")
        return self.paper_search_service.topic_planner.plan(
            topic=topic,
            days_back=days_back,
            max_papers=max_papers,
            sources=sources,
        )

    async def search_papers(
        self,
        request: SearchPapersRequest,
        *,
        graph_runtime: Any | None = None,
    ) -> SearchPapersResponse:
        if graph_runtime is None:
            raise RuntimeError("search_papers requires an initialized graph runtime")
        agent_request = self._build_discovery_only_request(
            message=request.topic,
            days_back=request.days_back,
            max_papers=request.max_papers,
            sources=list(request.sources),
            conversation_id=request.conversation_id,
            trigger="search_papers",
            metadata_update={
                "search_topic": request.topic,
                "response_contract": "search_papers",
            },
        )
        agent_response = await self.run_agent(agent_request, graph_runtime=graph_runtime)
        report = agent_response.report
        if report is None:
            raise RuntimeError("search_papers did not produce a research report")
        plan = self._recover_search_plan(
            report=report,
            topic=request.topic,
            days_back=request.days_back,
            max_papers=request.max_papers,
            sources=list(request.sources),
        )
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            report=report,
            papers=agent_response.papers,
            task_intent="research_search",
            metadata_update={
                "days_back": request.days_back,
                "max_papers": request.max_papers,
                "sources": list(request.sources),
                "report_id": report.report_id,
            },
        )
        return SearchPapersResponse(
            plan=plan,
            papers=agent_response.papers,
            report=report,
            warnings=list(agent_response.warnings),
        )

    async def create_task(
        self,
        request: CreateResearchTaskRequest,
        *,
        graph_runtime: Any | None = None,
    ) -> ResearchTaskResponse:
        """Facade: create a research task and optionally trigger autonomous execution.

        This method owns initial task construction and persistence. If
        ``run_immediately`` is set, it delegates to :meth:`run_task` which in
        turn delegates to the Runtime via :meth:`run_agent`.
        """
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
        self.memory_gateway.update_user_profile(
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
        on_progress: Any | None = None,
    ) -> ResearchAgentRunResponse:
        """Facade: delegate an agent run request to the Supervisor Graph Runtime.

        The Service only performs pre-flight user preference observation here;
        all execution, state mutation, and persistence happen inside the Runtime.
        """
        from runtime.research.supervisor_graph_runtime import ResearchSupervisorGraphRuntime

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
            on_progress=on_progress,
        )

    async def run_task(
        self,
        task_id: str,
        *,
        graph_runtime: Any | None = None,
        conversation_id: str | None = None,
    ) -> ResearchTaskResponse:
        """Facade: execute a previously created research task end-to-end.

        Handles lifecycle bookkeeping (status transitions, events, metrics)
        around a call to :meth:`run_agent`.  The actual research logic lives
        entirely inside the Runtime.
        """
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if graph_runtime is None:
            raise RuntimeError("run_task requires an initialized graph runtime")
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
            payload={"tool_name": "research_supervisor_graph"},
        )
        try:
            agent_request = ResearchAgentRunRequest(
                message=(
                    f"请继续完成研究任务“{running_task.topic}”，"
                    "统一通过 Supervisor 自主完成检索、导入、问答和报告更新。"
                ),
                mode="research",
                task_id=running_task.task_id,
                conversation_id=conversation_id,
                days_back=running_task.days_back,
                max_papers=running_task.max_papers,
                sources=list(running_task.sources),
                selected_paper_ids=list(running_task.workspace.must_read_paper_ids),
                selected_document_ids=list(running_task.imported_document_ids),
                auto_import=True,
                include_graph=True,
                include_embeddings=True,
                skill_name="research_report",
                reasoning_style="cot",
                metadata={
                    "task_id": running_task.task_id,
                    "task_topic": running_task.topic,
                    "route_mode": "research_discovery",
                    "routing_authority": "supervisor_llm",
                    "trigger": "run_task",
                },
            )
            response = await self.run_agent(
                agent_request,
                graph_runtime=graph_runtime,
            )
            completed_response = self.get_task(running_task.task_id)
            runtime_report = completed_response.report
            completed_task = completed_response.task
            self.observability_service.record_metric(
                metric_type="task_completed",
                payload={
                    "task_id": completed_task.task_id,
                    "paper_count": len(completed_response.papers),
                    "warning_count": len(response.warnings),
                },
            )
            self.append_runtime_event(
                conversation_id=conversation_id,
                event_type="task_completed",
                task_id=completed_task.task_id,
                correlation_id=completed_task.status_metadata.correlation_id,
                payload={
                    "paper_count": len(completed_response.papers),
                    "report_id": runtime_report.report_id if runtime_report is not None else None,
                    "warning_count": len(response.warnings),
                    "trigger": "run_task",
                },
            )
            return ResearchTaskResponse(
                task=completed_task,
                papers=completed_response.papers,
                report=runtime_report,
                warnings=list(response.warnings),
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

    # Thin compatibility facade; collaborators own application workflows.
    def list_conversations(self) -> list[ResearchConversation]:
        return self.conversation_service.list_conversations()

    def _topic_continuity_score(self, *args, **kwargs):
        return self.conversation_service._topic_continuity_score(*args, **kwargs)

    def _looks_like_general_chat(self, *args, **kwargs):
        return self.conversation_service._looks_like_general_chat(*args, **kwargs)

    def _infer_route_mode(self, *args, **kwargs):
        return self.conversation_service._infer_route_mode(*args, **kwargs)

    def _thread_metadata_update(self, *args, **kwargs):
        return self.conversation_service._thread_metadata_update(*args, **kwargs)

    def create_conversation(self, request: CreateResearchConversationRequest) -> ResearchConversationResponse:
        return self.conversation_service.create_conversation(request)

    def get_conversation(self, conversation_id: str) -> ResearchConversationResponse:
        return self.conversation_service.get_conversation(conversation_id)

    def rename_conversation(self, conversation_id: str, title: str) -> ResearchConversationResponse:
        return self.conversation_service.rename_conversation(conversation_id, title)

    def _conversation_task_id(self, *args, **kwargs):
        return self.conversation_service._conversation_task_id(*args, **kwargs)

    def delete_conversation(self, conversation_id: str) -> None:
        return self.conversation_service.delete_conversation(conversation_id)

    async def reset_state(self) -> None:
        return await self.conversation_service.reset_state()

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
        return self.conversation_service.record_search_turn(conversation_id, topic=topic, response=response, days_back=days_back, max_papers=max_papers, sources=sources)

    def record_task_turn(
        self,
        conversation_id: str,
        *,
        response: ResearchTaskResponse,
    ) -> ResearchConversationResponse:
        return self.conversation_service.record_task_turn(conversation_id, response=response)

    def record_import_turn(
        self,
        conversation_id: str,
        *,
        task_response: ResearchTaskResponse | None,
        import_response: ImportPapersResponse,
        selected_paper_ids: list[str] | None = None,
        notice: str | None = None,
    ) -> ResearchConversationResponse:
        return self.conversation_service.record_import_turn(conversation_id, task_response=task_response, import_response=import_response, selected_paper_ids=selected_paper_ids, notice=notice)

    def record_qa_turn(
        self,
        conversation_id: str,
        *,
        task_response: ResearchTaskResponse,
        ask_response: ResearchTaskAskResponse,
    ) -> ResearchConversationResponse:
        return self.conversation_service.record_qa_turn(conversation_id, task_response=task_response, ask_response=ask_response)

    def record_agent_turn(
        self,
        conversation_id: str,
        *,
        request: ResearchAgentRunRequest,
        response: ResearchAgentRunResponse,
    ) -> ResearchConversationResponse:
        return self.conversation_service.record_agent_turn(conversation_id, request=request, response=response)

    def _active_paper_ids_from_agent_turn(self, *args, **kwargs):
        return self.conversation_service._active_paper_ids_from_agent_turn(*args, **kwargs)

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
        return self.conversation_service.record_notice(conversation_id, task_response=task_response, notice=notice, kind=kind, active_job_id=active_job_id, last_error=last_error)

    def _task_response_from_agent_response(self, *args, **kwargs):
        return self.conversation_service._task_response_from_agent_response(*args, **kwargs)

    def _ask_response_from_agent_response(self, *args, **kwargs):
        return self.conversation_service._ask_response_from_agent_response(*args, **kwargs)

    def _build_message(self, *args, **kwargs):
        return self.conversation_service._build_message(*args, **kwargs)

    def _build_runtime_event(self, *args, **kwargs):
        return self.conversation_service._build_runtime_event(*args, **kwargs)

    def _build_context_summary(self, *args, **kwargs):
        return self.conversation_service._build_context_summary(*args, **kwargs)

    def _persist_runtime_conversation_snapshot(self, *args, **kwargs):
        return self.conversation_service._persist_runtime_conversation_snapshot(*args, **kwargs)

    def append_runtime_event(
        self,
        *,
        conversation_id: str | None,
        event_type: str,
        task_id: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        return self.conversation_service.append_runtime_event(conversation_id=conversation_id, event_type=event_type, task_id=task_id, correlation_id=correlation_id, payload=payload)

    def _record_conversation_messages(self, *args, **kwargs):
        return self.conversation_service._record_conversation_messages(*args, **kwargs)

    def _filter_persisted_conversation_messages(self, *args, **kwargs):
        return self.conversation_service._filter_persisted_conversation_messages(*args, **kwargs)

    def _set_conversation_active_job(self, *args, **kwargs):
        return self.conversation_service._set_conversation_active_job(*args, **kwargs)

    async def list_paper_figures(
        self,
        task_id: str,
        paper_id: str,
        *,
        graph_runtime: Any,
    ) -> ResearchPaperFigureListResponse:
        return await self.paper_service.list_paper_figures(task_id, paper_id, graph_runtime=graph_runtime)

    async def analyze_paper_figure(
        self,
        task_id: str,
        paper_id: str,
        request: AnalyzeResearchPaperFigureRequest,
        *,
        graph_runtime: Any,
        supervisor_instruction: str | None = None,
        skill_context: str | None = None,
        expert_context: dict[str, Any] | None = None,
    ) -> AnalyzeResearchPaperFigureResponse:
        return await self.paper_service.analyze_paper_figure(task_id, paper_id, request, graph_runtime=graph_runtime, supervisor_instruction=supervisor_instruction, skill_context=skill_context, expert_context=expert_context)

    def _load_cached_figure_payload(self, *args, **kwargs):
        return self.paper_service._load_cached_figure_payload(*args, **kwargs)

    def _load_cached_figure_target(self, *args, **kwargs):
        return self.paper_service._load_cached_figure_target(*args, **kwargs)

    def _persist_paper_figure_cache(self, *args, **kwargs):
        return self.paper_service._persist_paper_figure_cache(*args, **kwargs)

    def _resolve_imported_paper(self, *args, **kwargs):
        return self.paper_service._resolve_imported_paper(*args, **kwargs)

    async def _parse_imported_paper_document(self, *args, **kwargs):
        return await self.paper_service._parse_imported_paper_document(*args, **kwargs)

    def persist_runtime_state(
        self,
        *,
        task_response: ResearchTaskResponse | None,
        workspace: ResearchWorkspaceState,
        conversation_id: str | None = None,
        advanced_strategy: ResearchAdvancedStrategy | None = None,
    ) -> ResearchTaskResponse | None:
        return self.paper_service.persist_runtime_state(task_response=task_response, workspace=workspace, conversation_id=conversation_id, advanced_strategy=advanced_strategy)

    def get_job(self, job_id: str) -> ResearchJob:
        return self.paper_service.get_job(job_id)

    async def start_import_job(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        return await self.paper_service.start_import_job(request, graph_runtime=graph_runtime)

    async def start_todo_import_job(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchJob:
        return await self.paper_service.start_todo_import_job(task_id, todo_id, request, graph_runtime=graph_runtime)

    def update_todo_status(self, task_id: str, todo_id: str, status: str) -> ResearchTaskResponse:
        return self.paper_service.update_todo_status(task_id, todo_id, status)

    async def rerun_todo_search(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime: Any,
    ) -> ResearchTodoActionResponse:
        return await self.paper_service.rerun_todo_search(task_id, todo_id, request, graph_runtime=graph_runtime)

    async def import_from_todo(
        self,
        task_id: str,
        todo_id: str,
        request: ResearchTodoActionRequest,
        *,
        graph_runtime,
    ) -> ResearchTodoActionResponse:
        return await self.paper_service.import_from_todo(task_id, todo_id, request, graph_runtime=graph_runtime)

    async def import_papers(
        self,
        request: ImportPapersRequest,
        *,
        graph_runtime,
        progress_callback: ImportProgressCallback | None = None,
    ) -> ImportPapersResponse:
        return await self.paper_service.import_papers(request, graph_runtime=graph_runtime, progress_callback=progress_callback)

    def _emit_import_heartbeat(self, *args, **kwargs):
        return self.paper_service._emit_import_heartbeat(*args, **kwargs)

    async def _import_single_paper(self, *args, **kwargs):
        return await self.paper_service._import_single_paper(*args, **kwargs)

    async def _sync_imported_paper_to_zotero(self, *args, **kwargs):
        return await self.paper_service._sync_imported_paper_to_zotero(*args, **kwargs)

    def _resolve_import_candidates(self, *args, **kwargs):
        return self.paper_service._resolve_import_candidates(*args, **kwargs)

    async def _search_follow_up_papers(self, *args, **kwargs):
        return await self.paper_service._search_follow_up_papers(*args, **kwargs)

    async def _run_import_job(self, *args, **kwargs):
        return await self.paper_service._run_import_job(*args, **kwargs)

    async def _run_todo_import_job(self, *args, **kwargs):
        return await self.paper_service._run_todo_import_job(*args, **kwargs)

    async def _run_graph_backfill_for_import_results(self, *args, **kwargs):
        return await self.paper_service._run_graph_backfill_for_import_results(*args, **kwargs)

    def _format_evidence_citation(self, *args, **kwargs):
        return self.paper_service._format_evidence_citation(*args, **kwargs)

    def _load_task_and_todo(self, *args, **kwargs):
        return self.paper_service._load_task_and_todo(*args, **kwargs)

    def _find_todo(self, *args, **kwargs):
        return self.paper_service._find_todo(*args, **kwargs)

    def _replace_task_todo(self, *args, **kwargs):
        return self.paper_service._replace_task_todo(*args, **kwargs)

    def _resolve_todos_after_import(self, *args, **kwargs):
        return self.paper_service._resolve_todos_after_import(*args, **kwargs)

    def _build_todo_query(self, *args, **kwargs):
        return self.paper_service._build_todo_query(*args, **kwargs)

    def _select_todo_import_candidates(self, *args, **kwargs):
        return self.paper_service._select_todo_import_candidates(*args, **kwargs)

    def _merge_papers(self, *args, **kwargs):
        return self.paper_service._merge_papers(*args, **kwargs)

    def _paper_identity_keys(self, *args, **kwargs):
        return self.paper_service._paper_identity_keys(*args, **kwargs)

    def _rebuild_task_report(self, *args, **kwargs):
        return self.paper_service._rebuild_task_report(*args, **kwargs)

    def _extract_markdown_section(self, *args, **kwargs):
        return self.paper_service._extract_markdown_section(*args, **kwargs)

    def _append_todo_action_entry(self, *args, **kwargs):
        return self.paper_service._append_todo_action_entry(*args, **kwargs)

    def _merge_text_entries(self, *args, **kwargs):
        return self.paper_service._merge_text_entries(*args, **kwargs)

    async def ask_task_collection(
        self,
        task_id: str,
        request: ResearchTaskAskRequest,
        *,
        graph_runtime,
    ) -> ResearchTaskAskResponse:
        return await self.qa_service.ask_task_collection(task_id, request, graph_runtime=graph_runtime)

    async def _select_qa_route(self, *args, **kwargs):
        return await self.qa_service._select_qa_route(*args, **kwargs)

    def _restore_visual_anchor_from_workspace(self, *args, **kwargs):
        return self.qa_service._restore_visual_anchor_from_workspace(*args, **kwargs)

    def _extract_visual_anchor(self, *args, **kwargs):
        return self.qa_service._extract_visual_anchor(*args, **kwargs)

    async def _infer_cached_visual_anchor(self, *args, **kwargs):
        return await self.qa_service._infer_cached_visual_anchor(*args, **kwargs)

    async def _infer_or_discover_visual_anchor(self, *args, **kwargs):
        return await self.qa_service._infer_or_discover_visual_anchor(*args, **kwargs)

    async def _discover_figures_for_scope(self, *args, **kwargs):
        return await self.qa_service._discover_figures_for_scope(*args, **kwargs)

    def _refresh_existing_pool(self, *args, **kwargs):
        return self.qa_service._refresh_existing_pool(*args, **kwargs)

    def _apply_qa_follow_up(self, *args, **kwargs):
        return self.qa_service._apply_qa_follow_up(*args, **kwargs)

    def _upsert_follow_up_todo(self, *args, **kwargs):
        return self.qa_service._upsert_follow_up_todo(*args, **kwargs)

    def _append_qa_report_entry(self, *args, **kwargs):
        return self.qa_service._append_qa_report_entry(*args, **kwargs)

    def _prepend_unique_text(self, *args, **kwargs):
        return self.qa_service._prepend_unique_text(*args, **kwargs)

    def _compact_text(self, *args, **kwargs):
        return self.qa_service._compact_text(*args, **kwargs)
