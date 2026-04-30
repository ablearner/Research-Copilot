from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from core.utils import now_iso as _now_iso
from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.literature_scout_agent import LiteratureScoutAgent
from agents.preference_memory_agent import PreferenceMemoryAgent
from agents.research_knowledge_agent import ResearchKnowledgeAgent
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.research import (
    ResearchAdvancedStrategy,
    CreateResearchTaskRequest,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversationResponse,
    ResearchContextSummary,
    ResearchJob,
    ResearchReport,
    ResearchMessage,
    ResearchRuntimeEvent,
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
from services.research.capabilities import (
    PaperReader,
    ResearchQARouter,
    ResearchEvaluator,
    ResearchIntentResolver,
    ReviewWriter,
    WritingPolisher,
    PaperCurator,
)
from services.research.research_context import ResearchExecutionContext
from services.research.research_context_manager import ResearchContextManager
from services.research.research_discovery_capability import ResearchDiscoveryCapability
from services.research.research_memory_gateway import ResearchMemoryGateway
from services.research.paper_selector_service import PaperSelectorService
from services.research.paper_import_service import PaperImportService
from services.research.observability_service import ResearchObservabilityService
from services.research.paper_search_service import PaperSearchService
from services.research.research_report_service import ResearchReportService
from services.research.research_workspace import build_workspace_state
from services.research.conversation_manager import ConversationMixin, _preferred_answer_language_from_text
from services.research.paper_operations import PaperOperationsMixin
from services.research.qa_router import QARoutingMixin

logger = logging.getLogger(__name__)

INTERNAL_CONVERSATION_MESSAGE_TITLES = {
    "Manager 任务分解",
    "Research Workspace",
    "Agent 决策轨迹",
    "TODO 动作回执",
    "Research-Copilot",
}


class LiteratureResearchService(QARoutingMixin, ConversationMixin, PaperOperationsMixin):
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
        paper_reading_skill: PaperReader | None = None,
        evaluation_skill: ResearchEvaluator | None = None,
        review_writing_skill: ReviewWriter | None = None,
        writing_polish_skill: WritingPolisher | None = None,
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
        self.paper_reading_skill = paper_reading_skill or PaperReader()
        llm_adapter = getattr(paper_search_service, "llm_adapter", None)
        self.llm_adapter = llm_adapter
        self.literature_scout_agent = LiteratureScoutAgent(
            paper_search_service,
        )
        self.paper_curation_skill = PaperCurator(paper_search_service)
        self.research_knowledge_agent = ResearchKnowledgeAgent()
        self.research_writer_agent = ResearchWriterAgent(
            paper_search_service,
            llm_adapter=llm_adapter,
        )
        self.research_discovery_capability = ResearchDiscoveryCapability(
            literature_scout_agent=self.literature_scout_agent,
            research_writer_agent=self.research_writer_agent,
            curation_skill=self.paper_curation_skill,
        )
        self.qa_routing_skill = ResearchQARouter(llm_adapter=llm_adapter)
        self.user_intent_resolver = ResearchIntentResolver(llm_adapter=llm_adapter)
        self.chart_analysis_agent = ChartAnalysisAgent(
            llm_adapter=llm_adapter,
            storage_root=report_service.storage_root / "assets",
        )
        self.preference_memory_agent = PreferenceMemoryAgent(
            memory_manager=self.memory_manager,
            paper_search_service=self.paper_search_service,
            storage_root=report_service.storage_root,
        )
        self.memory_gateway = ResearchMemoryGateway(
            memory_manager=self.memory_manager,
            research_context_manager=self.research_context_manager,
            paper_reading_skill=self.paper_reading_skill,
            compact_text=lambda value: self._compact_text(value, limit=280),
        )
        self.evaluation_skill = evaluation_skill or ResearchEvaluator()
        self.review_writing_skill = review_writing_skill or ReviewWriter()
        self.writing_polish_skill = writing_polish_skill or WritingPolisher()
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
            on_progress=on_progress,
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
