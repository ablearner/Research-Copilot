"""Conversation management mixin for LiteratureResearchService.

Extracts conversation CRUD, turn recording, and thread management into
a cohesive mixin so that the main service file stays focused on task
lifecycle and paper operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from core.utils import now_iso as _now_iso, normalize_topic_text as _normalize_topic_text
from domain.schemas.research import (
    CreateResearchConversationRequest,
    ImportPapersResponse,
    ResearchAdvancedStrategy,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversation,
    ResearchConversationResponse,
    ResearchConversationSnapshot,
    ResearchMessage,
    ResearchRouteMode,
    ResearchStatusMetadata,
    ResearchTaskAskResponse,
    ResearchTaskResponse,
    ResearchThreadSnapshot,
    ResearchWorkspaceState,
    SearchPapersResponse,
)
from services.research.research_workspace import build_workspace_state

logger = logging.getLogger(__name__)


def _preferred_answer_language_from_text(text: str | None) -> str:
    value = str(text or "").strip()
    if not value:
        return "zh-CN"
    cjk_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in value if ("a" <= char.lower() <= "z"))
    if cjk_count > 0 and cjk_count >= max(1, latin_count // 2):
        return "zh-CN"
    return "en-US"


class ConversationMixin:
    """Mixin providing conversation management for research tasks.

    Assumes the host class exposes (from ``LiteratureResearchService``):

    * ``report_service``
    * ``memory_manager``
    * ``_job_tasks``
    * ``_build_message(...)``
    * ``_build_runtime_event(...)``
    * ``_build_context_summary(...)``
    * ``append_runtime_event(...)``
    * ``_record_conversation_messages(...)``
    * ``_filter_persisted_conversation_messages(...)``
    * ``_format_evidence_citation(...)``
    """

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
            "\u4f60\u597d",
            "\u60a8\u597d",
            "hello",
            "hi",
            "\u5929\u6c14",
            "\u7ffb\u8bd1",
            "\u8bb2\u4e2a\u7b11\u8bdd",
            "what is",
            "how are you",
        }
        research_markers = {
            "\u8bba\u6587",
            "paper",
            "\u6587\u732e",
            "\u8c03\u7814",
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
        title = (request.title or request.topic or "\u672a\u547d\u540d\u7814\u7a76\u4f1a\u8bdd").strip() or "\u672a\u547d\u540d\u7814\u7a76\u4f1a\u8bdd"
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
            content="\u8fd9\u91cc\u4f1a\u6301\u7eed\u8bb0\u5f55\u4f60\u7684\u7814\u7a76\u68c0\u7d22\u3001\u5bfc\u5165\u3001\u95ee\u7b54\u548c TODO \u6267\u884c\u5386\u53f2\u3002",
            meta="\u591a\u6e90\u68c0\u7d22\u3001\u5165\u5e93\u3001\u7814\u7a76\u96c6\u5408\u95ee\u7b54\u3001TODO \u95ed\u73af",
            created_at=now,
        )
        self.report_service.save_conversation(conversation)
        self.report_service.save_messages(conversation.conversation_id, [welcome])
        self.memory_gateway.update_user_profile(
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
        self.memory_gateway.clear_session(conversation_id)
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
                self._build_message(role="user", kind="topic", title="\u5f53\u524d\u7814\u7a76\u4e3b\u9898", content=topic),
                self._build_message(
                    role="assistant",
                    kind="report",
                    title="\u6587\u732e\u7efc\u8ff0\u7ed3\u679c",
                    meta=f"\u5019\u9009\u8bba\u6587 {response.report.paper_count} \u7bc7",
                    content=response.report.markdown,
                    payload={"report": response.report.model_dump(mode="json")},
                ),
                self._build_message(
                    role="assistant",
                    kind="candidates",
                    title="\u5019\u9009\u8bba\u6587\u6c60",
                    meta=f"\u5f53\u524d\u5171 {len(response.papers)} \u7bc7\uff0c\u53ef\u52fe\u9009\u540e\u5bfc\u5165",
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
                title="\u5f53\u524d\u7814\u7a76\u4e3b\u9898",
                meta="\u7814\u7a76\u4efb\u52a1\u5df2\u521b\u5efa",
                content=response.task.topic,
            )
        ]
        if report is not None:
            messages.append(
                self._build_message(
                    role="assistant",
                    kind="report",
                    title="\u6587\u732e\u7efc\u8ff0\u7ed3\u679c",
                    meta=f"\u5019\u9009\u8bba\u6587 {report.paper_count} \u7bc7",
                    content=report.markdown,
                    payload={"report": report.model_dump(mode="json")},
                )
            )
        if response.papers:
            messages.append(
                self._build_message(
                    role="assistant",
                    kind="candidates",
                    title="\u5019\u9009\u8bba\u6587\u6c60",
                    meta=f"\u5f53\u524d\u5171 {len(response.papers)} \u7bc7\uff0c\u53ef\u52fe\u9009\u540e\u5bfc\u5165",
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
            f"imported={import_response.imported_count} \u00b7 skipped={import_response.skipped_count} \u00b7 failed={import_response.failed_count}"
        ]
        for result in import_response.results[:5]:
            suffix = f" \u00b7 doc={result.document_id}" if result.document_id else ""
            error = f" \u00b7 {result.error_message}" if result.error_message else ""
            preview_lines.append(f"- {result.title} \u00b7 {result.status}{suffix}{error}")
        messages = [
            self._build_message(
                role="assistant",
                kind="import_result",
                title="\u5bfc\u5165\u7ed3\u679c",
                meta="\u5019\u9009\u8bba\u6587\u5df2\u8fdb\u5165\u6587\u6863\u94fe\u8def",
                content="\n".join(preview_lines),
                payload={"import_result": import_response.model_dump(mode="json")},
            )
        ]
        if notice:
            messages.append(self._build_message(role="assistant", kind="notice", title="\u7cfb\u7edf\u901a\u77e5", content=notice))
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
                    title="\u7814\u7a76\u96c6\u5408\u63d0\u95ee",
                    content=ask_response.qa.question,
                ),
                self._build_message(
                    role="assistant",
                    kind="answer",
                    title="\u7814\u7a76\u96c6\u5408\u56de\u7b54",
                    meta=(
                        f"route={qa_route} \u00b7 runtime={drilldown_runtime} \u00b7 "
                        f"evidence={len(ask_response.qa.evidence_bundle.evidences)} \u00b7 "
                        f"confidence={ask_response.qa.confidence if ask_response.qa.confidence is not None else 'empty'} \u00b7 "
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
                    title="\u7528\u6237\u7814\u7a76\u76ee\u6807" if request.mode != "qa" else "\u7814\u7a76\u96c6\u5408\u63d0\u95ee",
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
            messages=[self._build_message(role="assistant", kind=kind, title="\u7cfb\u7edf\u901a\u77e5", content=notice)],
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
