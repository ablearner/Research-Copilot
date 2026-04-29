from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.general_answer_agent import GeneralAnswerAgent
from agents.literature_scout_agent import LiteratureScoutAgent
from agents.paper_analysis_agent import PaperAnalysisAgent
from agents.preference_memory_agent import PreferenceMemoryAgent
from agents.research_knowledge_agent import ResearchKnowledgeAgent, merge_retrieval_hits
from agents.research_supervisor_agent import (
    ResearchSupervisorAgent,
    ResearchSupervisorDecision,
    ResearchSupervisorState,
)
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import (
    PaperCandidate,
    ResearchAdvancedStrategy,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchAgentTraceStep,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskResponse,
    ResearchTaskResponse,
    ResearchWorkspaceState,
)
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import AnalyzePapersFunctionOutput, RecommendPapersFunctionOutput
from domain.schemas.sub_manager import SubManagerState, TaskStep
from domain.schemas.unified_runtime import (
    UNIFIED_ACTION_OUTPUT_METADATA_KEY,
    UnifiedAgentDescriptor,
    UnifiedAgentResult,
    UnifiedAgentTask,
)
from reasoning import CoTReasoningAgent, PlanAndSolveReasoningAgent, ReasoningStrategySet
from services.research.research_context import ResearchExecutionContext
from services.research.unified_runtime import (
    build_phase1_unified_agent_registry,
    build_phase1_unified_blueprint,
    build_phase1_unified_runtime_context,
    serialize_unified_agent_messages,
    serialize_unified_agent_registry,
    serialize_unified_agent_results,
    serialize_unified_delegation_plan,
)
from typing import TYPE_CHECKING
from services.research.research_workspace import build_workspace_from_task, build_workspace_state
from services.research.capabilities import PaperAnalyzer, PaperReader, ResearchIntentResolver, PaperCurator
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.research_supervisor_tool_specs import (
    ResearchSupervisorActionRegistry,
    SupervisorActionToolOutput,
)
from context.compressor import ContextCompressor
from core.utils import normalize_topic_text as _normalize_topic_text_impl

# --- Supervisor tool classes (extracted to services.research.supervisor_tools) ---
from services.research.supervisor_tools import (  # noqa: E402
    AnalyzePaperFiguresTool,
    AnalyzePapersTool,
    AnswerQuestionTool,
    AnswerResearchQuestionTool,
    CompressContextTool,
    CreateResearchTaskTool,
    GeneralAnswerTool,
    ImportPapersTool,
    ImportRelevantPapersTool,
    RecommendFromPreferencesTool,
    ResearchAgentGraphState,
    ResearchAgentTool,
    ResearchAgentToolContext,
    ResearchToolResult,
    SearchLiteratureTool,
    SyncToZoteroTool,
    UnderstandChartTool,
    UnderstandDocumentTool,
    WriteReviewTool,
    _llm_stage_timeout_seconds,
    _message,
    _now_iso,
    _observation_envelope,
    _should_fallback_llm_stage,
    _update_runtime_progress,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from services.research.literature_research_service import LiteratureResearchService


class ResearchRuntimeBase:
    """Shared specialist runtime capabilities for high-level research orchestration."""

    def __init__(
        self,
        *,
        research_service: LiteratureResearchService,
        manager_agent: ResearchSupervisorAgent | None = None,
        max_steps: int = 8,
    ) -> None:
        self.research_service = research_service
        paper_reading_skill = getattr(research_service, "paper_reading_skill", None) or PaperReader()
        llm_adapter = getattr(getattr(research_service, "paper_search_service", None), "llm_adapter", None)
        if llm_adapter is None:
            llm_adapter = getattr(paper_reading_skill, "llm_adapter", None)
        self.manager_agent = manager_agent or ResearchSupervisorAgent(llm_adapter=llm_adapter)
        if getattr(self.manager_agent, "llm_adapter", None) is None:
            self.manager_agent.llm_adapter = llm_adapter
        self.user_intent_resolver = ResearchIntentResolver(llm_adapter=llm_adapter)
        self.max_steps = max_steps
        self.reasoning_strategies = ReasoningStrategySet(
            query_planning=PlanAndSolveReasoningAgent(llm_adapter=llm_adapter),
            answer_synthesis=CoTReasoningAgent(llm_adapter=llm_adapter),
        )
        self.literature_scout_agent = LiteratureScoutAgent(
            research_service.paper_search_service,
            reasoning_strategies=self.reasoning_strategies,
        )
        self.research_knowledge_agent = ResearchKnowledgeAgent(
            reasoning_strategies=self.reasoning_strategies,
        )
        self.research_writer_agent = ResearchWriterAgent(
            research_service.paper_search_service,
            llm_adapter=llm_adapter,
            reasoning_strategies=self.reasoning_strategies,
        )
        self.paper_analysis_agent = PaperAnalysisAgent(
            paper_analysis_skill=PaperAnalyzer(
                paper_reading_skill=paper_reading_skill,
                llm_adapter=llm_adapter,
            )
        )
        self.chart_analysis_agent = ChartAnalysisAgent(llm_adapter=llm_adapter)
        self.general_answer_agent = GeneralAnswerAgent(llm_adapter=llm_adapter)
        self.preference_memory_agent = getattr(research_service, "preference_memory_agent", None) or PreferenceMemoryAgent(
            memory_manager=research_service.memory_manager,
            paper_search_service=research_service.paper_search_service,
            storage_root=research_service.report_service.storage_root,
        )
        self.paper_curation_skill = PaperCurator(research_service.paper_search_service)
        self._context_compressor = ContextCompressor(
            llm_adapter=llm_adapter,
            target_budget_ratio=0.75,
        )
        self.tools: dict[str, ResearchAgentTool] = {
            SearchLiteratureTool.name: SearchLiteratureTool(
                literature_scout_agent=self.literature_scout_agent,
                research_writer_agent=self.research_writer_agent,
                curation_skill=self.paper_curation_skill,
            ),
            WriteReviewTool.name: WriteReviewTool(
                research_service=research_service,
                writer_agent=self.research_writer_agent,
            ),
            ImportPapersTool.name: ImportPapersTool(),
            SyncToZoteroTool.name: SyncToZoteroTool(),
            AnswerQuestionTool.name: AnswerQuestionTool(),
            GeneralAnswerTool.name: GeneralAnswerTool(general_answer_agent=self.general_answer_agent),
            RecommendFromPreferencesTool.name: RecommendFromPreferencesTool(
                preference_memory_agent=self.preference_memory_agent
            ),
            AnalyzePapersTool.name: AnalyzePapersTool(paper_analysis_agent=self.paper_analysis_agent),
            CompressContextTool.name: CompressContextTool(),
            UnderstandDocumentTool.name: UnderstandDocumentTool(),
            UnderstandChartTool.name: UnderstandChartTool(chart_analysis_agent=self.chart_analysis_agent),
            AnalyzePaperFiguresTool.name: AnalyzePaperFiguresTool(chart_analysis_agent=self.chart_analysis_agent),
        }
        self.action_tool_registry = ToolRegistry()
        self.action_tool_executor = ToolExecutor(self.action_tool_registry)
        self._action_invocations: dict[str, tuple[ResearchAgentToolContext, ResearchSupervisorDecision]] = {}
        ResearchSupervisorActionRegistry(self.action_tool_registry).register_many(
            self._build_action_tool_handlers(),
            replace=True,
        )

    def _manager_display_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _manager_trace_agent_name(self) -> str:
        return self._manager_display_name()

    def _hydrate_request_from_conversation(
        self,
        *,
        request: ResearchAgentRunRequest,
    ) -> tuple[ResearchAgentRunRequest, ResearchTaskResponse | None]:
        if not request.conversation_id:
            return request, None
        try:
            conversation_response = self.research_service.get_conversation(request.conversation_id)
        except KeyError:
            return request, None

        conversation = conversation_response.conversation
        snapshot = conversation.snapshot

        restored_task_response: ResearchTaskResponse | None = None
        if request.task_id:
            try:
                restored_task_response = self.research_service.get_task(request.task_id)
            except KeyError:
                restored_task_response = None
        elif conversation.task_id:
            try:
                restored_task_response = self.research_service.get_task(conversation.task_id)
            except KeyError:
                restored_task_response = snapshot.task_result
        elif snapshot.task_result is not None:
            restored_task_response = snapshot.task_result

        inherit_scope = self._should_inherit_snapshot_scope(request=request, snapshot=snapshot)
        selected_paper_ids = list(request.selected_paper_ids)
        if not selected_paper_ids and inherit_scope:
            selected_paper_ids = list(
                snapshot.selected_paper_ids
                or snapshot.active_paper_ids
                or []
            )
        selected_document_ids = list(request.selected_document_ids)
        if not selected_document_ids and selected_paper_ids and restored_task_response is not None:
            imported_document_ids = {
                str(item).strip()
                for item in restored_task_response.task.imported_document_ids
                if str(item).strip()
            }
            papers_by_id = {paper.paper_id: paper for paper in restored_task_response.papers}
            for paper_id in selected_paper_ids:
                paper = papers_by_id.get(paper_id)
                if paper is None:
                    continue
                document_id = str(paper.metadata.get("document_id") or "").strip()
                if not document_id:
                    continue
                if imported_document_ids and document_id not in imported_document_ids:
                    continue
                if document_id not in selected_document_ids:
                    selected_document_ids.append(document_id)
        if (
            not selected_document_ids
            and inherit_scope
            and snapshot.ask_result is not None
        ):
            selected_document_ids = [
                str(item).strip()
                for item in snapshot.ask_result.document_ids
                if str(item).strip()
            ]

        metadata = dict(request.metadata or {})
        metadata_context = (
            dict(metadata.get("context") or {})
            if isinstance(metadata.get("context"), dict)
            else {}
        )
        active_paper_ids = [
            str(item).strip()
            for item in metadata_context.get("active_paper_ids", [])
            if str(item).strip()
        ]
        if not active_paper_ids and inherit_scope:
            active_paper_ids = list(snapshot.active_paper_ids or snapshot.selected_paper_ids or [])
        if active_paper_ids:
            metadata_context["active_paper_ids"] = active_paper_ids
            metadata["context"] = metadata_context
        elif "context" in metadata and isinstance(metadata.get("context"), dict):
            metadata_context.pop("active_paper_ids", None)
            metadata["context"] = metadata_context

        heuristic_intent = self.user_intent_resolver.resolve(
            message=request.message,
            has_task=restored_task_response is not None,
            candidate_paper_count=len(restored_task_response.papers) if restored_task_response else 0,
            candidate_papers=None,
            active_paper_ids=active_paper_ids,
            selected_paper_ids=selected_paper_ids,
        )
        metadata_context["route_mode"] = self._route_mode_hint_for_request(
            request=request,
            snapshot=snapshot,
            inherit_scope=inherit_scope,
            intent_result=heuristic_intent,
        )
        metadata_context["active_thread_id"] = snapshot.active_thread_id
        metadata["context"] = metadata_context

        hydrated_request = request.model_copy(
            update={
                "task_id": request.task_id or (
                    restored_task_response.task.task_id if restored_task_response is not None else None
                ),
                "selected_paper_ids": selected_paper_ids,
                "selected_document_ids": selected_document_ids,
                "metadata": metadata,
            }
        )
        return hydrated_request, restored_task_response

    def _should_inherit_snapshot_scope(
        self,
        *,
        request: ResearchAgentRunRequest,
        snapshot,
    ) -> bool:
        if request.task_id or request.selected_paper_ids or request.selected_document_ids:
            return True
        if request.document_file_path or request.chart_image_path or request.document_id or request.chart_id:
            return False
        message = self._normalize_topic_text(request.message)
        if not message or self._looks_like_general_chat(message):
            return False
        if self._looks_like_new_discovery(message):
            return False
        if snapshot.active_route_mode == "general_chat":
            return False
        follow_up_markers = ("这篇", "该论文", "上一个", "这些论文", "this paper", "these papers", "p1", "p2")
        return any(marker in request.message.lower() for marker in follow_up_markers) or bool(snapshot.active_paper_ids)

    _INTENT_TO_ROUTE_MODE: dict[str, str] = {
        "literature_search": "research_discovery",
        "paper_import": "paper_follow_up",
        "sync_to_zotero": "paper_follow_up",
        "collection_qa": "research_follow_up",
        "single_paper_qa": "paper_follow_up",
        "paper_comparison": "paper_follow_up",
        "paper_recommendation": "paper_follow_up",
        "figure_qa": "chart_drilldown",
        "document_understanding": "document_drilldown",
        "general_answer": "general_chat",
        "general_follow_up": "research_follow_up",
    }

    def _route_mode_hint_for_request(
        self,
        *,
        request: ResearchAgentRunRequest,
        snapshot,
        inherit_scope: bool,
        intent_result: Any | None = None,
    ) -> str:
        # Primary: use LLM intent result if available and confident
        if intent_result is not None:
            confidence = getattr(intent_result, "confidence", 0.0)
            intent_name = getattr(intent_result, "intent", "")
            if confidence >= 0.7 and intent_name in self._INTENT_TO_ROUTE_MODE:
                return self._INTENT_TO_ROUTE_MODE[intent_name]

        # Fallback: keyword-based heuristics
        message = self._normalize_topic_text(request.message)
        if self._looks_like_general_chat(message):
            return "general_chat"
        if request.chart_image_path or request.chart_id:
            return "chart_drilldown"
        if request.document_file_path or request.document_id:
            return "document_drilldown"
        has_scoped_papers = bool(
            request.selected_paper_ids
            or request.selected_document_ids
            or snapshot.active_paper_ids
            or snapshot.selected_paper_ids
        )
        if inherit_scope and has_scoped_papers and self._looks_like_scoped_paper_follow_up(message):
            return "paper_follow_up"
        if self._looks_like_new_discovery(message):
            return "research_discovery"
        if inherit_scope and has_scoped_papers:
            return "paper_follow_up"
        return snapshot.active_route_mode or "research_follow_up"

    def _looks_like_general_chat(self, normalized_message: str) -> bool:
        return any(marker in normalized_message for marker in ("你好", "您好", "hello", "hi", "天气", "翻译"))

    def _looks_like_new_discovery(self, normalized_message: str) -> bool:
        return any(
            marker in normalized_message
            for marker in ("调研", "文献", "论文", "paper", "papers", "survey", "search", "find papers", "找相关文章")
        )

    def _looks_like_scoped_paper_follow_up(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        referential_markers = (
            "这篇",
            "该论文",
            "上一篇",
            "上一个",
            "这些论文",
            "这项工作",
            "本文",
            "this paper",
            "these papers",
            "current paper",
            "selected paper",
            "selected papers",
        )
        if any(marker in normalized_message for marker in referential_markers):
            return True
        paper_markers = ("论文", "paper", "papers", "文献", "work", "works")
        detail_markers = (
            "方法",
            "技术路线",
            "核心思路",
            "主要思路",
            "怎么做",
            "做法",
            "模型",
            "架构",
            "实验",
            "贡献",
            "创新点",
            "结果",
            "结论",
            "解释",
            "讲解",
            "method",
            "methods",
            "approach",
            "pipeline",
            "architecture",
            "experiment",
            "experiments",
            "contribution",
            "contributions",
            "results",
            "explain",
        )
        return any(marker in normalized_message for marker in paper_markers) and any(
            marker in normalized_message for marker in detail_markers
        )

    def _looks_like_scoped_recommendation_request(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        recommend_markers = ("推荐", "recommend", "suggest", "worth", "值得看", "值得读")
        paper_markers = ("论文", "paper", "papers", "work", "works", "文献")
        referential_markers = (
            "这篇",
            "这些",
            "当前",
            "候选",
            "已选",
            "勾选",
            "this paper",
            "these papers",
            "current papers",
            "selected papers",
            "candidate papers",
            "among",
        )
        curation_markers = (
            "精读",
            "先读",
            "先看",
            "优先读",
            "优先看",
            "必读",
            "代表性",
            "代表论文",
            "哪篇",
            "哪一个",
            "which one",
        )
        return (
            any(marker in normalized_message for marker in recommend_markers)
            and any(marker in normalized_message for marker in paper_markers)
            and (
                any(marker in normalized_message for marker in referential_markers)
                or any(marker in normalized_message for marker in curation_markers)
                or self._looks_like_scoped_paper_follow_up(normalized_message)
            )
        )

    def _looks_like_preference_recommendation_request(self, normalized_message: str) -> bool:
        recommend_markers = ("推荐", "recommend", "suggest", "worth", "值得看", "值得读")
        paper_markers = ("论文", "paper", "papers", "work", "works", "文献")
        return (
            any(marker in normalized_message for marker in recommend_markers)
            and any(marker in normalized_message for marker in paper_markers)
            and not self._looks_like_scoped_recommendation_request(normalized_message)
        )

    def _normalize_topic_text(self, text: str | None) -> str:
        return _normalize_topic_text_impl(text)

    def _build_tool_context(self, *, request: ResearchAgentRunRequest, graph_runtime: Any) -> ResearchAgentToolContext:
        hydrated_request, restored_task_response = self._hydrate_request_from_conversation(request=request)
        context = ResearchAgentToolContext(
            request=hydrated_request,
            research_service=self.research_service,
            graph_runtime=graph_runtime,
            warnings=[],
        )
        if restored_task_response is not None:
            context.task_response = restored_task_response
        elif hydrated_request.task_id:
            try:
                context.task_response = self.research_service.get_task(hydrated_request.task_id)
            except KeyError:
                context.warnings.append(f"research task not found: {hydrated_request.task_id}")
        context.execution_context = self.research_service.build_execution_context(
            graph_runtime=graph_runtime,
            conversation_id=hydrated_request.conversation_id,
            task=context.task_response.task if context.task_response else None,
            report=context.task_response.report if context.task_response else None,
            papers=context.task_response.papers if context.task_response else None,
            document_ids=context.task_response.task.imported_document_ids if context.task_response else [],
            selected_paper_ids=hydrated_request.selected_paper_ids,
            skill_name=hydrated_request.skill_name,
            reasoning_style=hydrated_request.reasoning_style,
            metadata=hydrated_request.metadata,
        )
        context.unified_runtime_context = build_phase1_unified_runtime_context(
            graph_runtime=graph_runtime,
            research_service=self.research_service,
        )
        context.unified_agent_registry = build_phase1_unified_agent_registry(
            graph_runtime=graph_runtime,
            research_service=self.research_service,
            legacy_delegates=self._legacy_unified_delegates(),
            execution_handlers=self._legacy_unified_execution_handlers(),
        )
        context.unified_blueprint = build_phase1_unified_blueprint(
            graph_runtime=graph_runtime,
            research_service=self.research_service,
        ).model_dump(mode="json")
        return context

    def _legacy_unified_delegates(self) -> dict[str, Any]:
        return {
            "ResearchSupervisorAgent": self.manager_agent,
            "LiteratureScoutAgent": self.literature_scout_agent,
            "ResearchKnowledgeAgent": self.research_knowledge_agent,
            "ResearchWriterAgent": self.research_writer_agent,
            "PaperAnalysisAgent": self.paper_analysis_agent,
            "PreferenceMemoryAgent": self.preference_memory_agent,
            "ChartAnalysisAgent": self.chart_analysis_agent,
            "GeneralAnswerAgent": self.general_answer_agent,
        }

    def _legacy_unified_execution_handlers(self) -> dict[str, Any]:
        return {
            "LiteratureScoutAgent": self._build_unified_execution_handler(
                agent_name="LiteratureScoutAgent",
                supported_task_types={"search_literature"},
            ),
            "ResearchKnowledgeAgent": self._build_unified_execution_handler(
                agent_name="ResearchKnowledgeAgent",
                supported_task_types={"import_papers", "sync_to_zotero", "answer_question", "compress_context"},
            ),
            "ResearchWriterAgent": self._build_unified_execution_handler(
                agent_name="ResearchWriterAgent",
                supported_task_types={"write_review"},
            ),
            "PaperAnalysisAgent": self._build_unified_execution_handler(
                agent_name="PaperAnalysisAgent",
                supported_task_types={"analyze_papers"},
            ),
            "ChartAnalysisAgent": self._build_unified_execution_handler(
                agent_name="ChartAnalysisAgent",
                supported_task_types={"understand_chart", "analyze_paper_figures"},
            ),
            "GeneralAnswerAgent": self._build_unified_execution_handler(
                agent_name="GeneralAnswerAgent",
                supported_task_types={"general_answer"},
            ),
            "PreferenceMemoryAgent": self._build_unified_execution_handler(
                agent_name="PreferenceMemoryAgent",
                supported_task_types={"recommend_from_preferences"},
            ),
        }

    def _build_unified_execution_handler(
        self,
        *,
        agent_name: str,
        supported_task_types: set[str],
    ):
        async def handler(task: UnifiedAgentTask, runtime_context, legacy_delegate):
            supervisor_context = runtime_context.metadata.get("supervisor_tool_context")
            decision = runtime_context.metadata.get("supervisor_decision")
            if supervisor_context is None or decision is None:
                return UnifiedAgentResult(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    task_type=task.task_type,
                    status="failed",
                    instruction=task.instruction,
                    payload={
                        "observation": "missing supervisor runtime context for unified execution",
                        "tool_metadata": {"reason": "missing_supervisor_runtime_context"},
                    },
                    context_slice=task.context_slice,
                    priority=task.priority,
                    expected_output_schema=task.expected_output_schema,
                    depends_on=task.depends_on,
                    retry_count=task.retry_count,
                    metadata={
                        "execution_engine": "unified_agent_registry",
                        "execution_adapter": "phase1_wrapped_action_tool",
                        "legacy_delegate_type": (
                            legacy_delegate.__class__.__name__ if legacy_delegate is not None else None
                        ),
                    },
                )
            if task.task_type not in supported_task_types:
                return UnifiedAgentResult(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    task_type=task.task_type,
                    status="skipped",
                    instruction=task.instruction,
                    payload={
                        "observation": f"{agent_name} does not support task_type={task.task_type}",
                        "tool_metadata": {"reason": "unsupported_task_type"},
                    },
                    context_slice=task.context_slice,
                    priority=task.priority,
                    expected_output_schema=task.expected_output_schema,
                    depends_on=task.depends_on,
                    retry_count=task.retry_count,
                    metadata={
                        "execution_engine": "unified_agent_registry",
                        "execution_adapter": "phase1_wrapped_action_tool",
                        "legacy_delegate_type": (
                            legacy_delegate.__class__.__name__ if legacy_delegate is not None else None
                        ),
                    },
                )
            execution_result = await self._execute_action_tool(
                action_name=task.task_type,
                context=supervisor_context,
                decision=decision,
            )
            result = self._action_tool_result_from_execution(execution_result)
            execution_metadata = {
                "execution_engine": "unified_agent_registry",
                "execution_adapter": "phase1_wrapped_action_tool",
                "legacy_delegate_type": (
                    legacy_delegate.__class__.__name__ if legacy_delegate is not None else None
                ),
                **self._action_tool_execution_metadata(execution_result),
            }
            if result is None:
                return UnifiedAgentResult(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    task_type=task.task_type,
                    status="failed",
                    instruction=task.instruction,
                    payload={
                        "observation": execution_result.error_message or f"tool execution failed: {task.task_type}",
                        "tool_metadata": {"reason": "tool_execution_failed"},
                    },
                    context_slice=task.context_slice,
                    priority=task.priority,
                    expected_output_schema=task.expected_output_schema,
                    depends_on=task.depends_on,
                    retry_count=task.retry_count,
                    metadata=execution_metadata,
                )
            return UnifiedAgentResult(
                task_id=task.task_id,
                agent_name=agent_name,
                task_type=task.task_type,
                status=result.status,  # type: ignore[arg-type]
                instruction=task.instruction,
                payload={
                    "observation": result.observation,
                    "tool_metadata": dict(result.metadata),
                },
                context_slice=task.context_slice,
                priority=task.priority,
                expected_output_schema=task.expected_output_schema,
                depends_on=task.depends_on,
                retry_count=task.retry_count,
                action_output=(
                    dict(result.metadata)
                    if UnifiedAgentResult.is_action_output_payload(result.metadata)
                    else None
                ),
                metadata=execution_metadata,
            )

        return handler

    def _build_action_tool_handlers(self) -> dict[str, Any]:
        return {
            name: self._build_action_tool_handler(name)
            for name in self.tools
        }

    def _build_action_tool_handler(self, action_name: str):
        async def handler(*, invocation_id: str):
            invocation = self._action_invocations.get(invocation_id)
            if invocation is None:
                raise RuntimeError(f"unknown supervisor action invocation: {invocation_id}")
            context, decision = invocation
            tool = self.tools.get(action_name)
            if tool is None:
                raise RuntimeError(f"supervisor action tool not registered: {action_name}")
            result = await tool.run(context, decision)
            return SupervisorActionToolOutput(
                status=result.status if result.status in {"succeeded", "failed", "skipped"} else "failed",
                observation=result.observation,
                metadata=dict(result.metadata),
            )

        return handler

    async def _execute_action_tool(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        decision: ResearchSupervisorDecision,
    ):
        invocation_id = f"supervisor_action_{uuid4().hex}"
        self._action_invocations[invocation_id] = (context, decision)
        try:
            execution_result = await self.action_tool_executor.execute_tool_call(
                action_name,
                {"invocation_id": invocation_id},
            )
            normalized = self._normalize_execution_result_metadata(
                action_name=action_name,
                context=context,
                execution_result=execution_result,
            )
            if normalized is not None:
                execution_result.output = normalized
            return execution_result
        finally:
            self._action_invocations.pop(invocation_id, None)

    def _normalize_execution_result_metadata(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        execution_result,
    ) -> SupervisorActionToolOutput | None:
        output = execution_result.output
        if output is None:
            return None
        if isinstance(output, SupervisorActionToolOutput):
            metadata = dict(output.metadata)
            metadata = self._with_standardized_observation(
                action_name=action_name,
                context=context,
                status=output.status,
                metadata=metadata,
            )
            return SupervisorActionToolOutput(
                status=output.status,
                observation=output.observation,
                metadata=metadata,
            )
        if isinstance(output, dict):
            metadata = self._with_standardized_observation(
                action_name=action_name,
                context=context,
                status=str(output.get("status") or "failed"),
                metadata=dict(output.get("metadata") or {}),
            )
            return SupervisorActionToolOutput(
                status=str(output.get("status") or "failed"),
                observation=str(output.get("observation") or execution_result.error_message or ""),
                metadata=metadata,
            )
        return None

    def _with_standardized_observation(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        status: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        next_actions = list((context.workspace.next_actions if context.workspace is not None else [])[:3])
        observation = dict(metadata.get("observation_envelope") or {})
        progress_made = status == "succeeded" and metadata.get("reason") not in {"route_mismatch"}
        confidence = metadata.get("confidence")
        if action_name == "answer_question" and context.qa_result is not None:
            confidence = context.qa_result.qa.confidence
            next_actions = list((context.qa_result.report.workspace.next_actions if context.qa_result.report is not None else next_actions)[:3])
        elif action_name == "general_answer":
            confidence = metadata.get("confidence")
        elif action_name == "search_literature":
            confidence = 0.82 if status == "succeeded" else 0.35
        elif action_name == "write_review":
            confidence = 0.78 if status == "succeeded" else 0.4
        elif action_name == "analyze_papers":
            confidence = 0.8 if status == "succeeded" else 0.4
        elif action_name == "recommend_from_preferences":
            confidence = 0.83 if status == "succeeded" else 0.35
        elif action_name == "compress_context":
            confidence = 0.74 if status == "succeeded" else 0.3
        missing_inputs = list(observation.get("missing_inputs") or [])
        reason = str(metadata.get("reason") or "").strip()
        if reason == "missing_task":
            missing_inputs.append("task")
        elif reason == "missing_document_file_path":
            missing_inputs.append("document_file_path")
        elif reason == "missing_chart_image_path":
            missing_inputs.append("chart_image_path")
        elif reason == "no_target_papers":
            missing_inputs.append("paper_scope")
        elif reason == "no_papers":
            missing_inputs.append("papers")
        elif reason == "missing_execution_context":
            missing_inputs.append("execution_context")
        suggested = list(observation.get("suggested_next_actions") or [])
        if not suggested:
            if status == "skipped" and reason in {"missing_task", "no_target_papers", "no_papers"}:
                suggested = ["clarify_request", "search_literature"]
            elif action_name == "answer_question":
                suggested = ["analyze_papers", "compress_context"]
            elif action_name == "search_literature":
                suggested = ["write_review", "import_papers", "answer_question"]
            elif action_name == "recommend_from_preferences":
                suggested = ["search_literature", "answer_question"]
            elif action_name == "general_answer" and metadata.get("reason") == "route_mismatch":
                suggested = [str(metadata.get("suggested_action") or "answer_question")]
            elif next_actions:
                suggested = next_actions
        state_delta = dict(observation.get("state_delta") or {})
        if action_name == "search_literature" and context.task is not None:
            state_delta.setdefault("task_id", context.task.task_id)
            state_delta.setdefault("paper_count", len(context.papers))
        if action_name == "answer_question" and context.qa_result is not None:
            state_delta.setdefault("qa_task_id", context.qa_result.task_id)
            state_delta.setdefault("paper_ids", list(context.qa_result.paper_ids))
        if action_name == "recommend_from_preferences" and context.preference_recommendation_result is not None:
            state_delta.setdefault(
                "recommended_paper_ids",
                [item.paper_id for item in context.preference_recommendation_result.recommendations],
            )
        if action_name == "general_answer":
            state_delta.setdefault("ignore_research_context", bool(metadata.get("ignore_research_context")))
        metadata["observation_envelope"] = _observation_envelope(
            progress_made=progress_made,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            missing_inputs=list(dict.fromkeys(item for item in missing_inputs if item)),
            suggested_next_actions=list(dict.fromkeys(str(item).strip() for item in suggested if str(item).strip()))[:4],
            state_delta=state_delta,
        )
        return metadata

    def _action_tool_result_from_execution(self, execution_result) -> ResearchToolResult | None:
        output = execution_result.output
        if output is None:
            return None
        if isinstance(output, SupervisorActionToolOutput):
            return ResearchToolResult(
                status=output.status,
                observation=output.observation,
                metadata=dict(output.metadata),
            )
        if isinstance(output, dict):
            return ResearchToolResult(
                status=str(output.get("status") or "failed"),
                observation=str(output.get("observation") or execution_result.error_message or ""),
                metadata=dict(output.get("metadata") or {}),
            )
        return None

    def _action_tool_execution_metadata(self, execution_result) -> dict[str, Any]:
        metadata = {
            "supervisor_action_call_id": execution_result.call_id,
            "supervisor_action_tool_status": execution_result.status,
            "supervisor_action_attempt_count": execution_result.attempt_count,
        }
        if execution_result.error_message:
            metadata["supervisor_action_error"] = execution_result.error_message
        return metadata

    async def _execute_unified_worker(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: ResearchSupervisorDecision,
        active_message: AgentMessage | None,
        worker_agent: str,
    ) -> UnifiedAgentResult | None:
        if active_message is None:
            return None
        registry = context.unified_agent_registry
        runtime_context = context.unified_runtime_context
        if registry is None or runtime_context is None:
            return None
        executor = registry.get(worker_agent)
        if executor is None:
            return None
        runtime_context.metadata.update(
            {
                "supervisor_tool_context": context,
                "supervisor_decision": decision,
                "supervisor_worker_agent": worker_agent,
                "supervisor_action_name": decision.action_name,
            }
        )
        task = UnifiedAgentTask.from_agent_message(
            active_message,
            preferred_skill_name=self._preferred_skill_name_for_message(
                context,
                active_message=active_message,
                worker_agent=worker_agent,
            ),
            available_tool_names=self._available_tool_names_for_agent(
                context,
                agent_name=worker_agent,
            ),
        )
        return await executor.execute(task, runtime_context)

    def _research_tool_result_from_unified_result(
        self,
        unified_result: UnifiedAgentResult,
    ) -> ResearchToolResult:
        payload = dict(unified_result.payload)
        tool_metadata = payload.get("tool_metadata")
        metadata = dict(tool_metadata) if isinstance(tool_metadata, dict) else {}
        metadata.update(dict(unified_result.metadata))
        action_output = unified_result.action_output or UnifiedAgentResult.extract_action_output(
            payload=payload,
            metadata=metadata,
        )
        if action_output is not None:
            metadata.setdefault(
                UNIFIED_ACTION_OUTPUT_METADATA_KEY,
                dict(action_output),
            )
            for key, value in action_output.items():
                metadata.setdefault(key, value)
        return ResearchToolResult(
            status=unified_result.status,
            observation=str(payload.get("observation") or ""),
            metadata=metadata,
        )

    async def _decide_next_action(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        return await self.manager_agent.decide_next_action_async(
            await self._state_from_context(state["context"], trace=state.get("trace", [])),
            pending_messages=state.get("pending_agent_messages", []),
            agent_messages=state.get("agent_messages", []),
            agent_results=state.get("agent_results", []),
            completed_task_ids=state.get("completed_agent_task_ids", []),
            failed_task_ids=state.get("failed_agent_task_ids", []),
            replanned_failure_task_ids=state.get("replanned_failure_task_ids", []),
            planner_runs=int(state.get("planner_runs", 0) or 0),
            replan_count=int(state.get("replan_count", 0) or 0),
            context_slice=self._context_slice(state["context"]),
            clarification_request=state.get("clarification_request"),
            active_plan_id=state.get("active_plan_id"),
        )

    def _route_decision(self, decision: ResearchSupervisorDecision) -> str:
        return "finalize" if decision.action_name in {"finalize", "clarify_request"} else decision.action_name

    def _progress_signature(self, state: ResearchAgentGraphState) -> str:
        context = state["context"]
        workspace = context.workspace
        workspace_metadata = workspace.metadata if workspace is not None else {}
        task = context.task
        report = context.report
        return "|".join(
            [
                str(task.task_id if task else ""),
                str(len(context.papers)),
                str(report.report_id if report else ""),
                str(len(task.imported_document_ids) if task else 0),
                str(len(task.todo_items) if task else 0),
                str(context.import_result.imported_count if context.import_result else 0),
                str(context.qa_result.qa.answer[:80] if context.qa_result else ""),
                str(context.paper_analysis_result is not None or bool(workspace_metadata.get("latest_paper_analysis"))),
                str(
                    context.preference_recommendation_result is not None
                    or bool(workspace_metadata.get("latest_preference_recommendations"))
                ),
                str(context.compressed_context_summary is not None or bool(workspace_metadata.get("context_compression"))),
                str(context.parsed_document is not None),
                str(context.chart_result is not None),
                str(len(state.get("agent_results", []))),
                str(len(state.get("trace", []))),
            ]
        )

    def _should_force_finalize(self, state: ResearchAgentGraphState) -> bool:
        if state.get("exhausted"):
            return True
        stagnant = int(state.get("stagnant_decision_count", 0) or 0)
        repeated = int(state.get("repeated_action_count", 0) or 0)
        if stagnant >= 2 or repeated >= 2:
            return True
        context = state["context"]
        request = context.request
        if request.mode == "qa" and context.qa_result is not None:
            return True
        if context.preference_recommendation_result is not None:
            return True
        if request.advanced_action in {"analyze", "compare", "recommend"} and (
            context.paper_analysis_result is not None or self._workspace_has(context, "latest_paper_analysis")
        ):
            return True
        latest_result = state.get("agent_results", [])[-1] if state.get("agent_results") else None
        latest_observation = {}
        if latest_result is not None:
            observation = latest_result.payload.get("observation_envelope")
            if isinstance(observation, dict):
                latest_observation = observation
        latest_next_actions = {
            str(item).strip()
            for item in latest_observation.get("suggested_next_actions", [])
            if str(item).strip()
        }
        if (
            latest_result is not None
            and latest_result.task_type == "search_literature"
            and latest_result.status == "succeeded"
            and context.task_response is not None
            and context.report is not None
            and request.mode != "qa"
            and not latest_next_actions.intersection({"write_review", "import_papers", "answer_question"})
        ):
            return True
        if (
            latest_result is not None
            and latest_result.task_type == "write_review"
            and latest_result.status == "succeeded"
            and context.task_response is not None
            and context.report is not None
            and request.mode != "qa"
            and not latest_next_actions.intersection({"import_papers", "answer_question"})
        ):
            return True
        if context.task_response is not None and context.report is not None and not request.auto_import and request.mode != "qa":
            return True
        if context.import_attempted and context.import_result is not None and not request.message.strip():
            return True
        # Fast-finalize: when a specialist produced a terminal user-facing
        # answer, skip the extra supervisor LLM roundtrip to decide "finalize".
        # Only truly single-shot actions go here; chainable actions
        # (import_papers, understand_document, understand_chart, compress_context)
        # must return to supervisor so it can decide the next step.
        _TERMINAL_TASK_TYPES = {
            "general_answer",
            "answer_question",
            "analyze_papers",
            "analyze_paper_figures",
            "sync_to_zotero",
        }
        if (
            latest_result is not None
            and latest_result.status in {"succeeded", "skipped"}
            and latest_result.task_type in _TERMINAL_TASK_TYPES
            and not latest_next_actions
        ):
            return True
        return False

    def _workspace_has(self, context: ResearchAgentToolContext, key: str) -> bool:
        workspace = context.workspace
        return bool(workspace is not None and workspace.metadata.get(key))

    def _forced_finalize_decision(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        stagnant = int(state.get("stagnant_decision_count", 0) or 0)
        repeated = int(state.get("repeated_action_count", 0) or 0)
        reason = "Research loop reached a natural stopping point."
        if stagnant >= 2:
            reason = "Research loop stopped because workspace progress stalled across consecutive decisions."
        elif repeated >= 2:
            reason = "Research loop stopped because the manager repeated the same action without enough new progress."
        elif state.get("exhausted"):
            reason = f"Research loop stopped after reaching max_steps={self.max_steps}."
        return ResearchSupervisorDecision(
            action_name="finalize",
            thought="Stop the loop and return the current workspace instead of continuing low-value work.",
            rationale=reason,
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=reason,
            metadata={
                "decision_source": "runtime_guardrail",
                "worker_agent": self._manager_trace_agent_name(),
                "loop_guardrail": {
                    "stagnant_decision_count": stagnant,
                    "repeated_action_count": repeated,
                    "progress_signature": state.get("progress_signature", ""),
                },
            },
        )

    def _on_decision(
        self,
        state: ResearchAgentGraphState,
        step_index: int,
        decision: ResearchSupervisorDecision,
    ) -> ResearchAgentGraphState:
        signature = self._progress_signature(state)
        previous_signature = str(state.get("progress_signature") or "")
        stagnant_count = int(state.get("stagnant_decision_count", 0) or 0)
        if previous_signature and signature == previous_signature:
            stagnant_count += 1
        else:
            stagnant_count = 0
        trace = state.get("trace", [])
        last_action = trace[-1].action_name if trace else None
        repeated_count = int(state.get("repeated_action_count", 0) or 0)
        if decision.action_name != "finalize" and decision.action_name == last_action:
            repeated_count += 1
        else:
            repeated_count = 0
        logger.info(
            "Research manager loop guard | step=%s | action=%s | stagnant=%s | repeated=%s | signature=%s",
            step_index,
            decision.action_name,
            stagnant_count,
            repeated_count,
            signature,
        )
        update = decision.metadata.get("state_update") if isinstance(decision.metadata, dict) else None
        if isinstance(update, dict):
            update = {
                **update,
                "progress_signature": signature,
                "stagnant_decision_count": stagnant_count,
                "repeated_action_count": repeated_count,
            }
            self._sync_execution_context(
                context=state["context"],
                agent_messages=update.get("agent_messages", state.get("agent_messages", [])),
                agent_results=update.get("agent_results", state.get("agent_results", [])),
                pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
            )
            return update  # type: ignore[return-value]
        return {
            "progress_signature": signature,
            "stagnant_decision_count": stagnant_count,
            "repeated_action_count": repeated_count,
        }

    async def search_literature_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, SearchLiteratureTool.name)

    async def write_review_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, WriteReviewTool.name)

    async def import_papers_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, ImportPapersTool.name)

    async def sync_to_zotero_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, SyncToZoteroTool.name)

    async def answer_question_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, AnswerQuestionTool.name)

    async def general_answer_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, GeneralAnswerTool.name)

    async def recommend_from_preferences_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, RecommendFromPreferencesTool.name)

    async def analyze_papers_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, AnalyzePapersTool.name)

    async def compress_context_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, CompressContextTool.name)

    async def understand_document_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, UnderstandDocumentTool.name)

    async def understand_chart_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, UnderstandChartTool.name)

    async def analyze_paper_figures_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, AnalyzePaperFiguresTool.name)

    async def _run_action_node(
        self,
        state: ResearchAgentGraphState,
        action_name: Literal[
            "search_literature",
            "write_review",
            "import_papers",
            "sync_to_zotero",
            "answer_question",
            "general_answer",
            "recommend_from_preferences",
            "analyze_papers",
            "compress_context",
            "understand_document",
            "understand_chart",
            "analyze_paper_figures",
        ],
    ) -> ResearchAgentGraphState:
        decision = state.get("current_decision")
        if decision is None or decision.action_name != action_name:
            return {}
        _update_runtime_progress(
            state["context"],
            stage=action_name,
            node=f"{action_name}_node",
            status="running",
            summary=f"Executing {action_name}.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0)},
        )
        trace = list(state.get("trace", []))
        step_index = state.get("current_step_index", 0)
        worker_agent = self._worker_agent_name(decision)
        active_message = self._active_message(decision)
        unified_result = await self._execute_unified_worker(
            context=state["context"],
            decision=decision,
            active_message=active_message,
            worker_agent=worker_agent,
        )
        if unified_result is not None:
            result = self._research_tool_result_from_unified_result(unified_result)
        else:
            execution_result = await self._execute_action_tool(
                action_name=action_name,
                context=state["context"],
                decision=decision,
            )
            result = self._action_tool_result_from_execution(execution_result)
            if result is None:
                trace.append(
                    ResearchAgentTraceStep(
                        step_index=step_index,
                        agent=worker_agent,
                        thought=decision.thought,
                        action_name=action_name,
                        phase=decision.phase,  # type: ignore[arg-type]
                        action_input=decision.action_input,
                        status="failed",
                        observation=execution_result.error_message or f"tool execution failed: {action_name}",
                        rationale=decision.rationale,
                        estimated_gain=decision.estimated_gain,
                        estimated_cost=decision.estimated_cost,
                        stop_signal=True,
                        workspace_summary=self._workspace_summary(state["context"]),
                        metadata=self._trace_metadata(
                            {
                                "reason": "tool_execution_failed",
                                **self._action_tool_execution_metadata(execution_result),
                            },
                            active_message=active_message,
                        ),
                    )
                )
                update = {
                    "trace": trace,
                    "failed": True,
                    **self._message_result_update(
                        state,
                        active_message=active_message,
                        worker_agent=worker_agent,
                        status="failed",
                        payload={
                            "reason": "tool_execution_failed",
                            **self._action_tool_execution_metadata(execution_result),
                        },
                    ),
                }
                self._sync_execution_context(
                    context=state["context"],
                    agent_messages=state.get("agent_messages", []),
                    agent_results=update.get("agent_results", state.get("agent_results", [])),
                    pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
                )
                return update

        status = result.status if result.status in {"succeeded", "failed", "skipped"} else "succeeded"
        trace.append(
            ResearchAgentTraceStep(
                step_index=step_index,
                agent=worker_agent,
                thought=decision.thought,
                action_name=action_name,
                phase=decision.phase,  # type: ignore[arg-type]
                action_input=decision.action_input,
                status=status,  # type: ignore[arg-type]
                observation=result.observation,
                rationale=decision.rationale,
                estimated_gain=decision.estimated_gain,
                estimated_cost=decision.estimated_cost,
                stop_signal=status == "failed",
                workspace_summary=self._workspace_summary(state["context"]),
                metadata=self._trace_metadata(
                    dict(result.metadata),
                    active_message=active_message,
                ),
            )
        )
        update = {
            "trace": trace,
            "failed": bool(state.get("failed")) or result.status == "failed",
            **self._message_result_update(
                state,
                active_message=active_message,
                worker_agent=worker_agent,
                status=status,
                payload=dict(result.metadata),
            ),
        }
        self._sync_execution_context(
            context=state["context"],
            agent_messages=state.get("agent_messages", []),
            agent_results=update.get("agent_results", state.get("agent_results", [])),
            pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
        )
        return update

    def route_after_action(self, state: ResearchAgentGraphState) -> str:
        del state
        return "manager"

    async def finalize_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        decision = state.get("current_decision")
        if decision is None:
            return {}
        _update_runtime_progress(
            state["context"],
            stage="finalize",
            node="finalize_node",
            status="running",
            summary="Finalizing supervisor response.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0)},
        )
        trace = list(state.get("trace", []))
        worker_agent = self._worker_agent_name(decision)
        active_message = self._active_message(decision)
        trace.append(
            ResearchAgentTraceStep(
                step_index=state.get("current_step_index", 0),
                agent=worker_agent,
                thought=decision.thought,
                action_name="finalize",
                phase=decision.phase,  # type: ignore[arg-type]
                action_input=decision.action_input,
                status="succeeded",
                observation="manager requested clarification and stopped the loop" if decision.action_name == "clarify_request" else "manager stopped the loop",
                rationale=decision.rationale,
                estimated_gain=decision.estimated_gain,
                estimated_cost=decision.estimated_cost,
                stop_signal=True,
                workspace_summary=self._workspace_summary(state["context"]),
                metadata=self._trace_metadata(
                    {
                        "stop_reason": decision.stop_reason,
                        **(
                            {"clarification_request": decision.stop_reason}
                            if decision.action_name == "clarify_request" and decision.stop_reason
                            else {}
                        ),
                    },
                    active_message=active_message,
                ),
            )
        )
        update = {
            "trace": trace,
            **self._message_result_update(
                state,
                active_message=active_message,
                worker_agent=worker_agent,
                status="succeeded",
                payload={
                    "stop_reason": decision.stop_reason,
                    **(
                        {"clarification_request": decision.stop_reason}
                        if decision.action_name == "clarify_request" and decision.stop_reason
                        else {}
                    ),
                },
            ),
        }
        self._sync_execution_context(
            context=state["context"],
            agent_messages=state.get("agent_messages", []),
            agent_results=update.get("agent_results", state.get("agent_results", [])),
            pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
        )
        return update

    def _context_slice(self, context: ResearchAgentToolContext):
        execution_context = context.execution_context
        if execution_context is None:
            return None
        return execution_context.context_slices.get("manager") or execution_context.context_slices.get("default")

    def _sync_execution_context(
        self,
        *,
        context: ResearchAgentToolContext,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        pending_messages: list[AgentMessage],
    ) -> None:
        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return
        research_context = execution_context.research_context.model_copy(deep=True)
        pending_ids = {message.task_id for message in pending_messages}
        result_by_task_id = {result.task_id: result for result in agent_results}
        task_plan: list[TaskStep] = []
        for message in agent_messages:
            result = result_by_task_id.get(message.task_id)
            status = "planned"
            if message.task_id in pending_ids:
                status = "queued"
            if result is not None:
                if result.status == "failed" or (result.evaluation is not None and not result.evaluation.passed):
                    status = "failed"
                elif result.status == "skipped":
                    status = "skipped"
                elif result.status == "succeeded":
                    status = "succeeded"
            task_plan.append(
                TaskStep(
                    task_id=message.task_id,
                    assigned_to=message.agent_to,
                    instruction=message.instruction,
                    task_type=message.task_type,
                    depends_on=list(message.depends_on),
                    context_slice=self._serialize_context_slice(message.context_slice),
                    expected_output_schema=dict(message.expected_output_schema),
                    priority=self._task_priority_from_message(message.priority),
                    retry_count=message.retry_count,
                    status=status,  # type: ignore[arg-type]
                    metadata=dict(message.metadata),
                )
            )

        research_context.current_task_plan = task_plan
        research_context.sub_manager_states = self._build_sub_manager_states(task_plan)
        execution_context.research_context = research_context
        execution_context.context_slices = self.research_service.build_context_slices(
            research_context,
            selected_paper_ids=context.request.selected_paper_ids,
        )
        if execution_context.session_id:
            self.research_service.memory_manager.save_context(
                execution_context.session_id,
                research_context,
            )

    def _active_message(self, decision: ResearchSupervisorDecision) -> AgentMessage | None:
        if not isinstance(decision.metadata, dict):
            return None
        active_message = decision.metadata.get("active_message")
        if isinstance(active_message, AgentMessage):
            return active_message
        if isinstance(active_message, dict):
            return AgentMessage.model_validate(active_message)
        return None

    def _worker_agent_name(self, decision: ResearchSupervisorDecision) -> str:
        if isinstance(decision.metadata, dict):
            worker_agent = decision.metadata.get("worker_agent")
            if isinstance(worker_agent, str) and worker_agent.strip():
                return worker_agent
        return self._manager_trace_agent_name()

    def _unified_agent_descriptor(
        self,
        context: ResearchAgentToolContext,
        *,
        agent_name: str,
    ) -> UnifiedAgentDescriptor | None:
        registry = context.unified_agent_registry
        if registry is None:
            return None
        executor = registry.get(agent_name)
        if executor is None:
            return None
        return executor.descriptor

    def _preferred_skill_name_for_message(
        self,
        context: ResearchAgentToolContext,
        *,
        active_message: AgentMessage,
        worker_agent: str,
    ) -> str | None:
        for key in ("preferred_skill_name", "skill_name"):
            raw_value = active_message.metadata.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
        request_skill_name = (context.request.skill_name or "").strip()
        if request_skill_name:
            return request_skill_name
        descriptor = self._unified_agent_descriptor(context, agent_name=worker_agent)
        if descriptor is not None and descriptor.skill_binding.profile_name:
            return descriptor.skill_binding.profile_name
        return None

    def _available_tool_names_for_agent(
        self,
        context: ResearchAgentToolContext,
        *,
        agent_name: str,
    ) -> list[str]:
        descriptor = self._unified_agent_descriptor(context, agent_name=agent_name)
        if descriptor is None:
            return []
        return list(descriptor.available_tool_names)

    def _trace_metadata(
        self,
        payload: dict[str, Any],
        *,
        active_message: AgentMessage | None,
    ) -> dict[str, Any]:
        metadata = dict(payload)
        if active_message is not None:
            metadata.setdefault("planner_task_id", active_message.task_id)
            metadata.setdefault("task_type", active_message.task_type)
            metadata.setdefault("agent_to", active_message.agent_to)
        return metadata

    def _message_result_update(
        self,
        state: ResearchAgentGraphState,
        *,
        active_message: AgentMessage | None,
        worker_agent: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> ResearchAgentGraphState:
        if active_message is None:
            return {}
        context = state["context"]
        result_payload = dict(payload or {})
        pending = [
            message
            for message in state.get("pending_agent_messages", [])
            if message.task_id != active_message.task_id
        ]
        if status == "failed":
            pending = []
        descriptor = self._unified_agent_descriptor(context, agent_name=worker_agent)
        unified_task = UnifiedAgentTask.from_agent_message(
            active_message,
            preferred_skill_name=self._preferred_skill_name_for_message(
                context,
                active_message=active_message,
                worker_agent=worker_agent,
            ),
            available_tool_names=self._available_tool_names_for_agent(
                context,
                agent_name=worker_agent,
            ),
        )
        action_output = UnifiedAgentResult.extract_action_output(payload=result_payload)
        result_metadata = {"plan_id": active_message.metadata.get("plan_id"), **result_payload}
        if action_output is not None:
            result_metadata.setdefault(
                UNIFIED_ACTION_OUTPUT_METADATA_KEY,
                dict(action_output),
            )
        if unified_task.preferred_skill_name:
            result_metadata.setdefault("preferred_skill_name", unified_task.preferred_skill_name)
        if unified_task.available_tool_names:
            result_metadata.setdefault("available_tool_names", list(unified_task.available_tool_names))
        if descriptor is not None:
            result_metadata.setdefault("execution_mode", descriptor.execution_mode)
            result_metadata.setdefault("agent_descriptor_name", descriptor.name)
        unified_result = UnifiedAgentResult(
            task_id=active_message.task_id,
            agent_name=worker_agent,
            task_type=active_message.task_type,
            status=status,  # type: ignore[arg-type]
            instruction=active_message.instruction,
            payload=result_payload,
            context_slice=active_message.context_slice,
            priority=active_message.priority,
            expected_output_schema=active_message.expected_output_schema,
            depends_on=list(active_message.depends_on),
            retry_count=active_message.retry_count,
            evaluation=result_payload.get("evaluation") if isinstance(result_payload, dict) else None,
            action_output=action_output,
            metadata=result_metadata,
        )
        result_message = unified_result.to_agent_result_message(reply_to=active_message.agent_from)
        completed = list(state.get("completed_agent_task_ids", []))
        failed = list(state.get("failed_agent_task_ids", []))
        if status in {"succeeded", "skipped"}:
            completed.append(active_message.task_id)
        if status == "failed":
            failed.append(active_message.task_id)
        return {
            "pending_agent_messages": pending,
            "agent_results": [*state.get("agent_results", []), result_message],
            "completed_agent_task_ids": self._dedupe_ids(completed),
            "failed_agent_task_ids": self._dedupe_ids(failed),
        }

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            deduped.append(value)
            seen.add(value)
        return deduped

    def _serialize_context_slice(self, context_slice: Any) -> dict[str, Any]:
        if hasattr(context_slice, "model_dump"):
            return context_slice.model_dump(mode="json")
        if isinstance(context_slice, dict):
            return dict(context_slice)
        return {}

    def _task_priority_from_message(self, priority: str) -> str:
        if priority in {"high", "critical"}:
            return "high"
        if priority == "low":
            return "low"
        return "normal"

    def _build_sub_manager_states(
        self,
        task_plan: list[TaskStep],
    ) -> dict[str, SubManagerState]:
        states = {
            "research": SubManagerState(name="research"),
            "writing": SubManagerState(name="writing"),
        }
        assignments = {
            "LiteratureScoutAgent": "research",
            "ChartAnalysisAgent": "research",
            "ResearchKnowledgeAgent": "research",
            "PreferenceMemoryAgent": "research",
            "ResearchWriterAgent": "writing",
            "PaperAnalysisAgent": "writing",
        }
        for task in task_plan:
            key = assignments.get(task.assigned_to)
            if key is None:
                continue
            state = states[key]
            if task.status in {"planned", "queued", "running"}:
                state.active_task_ids = self._dedupe_ids([*state.active_task_ids, task.task_id])
            if task.status == "succeeded":
                state.completed_task_ids = self._dedupe_ids([*state.completed_task_ids, task.task_id])
            if task.status == "failed":
                state.status = "failed"
            elif task.status in {"planned", "queued", "running"} and state.status != "failed":
                state.status = "running"
            elif (
                task.status == "succeeded"
                and not state.active_task_ids
                and state.status not in {"failed", "running"}
            ):
                state.status = "completed"
            state.last_task_plan_id = str(task.metadata.get("plan_id") or state.last_task_plan_id or "")
        for state in states.values():
            if state.status == "idle" and state.completed_task_ids:
                state.status = "completed"
            if not state.last_task_plan_id:
                state.last_task_plan_id = None
        return states

    def _serialize_task_plan(
        self,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
    ) -> list[dict[str, Any]]:
        result_by_task_id = {result.task_id: result for result in agent_results}
        serialized: list[dict[str, Any]] = []
        for message in agent_messages:
            result = result_by_task_id.get(message.task_id)
            serialized_status = "planned"
            if result is not None:
                serialized_status = (
                    "failed"
                    if result.evaluation is not None and not result.evaluation.passed
                    else result.status
                )
            serialized.append(
                {
                    "task_id": message.task_id,
                    "assigned_to": message.agent_to,
                    "task_type": message.task_type,
                    "instruction": message.instruction,
                    "depends_on": list(message.depends_on),
                    "priority": message.priority,
                    "retry_count": message.retry_count,
                    "status": serialized_status,
                    "evaluation": (
                        result.evaluation.model_dump(mode="json")
                        if result is not None and result.evaluation is not None
                        else None
                    ),
                }
            )
        return serialized

    async def _state_from_context(
        self,
        context: ResearchAgentToolContext,
        *,
        trace: list[ResearchAgentTraceStep] | None = None,
    ) -> ResearchSupervisorState:
        task = context.task
        papers = context.papers
        importable_paper_count = sum(
            1
            for paper in papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        )
        has_import_candidates = importable_paper_count > 0
        open_todo_count = sum(1 for item in (task.todo_items if task else []) if item.status == "open")
        workspace = context.workspace
        workspace_stage = workspace.current_stage if workspace is not None else self._derive_workspace_stage(context)
        evidence_gap_count = len(workspace.evidence_gaps) if workspace is not None else len(context.report.gaps) if context.report else 0
        resolved_trace = trace or []
        failed_actions = [step.action_name for step in resolved_trace if step.status == "failed"]
        workspace_metadata = dict(workspace.metadata) if workspace is not None else {}
        research_context = (
            context.execution_context.research_context
            if context.execution_context is not None
            else None
        )
        request_metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        metadata_context = (
            dict(request_metadata.get("context") or {})
            if isinstance(request_metadata.get("context"), dict)
            else {}
        )
        research_goal_lower = context.request.message.lower()
        conversation = None
        if context.request.conversation_id:
            conversation = context.research_service.report_service.load_conversation(context.request.conversation_id)
        snapshot = conversation.snapshot if conversation is not None else None
        metadata_comparison_dimension_values = request_metadata.get("comparison_dimensions")
        metadata_comparison_dimensions = [
            str(item).strip()
            for item in (metadata_comparison_dimension_values if isinstance(metadata_comparison_dimension_values, list) else [])
            if str(item).strip()
        ]
        comparison_dimensions = list(
            dict.fromkeys(
                [
                    *(item.strip() for item in context.request.comparison_dimensions if item.strip()),
                    *metadata_comparison_dimensions,
                ]
            )
        )
        recommendation_goal = str(
            context.request.recommendation_goal
            or request_metadata.get("recommendation_goal")
            or ""
        ).strip() or None
        recommendation_top_k = max(
            1,
            int(
                context.request.recommendation_top_k
                or request_metadata.get("recommendation_top_k")
                or 3
            ),
        )
        user_profile = context.research_service.memory_manager.load_user_profile()
        force_context_compression = bool(
            context.request.force_context_compression
            or request_metadata.get("force_context_compression")
        )
        active_paper_ids = self._active_paper_ids_for_manager(context)
        compare_requested = any(
            marker in research_goal_lower
            for marker in ("对比", "比较", "compare", "comparison", " vs ", "versus")
        ) or context.request.advanced_action == "compare" or bool(comparison_dimensions)
        recommend_requested = any(
            marker in research_goal_lower
            for marker in ("推荐", "recommend", "suggest", "建议阅读")
        ) or context.request.advanced_action == "recommend" or bool(recommendation_goal)
        preference_recommendation_requested = (
            context.request.advanced_action is None
            and self._looks_like_preference_recommendation_request(research_goal_lower)
            and not compare_requested
            and not bool(recommendation_goal)
        )
        explain_requested = any(
            marker in research_goal_lower
            for marker in ("分析", "讲解", "解释", "怎么理解", "analysis", "analyze", "explain")
        ) or context.request.advanced_action == "analyze"
        paper_detail_requested = any(
            marker in research_goal_lower
            for marker in (
                "方法",
                "用了什么方法",
                "技术路线",
                "核心思路",
                "主要思路",
                "怎么做的",
                "做法",
                "模型",
                "架构",
                "实验",
                "贡献",
                "创新点",
                "method",
                "methods",
                "approach",
                "pipeline",
                "architecture",
                "experiment",
                "experiments",
                "contribution",
                "contributions",
            )
        )
        paper_analysis_requested = (
            compare_requested
            or (recommend_requested and not preference_recommendation_requested)
            or context.request.advanced_action == "analyze"
            or (
                bool(context.request.selected_paper_ids or active_paper_ids)
                and (explain_requested or paper_detail_requested)
            )
        )
        analysis_focus = (
            "compare"
            if compare_requested
            else "recommend"
            if recommend_requested
            else "explain"
            if explain_requested or paper_detail_requested
            else None
        )
        context_compressed = bool(
            context.compressed_context_summary
            or workspace_metadata.get("context_compression")
        )
        session_history_count = len(research_context.session_history) if research_context is not None else 0
        context_size_large = False
        _exec_ctx = context.execution_context
        if _exec_ctx is not None and hasattr(_exec_ctx, "context_slices"):
            _manager_slice = (_exec_ctx.context_slices or {}).get("manager")
            if _manager_slice is not None:
                try:
                    import json as _json
                    _slice_data = _manager_slice.model_dump(mode="json") if hasattr(_manager_slice, "model_dump") else dict(_manager_slice)
                    context_size_large = len(_json.dumps(_slice_data, ensure_ascii=False, default=str)) > 80_000
                except Exception:
                    pass
        context_compression_needed = bool(task or papers) and not context_compressed and (
            force_context_compression
            or paper_analysis_requested
            or len(context.request.selected_paper_ids) >= 2
            or len(papers) >= 4
            or session_history_count >= 6
            or context_size_large
        )
        user_intent = await self.user_intent_resolver.resolve_async(
            message=context.request.message,
            has_task=task is not None,
            candidate_paper_count=len(papers),
            candidate_papers=self._candidate_paper_scope_for_manager(papers),
            active_paper_ids=active_paper_ids,
            selected_paper_ids=list(context.request.selected_paper_ids),
            has_visual_anchor=bool(context.request.chart_image_path or context.request.chart_id),
            has_document_input=bool(context.request.document_file_path),
        )
        route_mode = str(
            metadata_context.get("route_mode")
            or (snapshot.active_route_mode if snapshot is not None else "")
            or "research_follow_up"
        ).strip() or "research_follow_up"
        active_thread_id = str(
            metadata_context.get("active_thread_id")
            or (snapshot.active_thread_id if snapshot is not None else "")
        ).strip() or None
        active_thread_topic = None
        if snapshot is not None and active_thread_id:
            for thread in snapshot.thread_history:
                if thread.thread_id == active_thread_id:
                    active_thread_topic = thread.topic or None
                    break
        topic_continuity_score = self._topic_continuity_score(
            context.request.message,
            active_thread_topic or (snapshot.topic if snapshot is not None else ""),
        )
        new_topic_detected = (
            route_mode == "research_discovery"
            and bool(snapshot is not None and snapshot.topic)
            and topic_continuity_score < 0.42
        )
        should_ignore_research_context = route_mode == "general_chat" or (
            route_mode == "research_discovery" and new_topic_detected
        )
        return ResearchSupervisorState(
            goal=context.request.message,
            mode=context.request.mode,
            route_mode=route_mode,
            active_thread_id=active_thread_id,
            active_thread_topic=active_thread_topic,
            topic_continuity_score=topic_continuity_score,
            new_topic_detected=new_topic_detected,
            should_ignore_research_context=should_ignore_research_context,
            task_id=context.request.task_id or (task.task_id if task else None),
            has_task=task is not None,
            has_report=context.report is not None,
            paper_count=len(papers),
            imported_document_count=len(task.imported_document_ids) if task else 0,
            has_document_input=bool(context.request.document_file_path),
            has_chart_input=bool(context.request.chart_image_path),
            document_understood=context.document_attempted or context.parsed_document is not None,
            chart_understood=context.chart_attempted or context.chart_result is not None,
            has_import_candidates=has_import_candidates,
            importable_paper_count=importable_paper_count,
            selected_paper_count=len(context.request.selected_paper_ids),
            active_paper_ids=active_paper_ids,
            auto_import=context.request.auto_import,
            import_top_k=context.request.import_top_k,
            import_attempted=context.import_attempted or context.import_result is not None,
            answer_attempted=context.qa_result is not None,
            open_todo_count=open_todo_count,
            evidence_gap_count=evidence_gap_count,
            workspace_stage=workspace_stage,
            workspace_ready=bool(task and (task.imported_document_ids or context.report is not None)),
            paper_analysis_requested=paper_analysis_requested,
            preference_recommendation_requested=preference_recommendation_requested,
            known_interest_count=len(user_profile.interest_topics or user_profile.research_interests),
            analysis_focus=analysis_focus,
            comparison_dimensions=comparison_dimensions,
            recommendation_goal=recommendation_goal,
            recommendation_top_k=recommendation_top_k,
            force_context_compression=force_context_compression,
            context_compression_needed=context_compression_needed,
            paper_analysis_completed=bool(
                context.paper_analysis_result is not None or workspace_metadata.get("latest_paper_analysis")
            ),
            context_compressed=context_compressed,
            last_action_name=resolved_trace[-1].action_name if resolved_trace else None,
            failed_actions=failed_actions,
            candidate_papers=self._candidate_paper_scope_for_manager(papers),
            user_intent=user_intent.model_dump(mode="json"),
        )

    def _topic_continuity_score(self, current: str | None, previous: str | None) -> float:
        current_tokens = set(self._normalize_topic_text(current).split())
        previous_tokens = set(self._normalize_topic_text(previous).split())
        if not current_tokens or not previous_tokens:
            return 0.0
        return len(current_tokens & previous_tokens) / max(1, len(current_tokens | previous_tokens))

    def _active_paper_ids_for_manager(self, context: ResearchAgentToolContext) -> list[str]:
        metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        metadata_context = metadata.get("context")
        active_ids: list[str] = []
        if isinstance(metadata_context, dict):
            raw_active = metadata_context.get("active_paper_ids")
            if isinstance(raw_active, list):
                active_ids.extend(str(item).strip() for item in raw_active if str(item).strip())
        if context.execution_context is not None:
            active_ids.extend(
                str(item).strip()
                for item in getattr(context.execution_context.research_context, "active_papers", [])
                if str(item).strip()
            )
        return self._dedupe_ids(active_ids or context.request.selected_paper_ids)

    def _candidate_paper_scope_for_manager(self, papers: list[PaperCandidate], *, limit: int = 12) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for index, paper in enumerate(papers[:limit], start=1):
            candidates.append(
                {
                    "index": index,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "relevance_score": paper.relevance_score,
                    "citations": paper.citations,
                    "has_imported_document": bool(str(paper.metadata.get("document_id") or "").strip()),
                }
            )
        return candidates

    def _build_response(
        self,
        *,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        failed: bool,
        exhausted: bool,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        clarification_request: str | None,
        active_plan_id: str | None,
    ) -> ResearchAgentRunResponse:
        warnings = list(context.warnings or [])
        if context.task_response:
            warnings.extend(warning for warning in context.task_response.warnings if warning not in warnings)
        if context.import_result:
            for result in context.import_result.results:
                if result.status == "failed" and result.error_message:
                    warning = f"import failed: {result.title}: {result.error_message}"
                    if warning not in warnings:
                        warnings.append(warning)
        if clarification_request and clarification_request not in warnings:
            warnings.append(clarification_request)
        workspace = self._resolve_workspace(context=context, trace=trace, failed=failed, exhausted=exhausted)
        advanced_strategy = self._resolved_advanced_strategy(context, workspace=workspace)
        workspace = workspace.model_copy(
            update={
                "metadata": {
                    **workspace.metadata,
                    "advanced_strategy": advanced_strategy.model_dump(mode="json"),
                }
            }
        )
        context.task_response = self.research_service.persist_runtime_state(
            task_response=context.task_response,
            workspace=workspace,
            conversation_id=request.conversation_id,
            advanced_strategy=advanced_strategy,
        )
        messages = self._build_messages(
            request,
            context,
            trace,
            workspace,
            agent_messages=agent_messages,
            agent_results=agent_results,
            clarification_request=clarification_request,
            replan_count=replan_count,
        )
        self._log_internal_runtime_state(
            request=request,
            workspace=workspace,
            trace=trace,
            agent_messages=agent_messages,
            agent_results=agent_results,
            clarification_request=clarification_request,
        )
        next_actions = self._next_actions(context, workspace, clarification_request=clarification_request)
        status = "failed" if failed and not context.task_response else "partial" if failed or warnings else "succeeded"
        runtime_metadata = self._response_runtime_metadata()
        active_paper_ids = (
            list(context.execution_context.research_context.active_papers)
            if context.execution_context and context.execution_context.research_context
            else self._active_paper_ids_for_manager(context)
        )
        return ResearchAgentRunResponse(
            status=status,
            task=context.task,
            papers=context.papers,
            report=context.report,
            import_result=context.import_result,
            qa=context.qa_result.qa if context.qa_result else None,
            parsed_document=context.parsed_document,
            document_index_result=context.document_index_result,
            chart=getattr(context.chart_result, "chart", None) if context.chart_result is not None else None,
            chart_graph_text=getattr(context.chart_result, "graph_text", None) if context.chart_result is not None else None,
            messages=messages,
            trace=trace,
            warnings=warnings,
            next_actions=next_actions,
            workspace=workspace,
            metadata={
                **runtime_metadata,
                "tool_count": len(self.action_tool_registry.list_tools(include_disabled=False)),
                "supervisor_action_tool_engine": "tool_executor",
                "supervisor_worker_execution_engine": (
                    "unified_agent_registry" if context.unified_agent_registry is not None else "legacy_action_tools"
                ),
                "supervisor_action_trace_count": len(self.action_tool_executor.get_traces()),
                "unified_supervisor_mode": "pure_supervisor",
                "unified_runtime_blueprint": context.unified_blueprint or {},
                "unified_agent_registry": serialize_unified_agent_registry(context.unified_agent_registry),
                "task_id": context.task.task_id if context.task else None,
                "qa_task_id": context.qa_result.task_id if context.qa_result else None,
                "active_paper_ids": active_paper_ids,
                "route_mode": (
                    ((context.request.metadata or {}).get("context") or {}).get("route_mode")
                    if isinstance((context.request.metadata or {}).get("context"), dict)
                    else None
                ),
                "active_thread_id": (
                    ((context.request.metadata or {}).get("context") or {}).get("active_thread_id")
                    if isinstance((context.request.metadata or {}).get("context"), dict)
                    else None
                ),
                "session_id": context.execution_context.session_id if context.execution_context else None,
                "memory_enabled": context.execution_context.memory_enabled if context.execution_context else False,
                "has_document_tool_output": context.parsed_document is not None,
                "has_chart_tool_output": context.chart_result is not None,
                "has_general_answer": context.general_answer is not None,
                "preference_recommendations": (
                    context.preference_recommendation_result.model_dump(mode="json")
                    if context.preference_recommendation_result is not None
                    else workspace.metadata.get("latest_preference_recommendations")
                ),
                "workspace_stage": workspace.current_stage,
                "workspace_summary": workspace.status_summary,
                "stop_reason": workspace.stop_reason,
                "trace_steps": len(trace),
                "manager_decision_count": planner_runs,
                "recovery_decision_count": replan_count,
                "agent_message_count": len(agent_messages),
                "agent_result_count": len(agent_results),
                "active_decision_batch_id": active_plan_id,
                "clarification_requested": bool(clarification_request),
                "advanced_strategy": advanced_strategy.model_dump(mode="json"),
                "delegation_plan": self._serialize_task_plan(agent_messages, agent_results),
                "unified_delegation_plan": serialize_unified_delegation_plan(
                    agent_messages,
                    agent_results,
                    registry=context.unified_agent_registry,
                ),
                "agent_lane_states": (
                    {
                        name: state.model_dump(mode="json")
                        for name, state in context.execution_context.research_context.sub_manager_states.items()
                    }
                    if context.execution_context and context.execution_context.research_context
                    else {}
                ),
                "agent_messages": [message.model_dump(mode="json") for message in agent_messages],
                "agent_results": [result.model_dump(mode="json") for result in agent_results],
                "general_answer": context.general_answer,
                "general_answer_metadata": context.general_answer_metadata or {},
                "unified_agent_messages": serialize_unified_agent_messages(
                    agent_messages,
                    registry=context.unified_agent_registry,
                ),
                "unified_agent_results": serialize_unified_agent_results(
                    agent_results,
                    registry=context.unified_agent_registry,
                ),
            },
        )

    def _log_internal_runtime_state(
        self,
        *,
        request: ResearchAgentRunRequest,
        workspace: ResearchWorkspaceState,
        trace: list[ResearchAgentTraceStep],
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        clarification_request: str | None,
    ) -> None:
        logger.info(
            "Research agent internals | mode=%s | task=%s | workspace=%s | stop_reason=%s | clarification=%s | plan=%s | trace=%s",
            request.mode,
            request.task_id,
            workspace.status_summary or workspace.current_stage,
            workspace.stop_reason,
            clarification_request or "",
            [
                {
                    "task_id": message.task_id,
                    "agent": message.agent_to,
                    "type": message.task_type,
                    "priority": message.priority,
                    "status": next(
                        (
                            result.status
                            for result in agent_results
                            if result.task_id == message.task_id
                        ),
                        "planned",
                    ),
                }
                for message in agent_messages[-8:]
            ],
            [
                {
                    "step": step.step_index,
                    "agent": step.agent,
                    "action": step.action_name,
                    "phase": step.phase,
                    "status": step.status,
                }
                for step in trace
            ],
        )

    def _build_messages(
        self,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        workspace: ResearchWorkspaceState,
        *,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        clarification_request: str | None,
        replan_count: int,
    ) -> list[ResearchMessage]:
        executed_actions = {step.action_name for step in trace if step.status in {"succeeded", "skipped"}}
        messages = [
            _message(
                role="user",
                kind="topic" if request.mode != "qa" else "question",
                title="用户研究目标" if request.mode != "qa" else "研究集合提问",
                content=request.message,
            )
        ]
        if clarification_request:
            messages.append(
                _message(
                    role="assistant",
                    kind="warning",
                    title="需要澄清研究目标",
                    meta=f"{self._manager_display_name()} 请求用户澄清范围",
                    content=clarification_request,
                    payload={"clarification_request": clarification_request},
                )
            )
        if agent_messages:
            plan_lines = [
                (
                    f"- {message.task_id} · {message.agent_to} · {message.task_type} "
                    f"· priority={message.priority}"
                    f"{' · depends_on=' + ','.join(message.depends_on) if message.depends_on else ''}"
                )
                for message in agent_messages[-8:]
            ]
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Manager 决策轨迹",
                    meta=f"decisions={len(agent_messages)} · results={len(agent_results)} · recoveries={replan_count}",
                    content="\n".join(plan_lines),
                    payload={
                        "agent_messages": [message.model_dump(mode="json") for message in agent_messages],
                        "agent_results": [result.model_dump(mode="json") for result in agent_results],
                        "delegation_plan": self._serialize_task_plan(agent_messages, agent_results),
                        "unified_delegation_plan": serialize_unified_delegation_plan(
                            agent_messages,
                            agent_results,
                            registry=context.unified_agent_registry,
                        ),
                    },
                )
            )
        report_is_fresh = executed_actions & {"search_literature", "write_review"}
        if context.report and report_is_fresh:
            messages.append(
                _message(
                    role="assistant",
                    kind="report",
                    title="自主文献综述",
                    meta=f"候选论文 {context.report.paper_count} 篇",
                    content=context.report.markdown,
                    payload={"report": context.report.model_dump(mode="json")},
                )
            )
        if context.papers and report_is_fresh:
            messages.append(
                _message(
                    role="assistant",
                    kind="candidates",
                    title="候选论文池",
                    meta=f"当前共 {len(context.papers)} 篇",
                    payload={"papers": [paper.model_dump(mode="json") for paper in context.papers]},
                )
            )
        if context.import_result:
            lines = [
                f"imported={context.import_result.imported_count} · skipped={context.import_result.skipped_count} · failed={context.import_result.failed_count}"
            ]
            lines.extend(
                f"- {result.title} · {result.status}{' · doc=' + result.document_id if result.document_id else ''}"
                for result in context.import_result.results[:5]
            )
            messages.append(
                _message(
                    role="assistant",
                    kind="import_result",
                    title="自主导入结果",
                    meta=f"{self._manager_display_name()} 调用了论文导入工具",
                    content="\n".join(lines),
                    payload={"import_result": context.import_result.model_dump(mode="json")},
                )
            )
        if context.zotero_sync_results:
            synced_count = sum(1 for item in context.zotero_sync_results if str(item.get("status") or "") in {"imported", "reused"})
            lines = [f"synced={synced_count} · total={len(context.zotero_sync_results)}"]
            lines.extend(
                f"- {item.get('title', '')} · {item.get('status', 'unknown')}"
                for item in context.zotero_sync_results[:5]
            )
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Zotero 同步结果",
                    meta=f"{self._manager_display_name()} 调用了 Zotero 同步工具",
                    content="\n".join(lines),
                    payload={"zotero_sync_results": context.zotero_sync_results},
                )
            )
        if context.parsed_document:
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="文档理解结果",
                    meta=f"pages={len(context.parsed_document.pages)} · doc={context.parsed_document.id}",
                    content=(
                        f"已将文档 {context.parsed_document.filename} 解析为 "
                        f"{len(context.parsed_document.pages)} 页，并作为科研助手的证据工具输出。"
                    ),
                    payload={
                        "parsed_document": context.parsed_document.model_dump(mode="json"),
                        "document_index_result": context.document_index_result,
                    },
                )
            )
        if context.chart_result:
            chart = getattr(context.chart_result, "chart", None)
            chart_metadata = getattr(chart, "metadata", {}) or {} if chart is not None else {}
            chart_image_path = str(chart_metadata.get("image_path") or "").strip()
            chart_summary = getattr(chart, "summary", None) or "已完成图表结构化理解。"
            figure_answer = (getattr(context.chart_result, "answer", None) or "").strip()
            if figure_answer:
                display_content = figure_answer
                if chart_image_path:
                    display_content = f"{display_content}\n\n图片路径：{chart_image_path}"
                messages.append(
                    _message(
                        role="assistant",
                        kind="answer",
                        title="论文图表分析",
                        meta=f"chart_type={getattr(chart, 'chart_type', 'unknown')}",
                        content=display_content,
                        payload={
                            "chart": chart.model_dump(mode="json") if hasattr(chart, "model_dump") else None,
                            "graph_text": getattr(context.chart_result, "graph_text", None),
                            "image_path": chart_image_path or None,
                        },
                    )
                )
            else:
                if chart_image_path:
                    chart_summary = f"{chart_summary}\n图片路径：{chart_image_path}"
                messages.append(
                    _message(
                        role="assistant",
                        kind="notice",
                        title="图表理解结果",
                        meta=f"chart_type={getattr(chart, 'chart_type', 'unknown')}",
                        content=chart_summary,
                        payload={
                            "chart": chart.model_dump(mode="json") if hasattr(chart, "model_dump") else None,
                            "graph_text": getattr(context.chart_result, "graph_text", None),
                            "image_path": chart_image_path or None,
                        },
                    )
                )
        if context.qa_result:
            messages.append(
                _message(
                    role="assistant",
                    kind="answer",
                    title="研究集合回答",
                    meta=(
                        f"evidence={len(context.qa_result.qa.evidence_bundle.evidences)} · "
                        f"confidence={context.qa_result.qa.confidence if context.qa_result.qa.confidence is not None else 'empty'}"
                    ),
                    content=context.qa_result.qa.answer,
                    payload={"qa": context.qa_result.qa.model_dump(mode="json")},
                )
            )
        if context.general_answer:
            meta = context.general_answer_metadata or {}
            messages.append(
                _message(
                    role="assistant",
                    kind="answer",
                    title="通用回答",
                    meta=f"route=general_answer · confidence={meta.get('confidence', 'empty')}",
                    content=context.general_answer,
                    payload={"general_answer": meta},
                )
            )
        preference_recommendations_payload = None
        if context.preference_recommendation_result is not None:
            preference_recommendations_payload = context.preference_recommendation_result.model_dump(mode="json")
        if isinstance(preference_recommendations_payload, dict):
            recommendations = list(preference_recommendations_payload.get("recommendations") or [])
            topics_used = list((preference_recommendations_payload.get("metadata") or {}).get("topics_used") or [])
            resolved_sources = list((preference_recommendations_payload.get("metadata") or {}).get("resolved_sources") or [])
            topic_groups = list((preference_recommendations_payload.get("metadata") or {}).get("topic_groups") or [])
            content_lines: list[str] = []
            if topic_groups:
                for group in topic_groups[:4]:
                    topic_name = str(group.get("topic") or "其他").strip() or "其他"
                    papers = list(group.get("papers") or [])
                    content_lines.append(f"主题：{topic_name}")
                    for index, paper in enumerate(papers[:4], start=1):
                        paper_title = str(paper.get("title") or "").strip()
                        paper_source = str(paper.get("source") or "").strip()
                        paper_year = paper.get("year")
                        paper_url = str(paper.get("url") or "").strip()
                        paper_reason = str(paper.get("reason") or "").strip()
                        paper_explanation = str(paper.get("explanation") or "").strip()
                        meta_parts = [str(item) for item in (paper_year, paper_source) if item]
                        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        content_lines.append(f"{index}. {paper_title}{meta}")
                        if paper_url:
                            content_lines.append(f"链接：{paper_url}")
                        if paper_reason:
                            content_lines.append(f"推荐理由：{paper_reason}")
                        if paper_explanation:
                            content_lines.append(f"论文讲解：{paper_explanation}")
                    content_lines.append("")
                if content_lines and not content_lines[-1].strip():
                    content_lines.pop()
            else:
                for index, item in enumerate(recommendations[:5], start=1):
                    title = str(item.get("title") or "").strip()
                    source = str(item.get("source") or "").strip()
                    year = item.get("year")
                    url = str(item.get("url") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    meta_parts = [str(value) for value in (year, source) if value]
                    meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                    content_lines.append(f"{index}. {title}{meta}")
                    if url:
                        content_lines.append(f"链接：{url}")
                    if reason:
                        content_lines.append(f"推荐理由：{reason}")
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="长期兴趣论文推荐",
                    meta=(
                        f"recommended={len(recommendations)}"
                        f"{' · topics=' + ', '.join(topics_used[:3]) if topics_used else ''}"
                        f"{' · sources=' + ', '.join(resolved_sources[:3]) if resolved_sources else ''}"
                    ),
                    content="\n".join(content_lines) or "已生成基于长期兴趣的论文推荐。",
                    payload={"recommendations": preference_recommendations_payload},
                )
            )
        paper_analysis_payload = None
        if context.paper_analysis_result is not None:
            paper_analysis_payload = context.paper_analysis_result.model_dump(mode="json")
        if isinstance(paper_analysis_payload, dict):
            focus = str(paper_analysis_payload.get("focus") or "analysis")
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="论文分析结果",
                    meta=f"focus={focus}",
                    content=str(paper_analysis_payload.get("answer") or "").strip() or "已生成基于所选论文的分析结果。",
                    payload={"paper_analysis": paper_analysis_payload},
                )
            )
        compression_payload = context.compressed_context_summary
        if isinstance(compression_payload, dict):
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="上下文压缩摘要",
                    meta=(
                        f"papers={compression_payload.get('paper_count', 0)} · "
                        f"summaries={compression_payload.get('summary_count', 0)}"
                    ),
                    content="当前研究上下文已经压缩为更短的论文摘要视图，后续 QA、对比和推荐会复用它。",
                    payload={"context_compression": compression_payload},
                )
            )
        if workspace.status_summary or workspace.stop_reason:
            workspace_lines = []
            if workspace.stop_reason:
                workspace_lines.append(f"stop_reason: {workspace.stop_reason}")
            workspace_lines.extend(f"- {item}" for item in workspace.next_actions[:4])
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Research Workspace",
                    meta=workspace.status_summary,
                    content="\n".join(workspace_lines),
                    payload={"workspace": workspace.model_dump(mode="json")},
                )
            )
        trace_lines = [
            f"{step.step_index}. {step.agent} · {step.phase}:{step.action_name} · {step.status} · {step.observation}"
            for step in trace
        ]
        messages.append(
            _message(
                role="assistant",
                kind="notice",
                title="Agent 决策轨迹",
                meta=f"{len(trace)} step(s)",
                content="\n".join(trace_lines),
                payload={"trace": [step.model_dump(mode="json") for step in trace]},
            )
        )
        return messages

    def _next_actions(
        self,
        context: ResearchAgentToolContext,
        workspace: ResearchWorkspaceState,
        *,
        clarification_request: str | None = None,
    ) -> list[str]:
        actions: list[str] = list(workspace.next_actions)
        task = context.task
        if clarification_request:
            actions.insert(0, "补充更具体的研究子方向、评价维度、应用场景或时间范围后再继续。")
        if task and context.papers:
            actions.append(
                f"继续追问这个研究集合，{self._manager_display_name()} 会基于已导入文献和候选论文池回答。"
            )
        if task and not task.imported_document_ids:
            actions.append("导入开放 PDF 后可以获得更强的 grounded QA 证据。")
        if task and task.todo_items:
            actions.append("执行或关闭自动 TODO，让研究空间持续补证据。")
        if context.parsed_document:
            actions.append("可以继续围绕刚解析的文档提问，或让助手补充相关领域文献。")
        if context.chart_result:
            actions.append("可以让助手把图表结论和相关论文证据合并分析。")
        if context.preference_recommendation_result is not None:
            actions.append("可以继续追问推荐列表里的某篇论文，或让助手按其中一个主题继续做深入调研。")
        if workspace.metadata.get("latest_paper_analysis"):
            actions.append("可以继续基于这组论文追问实验差异、适用场景、失败边界或下一步阅读建议。")
        if not actions:
            actions.append("换一个更具体的研究目标，或扩大时间窗口和数据源。")
        deduped: list[str] = []
        for action in actions:
            normalized = action.strip()
            if not normalized or normalized in deduped:
                continue
            deduped.append(normalized)
        return deduped[:5]

    def _resolved_advanced_strategy(
        self,
        context: ResearchAgentToolContext,
        *,
        workspace: ResearchWorkspaceState,
    ) -> ResearchAdvancedStrategy:
        base_strategy = self._read_advanced_strategy(context.workspace or workspace)
        request_metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        updated = base_strategy.model_copy(deep=True)
        metadata_dimensions = request_metadata.get("comparison_dimensions")
        explicit_dimensions = [
            str(item).strip().lower()
            for item in [
                *context.request.comparison_dimensions,
                *(metadata_dimensions if isinstance(metadata_dimensions, list) else []),
            ]
            if str(item).strip()
        ]
        explicit_goal = str(
            context.request.recommendation_goal
            or request_metadata.get("recommendation_goal")
            or ""
        ).strip()
        explicit_action = context.request.advanced_action
        if explicit_action is not None:
            updated.action = explicit_action
        if explicit_action is None and (
            context.paper_analysis_result is not None or workspace.metadata.get("latest_paper_analysis")
        ):
            updated.action = "analyze"
        if explicit_dimensions:
            updated.comparison_dimensions = list(dict.fromkeys(explicit_dimensions))
        elif updated.action in {"analyze", "compare"} and not updated.comparison_dimensions:
            comparison_payload = workspace.metadata.get("latest_comparison")
            if isinstance(comparison_payload, dict):
                updated.comparison_dimensions = list(
                    dict.fromkeys(
                        str(item.get("dimension")).strip().lower()
                        for item in comparison_payload.get("table", [])
                        if isinstance(item, dict) and str(item.get("dimension") or "").strip()
                    )
                )
        if explicit_goal:
            updated.recommendation_goal = explicit_goal
        if explicit_action in {"analyze", "recommend"} or "recommendation_top_k" in request_metadata:
            updated.recommendation_top_k = max(
                1,
                min(
                    10,
                    int(
                        context.request.recommendation_top_k
                        or request_metadata.get("recommendation_top_k")
                        or updated.recommendation_top_k
                        or 3
                    ),
                ),
            )
        if (
            explicit_action in {"analyze", "compare", "recommend"}
            or "force_context_compression" in request_metadata
            or context.request.force_context_compression
        ):
            updated.force_context_compression = bool(
                context.request.force_context_compression
                or request_metadata.get("force_context_compression")
            )
        return updated

    def _read_advanced_strategy(
        self,
        workspace: ResearchWorkspaceState | None,
    ) -> ResearchAdvancedStrategy:
        if workspace is None:
            return ResearchAdvancedStrategy()
        payload = workspace.metadata.get("advanced_strategy")
        if not isinstance(payload, dict):
            return ResearchAdvancedStrategy()
        try:
            return ResearchAdvancedStrategy.model_validate(payload)
        except Exception:
            return ResearchAdvancedStrategy()

    def _derive_workspace_stage(self, context: ResearchAgentToolContext) -> str:
        if context.workspace is not None and context.workspace.current_stage:
            if (
                context.workspace.metadata.get("latest_paper_analysis")
                or context.workspace.metadata.get("latest_comparison")
                or context.workspace.metadata.get("latest_recommendations")
                or context.workspace.metadata.get("latest_preference_recommendations")
            ):
                return "complete"
            return context.workspace.current_stage
        if context.request.mode == "document" or context.parsed_document is not None:
            return "document"
        if context.request.mode == "chart" or context.chart_result is not None:
            return "chart"
        if context.qa_result is not None:
            return "qa"
        if context.preference_recommendation_result is not None:
            return "complete"
        if context.import_result is not None:
            return "ingest"
        if context.task is not None:
            return "qa" if context.task.imported_document_ids else "ingest" if context.papers else "discover"
        return "discover"

    def _workspace_summary(self, context: ResearchAgentToolContext) -> str:
        if context.workspace is not None and context.workspace.status_summary:
            return context.workspace.status_summary
        paper_count = len(context.papers)
        imported_document_count = len(context.task.imported_document_ids) if context.task else 0
        open_todo_count = sum(1 for item in (context.task.todo_items if context.task else []) if item.status == "open")
        gap_count = len(context.workspace.evidence_gaps) if context.workspace is not None else len(context.report.gaps) if context.report else 0
        return (
            f"stage={self._derive_workspace_stage(context)}; papers={paper_count}; "
            f"imported_docs={imported_document_count}; open_todos={open_todo_count}; evidence_gaps={gap_count}"
        )

    def _stop_reason_from_trace(
        self,
        trace: list[ResearchAgentTraceStep],
        *,
        failed: bool,
        exhausted: bool,
    ) -> str | None:
        for step in reversed(trace):
            stop_reason = step.metadata.get("stop_reason") if isinstance(step.metadata, dict) else None
            if isinstance(stop_reason, str) and stop_reason.strip():
                return stop_reason
        if failed:
            return "The manager stopped because a tool action failed and returned a partial workspace."
        if exhausted:
            return f"The manager reached max_steps={self.max_steps} before completing the research loop."
        return None

    def _workspace_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_engine": "llm_dynamic_decision_loop",
            "routing_engine": "llm_native_multi_agent",
        }

    def _response_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_agent": self._manager_display_name(),
            "manager_engine": "llm_dynamic_decision_loop",
            "routing_engine": "llm_native_multi_agent",
            "autonomy_mode": "manager_delegates_to_worker_agents",
            "agent_architecture": "llm_native_manager_worker_agents",
            "primary_agents": [
                "ResearchSupervisorAgent",
                "GeneralAnswerAgent",
                "ChartAnalysisAgent",
                "LiteratureScoutAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
                "PaperAnalysisAgent",
                "PreferenceMemoryAgent",
            ],
            "primary_runtime_workers": [],
            "primary_skills": [
                "PaperCurator",
                "TopicPlanner",
                "ResearchQueryRewriter",
                "PaperRanker",
                "SurveyWriter",
                "PaperAnalyzer",
            ],
        }

    def _resolve_workspace(
        self,
        *,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        failed: bool,
        exhausted: bool,
    ) -> ResearchWorkspaceState:
        stage = self._derive_workspace_stage(context)
        stop_reason = self._stop_reason_from_trace(trace, failed=failed, exhausted=exhausted) or (
            context.workspace.stop_reason if context.workspace is not None else None
        )
        metadata = {**self._workspace_runtime_metadata(), "trace_steps": len(trace)}
        extra_questions = [context.request.message] if context.request.message else []
        extra_findings = [context.qa_result.qa.answer] if context.qa_result is not None else []
        if context.preference_recommendation_result is not None:
            metadata["latest_preference_recommendations"] = context.preference_recommendation_result.model_dump(mode="json")
        if context.workspace is not None:
            next_actions = list(context.workspace.next_actions)
            if stop_reason:
                stop_action = f"Stop reason: {stop_reason}"
                if stop_action not in next_actions:
                    next_actions.append(stop_action)
            return context.workspace.model_copy(
                update={
                    "current_stage": stage,
                    "stop_reason": stop_reason,
                    "status_summary": context.workspace.status_summary or self._workspace_summary(context),
                    "next_actions": next_actions[:4],
                    "metadata": {**context.workspace.metadata, **metadata},
                }
            )
        if context.task is not None or context.report is not None:
            return build_workspace_from_task(
                task=context.task,
                report=context.report,
                papers=context.papers,
                extra_questions=extra_questions,
                extra_findings=extra_findings,
                stage=stage,  # type: ignore[arg-type]
                stop_reason=stop_reason,
                metadata=metadata,
            )
        return build_workspace_state(
            objective=context.request.message,
            stage=stage,  # type: ignore[arg-type]
            papers=context.papers,
            imported_document_ids=context.task.imported_document_ids if context.task else [],
            report=context.report,
            todo_items=context.task.todo_items if context.task else [],
            extra_questions=extra_questions,
            extra_findings=extra_findings,
            stop_reason=stop_reason,
            metadata=metadata,
        )


class ResearchSupervisorGraphRuntime(ResearchRuntimeBase):
    """High-level supervisor runtime backed by a LangGraph loop."""

    def __init__(self, **kwargs: Any) -> None:
        from langgraph.graph import END, StateGraph

        super().__init__(**kwargs)
        graph = StateGraph(ResearchAgentGraphState)
        graph.add_node("bootstrap_context_node", self.bootstrap_context_node)
        graph.add_node("supervisor_node", self.supervisor_node)
        graph.add_node("literature_scout_node", self.literature_scout_node)
        graph.add_node("write_review_node", self.write_review_specialist_node)
        graph.add_node("paper_import_node", self.paper_import_specialist_node)
        graph.add_node("zotero_sync_node", self.zotero_sync_specialist_node)
        graph.add_node("research_qa_node", self.research_qa_specialist_node)
        graph.add_node("general_answer_node", self.general_answer_specialist_node)
        graph.add_node("preference_memory_node", self.preference_memory_specialist_node)
        graph.add_node("paper_analysis_node", self.paper_analysis_specialist_node)
        graph.add_node("context_compression_node", self.context_compression_specialist_node)
        graph.add_node("document_specialist_node", self.document_specialist_node)
        graph.add_node("chart_specialist_node", self.chart_specialist_node)
        graph.add_node("paper_figure_analysis_node", self.paper_figure_analysis_specialist_node)
        graph.add_node("finalize_node", self.finalize_node)
        graph.add_edge("__start__", "bootstrap_context_node")
        graph.add_edge("finalize_node", END)
        self.graph = graph.compile()

    async def run(self, request: ResearchAgentRunRequest, *, graph_runtime: Any, on_progress: Any | None = None) -> ResearchAgentRunResponse:
        context = self._build_tool_context(request=request, graph_runtime=graph_runtime)
        if on_progress is not None:
            context.progress_callback = on_progress
        _update_runtime_progress(
            context,
            stage="supervisor_graph",
            node="bootstrap_context_node",
            status="started",
            summary="Supervisor graph initialized.",
        )
        initial_state: ResearchAgentGraphState = {
            "context": context,
            "trace": [],
            "failed": False,
            "exhausted": False,
            "current_step_index": 0,
            "pending_agent_messages": [],
            "agent_messages": [],
            "agent_results": [],
            "completed_agent_task_ids": [],
            "failed_agent_task_ids": [],
            "replanned_failure_task_ids": [],
            "planner_runs": 0,
            "replan_count": 0,
            "clarification_request": None,
            "active_plan_id": None,
            "progress_signature": "",
            "stagnant_decision_count": 0,
            "repeated_action_count": 0,
        }
        heartbeat_task = asyncio.create_task(self._heartbeat_runtime_progress(context))
        try:
            graph_state = await self.graph.ainvoke(initial_state)
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if graph_state.get("exhausted"):
            context.warnings.append(f"research supervisor graph reached max_steps={self.max_steps} before finalize")
        return self._build_response(
            request=request,
            context=context,
            trace=graph_state.get("trace", []),
            failed=bool(graph_state.get("failed")),
            exhausted=bool(graph_state.get("exhausted")),
            agent_messages=graph_state.get("agent_messages", []),
            agent_results=graph_state.get("agent_results", []),
            planner_runs=int(graph_state.get("planner_runs", 0) or 0),
            replan_count=int(graph_state.get("replan_count", 0) or 0),
            clarification_request=graph_state.get("clarification_request"),
            active_plan_id=graph_state.get("active_plan_id"),
        )

    async def _heartbeat_runtime_progress(self, context: ResearchAgentToolContext) -> None:
        while True:
            await asyncio.sleep(5.0)
            progress = dict(context.runtime_progress or {})
            if not progress:
                continue
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
                    "runtime_event": "supervisor_heartbeat",
                    **progress,
                },
            )

    async def _decide_next_action(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        decision = await super()._decide_next_action(state)
        metadata = dict(decision.metadata)
        if metadata.get("worker_agent") == "ResearchSupervisorAgent":
            metadata["worker_agent"] = self._manager_trace_agent_name()
        return ResearchSupervisorDecision(
            action_name=decision.action_name,
            thought=decision.thought,
            rationale=decision.rationale,
            phase=decision.phase,
            estimated_gain=decision.estimated_gain,
            estimated_cost=decision.estimated_cost,
            stop_reason=decision.stop_reason,
            action_input=dict(decision.action_input),
            metadata=metadata,
        )

    def _workspace_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_engine": "langgraph_supervisor",
            "routing_engine": "langgraph_supervisor_specialists",
        }

    def _manager_display_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _manager_trace_agent_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _response_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_agent": self._manager_display_name(),
            "manager_engine": "langgraph_supervisor",
            "routing_engine": "langgraph_supervisor_specialists",
            "autonomy_mode": "supervisor_delegates_to_specialists",
            "agent_architecture": "langgraph_supervisor_multi_agent",
            "primary_agents": [
                "ResearchSupervisorAgent",
                "GeneralAnswerAgent",
                "ChartAnalysisAgent",
                "LiteratureScoutAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
                "PaperAnalysisAgent",
                "PreferenceMemoryAgent",
            ],
            "primary_runtime_workers": [],
            "primary_skills": [
                "PaperCurator",
                "TopicPlanner",
                "ResearchQueryRewriter",
                "PaperRanker",
                "SurveyWriter",
                "PaperAnalyzer",
            ],
            "supervisor_graph": {
                "entry_node": "bootstrap_context_node",
                "decision_node": "supervisor_node",
                "specialist_nodes": [
                    "literature_scout_node",
                    "write_review_node",
                    "paper_import_node",
                    "zotero_sync_node",
                    "research_qa_node",
                    "general_answer_node",
                    "preference_memory_node",
                    "paper_analysis_node",
                    "context_compression_node",
                    "document_specialist_node",
                    "chart_specialist_node",
                    "paper_figure_analysis_node",
                ],
                "finalize_node": "finalize_node",
            },
        }

    async def bootstrap_context_node(self, state: ResearchAgentGraphState):
        from langgraph.types import Command

        _update_runtime_progress(
            state["context"],
            stage="supervisor_graph",
            node="bootstrap_context_node",
            status="running",
            summary="Bootstrapping supervisor graph context.",
        )

        return Command(
            update={
                "progress_signature": self._progress_signature(state),
                "trace": list(state.get("trace", [])),
            },
            goto="supervisor_node",
        )

    async def supervisor_node(self, state: ResearchAgentGraphState):
        from langgraph.types import Command

        _update_runtime_progress(
            state["context"],
            stage="supervisor_graph",
            node="supervisor_node",
            status="running",
            summary="Manager is choosing the next action.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0) + 1},
        )
        current_step_index = int(state.get("current_step_index", 0) or 0)
        if current_step_index >= self.max_steps:
            return Command(update={"exhausted": True}, goto="finalize_node")
        if self._should_force_finalize(state):
            decision = self._forced_finalize_decision(state)
            update = self._on_decision(state, current_step_index + 1, decision)
            return Command(
                update={
                    "current_step_index": current_step_index + 1,
                    "current_decision": decision,
                    **update,
                },
                goto="finalize_node",
            )
        step_index = current_step_index + 1
        decision = await self._decide_next_action(state)
        update = self._on_decision(state, step_index, decision)
        goto = self._goto_after_supervisor(decision)
        return Command(
            update={
                "current_step_index": step_index,
                "current_decision": decision,
                **update,
            },
            goto=goto,
        )

    async def literature_scout_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.search_literature_node)

    async def write_review_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.write_review_node)

    async def paper_import_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.import_papers_node)

    async def zotero_sync_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.sync_to_zotero_node)

    async def research_qa_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.answer_question_node)

    async def general_answer_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.general_answer_node)

    async def preference_memory_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.recommend_from_preferences_node)

    async def paper_analysis_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.analyze_papers_node)

    async def context_compression_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.compress_context_node)

    async def document_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.understand_document_node)

    async def chart_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.understand_chart_node)

    async def paper_figure_analysis_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.analyze_paper_figures_node)

    async def _run_specialist_and_route(self, state: ResearchAgentGraphState, runner):
        from langgraph.types import Command

        update = await runner(state)
        next_node = "finalize_node" if self._should_force_finalize({**state, **update}) else "supervisor_node"
        return Command(update=update, goto=next_node)

    def _goto_after_supervisor(self, decision: ResearchSupervisorDecision) -> str:
        action = self._route_decision(decision)
        mapping = {
            SearchLiteratureTool.name: "literature_scout_node",
            WriteReviewTool.name: "write_review_node",
            ImportPapersTool.name: "paper_import_node",
            SyncToZoteroTool.name: "zotero_sync_node",
            AnswerQuestionTool.name: "research_qa_node",
            GeneralAnswerTool.name: "general_answer_node",
            RecommendFromPreferencesTool.name: "preference_memory_node",
            AnalyzePapersTool.name: "paper_analysis_node",
            CompressContextTool.name: "context_compression_node",
            UnderstandDocumentTool.name: "document_specialist_node",
            UnderstandChartTool.name: "chart_specialist_node",
            AnalyzePaperFiguresTool.name: "paper_figure_analysis_node",
            "finalize": "finalize_node",
        }
        return mapping[action]
