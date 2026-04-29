from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.agent_message import (
    AgentMessage,
    AgentResultMessage,
    AgentTaskPriority,
    AgentTaskStatus,
)
from domain.schemas.research import (
    CreateResearchTaskRequest,
    PaperCandidate,
    ResearchReport,
    ResearchTaskAskRequest,
)
from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.sub_manager import TaskEvaluation


UnifiedAgentKind = Literal["orchestrator", "specialist", "runtime"]
UnifiedExecutionMode = Literal["tool_native", "service_native", "hybrid"]
UNIFIED_ACTION_OUTPUT_METADATA_KEY = "unified_action_output"


class UnifiedCapabilityBinding(BaseModel):
    profile_name: str | None = None
    service_names: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class UnifiedAgentDescriptor(BaseModel):
    name: str
    description: str
    kind: UnifiedAgentKind = "specialist"
    execution_mode: UnifiedExecutionMode = "hybrid"
    supported_task_types: list[str] = Field(default_factory=list)
    capability_binding: UnifiedCapabilityBinding = Field(default_factory=UnifiedCapabilityBinding)
    preferred_tool_names: list[str] = Field(default_factory=list)
    available_tool_names: list[str] = Field(default_factory=list)
    legacy_boundaries: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class UnifiedAgentTask(BaseModel):
    task_id: str
    agent_from: str
    agent_to: str
    task_type: str
    instruction: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    context_slice: ResearchContextSlice | dict[str, Any] = Field(default_factory=dict)
    priority: AgentTaskPriority = "medium"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=10)
    preferred_skill_name: str | None = None
    available_tool_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent_message(
        cls,
        message: AgentMessage,
        *,
        preferred_skill_name: str | None = None,
        available_tool_names: list[str] | None = None,
    ) -> "UnifiedAgentTask":
        skill_name = preferred_skill_name
        if skill_name is None:
            raw_value = message.metadata.get("skill_name")
            if isinstance(raw_value, str) and raw_value.strip():
                skill_name = raw_value
        return cls(
            task_id=message.task_id,
            agent_from=message.agent_from,
            agent_to=message.agent_to,
            task_type=message.task_type,
            instruction=message.instruction,
            payload=dict(message.payload),
            context_slice=message.context_slice,
            priority=message.priority,
            expected_output_schema=dict(message.expected_output_schema),
            depends_on=list(message.depends_on),
            retry_count=message.retry_count,
            preferred_skill_name=skill_name,
            available_tool_names=list(available_tool_names or []),
            metadata=dict(message.metadata),
        )


class UnifiedAgentResult(BaseModel):
    task_id: str
    agent_name: str
    task_type: str
    status: AgentTaskStatus
    instruction: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    context_slice: ResearchContextSlice | dict[str, Any] = Field(default_factory=dict)
    priority: AgentTaskPriority = "medium"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=10)
    evaluation: TaskEvaluation | None = None
    action_output: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent_result_message(cls, result: AgentResultMessage) -> "UnifiedAgentResult":
        action_output = cls.extract_action_output(
            payload=result.payload,
            metadata=result.metadata,
        )
        return cls(
            task_id=result.task_id,
            agent_name=result.agent_from,
            task_type=result.task_type,
            status=result.status,
            instruction=result.instruction,
            payload=dict(result.payload),
            context_slice=result.context_slice,
            priority=result.priority,
            expected_output_schema=dict(result.expected_output_schema),
            depends_on=list(result.depends_on),
            retry_count=result.retry_count,
            evaluation=result.evaluation,
            action_output=action_output,
            metadata=dict(result.metadata),
        )

    def to_agent_result_message(self, *, reply_to: str = "ResearchSupervisorAgent") -> AgentResultMessage:
        metadata = dict(self.metadata)
        if self.action_output:
            metadata.setdefault(
                UNIFIED_ACTION_OUTPUT_METADATA_KEY,
                dict(self.action_output),
            )
        return AgentResultMessage(
            task_id=self.task_id,
            agent_from=self.agent_name,
            agent_to=reply_to,
            task_type=self.task_type,
            status=self.status,
            instruction=self.instruction,
            payload=dict(self.payload),
            context_slice=self.context_slice,
            priority=self.priority,
            expected_output_schema=dict(self.expected_output_schema),
            depends_on=list(self.depends_on),
            retry_count=self.retry_count,
            evaluation=self.evaluation,
            metadata=metadata,
        )

    @staticmethod
    def is_action_output_payload(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        adapter_name = value.get("unified_input_adapter")
        return isinstance(adapter_name, str) and bool(adapter_name.strip())

    @classmethod
    def extract_action_output(
        cls,
        *,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        candidates: list[Any] = []
        if isinstance(metadata, dict):
            candidates.append(metadata.get(UNIFIED_ACTION_OUTPUT_METADATA_KEY))
        if isinstance(payload, dict):
            candidates.append(payload.get(UNIFIED_ACTION_OUTPUT_METADATA_KEY))
            candidates.append(payload.get("action_output"))
            candidates.append(payload.get("tool_metadata"))
            candidates.append(payload)
        for candidate in candidates:
            if cls.is_action_output_payload(candidate):
                return dict(candidate)
        return None


class UnifiedRuntimeBlueprint(BaseModel):
    name: str = "research-copilot-unified-runtime-phase1"
    target_state: str = (
        "skill profiles select behavior, a single agent envelope carries tasks, "
        "and all executable capabilities converge on ToolRegistry/ToolExecutor."
    )
    agent_descriptors: list[UnifiedAgentDescriptor] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    capability_profile_names: list[str] = Field(default_factory=list)
    unresolved_boundaries: list[str] = Field(default_factory=list)
    migration_stages: list[str] = Field(default_factory=list)


class UnifiedActionOutput(BaseModel):
    unified_input_adapter: str

    def to_metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class UnifiedReviewDraftInput(BaseModel):
    topic: str
    task_id: str
    curated_papers: list[PaperCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    must_read_ids: list[str] = Field(default_factory=list)
    ingest_candidate_ids: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    round_index: int = 0
    refinement_used: bool = False
    max_papers: int = Field(default=1, ge=1)
    report: ResearchReport | None = None


class UnifiedLiteratureSearchInput(BaseModel):
    topic: str
    days_back: int = Field(default=90, ge=1, le=3650)
    max_papers: int = Field(default=15, ge=1, le=100)
    sources: list[str] = Field(default_factory=list)
    run_immediately: bool = True
    conversation_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list)
    skill_name: str | None = None
    reasoning_style: str | None = "cot"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_create_research_task_request(self) -> CreateResearchTaskRequest:
        return CreateResearchTaskRequest(
            topic=self.topic,
            days_back=self.days_back,
            max_papers=self.max_papers,
            sources=list(self.sources),
            run_immediately=self.run_immediately,
            conversation_id=self.conversation_id,
        )


class UnifiedCollectionQAInput(BaseModel):
    task_id: str
    question: str
    top_k: int = Field(default=10, ge=1, le=100)
    paper_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    image_path: str | None = None
    page_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    chart_id: str | None = None
    return_citations: bool = True
    min_length: int = Field(default=600, ge=100, le=5000)
    skill_name: str | None = None
    reasoning_style: str | None = "cot"
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None

    def to_research_task_ask_request(self) -> ResearchTaskAskRequest:
        return ResearchTaskAskRequest(
            question=self.question,
            top_k=self.top_k,
            paper_ids=list(self.paper_ids),
            document_ids=list(self.document_ids),
            image_path=self.image_path,
            page_id=self.page_id,
            page_number=self.page_number,
            chart_id=self.chart_id,
            return_citations=self.return_citations,
            min_length=self.min_length,
            skill_name=self.skill_name,
            reasoning_style=self.reasoning_style,
            metadata=dict(self.metadata),
            conversation_id=self.conversation_id,
        )


class UnifiedPaperImportInput(BaseModel):
    task_id: str
    paper_ids: list[str] = Field(default_factory=list)
    selected_paper_ids: list[str] = Field(default_factory=list)
    import_top_k: int = Field(default=3, ge=0, le=20)
    include_graph: bool = True
    include_embeddings: bool = True
    skill_name: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_paper_ids(self, papers: list[PaperCandidate]) -> list[str]:
        if self.paper_ids:
            allowed = set(self.paper_ids)
            return [
                paper.paper_id
                for paper in papers
                if paper.paper_id in allowed and paper.ingest_status not in {"ingested", "unavailable"}
            ]
        if self.selected_paper_ids:
            allowed = set(self.selected_paper_ids)
            return [
                paper.paper_id
                for paper in papers
                if paper.paper_id in allowed and paper.ingest_status not in {"ingested", "unavailable"}
            ]
        if self.import_top_k <= 0:
            return []
        candidates = [
            paper
            for paper in papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        ]
        candidates.sort(
            key=lambda paper: (
                float(paper.relevance_score or 0.0),
                int(paper.year or 0),
                int(paper.citations or 0),
            ),
            reverse=True,
        )
        return [paper.paper_id for paper in candidates[: self.import_top_k]]


class UnifiedDocumentUnderstandingInput(BaseModel):
    file_path: str = ""
    document_id: str | None = None
    include_graph: bool = True
    include_embeddings: bool = True
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None


class UnifiedChartUnderstandingInput(BaseModel):
    image_path: str = ""
    document_id: str
    page_id: str
    page_number: int | None = Field(default=None, ge=1)
    chart_id: str
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None


class UnifiedContextCompressionInput(BaseModel):
    task_id: str
    selected_paper_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_selected_paper_ids(self) -> list[str]:
        if self.paper_ids:
            return list(dict.fromkeys(self.paper_ids))
        return list(dict.fromkeys(self.selected_paper_ids))


class UnifiedPaperAnalysisInput(BaseModel):
    question: str
    analysis_focus: str | None = None
    comparison_dimensions: list[str] = Field(default_factory=list)
    recommendation_goal: str | None = None
    papers: list[PaperCandidate] = Field(default_factory=list)
    task_topic: str = ""
    report_highlights: list[str] = Field(default_factory=list)

    def resolved_question(self) -> str:
        focus = (self.analysis_focus or "").strip().lower()
        if focus == "recommend" and self.recommendation_goal:
            return self.recommendation_goal.strip()
        if focus == "compare" and self.comparison_dimensions:
            dimensions = ", ".join(item.strip() for item in self.comparison_dimensions if item.strip())
            if dimensions:
                return f"{self.question.strip()}，重点比较：{dimensions}"
        return self.question.strip()


class UnifiedLiteratureSearchOutput(UnifiedActionOutput):
    unified_input_adapter: str = "literature_search_input"
    task_id: str
    paper_count: int = Field(default=0, ge=0)
    report_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class UnifiedReviewDraftOutput(UnifiedActionOutput):
    unified_input_adapter: str = "review_draft_input"
    task_id: str
    report_id: str
    report_word_count: int = Field(default=0, ge=0)
    report_has_citations: bool = False
    report_has_key_sections: bool = False
    retry_count: int = Field(default=0, ge=0)
    issues: list[str] = Field(default_factory=list)


class UnifiedCollectionQAOutput(UnifiedActionOutput):
    unified_input_adapter: str = "collection_qa_input"
    task_id: str
    paper_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    confidence: float | None = None


class UnifiedPaperImportOutput(UnifiedActionOutput):
    unified_input_adapter: str = "paper_import_input"
    paper_ids: list[str] = Field(default_factory=list)
    imported_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)


class UnifiedDocumentUnderstandingOutput(UnifiedActionOutput):
    unified_input_adapter: str = "document_understanding_input"
    document_id: str
    page_count: int = Field(default=0, ge=0)
    index_status: str | None = None


class UnifiedChartUnderstandingOutput(UnifiedActionOutput):
    unified_input_adapter: str = "chart_understanding_input"
    chart_id: str
    chart_type: str | None = None
    document_id: str


class UnifiedPaperAnalysisOutput(UnifiedActionOutput):
    unified_input_adapter: str = "paper_analysis_input"
    task_id: str
    paper_count: int = Field(default=0, ge=0)
    analyzed_paper_ids: list[str] = Field(default_factory=list)
    analysis_focus: str
    recommended_paper_ids: list[str] = Field(default_factory=list)


class UnifiedContextCompressionOutput(UnifiedActionOutput):
    unified_input_adapter: str = "context_compression_input"
    paper_count: int = Field(default=0, ge=0)
    summary_count: int = Field(default=0, ge=0)
    levels: list[str] = Field(default_factory=list)
    compressed_paper_ids: list[str] = Field(default_factory=list)
