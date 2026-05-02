from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import BoundingBox, ParsedDocument


PaperSource = Literal["arxiv", "openalex", "semantic_scholar", "ieee", "zotero"]

DEFAULT_AGENT_REASONING_STYLE = "react"


def normalize_reasoning_style(style: str | None) -> str:
    normalized = (style or DEFAULT_AGENT_REASONING_STYLE).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return DEFAULT_AGENT_REASONING_STYLE
    aliases = {
        "chain_of_thought": "react",
        "chainofthought": "react",
        "cot": "react",
        "planandsolve": "plan_and_execute",
        "plan_and_solve": "plan_and_execute",
        "plan_and_execute": "plan_and_execute",
        "react": "react",
        "auto": "auto",
    }
    return aliases.get(normalized, normalized)

ResearchTaskStatus = Literal["created", "running", "completed", "failed"]
PaperIngestStatus = Literal["not_selected", "selected", "ingested", "unavailable"]
ResearchTodoStatus = Literal["open", "done", "dismissed"]
ResearchTodoPriority = Literal["high", "medium", "low"]
ResearchComposerMode = Literal["research", "qa"]
ResearchWorkspaceStage = Literal["discover", "ingest", "qa", "document", "chart", "complete"]
ResearchDecisionPhase = Literal["observe", "plan", "act", "reflect", "commit"]
ResearchMessageRole = Literal["assistant", "user", "system"]
ResearchMessageKind = Literal[
    "welcome",
    "topic",
    "report",
    "candidates",
    "import_result",
    "question",
    "answer",
    "notice",
    "warning",
    "error",
    "job",
]
ResearchJobKind = Literal["paper_import", "todo_import"]
ResearchJobStatus = Literal["queued", "running", "completed", "failed"]
ResearchAgentMode = Literal["auto", "research", "qa", "import", "document", "chart"]
ResearchAgentActionStatus = Literal["planned", "succeeded", "failed", "skipped"]
ResearchAdvancedAction = Literal["discover", "analyze", "compare", "recommend"]
ResearchLifecycleStatus = Literal["queued", "running", "waiting_input", "completed", "failed", "cancelled"]
ResearchRouteMode = Literal[
    "general_chat",
    "research_discovery",
    "research_follow_up",
    "paper_follow_up",
    "document_drilldown",
    "chart_drilldown",
]
ResearchRuntimeEventType = Literal[
    "agent_started",
    "agent_routed",
    "tool_called",
    "tool_succeeded",
    "tool_failed",
    "memory_updated",
    "task_completed",
    "task_failed",
]


class ResearchStatusMetadata(BaseModel):
    lifecycle_status: ResearchLifecycleStatus = "queued"
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = Field(default=0, ge=0)
    correlation_id: str | None = None


class ResearchContextSummary(BaseModel):
    summary_version: int = Field(default=1, ge=1)
    objective: str = ""
    current_stage: ResearchWorkspaceStage = "discover"
    topic: str | None = None
    paper_count: int = Field(default=0, ge=0)
    imported_document_count: int = Field(default=0, ge=0)
    selected_paper_count: int = Field(default=0, ge=0)
    key_findings: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    status_summary: str = ""
    last_user_message: str | None = None
    last_updated_at: str | None = None


class ResearchRuntimeEvent(BaseModel):
    event_id: str
    event_type: ResearchRuntimeEventType
    task_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchTopicPlan(BaseModel):
    topic: str
    normalized_topic: str
    queries: list[str] = Field(default_factory=list)
    days_back: int = Field(default=90, ge=1, le=3650)
    max_papers: int = Field(default=15, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperCandidate(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    source: PaperSource
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    url: str | None = None
    citations: int | None = None
    is_open_access: bool | None = None
    published_at: str | None = None
    relevance_score: float | None = Field(default=None, ge=0)
    summary: str | None = None
    ingest_status: PaperIngestStatus = "not_selected"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchCluster(BaseModel):
    name: str
    paper_ids: list[str] = Field(default_factory=list)
    description: str | None = None


class ResearchWorkspaceState(BaseModel):
    objective: str = ""
    current_stage: ResearchWorkspaceStage = "discover"
    research_questions: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    must_read_paper_ids: list[str] = Field(default_factory=list)
    ingest_candidate_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    status_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchReport(BaseModel):
    report_id: str
    task_id: str | None = None
    topic: str
    generated_at: str
    markdown: str
    paper_count: int = Field(default=0, ge=0)
    source_counts: dict[str, int] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list)
    clusters: list[ResearchCluster] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    workspace: ResearchWorkspaceState = Field(default_factory=ResearchWorkspaceState)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTodoItem(BaseModel):
    todo_id: str
    content: str
    rationale: str | None = None
    status: ResearchTodoStatus = "open"
    priority: ResearchTodoPriority = "medium"
    created_at: str
    question: str | None = None
    source: Literal["qa_follow_up", "evidence_gap"] = "qa_follow_up"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTask(BaseModel):
    task_id: str
    topic: str
    status: ResearchTaskStatus = "created"
    created_at: str
    updated_at: str
    days_back: int = Field(default=90, ge=1, le=3650)
    max_papers: int = Field(default=15, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=lambda: ["arxiv", "openalex"])
    paper_count: int = Field(default=0, ge=0)
    imported_document_ids: list[str] = Field(default_factory=list)
    todo_items: list[ResearchTodoItem] = Field(default_factory=list)
    report_id: str | None = None
    workspace: ResearchWorkspaceState = Field(default_factory=ResearchWorkspaceState)
    status_metadata: ResearchStatusMetadata = Field(default_factory=ResearchStatusMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchPapersRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    days_back: int = Field(default=90, ge=1, le=3650)
    max_papers: int = Field(default=15, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=lambda: ["arxiv", "openalex"])
    conversation_id: str | None = None


class SearchPapersResponse(BaseModel):
    plan: ResearchTopicPlan
    papers: list[PaperCandidate] = Field(default_factory=list)
    report: ResearchReport
    warnings: list[str] = Field(default_factory=list)


class CreateResearchTaskRequest(SearchPapersRequest):
    run_immediately: bool = True


class ResearchTaskResponse(BaseModel):
    task: ResearchTask
    papers: list[PaperCandidate] = Field(default_factory=list)
    report: ResearchReport | None = None
    warnings: list[str] = Field(default_factory=list)


class ImportPapersRequest(BaseModel):
    task_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    papers: list[PaperCandidate] = Field(default_factory=list)
    include_graph: bool = True
    include_embeddings: bool = True
    fast_mode: bool = True
    skill_name: str | None = None
    conversation_id: str | None = None
    question: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    reasoning_style: str | None = "cot"


class ImportedPaperResult(BaseModel):
    paper_id: str
    title: str
    status: Literal["imported", "skipped", "failed"]
    document_id: str | None = None
    storage_uri: str | None = None
    parsed: bool = False
    indexed: bool = False
    graph_pending: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportPapersResponse(BaseModel):
    results: list[ImportedPaperResult] = Field(default_factory=list)
    imported_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)


class ResearchPaperFigurePreview(BaseModel):
    figure_id: str
    paper_id: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    chart_id: str
    title: str | None = None
    caption: str | None = None
    source: Literal["chart_candidate", "page_fallback"] = "chart_candidate"
    bbox: BoundingBox | None = None
    image_path: str | None = None
    preview_data_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPaperFigureListResponse(BaseModel):
    task_id: str
    paper_id: str
    document_id: str
    figures: list[ResearchPaperFigurePreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResearchPaperFigureRequest(BaseModel):
    figure_id: str | None = None
    page_id: str
    chart_id: str
    image_path: str | None = None
    question: str | None = None


class AnalyzeResearchPaperFigureResponse(BaseModel):
    task_id: str
    paper_id: str
    figure: ResearchPaperFigurePreview
    chart: ChartSchema
    graph_text: str
    answer: str
    key_points: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTaskAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    paper_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    image_path: str | None = None
    page_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    chart_id: str | None = None
    return_citations: bool = True
    min_length: int = Field(default=400, ge=100, le=5000)
    skill_name: str | None = None
    reasoning_style: str | None = "cot"
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class ResearchTaskAskResponse(BaseModel):
    task_id: str
    paper_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    scope_mode: Literal["all_imported", "selected_papers", "selected_documents", "metadata_only"] = "all_imported"
    qa: QAResponse
    report: ResearchReport | None = None
    todo_items: list[ResearchTodoItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UpdateResearchTodoRequest(BaseModel):
    status: ResearchTodoStatus


class ResearchTodoActionRequest(BaseModel):
    max_papers: int = Field(default=5, ge=1, le=20)
    include_graph: bool = True
    include_embeddings: bool = True
    skill_name: str | None = None
    conversation_id: str | None = None


class ResearchTodoActionResponse(BaseModel):
    task: ResearchTask
    todo: ResearchTodoItem
    papers: list[PaperCandidate] = Field(default_factory=list)
    report: ResearchReport | None = None
    warnings: list[str] = Field(default_factory=list)
    import_result: ImportPapersResponse | None = None


class ResearchMessage(BaseModel):
    message_id: str
    role: ResearchMessageRole
    kind: ResearchMessageKind
    title: str
    content: str = ""
    meta: str | None = None
    created_at: str
    citations: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchAdvancedStrategy(BaseModel):
    action: ResearchAdvancedAction = "discover"
    comparison_dimensions: list[str] = Field(default_factory=list)
    recommendation_goal: str | None = None
    recommendation_top_k: int = Field(default=3, ge=1, le=10)
    force_context_compression: bool = False


class ResearchThreadSnapshot(BaseModel):
    thread_id: str
    route_mode: ResearchRouteMode = "research_discovery"
    topic: str = ""
    task_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
    last_user_message: str | None = None
    last_updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchConversationSnapshot(BaseModel):
    topic: str = ""
    days_back: int = Field(default=180, ge=1, le=3650)
    max_papers: int = Field(default=12, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=lambda: ["arxiv", "openalex"])
    composer_mode: ResearchComposerMode = "research"
    active_route_mode: ResearchRouteMode = "research_discovery"
    active_thread_id: str | None = None
    thread_history: list[ResearchThreadSnapshot] = Field(default_factory=list)
    advanced_strategy: ResearchAdvancedStrategy = Field(default_factory=ResearchAdvancedStrategy)
    selected_paper_ids: list[str] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
    workspace: ResearchWorkspaceState = Field(default_factory=ResearchWorkspaceState)
    search_result: SearchPapersResponse | None = None
    task_result: ResearchTaskResponse | None = None
    import_result: ImportPapersResponse | None = None
    ask_result: ResearchTaskAskResponse | None = None
    last_error: str | None = None
    last_notice: str | None = None
    active_job_id: str | None = None
    context_summary: ResearchContextSummary = Field(default_factory=ResearchContextSummary)
    recent_events: list[ResearchRuntimeEvent] = Field(default_factory=list)


class ResearchConversation(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    task_id: str | None = None
    message_count: int = Field(default=0, ge=0)
    last_message_preview: str | None = None
    snapshot: ResearchConversationSnapshot = Field(default_factory=ResearchConversationSnapshot)
    status_metadata: ResearchStatusMetadata = Field(default_factory=ResearchStatusMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchConversationResponse(BaseModel):
    conversation: ResearchConversation
    messages: list[ResearchMessage] = Field(default_factory=list)


class CreateResearchConversationRequest(BaseModel):
    title: str | None = None
    topic: str | None = None
    days_back: int = Field(default=180, ge=1, le=3650)
    max_papers: int = Field(default=12, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=lambda: ["arxiv", "openalex"])


class RenameResearchConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ResearchJob(BaseModel):
    job_id: str
    kind: ResearchJobKind
    status: ResearchJobStatus = "queued"
    created_at: str
    updated_at: str
    task_id: str | None = None
    conversation_id: str | None = None
    progress_message: str | None = None
    progress_current: int | None = Field(default=None, ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    error_message: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    status_metadata: ResearchStatusMetadata = Field(default_factory=ResearchStatusMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchAgentRunRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    mode: ResearchAgentMode = "auto"
    task_id: str | None = None
    conversation_id: str | None = None
    days_back: int = Field(default=180, ge=1, le=3650)
    max_papers: int = Field(default=12, ge=1, le=100)
    sources: list[PaperSource] = Field(default_factory=lambda: ["arxiv", "openalex", "semantic_scholar"])
    selected_paper_ids: list[str] = Field(default_factory=list)
    selected_document_ids: list[str] = Field(default_factory=list)
    advanced_action: ResearchAdvancedAction | None = None
    comparison_dimensions: list[str] = Field(default_factory=list)
    recommendation_goal: str | None = None
    recommendation_top_k: int = Field(default=3, ge=1, le=10)
    force_context_compression: bool = False
    auto_import: bool = True
    import_top_k: int = Field(default=3, ge=0, le=20)
    include_graph: bool = True
    include_embeddings: bool = True
    top_k: int = Field(default=10, ge=1, le=100)
    skill_name: str | None = "research_report"
    reasoning_style: str | None = "cot"
    document_file_path: str | None = None
    document_id: str | None = None
    chart_image_path: str | None = None
    page_id: str | None = None
    page_number: int = Field(default=1, ge=1)
    chart_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchAgentTraceStep(BaseModel):
    step_index: int
    agent: str = "ResearchSupervisorAgent"
    thought: str
    action_name: str
    phase: ResearchDecisionPhase = "act"
    action_input: dict[str, Any] = Field(default_factory=dict)
    status: ResearchAgentActionStatus
    observation: str = ""
    rationale: str = ""
    estimated_gain: float | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    stop_signal: bool = False
    workspace_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchAgentRunResponse(BaseModel):
    status: Literal["succeeded", "partial", "failed"]
    task: ResearchTask | None = None
    papers: list[PaperCandidate] = Field(default_factory=list)
    report: ResearchReport | None = None
    import_result: ImportPapersResponse | None = None
    qa: QAResponse | None = None
    parsed_document: ParsedDocument | None = None
    document_index_result: dict[str, Any] | None = None
    chart: ChartSchema | None = None
    chart_graph_text: str | None = None
    messages: list[ResearchMessage] = Field(default_factory=list)
    trace: list[ResearchAgentTraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    workspace: ResearchWorkspaceState = Field(default_factory=ResearchWorkspaceState)
    metadata: dict[str, Any] = Field(default_factory=dict)
