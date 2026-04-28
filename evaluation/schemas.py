from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


CaseKind = Literal[
    "ask_document",
    "ask_fused",
    "chart_understand",
    "search_literature",
    "import_and_qa",
    "write_review",
    "multi_turn_session",
]


class EvaluationCase(BaseModel):
    id: str
    kind: CaseKind
    question: str | None = None
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    image_path: str | None = None
    page_id: str | None = None
    page_number: int = Field(default=1, ge=1)
    chart_id: str | None = None
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None
    reasoning_style: str | None = None

    expected_route: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    min_keyword_recall: float = Field(default=0.5, ge=0, le=1)
    expected_evidence_ids: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_retrieval_keywords: list[str] = Field(default_factory=list)
    expected_tool_names: list[str] = Field(default_factory=list)
    grounding_keywords: list[str] = Field(default_factory=list)
    require_nonempty_answer: bool = True
    require_evidence: bool | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "EvaluationCase":
        question_required_kinds = {"ask_document", "ask_fused", "search_literature", "import_and_qa", "write_review", "multi_turn_session"}
        if self.kind in question_required_kinds and not self.question:
            raise ValueError(f"question is required for case kind={self.kind}")
        if self.kind == "ask_fused" and not self.image_path:
            raise ValueError("image_path is required for ask_fused cases")
        if self.kind == "chart_understand":
            required = {"image_path": self.image_path, "document_id": self.document_id, "page_id": self.page_id, "chart_id": self.chart_id}
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"chart_understand case is missing fields: {', '.join(missing)}")
        return self

    @property
    def resolved_document_ids(self) -> list[str]:
        document_ids = list(self.document_ids)
        if self.document_id and self.document_id not in document_ids:
            document_ids.append(self.document_id)
        return document_ids

    @property
    def needs_evidence(self) -> bool:
        if self.require_evidence is not None:
            return self.require_evidence
        return self.kind in {"ask_document", "ask_fused", "import_and_qa"}


class CaseMetricResult(BaseModel):
    case_id: str
    kind: CaseKind
    actual_route: str
    task_success: bool
    answer: str | None = None
    keyword_recall: float | None = None
    reference_precision: float | None = None
    reference_recall: float | None = None
    reference_f1: float | None = None
    polarity_correct: bool | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    hit_at_k: bool | None = None
    recall_at_k: float | None = None
    hit_at_5: bool | None = None
    recall_at_5: float | None = None
    matched_evidence_ids: list[str] = Field(default_factory=list)
    matched_source_ids: list[str] = Field(default_factory=list)
    matched_retrieval_keywords: list[str] = Field(default_factory=list)
    groundedness: float | None = None
    route_correct: bool | None = None
    tool_call_success_rate: float | None = None
    tool_call_success_count: int = 0
    tool_call_total: int = 0
    validation_retry: bool = False
    latency_ms: float | None = None
    step_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregateMetrics(BaseModel):
    total_cases: int
    task_success_rate: float
    answer_keyword_recall: float | None = None
    reference_answer_precision: float | None = None
    reference_answer_recall: float | None = None
    reference_answer_f1: float | None = None
    answer_polarity_accuracy: float | None = None
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    hit_at_5: float | None = None
    recall_at_5: float | None = None
    groundedness: float | None = None
    route_accuracy: float | None = None
    tool_call_success_rate: float | None = None
    validation_retry_rate: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    average_steps_per_task: float | None = None
    insufficient_answer_rate: float | None = None
    warning_case_rate: float | None = None
    avg_warning_count: float | None = None
    error_free_rate: float | None = None


class CoreMetricSummary(BaseModel):
    recall_k: int = 5
    recall_at_k: float | None = None
    groundedness: float | None = None
    answer_keyword_recall: float | None = None
    route_accuracy: float | None = None
    tool_call_success_rate: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None


class EvaluationReport(BaseModel):
    runtime_mode: Literal["sample", "live"]
    metrics: AggregateMetrics
    core_6_metrics: CoreMetricSummary = Field(default_factory=CoreMetricSummary)
    cases: list[CaseMetricResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
