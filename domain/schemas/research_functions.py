from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.paper_knowledge import PaperFigureInsight, PaperFormulaInsight, PaperKnowledgeCard
from domain.schemas.research import PaperSource
from domain.schemas.research_context import QAPair, ResearchContext, ResearchContextPaperMeta, ResearchUserPreferences
from domain.schemas.sub_manager import TaskEvaluation, TaskStep


ResearchSortBy = Literal["relevance", "date", "citations"]


class ResearchDateRange(BaseModel):
    start_date: date | None = None
    end_date: date | None = None


class SearchPaperResult(BaseModel):
    id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None
    source: PaperSource | str


class SearchPapersFunctionInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    source: list[PaperSource | str] = Field(default_factory=list)
    date_range: ResearchDateRange | None = None
    max_results: int = Field(default=10, ge=1, le=100)
    sort_by: ResearchSortBy = "relevance"


class SearchPapersFunctionOutput(BaseModel):
    papers: list[SearchPaperResult] = Field(default_factory=list)


class ExtractPaperStructureFunctionInput(BaseModel):
    paper_id: str = Field(min_length=1)


class ExtractPaperStructureFunctionOutput(BaseModel):
    contribution: str = ""
    method: str = ""
    experiment: str = ""
    limitation: str = ""
    key_formulas: list[PaperFormulaInsight] = Field(default_factory=list)
    figures: list[PaperFigureInsight] = Field(default_factory=list)
    knowledge_card: PaperKnowledgeCard | None = None


class ComparisonTableRow(BaseModel):
    dimension: str
    values: dict[str, str] = Field(default_factory=dict)


class PaperAnalysisNote(BaseModel):
    paper_id: str
    title: str
    summary: str = ""
    relevance_to_question: str = ""
    strengths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ComparePapersFunctionInput(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, min_length=2)
    dimensions: list[str] = Field(default_factory=list)


class ComparePapersFunctionOutput(BaseModel):
    table: list[ComparisonTableRow] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzePapersFunctionInput(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    paper_ids: list[str] = Field(default_factory=list, min_length=1)


class AnalyzePapersFunctionOutput(BaseModel):
    answer: str = ""
    focus: Literal["analysis", "compare", "recommend", "explain"] = "analysis"
    key_points: list[str] = Field(default_factory=list)
    recommended_paper_ids: list[str] = Field(default_factory=list)
    paper_notes: list[PaperAnalysisNote] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateReviewFunctionInput(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, min_length=1)
    style: Literal["academic", "concise", "beginner"] = "academic"
    min_length: int = Field(default=800, ge=200, le=5000)
    include_citations: bool = True


class GenerateReviewFunctionOutput(BaseModel):
    review_text: str
    citations: list[str] = Field(default_factory=list)
    word_count: int = Field(default=0, ge=0)


class AnswerCitation(BaseModel):
    paper_id: str
    section_id: str | None = None
    evidence_text: str = ""
    rationale: str | None = None


class RelatedSection(BaseModel):
    paper_id: str
    section_id: str | None = None
    heading: str | None = None
    relevance_score: float | None = Field(default=None, ge=0)


class AskPaperFunctionInput(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    paper_ids: list[str] = Field(default_factory=list)
    return_citations: bool = True
    min_length: int = Field(default=400, ge=50, le=5000)


class AskPaperFunctionOutput(BaseModel):
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    related_sections: list[RelatedSection] = Field(default_factory=list)
    extended_analysis: str = ""


class RecommendedPaper(BaseModel):
    paper_id: str
    title: str
    reason: str
    source: PaperSource | str | None = None
    year: int | None = None
    url: str | None = None


class RecommendPapersFunctionInput(BaseModel):
    based_on_context: str = ""
    based_on_history: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)


class RecommendPapersFunctionOutput(BaseModel):
    recommendations: list[RecommendedPaper] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateResearchContextFunctionInput(BaseModel):
    topic: str = ""
    keywords: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    known_conclusions: list[str] = Field(default_factory=list)
    selected_papers: list[str] = Field(default_factory=list)
    imported_papers: list[ResearchContextPaperMeta] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    session_history: list[QAPair] = Field(default_factory=list)
    user_preferences: ResearchUserPreferences | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecomposeTaskFunctionInput(BaseModel):
    user_request: str = Field(min_length=1, max_length=4000)
    context: ResearchContext = Field(default_factory=ResearchContext)


class DecomposeTaskFunctionOutput(BaseModel):
    task_plan: list[TaskStep] = Field(default_factory=list)
    assigned_sub_manager: str | None = None
    parallel_allowed: bool = False
    clarification_needed: bool = False
    clarification_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPlanStep(BaseModel):
    step_id: str
    function_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    description: str | None = None


class ExecuteResearchPlanFunctionInput(BaseModel):
    plan_steps: list[ResearchPlanStep] = Field(default_factory=list)
    parallel: bool = False


class ResearchPlanStepResult(BaseModel):
    step_id: str
    function_name: str
    status: Literal["succeeded", "failed", "skipped"]
    output: dict[str, Any] | None = None
    error_message: str | None = None


class ExecuteResearchPlanFunctionOutput(BaseModel):
    step_results: list[ResearchPlanStepResult] = Field(default_factory=list)
    summary_report: str = ""


class EvaluateResultFunctionInput(BaseModel):
    result: dict[str, Any] = Field(default_factory=dict)
    task_instruction: str = Field(min_length=1, max_length=4000)
    expected_schema: dict[str, Any] = Field(default_factory=dict)


class EvaluateResultFunctionOutput(TaskEvaluation):
    pass


RESEARCH_FUNCTION_SCHEMAS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "search_papers": (SearchPapersFunctionInput, SearchPapersFunctionOutput),
    "extract_paper_structure": (
        ExtractPaperStructureFunctionInput,
        ExtractPaperStructureFunctionOutput,
    ),
    "analyze_papers": (AnalyzePapersFunctionInput, AnalyzePapersFunctionOutput),
    "compare_papers": (ComparePapersFunctionInput, ComparePapersFunctionOutput),
    "generate_review": (GenerateReviewFunctionInput, GenerateReviewFunctionOutput),
    "ask_paper": (AskPaperFunctionInput, AskPaperFunctionOutput),
    "recommend_papers": (RecommendPapersFunctionInput, RecommendPapersFunctionOutput),
    "update_research_context": (UpdateResearchContextFunctionInput, ResearchContext),
    "decompose_task": (DecomposeTaskFunctionInput, DecomposeTaskFunctionOutput),
    "evaluate_result": (EvaluateResultFunctionInput, EvaluateResultFunctionOutput),
    "execute_research_plan": (
        ExecuteResearchPlanFunctionInput,
        ExecuteResearchPlanFunctionOutput,
    ),
}
