from __future__ import annotations

from typing import Any

from domain.schemas.agent_message import AgentMessage
from domain.schemas.research import PaperCandidate, ResearchTaskResponse
from domain.schemas.unified_runtime import (
    UnifiedChartUnderstandingInput,
    UnifiedChartUnderstandingOutput,
    UnifiedCollectionQAInput,
    UnifiedCollectionQAOutput,
    UnifiedContextCompressionInput,
    UnifiedContextCompressionOutput,
    UnifiedDocumentUnderstandingInput,
    UnifiedDocumentUnderstandingOutput,
    UnifiedLiteratureSearchInput,
    UnifiedLiteratureSearchOutput,
    UnifiedPaperAnalysisInput,
    UnifiedPaperAnalysisOutput,
    UnifiedPaperImportInput,
    UnifiedPaperImportOutput,
    UnifiedReviewDraftInput,
    UnifiedReviewDraftOutput,
)


def resolve_active_message(decision: Any) -> AgentMessage | None:
    if not isinstance(getattr(decision, "metadata", None), dict):
        return None
    active_message = decision.metadata.get("active_message")
    if isinstance(active_message, AgentMessage):
        return active_message
    if isinstance(active_message, dict):
        return AgentMessage.model_validate(active_message)
    return None


def build_literature_search_input(*, context: Any, decision: Any) -> UnifiedLiteratureSearchInput:
    request = context.request
    task_payload = dict(getattr(decision, "action_input", {}) or {})
    # Prefer extracted_topic from intent (source names already stripped)
    # over the raw user message which may contain "arxiv", "ieee" etc.
    user_intent = (request.metadata or {}).get("user_intent") or {}
    extracted_topic = str(user_intent.get("extracted_topic") or "").strip()
    source_constraints = (
        list(task_payload.get("source_constraints") or [])
        or list(user_intent.get("source_constraints") or [])
    )
    topic = str(
        task_payload.get("evidence_gap_query")
        or task_payload.get("query")
        or task_payload.get("goal")
        or extracted_topic
        or request.message
    ).strip()
    resolved_sources = source_constraints if source_constraints else list(request.sources)
    requested_paper_count = task_payload.get("requested_paper_count")
    max_papers = int(requested_paper_count) if requested_paper_count is not None else int(request.max_papers or 5)
    return UnifiedLiteratureSearchInput(
        topic=topic or request.message.strip(),
        days_back=request.days_back,
        max_papers=max_papers,
        sources=resolved_sources or list(request.sources),
        run_immediately=True,
        conversation_id=request.conversation_id,
        selected_paper_ids=list(request.selected_paper_ids),
        skill_name=request.skill_name,
        reasoning_style=request.reasoning_style,
        metadata={
            **request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "create_research_task",
            "task_payload": task_payload,
            "source_constraints": source_constraints,
        },
    )


def build_literature_search_output(*, task_response: Any) -> UnifiedLiteratureSearchOutput:
    return UnifiedLiteratureSearchOutput(
        task_id=task_response.task.task_id,
        paper_count=len(task_response.papers),
        report_id=task_response.report.report_id if task_response.report else None,
        warnings=list(task_response.warnings),
    )


def build_review_draft_input(*, context: Any) -> UnifiedReviewDraftInput:
    task_response = context.task_response
    assert task_response is not None
    task = task_response.task
    return UnifiedReviewDraftInput(
        topic=task.topic,
        task_id=task.task_id,
        curated_papers=list(task_response.papers),
        warnings=[],
        must_read_ids=list(task.workspace.must_read_paper_ids),
        ingest_candidate_ids=list(task.workspace.ingest_candidate_ids),
        trace=[],
        round_index=0,
        refinement_used=False,
        max_papers=max(len(task_response.papers), 1),
        report=task_response.report,
    )


def build_review_draft_output(
    *,
    task_id: str,
    report_id: str,
    quality: dict[str, Any],
    retry_count: int,
) -> UnifiedReviewDraftOutput:
    return UnifiedReviewDraftOutput(
        task_id=task_id,
        report_id=report_id,
        report_word_count=int(quality["word_count"]),
        report_has_citations=bool(quality["has_citations"]),
        report_has_key_sections=bool(quality["has_key_sections"]),
        retry_count=retry_count,
        issues=list(quality["issues"]),
    )


def build_collection_qa_input(
    *,
    context: Any,
    task_id: str,
    active_message: AgentMessage | None,
) -> UnifiedCollectionQAInput:
    request = context.request
    payload = dict(active_message.payload or {}) if active_message is not None else {}
    question = str(payload.get("question") or payload.get("goal") or request.message).strip()
    if not question:
        question = request.message.strip()
    paper_ids = [
        str(item).strip()
        for item in (payload.get("paper_ids") or request.selected_paper_ids)
        if str(item).strip()
    ]
    document_ids = [
        str(item).strip()
        for item in (payload.get("document_ids") or request.selected_document_ids)
        if str(item).strip()
    ]
    image_path = str(payload.get("image_path") or request.chart_image_path or "").strip() or None
    page_id = str(payload.get("page_id") or request.page_id or "").strip() or None
    chart_id = str(payload.get("chart_id") or request.chart_id or "").strip() or None
    raw_page_number = payload.get("page_number")
    if raw_page_number is None:
        raw_page_number = request.page_number
    try:
        page_number = int(raw_page_number) if raw_page_number is not None else None
    except (TypeError, ValueError):
        page_number = None
    if page_number is not None and page_number < 1:
        page_number = None
    return UnifiedCollectionQAInput(
        task_id=task_id,
        question=question,
        top_k=request.top_k,
        paper_ids=paper_ids,
        document_ids=document_ids,
        image_path=image_path,
        page_id=page_id,
        page_number=page_number,
        chart_id=chart_id,
        return_citations=True,
        min_length=600,
        skill_name=request.skill_name,
        reasoning_style=request.reasoning_style,
        metadata={
            **request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "answer_research_question",
            "routing_authority": str(payload.get("routing_authority") or "supervisor_llm"),
            "preferred_qa_route": str(payload.get("qa_route") or "").strip() or None,
            "task_payload": payload,
        },
        conversation_id=request.conversation_id,
    )


def build_collection_qa_output(*, qa_result: Any) -> UnifiedCollectionQAOutput:
    return UnifiedCollectionQAOutput(
        task_id=qa_result.task_id,
        paper_ids=list(qa_result.paper_ids),
        document_ids=list(qa_result.document_ids),
        evidence_count=len(qa_result.qa.evidence_bundle.evidences),
        confidence=qa_result.qa.confidence,
    )


def build_paper_import_input(*, context: Any, decision: Any) -> UnifiedPaperImportInput:
    request = context.request
    active_message = resolve_active_message(decision)
    payload = dict(active_message.payload or {}) if active_message is not None else {}
    explicit_paper_ids = [
        str(item).strip()
        for item in payload.get("paper_ids") or []
        if str(item).strip()
    ]
    return UnifiedPaperImportInput(
        task_id=context.task.task_id if context.task is not None else "",
        paper_ids=explicit_paper_ids,
        selected_paper_ids=list(request.selected_paper_ids),
        import_top_k=int(payload.get("import_top_k") or request.import_top_k),
        include_graph=request.include_graph,
        include_embeddings=request.include_embeddings,
        skill_name=request.skill_name,
        conversation_id=request.conversation_id,
        metadata={
            **request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "import_relevant_papers",
            "task_payload": payload,
        },
    )


def build_paper_import_output(*, paper_ids: list[str], import_result: Any) -> UnifiedPaperImportOutput:
    return UnifiedPaperImportOutput(
        paper_ids=list(paper_ids),
        imported_count=import_result.imported_count,
        skipped_count=import_result.skipped_count,
        failed_count=import_result.failed_count,
    )


def build_document_understanding_input(*, context: Any, decision: Any) -> UnifiedDocumentUnderstandingInput:
    request = context.request
    task_payload = dict(getattr(decision, "action_input", {}) or {})
    return UnifiedDocumentUnderstandingInput(
        file_path=(request.document_file_path or "").strip(),
        document_id=request.document_id,
        include_graph=request.include_graph,
        include_embeddings=request.include_embeddings,
        session_id=context.execution_context.session_id if context.execution_context else None,
        metadata={
            **request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "understand_document",
            "task_payload": task_payload,
        },
        skill_name=request.skill_name,
    )


def build_document_understanding_output(
    *,
    parsed_document: Any,
    document_index_result: Any | None,
) -> UnifiedDocumentUnderstandingOutput:
    if isinstance(document_index_result, dict):
        index_status = document_index_result.get("status")
    else:
        index_status = getattr(document_index_result, "status", None)
    return UnifiedDocumentUnderstandingOutput(
        document_id=parsed_document.id,
        page_count=len(parsed_document.pages),
        index_status=index_status,
    )


def build_chart_understanding_input(*, context: Any, decision: Any) -> UnifiedChartUnderstandingInput:
    request = context.request
    page_id = request.page_id or "page-1"
    task_payload = dict(getattr(decision, "action_input", {}) or {})
    return UnifiedChartUnderstandingInput(
        image_path=(request.chart_image_path or "").strip(),
        document_id=request.document_id or (context.task.task_id if context.task else "research_document"),
        page_id=page_id,
        page_number=request.page_number,
        chart_id=request.chart_id or f"chart_{page_id}",
        session_id=context.execution_context.session_id if context.execution_context else None,
        context={
            **request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "understand_chart",
            "research_task_id": context.task.task_id if context.task else None,
            "task_payload": task_payload,
        },
        skill_name=request.skill_name,
    )


def build_chart_understanding_output(
    *,
    chart_result: Any,
    chart_input: UnifiedChartUnderstandingInput,
) -> UnifiedChartUnderstandingOutput:
    chart = getattr(chart_result, "chart", None)
    return UnifiedChartUnderstandingOutput(
        chart_id=getattr(chart, "id", chart_input.chart_id),
        chart_type=getattr(chart, "chart_type", None),
        document_id=getattr(chart, "document_id", chart_input.document_id),
    )


def build_paper_analysis_input(
    *,
    context: Any,
    task_response: ResearchTaskResponse,
    payload: dict[str, Any],
    papers: list[PaperCandidate],
) -> UnifiedPaperAnalysisInput:
    analysis_focus = str(payload.get("analysis_focus") or "").strip().lower() or None
    comparison_dimensions = [
        str(item).strip()
        for item in payload.get("dimensions") or []
        if str(item).strip()
    ]
    recommendation_goal = str(payload.get("recommendation_goal") or "").strip() or None
    question = str(payload.get("goal") or context.request.message).strip() or context.request.message.strip()
    return UnifiedPaperAnalysisInput(
        question=question,
        analysis_focus=analysis_focus,
        comparison_dimensions=comparison_dimensions,
        recommendation_goal=recommendation_goal,
        papers=papers,
        task_topic=task_response.task.topic,
        report_highlights=list(task_response.report.highlights) if task_response.report else [],
    )


def build_paper_analysis_output(
    *,
    task_id: str,
    analysis: Any,
    analyzed_papers: list[PaperCandidate],
) -> UnifiedPaperAnalysisOutput:
    return UnifiedPaperAnalysisOutput(
        task_id=task_id,
        paper_count=len(analyzed_papers),
        analyzed_paper_ids=[paper.paper_id for paper in analyzed_papers],
        analysis_focus=analysis.focus,
        recommended_paper_ids=list(analysis.recommended_paper_ids),
    )


def build_context_compression_input(*, context: Any, decision: Any) -> UnifiedContextCompressionInput:
    active_message = resolve_active_message(decision)
    payload = dict(active_message.payload or {}) if active_message is not None else {}
    explicit_paper_ids = [
        str(item).strip()
        for item in payload.get("paper_ids") or []
        if str(item).strip()
    ]
    return UnifiedContextCompressionInput(
        task_id=context.task.task_id if context.task is not None else "",
        selected_paper_ids=list(context.request.selected_paper_ids),
        paper_ids=explicit_paper_ids,
        metadata={
            **context.request.metadata,
            "agent_runtime": "research_agent",
            "manager_action": "compress_context",
            "task_payload": payload,
        },
    )


def build_context_compression_output(*, compression_summary: dict[str, Any]) -> UnifiedContextCompressionOutput:
    return UnifiedContextCompressionOutput(
        paper_count=int(compression_summary["paper_count"]),
        summary_count=int(compression_summary["summary_count"]),
        levels=list(compression_summary["levels"]),
        compressed_paper_ids=list(compression_summary["compressed_paper_ids"]),
    )
