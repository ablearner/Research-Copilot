"""Construct bounded worker inputs without copying manager prompt state."""
from copy import deepcopy
from typing import Any

from domain.schemas.research_context import (
    CompressedPaperSummary, QAPair, ResearchContext, ResearchContextSlice, ResearchContextPaperMeta,
    ResearchUserPreferences,
)
from domain.schemas.worker_task import WorkerArtifactReference, WorkerTaskInput
from runtime.research.unified_action_adapters import resolve_active_message


_TEXT_LIMIT = 1600
_PAPER_LIMIT = 8
_HISTORY_LIMIT = 4
_DEPENDENCY_LIMIT = 32


def _text(value: Any, limit: int = _TEXT_LIMIT) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _ids(value: Any) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value
                             if isinstance(item, str) and item.strip())) if isinstance(value, list) else []


def _selection(sources: list[dict[str, Any]], research: Any, available_papers: list[ResearchContextPaperMeta]) -> list[str]:
    for source in sources:
        paper_key = next((key for key in ('paper_ids', 'selected_paper_ids') if key in source), None)
        document_key = next((key for key in ('document_ids', 'selected_document_ids') if key in source), None)
        if paper_key is None and document_key is None:
            continue
        papers = _ids(source.get(paper_key))
        documents = _ids(source.get(document_key))
        if documents:
            papers.extend(paper.paper_id for paper in available_papers if paper.document_id in documents)
        # A specified but unknown document must not widen to the collection.
        if papers or documents:
            return list(dict.fromkeys(papers))
        return list(dict.fromkeys(paper.paper_id for paper in available_papers))
    return list(dict.fromkeys(research.selected_papers or research.active_papers
                              or [paper.paper_id for paper in available_papers]))


def _context_slice(context: Any, decision: Any, payload: dict[str, Any]) -> ResearchContextSlice:
    level = 'section' if decision.action_name == 'write_review' else 'document'
    result = ResearchContextSlice(summary_level=level)
    if decision.action_name == 'general_answer' and payload.get('ignore_research_context'):
        return result
    execution = getattr(context, 'execution_context', None)
    research = getattr(execution, 'research_context', None)
    if research is None:
        task = getattr(context, 'task', None)
        workspace = getattr(task, 'workspace', None)
        research = ResearchContext(
            research_topic=getattr(task, 'topic', '') or '',
            research_goals=list(getattr(workspace, 'research_questions', []) or []),
        )
    request = context.request
    request_scope = {}
    for key in ('selected_paper_ids', 'selected_document_ids'):
        value = getattr(request, key, [])
        if value:
            request_scope[key] = value
    available = {paper.paper_id: paper for paper in research.imported_papers}
    # Search candidates have not necessarily reached ResearchContext.imported_papers.
    for paper in getattr(context, 'papers', []) or []:
        if paper.paper_id not in available:
            available[paper.paper_id] = ResearchContextPaperMeta(
                paper_id=paper.paper_id, title=paper.title, authors=list(paper.authors),
                year=paper.year, source=paper.source, summary=paper.summary,
                document_id=paper.metadata.get('document_id') or None,
            )
    available_papers = list(available.values())
    selected = _selection([payload, decision.action_input, request_scope], research, available_papers)
    if decision.action_name in {'search_literature', 'recommend_from_preferences'} and not any(
        key in source for source in (payload, request_scope)
        for key in ('paper_ids', 'selected_paper_ids', 'document_ids', 'selected_document_ids')
    ):
        selected = []
    allowed = set(selected)
    papers = [paper for paper in available_papers if paper.paper_id in allowed][:_PAPER_LIMIT]
    result.selected_papers = selected
    result.research_topic = _text(research.research_topic)
    result.research_goals = [_text(goal, 300) for goal in research.research_goals[:6]]
    result.imported_papers = [ResearchContextPaperMeta(
        paper_id=paper.paper_id, title=_text(paper.title, 300), authors=[_text(a, 100) for a in paper.authors[:8]],
        year=paper.year, source=paper.source, document_id=paper.document_id, summary=_text(paper.summary),
        tags=[_text(tag, 80) for tag in paper.tags[:8]],
    ) for paper in papers]
    summaries = [s for s in research.paper_summaries if s.paper_id in allowed and s.level == level]
    if level == 'section':
        covered = {s.paper_id for s in summaries}
        summaries.extend(s for s in research.paper_summaries
                         if s.paper_id in allowed - covered and s.level == 'document')
    result.relevant_summaries = [CompressedPaperSummary(
        paper_id=s.paper_id, level=s.level, summary=_text(s.summary),
        source_section_ids=list(s.source_section_ids[:12]), relevance_score=s.relevance_score,
    ) for s in summaries[:_PAPER_LIMIT]]
    citations = allowed | {p.document_id for p in available_papers if p.paper_id in allowed and p.document_id}
    history = [q for q in research.session_history if q.citations and set(q.citations) <= citations]
    result.session_history = [QAPair(question=_text(q.question, 400), answer=_text(q.answer, 800),
        citations=list(q.citations[:20]), created_at=q.created_at) for q in history[-_HISTORY_LIMIT:]]
    preferences = research.user_preferences
    result.user_preferences = ResearchUserPreferences(
        review_style=preferences.review_style, answer_language=_text(preferences.answer_language, 40),
        citation_style=_text(preferences.citation_style, 80),
        max_selected_papers=preferences.max_selected_papers, max_history_turns=preferences.max_history_turns,
    )
    return result


def build_worker_task_input(*, context: Any, decision: Any, agent_results: list) -> tuple[WorkerTaskInput, dict[str, Any]]:
    """Return prompt data and invocation-local, detached dependency payloads.

    All explicit paper IDs survive; only their descriptive material is capped.
    Full dependency results are available through references, never auto-expanded.
    """
    message = resolve_active_message(decision)
    payload = {**decision.action_input, **(message.payload if message else {})}
    summaries = []
    references = []
    artifacts = {}
    results = {result.task_id: result for result in agent_results}
    for task_id in dict.fromkeys(message.depends_on if message else []):
        dependency = results.get(task_id)
        if dependency is None:
            if len(summaries) < _DEPENDENCY_LIMIT:
                summaries.append({'task_id': task_id, 'status': 'missing'})
            continue
        key = f'result:{task_id}'
        artifacts[key] = deepcopy(getattr(context, 'result_artifacts', {}).get(task_id, dependency.payload))
        summary = next((_text(dependency.payload.get(name), 1200)
                        for name in ('summary', 'answer', 'message')
                        if isinstance(dependency.payload.get(name), str)), '')
        references.append(WorkerArtifactReference(key=key, task_id=task_id))
        if len(summaries) < _DEPENDENCY_LIMIT:
            summaries.append({'task_id': task_id, 'status': dependency.status,
                              'task_type': dependency.task_type, 'summary': summary})
    return WorkerTaskInput(
        task_id=message.task_id if message else None, action_name=decision.action_name,
        instruction=message.instruction if message else context.request.message,
        context_slice=_context_slice(context, decision, payload),
        dependency_summaries=summaries, artifact_refs=references,
    ), artifacts
