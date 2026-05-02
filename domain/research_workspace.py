from __future__ import annotations

from typing import Iterable

from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTodoItem,
    ResearchTopicPlan,
    ResearchWorkspaceStage,
    ResearchWorkspaceState,
)


def _compact(text: str, *, limit: int = 140) -> str:
    normalized = " ".join((text or "").strip().split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 1, 1)].rstrip()}…"


def _unique(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    for value in values:
        normalized = _compact(str(value))
        if not normalized or normalized in deduped:
            continue
        deduped.append(normalized)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _default_hypotheses(
    *,
    objective: str,
    papers: list[PaperCandidate],
    imported_document_ids: list[str],
) -> list[str]:
    hypotheses: list[str] = []
    lowered_titles = " ".join((paper.title or "").lower() for paper in papers[:5])
    if "survey" in lowered_titles or "benchmark" in lowered_titles:
        hypotheses.append("Survey or benchmark papers are likely to provide the fastest high-coverage evidence.")
    if papers and not imported_document_ids:
        hypotheses.append("Importing open-access PDFs should materially improve grounded collection QA quality.")
    if objective and ("对比" in objective or "compare" in objective.lower()):
        hypotheses.append("A structured comparison table will likely be more useful than a free-form summary alone.")
    return _unique(hypotheses, limit=3)


def _default_next_actions(
    *,
    stage: ResearchWorkspaceStage,
    imported_document_ids: list[str],
    todo_items: list[ResearchTodoItem],
    report: ResearchReport | None,
    stop_reason: str | None,
) -> list[str]:
    actions: list[str] = []
    open_todos = [item for item in todo_items if item.status == "open"]
    if stage == "discover" and not imported_document_ids:
        actions.append("Import the highest-value open-access papers so the assistant can answer with grounded document evidence.")
    if stage in {"ingest", "qa"} and imported_document_ids:
        actions.append("Ask a focused collection question to turn the imported paper set into concrete conclusions.")
    if open_todos:
        actions.append(f"Execute or dismiss the {len(open_todos)} open TODO items to keep the research workspace moving.")
    if report and report.gaps:
        actions.append("Use the current evidence gaps to drive the next retrieval or follow-up import cycle.")
    if stop_reason:
        actions.append(f"Stop reason: {stop_reason}")
    return _unique(actions, limit=4)


def _status_summary(
    *,
    stage: ResearchWorkspaceStage,
    paper_count: int,
    imported_document_count: int,
    open_todo_count: int,
    gap_count: int,
) -> str:
    return (
        f"stage={stage}; papers={paper_count}; imported_docs={imported_document_count}; "
        f"open_todos={open_todo_count}; evidence_gaps={gap_count}"
    )


def build_workspace_state(
    *,
    objective: str,
    stage: ResearchWorkspaceStage,
    papers: list[PaperCandidate] | None = None,
    imported_document_ids: list[str] | None = None,
    report: ResearchReport | None = None,
    plan: ResearchTopicPlan | None = None,
    todo_items: list[ResearchTodoItem] | None = None,
    must_read_ids: list[str] | None = None,
    ingest_candidate_ids: list[str] | None = None,
    extra_questions: list[str] | None = None,
    extra_findings: list[str] | None = None,
    stop_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ResearchWorkspaceState:
    resolved_papers = list(papers or [])
    resolved_document_ids = list(imported_document_ids or [])
    resolved_todos = list(todo_items or [])
    key_findings = _unique(
        [
            *(report.highlights if report else []),
            *(extra_findings or []),
        ],
        limit=5,
    )
    evidence_gaps = _unique(report.gaps if report else [], limit=4)
    research_questions = _unique(
        [
            objective,
            *(plan.queries if plan else []),
            *(extra_questions or []),
        ],
        limit=5,
    )
    hypotheses = _default_hypotheses(
        objective=objective,
        papers=resolved_papers,
        imported_document_ids=resolved_document_ids,
    )
    next_actions = _default_next_actions(
        stage=stage,
        imported_document_ids=resolved_document_ids,
        todo_items=resolved_todos,
        report=report,
        stop_reason=stop_reason,
    )
    open_todo_count = sum(1 for item in resolved_todos if item.status == "open")
    return ResearchWorkspaceState(
        objective=_compact(objective, limit=180),
        current_stage=stage,
        research_questions=research_questions,
        hypotheses=hypotheses,
        key_findings=key_findings,
        evidence_gaps=evidence_gaps,
        must_read_paper_ids=list(must_read_ids or []),
        ingest_candidate_ids=list(ingest_candidate_ids or []),
        document_ids=resolved_document_ids,
        next_actions=next_actions,
        stop_reason=stop_reason,
        status_summary=_status_summary(
            stage=stage,
            paper_count=len(resolved_papers),
            imported_document_count=len(resolved_document_ids),
            open_todo_count=open_todo_count,
            gap_count=len(evidence_gaps),
        ),
        metadata={
            "paper_count": len(resolved_papers),
            "imported_document_count": len(resolved_document_ids),
            "open_todo_count": open_todo_count,
            "gap_count": len(evidence_gaps),
            **(metadata or {}),
        },
    )


def build_workspace_from_task(
    *,
    task: ResearchTask | None,
    report: ResearchReport | None = None,
    papers: list[PaperCandidate] | None = None,
    plan: ResearchTopicPlan | None = None,
    extra_questions: list[str] | None = None,
    extra_findings: list[str] | None = None,
    stage: ResearchWorkspaceStage | None = None,
    stop_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ResearchWorkspaceState:
    if task is None:
        return build_workspace_state(
            objective="",
            stage=stage or "discover",
            report=report,
            papers=papers,
            imported_document_ids=[],
            plan=plan,
            stop_reason=stop_reason,
            metadata=metadata,
        )
    resolved_stage = stage or (
        "qa"
        if task.imported_document_ids
        else "ingest"
        if task.paper_count
        else "discover"
    )
    workspace = task.workspace
    must_read_ids = list(workspace.must_read_paper_ids)
    ingest_candidate_ids = list(workspace.ingest_candidate_ids)
    papers_by_id = {paper.paper_id: paper for paper in papers or []}
    if papers_by_id and ingest_candidate_ids:
        ingest_candidate_ids = [
            paper_id
            for paper_id in ingest_candidate_ids
            if paper_id not in papers_by_id
            or papers_by_id[paper_id].ingest_status != "ingested"
        ]
    return build_workspace_state(
        objective=task.topic,
        stage=resolved_stage,
        papers=papers,
        imported_document_ids=task.imported_document_ids,
        report=report,
        plan=plan,
        todo_items=task.todo_items,
        must_read_ids=must_read_ids,
        ingest_candidate_ids=ingest_candidate_ids,
        extra_questions=extra_questions,
        extra_findings=extra_findings,
        stop_reason=stop_reason,
        metadata=metadata,
    )
