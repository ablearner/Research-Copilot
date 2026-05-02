from domain.schemas.research import PaperCandidate, ResearchTask
from tools.research.paper_selector import PaperSelectorService


def test_paper_selector_service_resolves_selected_imported_paper_scope() -> None:
    service = PaperSelectorService()
    task = ResearchTask(
        task_id="task-scope-1",
        topic="agentic review",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        imported_document_ids=["doc-a", "doc-b"],
    )
    papers = [
        PaperCandidate(
            paper_id="paper-a",
            title="Paper A",
            source="arxiv",
            ingest_status="ingested",
            metadata={"document_id": "doc-a"},
        ),
        PaperCandidate(
            paper_id="paper-b",
            title="Paper B",
            source="openalex",
            ingest_status="ingested",
            metadata={"document_id": "doc-b"},
        ),
    ]

    scope = service.resolve_qa_scope(
        task=task,
        papers=papers,
        requested_paper_ids=["paper-b"],
    )

    assert scope.scope_mode == "selected_papers"
    assert scope.paper_ids == ["paper-b"]
    assert scope.document_ids == ["doc-b"]
    assert [paper.title for paper in scope.papers] == ["Paper B"]
    assert scope.warnings == []
    assert scope.metadata["paper_count"] == 1
    assert scope.metadata["document_count"] == 1
    assert scope.metadata["matched_document_ids"] == ["doc-b"]
    assert "scope=selected_papers" in scope.metadata["selection_summary"]


def test_paper_selector_service_warns_for_unimported_selected_paper() -> None:
    service = PaperSelectorService()
    task = ResearchTask(
        task_id="task-scope-2",
        topic="agentic review",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        imported_document_ids=[],
    )
    papers = [
        PaperCandidate(
            paper_id="paper-a",
            title="Paper A",
            source="arxiv",
            ingest_status="selected",
        ),
    ]

    scope = service.resolve_qa_scope(
        task=task,
        papers=papers,
        requested_paper_ids=["paper-a"],
    )

    assert scope.scope_mode == "metadata_only"
    assert scope.paper_ids == ["paper-a"]
    assert scope.document_ids == []
    assert "当前只能基于元数据回答" in scope.warnings[0]
    assert scope.metadata["metadata_only_paper_ids"] == ["paper-a"]


def test_paper_selector_service_ignores_unknown_document_scope() -> None:
    service = PaperSelectorService()
    task = ResearchTask(
        task_id="task-scope-3",
        topic="agentic review",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        imported_document_ids=["doc-a"],
    )
    papers = [
        PaperCandidate(
            paper_id="paper-a",
            title="Paper A",
            source="arxiv",
            ingest_status="ingested",
            metadata={"document_id": "doc-a"},
        ),
    ]

    scope = service.resolve_qa_scope(
        task=task,
        papers=papers,
        requested_document_ids=["doc-missing"],
    )

    assert scope.scope_mode == "metadata_only"
    assert scope.paper_ids == []
    assert scope.document_ids == []
    assert scope.metadata["missing_document_ids"] == ["doc-missing"]
    assert "当前研究任务未登记该文档" in scope.warnings[0]
