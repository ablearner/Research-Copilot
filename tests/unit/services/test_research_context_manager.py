from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask, ResearchWorkspaceState
from domain.schemas.sub_manager import SubManagerState, TaskStep
from memory.research_context_manager import ResearchContextManager


def test_research_context_manager_builds_context_from_artifacts() -> None:
    manager = ResearchContextManager()
    task = ResearchTask(
        task_id="task-1",
        topic="Agentic academic assistants",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        workspace=ResearchWorkspaceState(
            objective="improve structured review quality",
            research_questions=["how to strengthen citation grounding"],
            key_findings=["current flow is too linear"],
            evidence_gaps=["paper-scoped QA is missing"],
        ),
    )
    report = ResearchReport(
        report_id="report-1",
        task_id="task-1",
        topic="Agentic academic assistants",
        generated_at="2026-04-20T00:00:00+00:00",
        markdown="# report",
        highlights=["multi-agent plan is needed"],
        gaps=["memory continuity is weak"],
    )
    papers = [
        PaperCandidate(
            paper_id="paper-1",
            title="A",
            source="arxiv",
            ingest_status="ingested",
            metadata={"document_id": "doc-1"},
        ),
        PaperCandidate(
            paper_id="paper-2",
            title="B",
            source="openalex",
            ingest_status="selected",
        ),
    ]

    context = manager.build_from_artifacts(
        task=task,
        report=report,
        papers=papers,
        selected_paper_ids=["paper-1"],
        history_entries=[{"question": "q1", "answer": "a1", "document_ids": ["doc-1"]}],
    )

    assert context.research_topic == "Agentic academic assistants"
    assert context.selected_papers == ["paper-1"]
    assert context.imported_papers[0].document_id == "doc-1"
    assert "current flow is too linear" in context.known_conclusions
    assert "memory continuity is weak" in context.open_questions
    assert context.session_history[0].question == "q1"
    assert set(context.sub_manager_states) == {"writing", "research"}


def test_research_context_manager_slices_selected_papers() -> None:
    manager = ResearchContextManager()
    context = manager.update_context(
        topic="paper review",
        goals=["structured extraction"],
        selected_papers=["paper-1", "paper-2"],
    )

    sliced = manager.slice_for_agent(
        context,
        paper_ids=["paper-2"],
        max_papers=1,
        max_history_turns=2,
    )

    assert sliced.research_topic == "paper review"
    assert sliced.selected_papers == ["paper-2"]


def test_research_context_manager_slices_task_plan_for_sub_manager() -> None:
    manager = ResearchContextManager()
    context = manager.update_context(
        topic="agentic review",
        current_task_plan=[
            TaskStep(
                task_id="task-1",
                assigned_to="research",
                instruction="search papers",
                task_type="search",
                metadata={"sub_manager": "research"},
            ),
            TaskStep(
                task_id="task-2",
                assigned_to="writing",
                instruction="draft review",
                task_type="review",
                metadata={"sub_manager": "writing"},
            ),
        ],
        sub_manager_states={
            "research": SubManagerState(
                name="research",
                status="running",
                active_task_ids=["task-1"],
            )
        },
    )

    sliced = manager.slice_for_agent(
        context,
        agent_scope="sub_manager",
        sub_manager_key="research",
    )

    assert [step.task_id for step in sliced.current_task_plan] == ["task-1"]
    assert sliced.sub_manager_state is not None
    assert sliced.sub_manager_state.status == "running"
    assert sliced.context_scope == "sub_manager"


def test_research_context_manager_compresses_papers_into_multi_level_summaries() -> None:
    manager = ResearchContextManager()

    summaries = manager.compress_papers(
        papers=[
            PaperCandidate(
                paper_id="paper-1",
                title="Structured Comparison for Research Agents",
                abstract=(
                    "This paper proposes a structured comparison pipeline. "
                    "The method coordinates planning, retrieval, and synthesis. "
                    "Experiments compare multiple agent settings."
                ),
                source="arxiv",
                year=2026,
                citations=10,
            )
        ],
        selected_paper_ids=["paper-1"],
    )

    assert {summary.level for summary in summaries} == {"paragraph", "section", "document"}
    assert all(summary.paper_id == "paper-1" for summary in summaries)
    assert any("贡献" in summary.summary for summary in summaries if summary.level != "paragraph")
