from __future__ import annotations

from types import SimpleNamespace

from memory.research_memory_gateway import ResearchMemoryGateway


class _MemoryManagerStub:
    def __init__(self) -> None:
        self.saved: list[tuple[str, object]] = []
        self.turns: list[dict[str, object]] = []
        self.active_papers: list[tuple[str, list[str]]] = []
        self.profile_updates: list[dict[str, object]] = []
        self.working_memory = SimpleNamespace(
            append_intermediate_step=self._append_intermediate_step,
        )
        self.intermediate_steps: list[dict[str, object]] = []

    def load_user_profile(self):
        return {"profile": "ok"}

    def update_user_profile(self, **kwargs):
        self.profile_updates.append(dict(kwargs))
        return kwargs

    def hydrate_context(self, session_id: str, *, base_context: object):
        return SimpleNamespace(
            research_topic=getattr(base_context, "research_topic", ""),
            research_goals=list(getattr(base_context, "research_goals", [])),
            imported_papers=list(getattr(base_context, "imported_papers", [])),
            known_conclusions=list(getattr(base_context, "known_conclusions", [])),
            open_questions=list(getattr(base_context, "open_questions", [])),
            paper_summaries=list(getattr(base_context, "paper_summaries", [])),
            current_task_plan=[],
            sub_manager_states={},
            metadata={},
            session_id=session_id,
        )

    def save_context(self, session_id: str, context: object):
        self.saved.append((session_id, context))
        return context

    def set_active_papers(self, session_id: str, paper_ids: list[str]):
        self.active_papers.append((session_id, list(paper_ids)))
        return paper_ids

    def record_turn(self, session_id: str, **kwargs):
        self.turns.append({"session_id": session_id, **kwargs})
        return kwargs

    def update_paper_knowledge(self, record):
        return record

    def _append_intermediate_step(self, **kwargs):
        self.intermediate_steps.append(dict(kwargs))


class _ResearchContextManagerStub:
    def build_from_artifacts(self, **kwargs):
        return SimpleNamespace(
            research_topic=str(getattr(kwargs.get("task"), "topic", "") or ""),
            research_goals=[str(getattr(kwargs.get("task"), "topic", "") or "")],
            imported_papers=[str(item.paper_id) for item in kwargs.get("papers") or []],
            known_conclusions=[],
            open_questions=[],
            paper_summaries=["summary"],
        )

    def compress_papers(self, **kwargs):
        return ["summary"] if kwargs.get("papers") else []

    def update_context(self, **kwargs):
        current_context = kwargs["current_context"]
        current_context.metadata = dict(kwargs.get("metadata") or {})
        return current_context


class _SessionMemoryStub:
    def __init__(self) -> None:
        self.updated: list[dict[str, object]] = []
        self.turns: list[dict[str, object]] = []

    def update_research_context(self, **kwargs):
        self.updated.append(dict(kwargs))

    def append_research_turn(self, **kwargs):
        self.turns.append(dict(kwargs))


def test_research_memory_gateway_persists_context_and_session_memory_updates() -> None:
    memory_manager = _MemoryManagerStub()
    context_manager = _ResearchContextManagerStub()
    session_memory = _SessionMemoryStub()
    gateway = ResearchMemoryGateway(
        memory_manager=memory_manager,
        research_context_manager=context_manager,
        paper_reading_skill=object(),
        compact_text=lambda value: str(value or "")[:280],
    )

    task = SimpleNamespace(task_id="task-1", topic="agentic qa", paper_count=1)
    report = SimpleNamespace(highlights=["h1"], gaps=["g1"], markdown="report")
    paper = SimpleNamespace(paper_id="paper-1")

    gateway.persist_research_update(
        session_id="session-1",
        conversation_id="conv-1",
        graph_runtime=SimpleNamespace(session_memory=session_memory),
        task=task,
        report=report,
        papers=[paper],
        document_ids=["doc-1"],
        selected_paper_ids=["paper-1"],
        task_intent="research_search",
        question="what changed?",
        answer="the report changed",
        retrieval_summary="retrieved evidence",
        metadata_update={"route_mode": "research"},
    )

    assert memory_manager.saved
    assert memory_manager.intermediate_steps[0]["step_type"] == "retrieve"
    assert memory_manager.turns[0]["question"] == "what changed?"
    assert session_memory.updated[0]["current_task_intent"] == "research_search"
    assert session_memory.turns[0]["conversation_id"] == "conv-1"
