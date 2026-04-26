import shutil

from domain.schemas.paper_knowledge import PaperKnowledgeCard, PaperKnowledgeRecord
from domain.schemas.research_context import ResearchContext
from domain.schemas.research_memory import LongTermMemoryQuery, LongTermMemoryRecord
from domain.schemas.sub_manager import TaskStep
from memory.long_term_memory import (
    InMemoryLongTermMemoryStore,
    JsonLongTermMemoryStore,
    LongTermMemory,
    deterministic_memory_vector,
)
from memory.memory_manager import MemoryManager
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from memory.session_memory import JsonSessionMemoryStore, SessionMemory
from memory.working_memory import WorkingMemory


def test_working_memory_keeps_only_recent_turns() -> None:
    memory = WorkingMemory(max_turns=2)

    memory.push_turn("s1", question="q1", answer="a1")
    memory.push_turn("s1", question="q2", answer="a2")
    state = memory.push_turn("s1", question="q3", answer="a3")

    assert [item.question for item in state.recent_history] == ["q2", "q3"]


def test_session_memory_json_store_persists_summary(tmp_path) -> None:
    memory = SessionMemory(JsonSessionMemoryStore(tmp_path / "session"))
    context = ResearchContext(
        research_topic="GraphRAG for literature review",
        research_goals=["compare retrieval strategies"],
        open_questions=["how to keep citations grounded"],
    )

    memory.update_context("session-1", context)
    memory.append_read_paper("session-1", "paper-1")
    memory.append_question("session-1", "GraphRAG compared with vector-only?")
    memory.append_conclusion("session-1", "Hybrid retrieval is more stable.")
    finalized = memory.finalize_session("session-1")

    assert finalized.summary is not None
    assert "研究主题" in finalized.summary.summary_text

    reloaded = memory.load("session-1")
    assert reloaded.summary is not None
    assert reloaded.summary.conclusions == ["Hybrid retrieval is more stable."]


def test_session_memory_json_store_recreates_missing_directory_on_save(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    store = JsonSessionMemoryStore(memory_root / "sessions")
    memory = SessionMemory(store)

    shutil.rmtree(memory_root)

    memory.update_context(
        "session-reset",
        ResearchContext(research_topic="resilient session memory"),
    )

    expected_path = store.root_dir / f"{store._path_for('session-reset').name}"
    assert expected_path.exists()
    assert memory.load("session-reset").context.research_topic == "resilient session memory"


def test_session_memory_clear_removes_persisted_record(tmp_path) -> None:
    memory = SessionMemory(JsonSessionMemoryStore(tmp_path / "session"))

    memory.update_context("session-clear", ResearchContext(research_topic="temporary context"))
    memory.clear("session-clear")

    assert memory.load("session-clear").context.research_topic == ""


def test_paper_knowledge_store_recreates_missing_directory_on_upsert(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    store = JsonPaperKnowledgeStore(memory_root / "paper_knowledge")
    memory = PaperKnowledgeMemory(store)

    shutil.rmtree(memory_root)

    record = PaperKnowledgeRecord(
        paper_id="paper-1",
        title="Robust Agents",
        knowledge_card=PaperKnowledgeCard(
            paper_id="paper-1",
            title="Robust Agents",
            summary="Recover storage roots after reset.",
        ),
    )

    memory.upsert(record)

    expected_path = store.root_dir / f"{store._path_for('paper-1').name}"
    assert expected_path.exists()
    assert memory.load("paper-1") is not None


def test_memory_manager_finalize_promotes_summary_to_long_term() -> None:
    manager = MemoryManager(
        session_memory=SessionMemory(),
        long_term_memory=LongTermMemory(InMemoryLongTermMemoryStore()),
    )
    manager.save_context(
        "session-2",
        ResearchContext(
            research_topic="agentic literature review",
            research_goals=["planner design"],
        ),
    )
    manager.record_read_paper("session-2", "paper-2")
    manager.record_conclusion("session-2", "Planner-agent split improves controllability.")
    manager.finalize_session("session-2")

    result = manager.long_term_memory.search(
        LongTermMemoryQuery(
            query="agentic literature review",
            topic="agentic literature review",
            top_k=3,
        )
    )

    assert result.records
    assert result.records[0].memory_type == "session_summary"
    assert result.records[0].memory_id == "session_summary:session-2"
    assert result.records[0].source_session_id == "session-2"
    assert result.records[0].context_snapshot["research_topic"] == "agentic literature review"


def test_json_long_term_memory_store_persists_records(tmp_path) -> None:
    memory = LongTermMemory(JsonLongTermMemoryStore(tmp_path / "long_term"))

    memory.upsert(
        LongTermMemoryRecord(
            memory_id="ltm:user-profile",
            memory_type="user_profile",
            topic="local-user",
            content="GraphRAG | agent memory",
            keywords=["GraphRAG", "agent memory", "user_profile"],
        )
    )

    reloaded = LongTermMemory(JsonLongTermMemoryStore(tmp_path / "long_term"))
    result = reloaded.search(
        LongTermMemoryQuery(
            query="local-user",
            topic="local-user",
            keywords=["user_profile"],
            top_k=3,
        )
    )

    assert result.records
    assert result.records[0].memory_id == "ltm:user-profile"


def test_deterministic_memory_vector_is_stable_and_sized() -> None:
    first = deterministic_memory_vector("GraphRAG grounded retrieval", size=8)
    second = deterministic_memory_vector("GraphRAG grounded retrieval", size=8)

    assert first == second
    assert len(first) == 8
    assert any(value != 0 for value in first)


def test_memory_manager_hydrates_task_plan_and_turn_history() -> None:
    manager = MemoryManager(
        session_memory=SessionMemory(),
        long_term_memory=LongTermMemory(InMemoryLongTermMemoryStore()),
    )
    manager.save_context(
        "session-3",
        ResearchContext(
            research_topic="multi-agent review",
            selected_papers=["paper-1"],
            current_task_plan=[
                TaskStep(
                    task_id="task-1",
                    assigned_to="research",
                    instruction="search evidence",
                    task_type="search",
                )
            ],
        ),
    )
    manager.record_turn(
        "session-3",
        question="最关键的方法差异是什么？",
        answer="差异主要在检索调度方式。",
        selected_paper_ids=["paper-2"],
    )

    hydrated = manager.hydrate_context("session-3")

    assert hydrated.selected_papers == ["paper-1", "paper-2"]
    assert hydrated.active_papers == ["paper-2"]
    assert hydrated.current_task_plan[0].task_id == "task-1"
    assert hydrated.session_history[-1].question == "最关键的方法差异是什么？"


def test_memory_manager_clear_session_drops_working_and_session_state() -> None:
    manager = MemoryManager(
        session_memory=SessionMemory(),
        long_term_memory=LongTermMemory(InMemoryLongTermMemoryStore()),
    )
    manager.save_context(
        "session-4",
        ResearchContext(
            research_topic="clear me",
            selected_papers=["paper-1"],
        ),
    )
    manager.record_turn(
        "session-4",
        question="q1",
        answer="a1",
        selected_paper_ids=["paper-2"],
    )

    manager.clear_session("session-4")

    hydrated = manager.hydrate_context("session-4")
    assert hydrated.research_topic == ""
    assert hydrated.selected_papers == []
    assert hydrated.session_history == []
