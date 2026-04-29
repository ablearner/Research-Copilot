import json
from pathlib import Path

from rag_runtime.memory import SQLiteSessionMemoryStore, SessionMemorySnapshot


def test_sqlite_session_memory_store_round_trip(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kepler.db")
    store = SQLiteSessionMemoryStore(db_path=db_path)

    snapshot = SessionMemorySnapshot(
        session_id="s1",
        current_document_id="doc1",
        last_answer_summary="answer",
        current_task_intent="chart_qa:test",
        metadata={"chart_turns": [{"question": "q1", "answer": "a1"}]},
    )
    store.put(snapshot)

    loaded = store.get("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.current_document_id == "doc1"
    assert loaded.last_answer_summary == "answer"
    assert loaded.metadata["chart_turns"][0]["answer"] == "a1"


def test_sqlite_session_memory_store_delete(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kepler.db")
    store = SQLiteSessionMemoryStore(db_path=db_path)

    store.put(SessionMemorySnapshot(session_id="s1", metadata={"key": "val"}))
    assert store.get("s1") is not None

    store.delete("s1")
    assert store.get("s1") is None


def test_sqlite_session_memory_store_update_overwrites(tmp_path: Path) -> None:
    db_path = str(tmp_path / "kepler.db")
    store = SQLiteSessionMemoryStore(db_path=db_path)

    store.put(SessionMemorySnapshot(session_id="s1", last_answer_summary="v1"))
    store.put(SessionMemorySnapshot(session_id="s1", last_answer_summary="v2"))

    loaded = store.get("s1")
    assert loaded is not None
    assert loaded.last_answer_summary == "v2"


def test_sqlite_session_memory_store_shares_db_with_research_store(tmp_path: Path) -> None:
    """Verify session_memory table coexists with research data tables in the same db."""
    import sqlite3

    db_path = str(tmp_path / "kepler.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, data_json TEXT)")
    conn.execute("INSERT INTO tasks VALUES ('t1', '{}')")
    conn.commit()

    store = SQLiteSessionMemoryStore(db_path=db_path)
    store.put(SessionMemorySnapshot(session_id="s1"))

    assert store.get("s1") is not None
    row = conn.execute("SELECT task_id FROM tasks WHERE task_id = 't1'").fetchone()
    assert row is not None
    conn.close()
