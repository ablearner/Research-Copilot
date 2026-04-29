from pathlib import Path

from domain.schemas.research_memory import LongTermMemoryQuery, LongTermMemoryRecord
from memory.long_term_memory import SQLiteLongTermMemoryStore


def test_upsert_and_search_by_topic(tmp_path: Path) -> None:
    store = SQLiteLongTermMemoryStore(db_path=tmp_path / "kepler.db")
    store.upsert(
        LongTermMemoryRecord(
            memory_id="m1",
            topic="GraphRAG",
            content="Graph-based retrieval augmented generation improves factual accuracy",
            keywords=["graph", "rag", "retrieval"],
        )
    )
    result = store.search(LongTermMemoryQuery(query="graph retrieval"))
    assert len(result.records) >= 1
    assert result.records[0].memory_id == "m1"
    assert result.records[0].score is not None and result.records[0].score > 0


def test_fts5_chinese_search(tmp_path: Path) -> None:
    store = SQLiteLongTermMemoryStore(db_path=tmp_path / "kepler.db")
    store.upsert(
        LongTermMemoryRecord(
            memory_id="m2",
            topic="长期记忆",
            content="基于向量检索的长期记忆存储方案",
            keywords=["记忆", "向量"],
        )
    )
    result = store.search(LongTermMemoryQuery(query="记忆"))
    assert len(result.records) >= 1
    assert result.records[0].memory_id == "m2"


def test_upsert_overwrites_existing(tmp_path: Path) -> None:
    store = SQLiteLongTermMemoryStore(db_path=tmp_path / "kepler.db")
    store.upsert(LongTermMemoryRecord(memory_id="m1", topic="v1", content="old content"))
    store.upsert(LongTermMemoryRecord(memory_id="m1", topic="v2", content="new content"))

    result = store.search(LongTermMemoryQuery(query="new content"))
    assert len(result.records) == 1
    assert result.records[0].topic == "v2"


def test_enforce_record_limit(tmp_path: Path) -> None:
    store = SQLiteLongTermMemoryStore(db_path=tmp_path / "kepler.db", max_records=3)
    for i in range(5):
        store.upsert(LongTermMemoryRecord(memory_id=f"m{i}", content=f"record {i}"))
    count = store._conn.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()[0]
    assert count <= 3


def test_min_score_filters_results(tmp_path: Path) -> None:
    store = SQLiteLongTermMemoryStore(db_path=tmp_path / "kepler.db")
    store.upsert(LongTermMemoryRecord(memory_id="m1", content="completely unrelated xyz"))
    result = store.search(LongTermMemoryQuery(query="quantum physics", min_score=5.0))
    assert len(result.records) == 0


def test_shares_db_with_other_tables(tmp_path: Path) -> None:
    """Verify long_term_memory table coexists with other tables in the same db."""
    import sqlite3

    db_path = tmp_path / "kepler.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO tasks VALUES ('t1')")
    conn.execute("CREATE TABLE IF NOT EXISTS session_memory (session_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO session_memory VALUES ('s1')")
    conn.commit()

    store = SQLiteLongTermMemoryStore(db_path=db_path)
    store.upsert(LongTermMemoryRecord(memory_id="m1", content="test"))

    assert store.search(LongTermMemoryQuery(query="test")).records
    assert conn.execute("SELECT * FROM tasks").fetchone() is not None
    assert conn.execute("SELECT * FROM session_memory").fetchone() is not None
    conn.close()
