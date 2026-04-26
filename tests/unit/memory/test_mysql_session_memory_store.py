import json
from datetime import datetime

from rag_runtime.memory import MySQLSessionMemoryStore, SessionMemorySnapshot


class CursorStub:
    def __init__(self, row=None):
        self.row = row
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ConnectionStub:
    def __init__(self, row=None):
        self.row = row
        self.cursor_instance = CursorStub(row=row)
        self.commit_count = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commit_count += 1


def test_mysql_session_memory_store_put_serializes_metadata() -> None:
    store = MySQLSessionMemoryStore(host="localhost", port=3306, user="root", password="1", database="ecs")
    store.connection = ConnectionStub()

    snapshot = SessionMemorySnapshot(session_id="s1", metadata={"chart_turns": [{"question": "q1"}]})
    store.put(snapshot)

    _, params = store.connection.cursor_instance.executed[-1]
    assert json.loads(params[5])["chart_turns"][0]["question"] == "q1"


def test_mysql_session_memory_store_get_deserializes_snapshot() -> None:
    row = {
        "session_id": "s1",
        "current_document_id": "doc1",
        "last_retrieval_summary": None,
        "last_answer_summary": "answer",
        "current_task_intent": "chart_qa:test",
        "metadata_json": json.dumps({"chart_turns": [{"question": "q1", "answer": "a1"}]}),
        "updated_at": datetime.utcnow(),
    }
    store = MySQLSessionMemoryStore(host="localhost", port=3306, user="root", password="1", database="ecs")
    store.connection = ConnectionStub(row=row)

    snapshot = store.get("s1")

    assert snapshot is not None
    assert snapshot.session_id == "s1"
    assert snapshot.metadata["chart_turns"][0]["answer"] == "a1"
