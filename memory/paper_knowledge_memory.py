from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Protocol

from domain.schemas.paper_knowledge import PaperKnowledgeRecord


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _storage_name(key: str) -> str:
    return sha1(key.encode("utf-8")).hexdigest()


class PaperKnowledgeStore(Protocol):
    def get(self, paper_id: str) -> PaperKnowledgeRecord | None:
        ...

    def put(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        ...


class JsonPaperKnowledgeStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self._ensure_root_dir()

    def _ensure_root_dir(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, paper_id: str) -> Path:
        return self.root_dir / f"{_storage_name(paper_id)}.json"

    def get(self, paper_id: str) -> PaperKnowledgeRecord | None:
        path = self._path_for(paper_id)
        if not path.exists():
            return None
        return PaperKnowledgeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        path = self._path_for(record.paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record


class InMemoryPaperKnowledgeStore:
    def __init__(self) -> None:
        self._records: dict[str, PaperKnowledgeRecord] = {}

    def get(self, paper_id: str) -> PaperKnowledgeRecord | None:
        return self._records.get(paper_id)

    def put(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        self._records[record.paper_id] = record
        return record


class SQLitePaperKnowledgeStore:
    """SQLite store for per-paper knowledge cards."""

    def __init__(self, db_path: str | Path) -> None:
        import sqlite3

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_paper_knowledge (
                paper_id TEXT PRIMARY KEY,
                document_id TEXT,
                title TEXT,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_research_paper_knowledge_document_id
            ON research_paper_knowledge(document_id);
            """
        )

    def get(self, paper_id: str) -> PaperKnowledgeRecord | None:
        row = self._conn.execute(
            "SELECT payload_json FROM research_paper_knowledge WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        return PaperKnowledgeRecord.model_validate_json(row[0])

    def put(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO research_paper_knowledge "
                "(paper_id, document_id, title, payload_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.paper_id,
                    record.document_id,
                    record.title,
                    record.model_dump_json(),
                    record.updated_at.isoformat(),
                ),
            )
        return record


class PaperKnowledgeMemory:
    def __init__(self, store: PaperKnowledgeStore) -> None:
        self.store = store

    def load(self, paper_id: str) -> PaperKnowledgeRecord | None:
        return self.store.get(paper_id)

    def upsert(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        updated = record.model_copy(update={"updated_at": utc_now()})
        return self.store.put(updated)

    def append_user_annotation(self, paper_id: str, annotation: str) -> PaperKnowledgeRecord | None:
        existing = self.store.get(paper_id)
        if existing is None:
            return None
        annotations = [*existing.user_annotations, annotation]
        return self.upsert(existing.model_copy(update={"user_annotations": annotations}))
