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
