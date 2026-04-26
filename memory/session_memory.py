from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Protocol

from domain.schemas.research_context import ResearchContext
from domain.schemas.research_memory import ResearchSessionSummary, SessionMemoryRecord


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _storage_name(key: str) -> str:
    return sha1(key.encode("utf-8")).hexdigest()


class SessionMemoryStore(Protocol):
    def get(self, session_id: str) -> SessionMemoryRecord | None:
        ...

    def put(self, record: SessionMemoryRecord) -> SessionMemoryRecord:
        ...

    def delete(self, session_id: str) -> None:
        ...


class InMemorySessionMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, SessionMemoryRecord] = {}

    def get(self, session_id: str) -> SessionMemoryRecord | None:
        return self._records.get(session_id)

    def put(self, record: SessionMemoryRecord) -> SessionMemoryRecord:
        self._records[record.session_id] = record
        return record

    def delete(self, session_id: str) -> None:
        self._records.pop(session_id, None)


class JsonSessionMemoryStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self._ensure_root_dir()

    def _ensure_root_dir(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.root_dir / f"{_storage_name(session_id)}.json"

    def get(self, session_id: str) -> SessionMemoryRecord | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        return SessionMemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, record: SessionMemoryRecord) -> SessionMemoryRecord:
        path = self._path_for(record.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def delete(self, session_id: str) -> None:
        path = self._path_for(session_id)
        if path.exists():
            path.unlink()


class SessionMemory:
    def __init__(self, store: SessionMemoryStore | None = None) -> None:
        self.store = store or InMemorySessionMemoryStore()

    def load(self, session_id: str) -> SessionMemoryRecord:
        return self.store.get(session_id) or SessionMemoryRecord(session_id=session_id)

    def save(self, record: SessionMemoryRecord) -> SessionMemoryRecord:
        updated = record.model_copy(update={"updated_at": utc_now()})
        return self.store.put(updated)

    def clear(self, session_id: str) -> None:
        self.store.delete(session_id)

    def update_context(self, session_id: str, context: ResearchContext) -> SessionMemoryRecord:
        record = self.load(session_id)
        return self.save(record.model_copy(update={"context": context}))

    def append_read_paper(self, session_id: str, paper_id: str) -> SessionMemoryRecord:
        record = self.load(session_id)
        read_paper_ids = list(dict.fromkeys([*record.read_paper_ids, paper_id]))
        return self.save(record.model_copy(update={"read_paper_ids": read_paper_ids}))

    def append_question(self, session_id: str, question: str) -> SessionMemoryRecord:
        record = self.load(session_id)
        questions = [*record.questions, question]
        return self.save(record.model_copy(update={"questions": questions[-50:]}))

    def append_conclusion(self, session_id: str, conclusion: str) -> SessionMemoryRecord:
        record = self.load(session_id)
        conclusions = [*record.conclusions, conclusion]
        return self.save(record.model_copy(update={"conclusions": conclusions[-50:]}))

    def finalize_session(self, session_id: str) -> SessionMemoryRecord:
        record = self.load(session_id)
        topic = record.context.research_topic
        key_papers = record.read_paper_ids[:10]
        questions = record.questions[-8:]
        conclusions = list(
            dict.fromkeys(
                [*record.context.known_conclusions, *record.conclusions]
            )
        )[:8]
        summary_lines = []
        if topic:
            summary_lines.append(f"研究主题：{topic}")
        if key_papers:
            summary_lines.append(f"已阅读论文：{', '.join(key_papers[:5])}")
        if conclusions:
            summary_lines.append(f"阶段性结论：{'；'.join(conclusions[:3])}")
        open_questions = record.context.open_questions[:3]
        if open_questions:
            summary_lines.append(f"待解决问题：{'；'.join(open_questions)}")
        if not summary_lines:
            summary_lines.append("本次研究会话信息不足，暂无法生成完整摘要。")
        summary = ResearchSessionSummary(
            session_id=session_id,
            research_topic=topic,
            key_papers=key_papers,
            questions=questions,
            conclusions=conclusions,
            summary_text="\n".join(summary_lines),
            metadata={"generated_by": "SessionMemory"},
        )
        return self.save(record.model_copy(update={"summary": summary}))
