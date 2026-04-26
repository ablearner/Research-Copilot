from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

import json

from pydantic import BaseModel, Field

from domain.schemas.api import QAResponse
from rag_runtime.state import ChartDocRAGState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionMemorySnapshot(BaseModel):
    session_id: str
    current_document_id: str | None = None
    last_retrieval_summary: str | None = None
    last_answer_summary: str | None = None
    current_task_intent: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMemoryStore(Protocol):
    def get(self, session_id: str) -> SessionMemorySnapshot | None:
        ...

    def put(self, snapshot: SessionMemorySnapshot) -> SessionMemorySnapshot:
        ...

    def delete(self, session_id: str) -> None:
        ...


class InMemorySessionMemoryStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, SessionMemorySnapshot] = {}

    def get(self, session_id: str) -> SessionMemorySnapshot | None:
        return self._snapshots.get(session_id)

    def put(self, snapshot: SessionMemorySnapshot) -> SessionMemorySnapshot:
        self._snapshots[snapshot.session_id] = snapshot
        return snapshot

    def delete(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)


class MySQLSessionMemoryStore:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
        table_name: str = "session_memory",
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = "utf8mb4" if charset.lower() in {"utf8", "utf8mb3"} else charset
        self.table_name = table_name
        self.connection: Any | None = None

    def _ensure_connection(self) -> None:
        if self.connection is not None:
            return
        import pymysql

        self.connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    session_id VARCHAR(191) PRIMARY KEY,
                    current_document_id VARCHAR(191) NULL,
                    last_retrieval_summary LONGTEXT NULL,
                    last_answer_summary LONGTEXT NULL,
                    current_task_intent LONGTEXT NULL,
                    metadata_json LONGTEXT NOT NULL,
                    updated_at DATETIME NOT NULL
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                """
            )
        self.connection.commit()

    def get(self, session_id: str) -> SessionMemorySnapshot | None:
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {self.table_name} WHERE session_id = %s", (session_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return SessionMemorySnapshot(
            session_id=row["session_id"],
            current_document_id=row.get("current_document_id"),
            last_retrieval_summary=row.get("last_retrieval_summary"),
            last_answer_summary=row.get("last_answer_summary"),
            current_task_intent=row.get("current_task_intent"),
            updated_at=row.get("updated_at") or utc_now(),
            metadata=json.loads(row.get("metadata_json") or "{}"),
        )

    def put(self, snapshot: SessionMemorySnapshot) -> SessionMemorySnapshot:
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table_name} (
                    session_id, current_document_id, last_retrieval_summary, last_answer_summary,
                    current_task_intent, metadata_json, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    current_document_id = VALUES(current_document_id),
                    last_retrieval_summary = VALUES(last_retrieval_summary),
                    last_answer_summary = VALUES(last_answer_summary),
                    current_task_intent = VALUES(current_task_intent),
                    metadata_json = VALUES(metadata_json),
                    updated_at = VALUES(updated_at)
                """,
                (
                    snapshot.session_id,
                    snapshot.current_document_id,
                    snapshot.last_retrieval_summary,
                    snapshot.last_answer_summary,
                    snapshot.current_task_intent,
                    json.dumps(snapshot.metadata, ensure_ascii=False),
                    snapshot.updated_at.replace(tzinfo=None),
                ),
            )
        self.connection.commit()
        return snapshot

    def delete(self, session_id: str) -> None:
        self._ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self.table_name} WHERE session_id = %s", (session_id,))
        self.connection.commit()


class GraphSessionMemory:
    """Structured session memory built from graph state instead of raw chat concatenation."""

    def __init__(self, store: SessionMemoryStore | None = None) -> None:
        self.store = store or InMemorySessionMemoryStore()

    def load(self, session_id: str | None) -> SessionMemorySnapshot | None:
        if not session_id:
            return None
        return self.store.get(session_id)

    def clear(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.store.delete(session_id)

    def update_from_state(self, state: ChartDocRAGState) -> SessionMemorySnapshot | None:
        session_id = state.get("session_id")
        if not session_id:
            return None
        existing = self.store.get(session_id) or SessionMemorySnapshot(session_id=session_id)
        updated = existing.model_copy(
            update={
                "current_document_id": state.get("document_id")
                or (state.get("document_ids") or [None])[0]
                or existing.current_document_id,
                "last_retrieval_summary": self._retrieval_summary(state) or existing.last_retrieval_summary,
                "last_answer_summary": self._answer_summary(state) or existing.last_answer_summary,
                "current_task_intent": self._task_intent(state) or existing.current_task_intent,
                "updated_at": utc_now(),
                "metadata": {
                    **existing.metadata,
                    "last_request_id": state.get("request_id"),
                    "last_thread_id": state.get("thread_id"),
                    "warning_count": len(state.get("warnings", [])),
                },
            }
        )
        return self.store.put(updated)

    def as_prompt_context(self, snapshot: SessionMemorySnapshot | None) -> dict[str, Any]:
        if snapshot is None:
            return {"memory_enabled": False}
        return {
            "memory_enabled": True,
            "current_document_id": snapshot.current_document_id,
            "last_retrieval_summary": snapshot.last_retrieval_summary,
            "last_answer_summary": snapshot.last_answer_summary,
            "current_task_intent": snapshot.current_task_intent,
            "updated_at": snapshot.updated_at.isoformat(),
        }

    def append_chart_turn(
        self,
        *,
        session_id: str,
        image_path: str,
        question: str,
        answer: str,
        chart_id: str | None = None,
        document_id: str | None = None,
        page_id: str | None = None,
    ) -> SessionMemorySnapshot:
        existing = self.store.get(session_id) or SessionMemorySnapshot(session_id=session_id)
        chart_turns = list(existing.metadata.get("chart_turns") or [])
        chart_turns.append(
            {
                "image_path": image_path,
                "chart_id": chart_id,
                "document_id": document_id,
                "page_id": page_id,
                "question": question,
                "answer": answer,
                "updated_at": utc_now().isoformat(),
            }
        )
        chart_turns = chart_turns[-10:]
        updated = existing.model_copy(
            update={
                "current_document_id": document_id or existing.current_document_id,
                "last_answer_summary": answer[:280],
                "current_task_intent": f"chart_qa:{question[:160]}",
                "updated_at": utc_now(),
                "metadata": {**existing.metadata, "chart_turns": chart_turns},
            }
        )
        return self.store.put(updated)

    def chart_history(self, session_id: str | None, image_path: str | None = None) -> list[dict[str, str]]:
        if not session_id:
            return []
        snapshot = self.store.get(session_id)
        if snapshot is None:
            return []
        turns = snapshot.metadata.get("chart_turns") or []
        history: list[dict[str, str]] = []
        for turn in turns:
            if image_path and turn.get("image_path") != image_path:
                continue
            history.append({"question": str(turn.get("question") or ""), "answer": str(turn.get("answer") or "")})
        return history[-8:]

    def update_research_context(
        self,
        *,
        session_id: str | None,
        current_document_id: str | None = None,
        last_retrieval_summary: str | None = None,
        last_answer_summary: str | None = None,
        current_task_intent: str | None = None,
        metadata_update: dict[str, Any] | None = None,
    ) -> SessionMemorySnapshot | None:
        if not session_id:
            return None
        existing = self.store.get(session_id) or SessionMemorySnapshot(session_id=session_id)
        cleaned_metadata = {
            key: value
            for key, value in (metadata_update or {}).items()
            if value is not None
        }
        updated = existing.model_copy(
            update={
                "current_document_id": current_document_id or existing.current_document_id,
                "last_retrieval_summary": last_retrieval_summary or existing.last_retrieval_summary,
                "last_answer_summary": last_answer_summary or existing.last_answer_summary,
                "current_task_intent": current_task_intent or existing.current_task_intent,
                "updated_at": utc_now(),
                "metadata": {
                    **existing.metadata,
                    **cleaned_metadata,
                },
            }
        )
        return self.store.put(updated)

    def append_research_turn(
        self,
        *,
        session_id: str | None,
        question: str,
        answer: str,
        task_id: str | None = None,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMemorySnapshot | None:
        if not session_id:
            return None
        existing = self.store.get(session_id) or SessionMemorySnapshot(session_id=session_id)
        research_turns = list(existing.metadata.get("research_turns") or [])
        research_turns.append(
            {
                "question": question,
                "answer": answer,
                "task_id": task_id,
                "conversation_id": conversation_id,
                "document_ids": list(document_ids or []),
                "updated_at": utc_now().isoformat(),
                "metadata": metadata or {},
            }
        )
        research_turns = research_turns[-10:]
        updated = existing.model_copy(
            update={
                "current_document_id": (document_ids or [None])[0] or existing.current_document_id,
                "last_answer_summary": answer[:280] or existing.last_answer_summary,
                "current_task_intent": f"research_qa:{question[:160]}",
                "updated_at": utc_now(),
                "metadata": {
                    **existing.metadata,
                    "research_turns": research_turns,
                    **({"last_task_id": task_id} if task_id else {}),
                    **({"last_conversation_id": conversation_id} if conversation_id else {}),
                },
            }
        )
        return self.store.put(updated)

    def research_history(self, session_id: str | None) -> list[dict[str, Any]]:
        if not session_id:
            return []
        snapshot = self.store.get(session_id)
        if snapshot is None:
            return []
        turns = snapshot.metadata.get("research_turns") or []
        history: list[dict[str, Any]] = []
        for turn in turns:
            history.append(
                {
                    "question": str(turn.get("question") or ""),
                    "answer": str(turn.get("answer") or ""),
                    "task_id": str(turn.get("task_id") or "") or None,
                    "conversation_id": str(turn.get("conversation_id") or "") or None,
                    "document_ids": list(turn.get("document_ids") or []),
                    "updated_at": str(turn.get("updated_at") or ""),
                }
            )
        return history[-6:]

    def _retrieval_summary(self, state: ChartDocRAGState) -> str | None:
        retrieval_meta = state.get("metadata", {})
        vector_meta = retrieval_meta.get("vector_retrieval") or {}
        graph_meta = retrieval_meta.get("graph_retrieval") or {}
        summary_meta = retrieval_meta.get("summary_retrieval") or {}
        merged_hit_count = retrieval_meta.get("merged_hit_count")
        counts = [
            f"vector_hits={len((vector_meta.get('hits') or []))}" if isinstance(vector_meta, dict) else None,
            f"graph_hits={len((graph_meta.get('hits') or []))}" if isinstance(graph_meta, dict) else None,
            f"summary_hits={len((summary_meta.get('hits') or []))}" if isinstance(summary_meta, dict) else None,
            f"merged_hits={merged_hit_count}" if merged_hit_count is not None else None,
        ]
        summary = ", ".join(item for item in counts if item)
        return summary or None

    def _answer_summary(self, state: ChartDocRAGState) -> str | None:
        final_answer = state.get("final_answer")
        if isinstance(final_answer, QAResponse):
            text = final_answer.answer.strip()
            return text[:280] if text else None
        if isinstance(final_answer, dict):
            text = str(final_answer.get("answer") or "").strip()
            return text[:280] if text else None
        return None

    def _task_intent(self, state: ChartDocRAGState) -> str | None:
        if state.get("task_intent"):
            return str(state["task_intent"])
        task_type = state.get("task_type")
        question = (state.get("user_input") or "").strip()
        if task_type == "ask" and question:
            return f"qa:{question[:160]}"
        return task_type
