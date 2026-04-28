"""SQLite-backed storage backend for Kepler research data."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from domain.schemas.research import (
    PaperCandidate,
    ResearchConversation,
    ResearchJob,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
)

from .migrations import run_migrations


class SQLiteStore:
    """Transactional storage backend using SQLite with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Run migrations on the main connection
        conn = self._get_conn()
        run_migrations(conn)

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local connection with WAL mode."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # ── Tasks ──

    def save_task(self, task: ResearchTask) -> None:
        data = json.dumps(task.model_dump(mode="json"), ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks (task_id, topic, status, created_at, updated_at, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task.task_id, task.topic, task.status, task.created_at, task.updated_at, data),
            )

    def load_task(self, task_id: str) -> ResearchTask | None:
        row = self._get_conn().execute(
            "SELECT data_json FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return ResearchTask.model_validate(json.loads(row[0]))

    # ── Papers ──

    def save_papers(self, task_id: str, papers: list[PaperCandidate]) -> None:
        data = json.dumps(
            [p.model_dump(mode="json") for p in papers], ensure_ascii=False
        )
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO papers (task_id, data_json) VALUES (?, ?)",
                (task_id, data),
            )

    def load_papers(self, task_id: str) -> list[PaperCandidate]:
        row = self._get_conn().execute(
            "SELECT data_json FROM papers WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return []
        return [PaperCandidate.model_validate(item) for item in json.loads(row[0])]

    # ── Reports ──

    def save_report(self, report: ResearchReport) -> None:
        data = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports (report_id, task_id, created_at, data_json) "
                "VALUES (?, ?, ?, ?)",
                (report.report_id, report.task_id, report.generated_at, data),
            )

    def load_report(self, task_id: str, report_id: str | None = None) -> ResearchReport | None:
        if report_id:
            row = self._get_conn().execute(
                "SELECT data_json FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        else:
            row = self._get_conn().execute(
                "SELECT data_json FROM reports WHERE task_id = ? ORDER BY created_at DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchReport.model_validate(json.loads(row[0]))

    # ── Conversations ──

    def save_conversation(self, conversation: ResearchConversation) -> None:
        data = json.dumps(conversation.model_dump(mode="json"), ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations "
                "(conversation_id, title, task_id, created_at, updated_at, snapshot_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation.conversation_id,
                    conversation.title,
                    getattr(conversation, "task_id", None),
                    conversation.created_at,
                    conversation.updated_at,
                    data,
                ),
            )

    def load_conversation(self, conversation_id: str) -> ResearchConversation | None:
        row = self._get_conn().execute(
            "SELECT snapshot_json FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return ResearchConversation.model_validate(json.loads(row[0]))

    def list_conversations(self) -> list[ResearchConversation]:
        rows = self._get_conn().execute(
            "SELECT snapshot_json FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [ResearchConversation.model_validate(json.loads(r[0])) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )

    # ── Messages ──

    def save_messages(self, conversation_id: str, messages: list[ResearchMessage]) -> None:
        data = json.dumps(
            [m.model_dump(mode="json") for m in messages], ensure_ascii=False
        )
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO messages (conversation_id, data_json) VALUES (?, ?)",
                (conversation_id, data),
            )

    def load_messages(self, conversation_id: str) -> list[ResearchMessage]:
        row = self._get_conn().execute(
            "SELECT data_json FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return []
        return [ResearchMessage.model_validate(item) for item in json.loads(row[0])]

    # ── Jobs ──

    def save_job(self, job: ResearchJob) -> None:
        data = json.dumps(job.model_dump(mode="json"), ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs "
                "(job_id, conversation_id, task_id, created_at, updated_at, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job.job_id, job.conversation_id, job.task_id, job.created_at, job.updated_at, data),
            )

    def load_job(self, job_id: str) -> ResearchJob | None:
        row = self._get_conn().execute(
            "SELECT data_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return ResearchJob.model_validate(json.loads(row[0]))

    def list_jobs(
        self, *, conversation_id: str | None = None, task_id: str | None = None
    ) -> list[ResearchJob]:
        if conversation_id:
            rows = self._get_conn().execute(
                "SELECT data_json FROM jobs WHERE conversation_id = ? ORDER BY updated_at DESC",
                (conversation_id,),
            ).fetchall()
        elif task_id:
            rows = self._get_conn().execute(
                "SELECT data_json FROM jobs WHERE task_id = ? ORDER BY updated_at DESC",
                (task_id,),
            ).fetchall()
        else:
            rows = self._get_conn().execute(
                "SELECT data_json FROM jobs ORDER BY updated_at DESC"
            ).fetchall()
        return [ResearchJob.model_validate(json.loads(r[0])) for r in rows]

    def delete_jobs(
        self, *, conversation_id: str | None = None, task_id: str | None = None
    ) -> None:
        with self._get_conn() as conn:
            if conversation_id:
                conn.execute(
                    "DELETE FROM jobs WHERE conversation_id = ?", (conversation_id,)
                )
            elif task_id:
                conn.execute("DELETE FROM jobs WHERE task_id = ?", (task_id,))

    # ── Cleanup ──

    def delete_task_artifacts(self, task_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM papers WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM reports WHERE task_id = ?", (task_id,))

    def clear_all(self) -> None:
        with self._get_conn() as conn:
            for table in ("conversations", "tasks", "reports", "papers", "messages", "jobs"):
                conn.execute(f"DELETE FROM {table}")
