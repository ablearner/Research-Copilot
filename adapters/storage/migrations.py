"""Schema versioning for SQLite storage backend."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

MIGRATIONS: list[tuple[str, str]] = [
    (
        "001_initial",
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT,
            task_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            snapshot_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            topic TEXT,
            status TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            task_id TEXT,
            created_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS papers (
            task_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (task_id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            conversation_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (conversation_id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            task_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic);
        CREATE INDEX IF NOT EXISTS idx_reports_task ON reports(task_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_conv ON jobs(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_task ON jobs(task_id);
        """,
    ),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations in order."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (id TEXT PRIMARY KEY, applied_at TEXT)"
    )
    applied = {row[0] for row in conn.execute("SELECT id FROM _migrations")}
    for migration_id, sql in MIGRATIONS:
        if migration_id not in applied:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO _migrations VALUES (?, ?)",
                (migration_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
