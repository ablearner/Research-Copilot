"""Attempt-private SQLite memory; only published business results may advance its snapshot."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from sqlalchemy import select
from adapters.storage.mysql import tables as t
from domain.schemas.identity import RequestContext
from services.identity import require_membership, ResourceNotFound


class MemoryCommitStore:
    def __init__(self, db, root):
        self.db, self.root = db, Path(root).resolve()

    async def prepare(self, task, versions):
        scope = {
            key: task[key] for key in ("user_id", "workspace_id", "project_id", "conversation_id")
        }
        scope["documents"] = versions
        digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        base = self.root / task["workspace_id"] / "memory" / digest
        attempt = base / task["id"] / str(task["execution_epoch"])
        manifest = dict(scope, task_id=task["id"], epoch=task["execution_epoch"])

        def stage():
            attempt.mkdir(parents=True, exist_ok=True)
            target = attempt / "memory.sqlite"
            current = base / "current.sqlite"
            if current.exists() and not target.exists():
                self.backup(current, target)
            temporary = attempt / "pending.tmp"
            temporary.write_text(json.dumps(manifest))
            os.replace(temporary, attempt / "pending.json")
            return target

        return await asyncio.to_thread(stage)

    @staticmethod
    def backup(source, target):
        temporary = target.with_suffix(".tmp")
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst)
        os.replace(temporary, target)

    async def commit(self, path):
        path = Path(path)
        pending = path.parent / "pending.json"
        if not pending.exists():
            return False
        manifest = json.loads(pending.read_text())
        async with self.db.transaction() as c:
            task = (
                (
                    await c.execute(
                        select(t.research_tasks).where(
                            t.research_tasks.c.id == manifest["task_id"],
                            t.research_tasks.c.workspace_id == manifest["workspace_id"],
                            t.research_tasks.c.execution_epoch == manifest["epoch"],
                            t.research_tasks.c.status.in_(["succeeded", "partially_succeeded"]),
                        )
                    )
                )
                .mappings()
                .first()
            )
            if not task:
                return False
            try:
                await require_membership(
                    c, RequestContext(task["user_id"], task["workspace_id"], "")
                )
            except ResourceNotFound:
                return False
            if not (
                await c.execute(
                    select(t.projects.c.id).where(
                        t.projects.c.id == task["project_id"], t.projects.c.status == "active"
                    )
                )
            ).first():
                return False
            for doc_id, version in manifest["documents"].items():
                valid = (
                    await c.execute(
                        select(t.documents.c.id).where(
                            t.documents.c.workspace_id == manifest["workspace_id"],
                            t.documents.c.id == doc_id,
                            t.documents.c.deleted_at.is_(None),
                            t.documents.c.active_version == version,
                        )
                    )
                ).first()
                if not valid:
                    return False
        watermark = path.parents[2] / "published.json"
        stamp = [task["updated_at"], task["id"], manifest["epoch"]]
        if watermark.exists() and json.loads(watermark.read_text())["stamp"] > stamp:
            await asyncio.to_thread(pending.rename, pending.with_name("superseded.json"))
            return False
        if path.exists():
            await asyncio.to_thread(self.backup, path, path.parents[2] / "current.sqlite")
        temporary = watermark.with_suffix(".tmp")
        temporary.write_text(json.dumps({"stamp": stamp}))
        os.replace(temporary, watermark)
        await asyncio.to_thread(pending.rename, pending.with_name("committed.json"))
        return True

    async def retry_pending(self, conversation_id=None):
        for pending in self.root.glob("*/memory/*/*/*/pending.json"):
            try:
                manifest = json.loads(pending.read_text())
                if conversation_id and manifest.get("conversation_id") != conversation_id:
                    continue
                await self.commit(pending.with_name("memory.sqlite"))
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                # Corrupt local metadata must not prevent unrelated tasks from recovering.
                # Database and I/O errors still propagate as dependency failures.
                logging.getLogger(__name__).exception("Corrupt memory manifest: %s", pending)
                pending.rename(pending.with_name("corrupt.manifest"))

    async def cleanup_document(self, workspace_id, document_id):
        def clean():
            for scope in (self.root / workspace_id / "memory").glob("*"):
                for manifest in scope.glob("*/*/*.json"):
                    if document_id in json.loads(manifest.read_text()).get("documents", {}):
                        shutil.rmtree(scope, ignore_errors=True)
                        break

        await asyncio.to_thread(clean)
