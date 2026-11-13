"""Scoped business records and private artifacts. Database is the visibility authority."""

import hashlib
import logging
from uuid import uuid4
from sqlalchemy import select, insert, update, func
from adapters.storage.mysql import tables as t
from services.identity import require_membership, ResourceNotFound, PermissionDenied


class BusinessService:
    def __init__(self, db, objects, max_file_bytes=50 * 1024 * 1024):
        self.db, self.objects, self.max_file_bytes = db, objects, max_file_bytes

    async def _auth(self, conn, ctx, write=False):
        role = await require_membership(conn, ctx)
        if write and role not in ("owner", "admin", "editor"):
            raise PermissionDenied()

    async def _resource(self, conn, ctx, table, rid, lock=False):
        query = select(table).where(table.c.workspace_id == ctx.workspace_id, table.c.id == rid)
        if "status" in table.c:
            query = query.where(table.c.status != "deleted")
        if lock:
            query = query.with_for_update()
        row = (await conn.execute(query)).mappings().first()
        if not row:
            raise ResourceNotFound("Resource not found")
        return dict(row)

    async def upload(self, ctx, project_id, filename, data):
        if not data.startswith(b"%PDF-") or len(data) > self.max_file_bytes:
            raise ValueError("A PDF within the upload size limit is required")
        digest = hashlib.sha256(data).hexdigest()
        file_id, doc_id, version_id, paper_id = [str(uuid4()) for _ in range(4)]
        key = f"workspaces/{ctx.workspace_id}/documents/{doc_id}/{version_id}/source.pdf"
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx, True)
            await self._resource(conn, ctx, t.projects, project_id)
        await self.objects.put(key, data, "application/pdf")
        try:
            async with self.db.transaction() as conn:
                await self._auth(conn, ctx, True)
                await conn.execute(
                    select(t.workspaces.c.id)
                    .where(t.workspaces.c.id == ctx.workspace_id)
                    .with_for_update()
                )
                await self._resource(conn, ctx, t.projects, project_id)
                existing = (
                    (
                        await conn.execute(
                            select(
                                t.documents.c.id,
                                t.documents.c.paper_id,
                                t.file_objects.c.id.label("file_id"),
                            )
                            .join(
                                t.document_versions,
                                t.document_versions.c.document_id == t.documents.c.id,
                            )
                            .join(
                                t.file_objects, t.file_objects.c.id == t.document_versions.c.file_id
                            )
                            .where(
                                t.documents.c.workspace_id == ctx.workspace_id,
                                t.documents.c.deleted_at.is_(None),
                                t.file_objects.c.sha256 == digest,
                                t.file_objects.c.status == "available",
                            )
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing:
                    doc_id, paper_id, file_id = (
                        existing["id"],
                        existing["paper_id"],
                        existing["file_id"],
                    )
                else:
                    await conn.execute(
                        insert(t.papers).values(
                            id=paper_id,
                            workspace_id=ctx.workspace_id,
                            title=filename[:512],
                            metadata_json={},
                        )
                    )
                    await conn.execute(
                        insert(t.file_objects).values(
                            id=file_id,
                            workspace_id=ctx.workspace_id,
                            object_key=key,
                            sha256=digest,
                            size=len(data),
                            mime="application/pdf",
                            status="available",
                        )
                    )
                    await conn.execute(
                        insert(t.documents).values(
                            id=doc_id,
                            workspace_id=ctx.workspace_id,
                            paper_id=paper_id,
                            status="uploaded",
                        )
                    )
                    await conn.execute(
                        insert(t.document_versions).values(
                            id=version_id,
                            workspace_id=ctx.workspace_id,
                            document_id=doc_id,
                            file_id=file_id,
                            sha256=digest,
                            parser_version="pymupdf-v1",
                            embedding_version="configured",
                        )
                    )
                link = (
                    await conn.execute(
                        select(t.project_papers.c.id).where(
                            t.project_papers.c.workspace_id == ctx.workspace_id,
                            t.project_papers.c.project_id == project_id,
                            t.project_papers.c.workspace_id == ctx.workspace_id,
                            t.project_papers.c.paper_id == paper_id,
                        )
                    )
                ).first()
                if not link:
                    await conn.execute(
                        insert(t.project_papers).values(
                            id=str(uuid4()),
                            workspace_id=ctx.workspace_id,
                            project_id=project_id,
                            paper_id=paper_id,
                        )
                    )
            if existing:
                try:
                    await self.objects.delete(key)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Orphan object pending reconciliation: %s", key
                    )
        except Exception:
            # Do not delete the object on an ambiguous commit: later orphan collection reconciles it.
            raise
        return {
            "document_id": doc_id,
            "paper_id": paper_id,
            "file_id": file_id,
            "duplicate": bool(existing),
        }

    async def documents(self, ctx, project_id, *, document_ids=None):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx)
            await self._resource(conn, ctx, t.projects, project_id)
            latest = t.document_versions.alias("latest_version")
            latest_id = (
                select(latest.c.id)
                .where(
                    latest.c.document_id == t.documents.c.id,
                    latest.c.workspace_id == ctx.workspace_id,
                )
                .order_by(latest.c.created_at.desc(), latest.c.id.desc())
                .limit(1)
                .correlate(t.documents)
                .scalar_subquery()
            )
            rows = (
                (
                    await conn.execute(
                        select(
                            t.documents,
                            t.papers.c.title,
                            t.document_versions.c.file_id,
                            t.document_versions.c.parse_status,
                            t.document_versions.c.index_status,
                        )
                        .join(
                            t.project_papers, t.project_papers.c.paper_id == t.documents.c.paper_id
                        )
                        .join(t.papers, t.papers.c.id == t.documents.c.paper_id)
                        .join(
                            t.document_versions,
                            t.document_versions.c.id
                            == func.coalesce(t.documents.c.active_version, latest_id),
                        )
                        .where(
                            t.documents.c.workspace_id == ctx.workspace_id,
                            t.project_papers.c.project_id == project_id,
                            t.project_papers.c.workspace_id == ctx.workspace_id,
                            t.documents.c.deleted_at.is_(None),
                            t.documents.c.id.in_(document_ids)
                            if document_ids is not None
                            else True,
                        )
                        .order_by(t.documents.c.created_at, t.documents.c.id)
                        .limit(100)
                    )
                )
                .mappings()
                .all()
            )
            return [dict(r) for r in rows]

    async def file(self, ctx, file_id):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx)
            row = (
                (
                    await conn.execute(
                        select(t.file_objects)
                        .join(
                            t.document_versions,
                            t.document_versions.c.file_id == t.file_objects.c.id,
                        )
                        .join(t.documents, t.documents.c.id == t.document_versions.c.document_id)
                        .where(
                            t.file_objects.c.id == file_id,
                            t.file_objects.c.workspace_id == ctx.workspace_id,
                            t.file_objects.c.status == "available",
                            t.documents.c.deleted_at.is_(None),
                        )
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                raise ResourceNotFound()
        return await self.objects.get(row["object_key"]), row["mime"]

    async def delete_document(self, ctx, document_id):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx, True)
            await self._resource(conn, ctx, t.documents, document_id, True)
            await conn.execute(
                update(t.documents)
                .where(t.documents.c.id == document_id)
                .values(status="deleted", deleted_at=t.now_ms(), version=t.documents.c.version + 1)
            )
            await conn.execute(
                insert(t.outbox_events).values(
                    id=str(uuid4()),
                    workspace_id=ctx.workspace_id,
                    kind="document.cleanup",
                    payload={"document_id": document_id, "workspace_id": ctx.workspace_id},
                )
            )
            await conn.execute(
                insert(t.audit_events).values(
                    id=str(uuid4()),
                    workspace_id=ctx.workspace_id,
                    actor_id=ctx.user_id,
                    action="document.delete",
                    resource_id=document_id,
                    request_id=ctx.request_id,
                    outcome="success",
                )
            )

    async def create_conversation(self, ctx, project_id, title="New research"):
        cid = str(uuid4())
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx, True)
            await self._resource(conn, ctx, t.projects, project_id)
            await conn.execute(
                insert(t.conversations).values(
                    id=cid, workspace_id=ctx.workspace_id, project_id=project_id, title=title[:255]
                )
            )
            return await self._resource(conn, ctx, t.conversations, cid)

    async def conversations(self, ctx, project_id):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx)
            await self._resource(conn, ctx, t.projects, project_id)
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        select(t.conversations)
                        .where(
                            t.conversations.c.workspace_id == ctx.workspace_id,
                            t.conversations.c.project_id == project_id,
                            t.conversations.c.status != "deleted",
                        )
                        .order_by(t.conversations.c.updated_at.desc(), t.conversations.c.id)
                        .limit(100)
                    )
                ).mappings()
            ]

    async def add_message(self, ctx, cid, role, content, task_id=None):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx, True)
            conv = await self._resource(conn, ctx, t.conversations, cid, True)
            await self._resource(conn, ctx, t.projects, conv["project_id"])
            if task_id:
                task = await self._resource(conn, ctx, t.research_tasks, task_id)
                if task["conversation_id"] != cid or task["project_id"] != conv["project_id"]:
                    raise ResourceNotFound()
            seq = (
                await conn.execute(
                    select(func.coalesce(func.max(t.messages.c.seq), 0)).where(
                        t.messages.c.conversation_id == cid
                    )
                )
            ).scalar_one() + 1
            mid = str(uuid4())
            await conn.execute(
                insert(t.messages).values(
                    id=mid,
                    workspace_id=ctx.workspace_id,
                    conversation_id=cid,
                    seq=seq,
                    role=role,
                    content=content,
                    task_id=task_id,
                )
            )
            await conn.execute(
                update(t.conversations)
                .where(t.conversations.c.id == cid)
                .values(updated_at=t.now_ms())
            )
            return dict(
                (await conn.execute(select(t.messages).where(t.messages.c.id == mid)))
                .mappings()
                .one()
            )

    async def messages(self, ctx, cid, after=0, limit=50, *, latest=False, before=None):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx)
            conv = await self._resource(conn, ctx, t.conversations, cid)
            await self._resource(conn, ctx, t.projects, conv["project_id"])
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        select(t.messages)
                        .where(
                            t.messages.c.workspace_id == ctx.workspace_id,
                            t.messages.c.conversation_id == cid,
                            t.messages.c.seq > after,
                            *([t.messages.c.seq < before] if before is not None else []),
                        )
                        .order_by(t.messages.c.seq.desc() if latest else t.messages.c.seq)
                        .limit(max(1, min(limit, 100)))
                    )
                ).mappings()
            ]

    async def reports(self, ctx, project_id):
        async with self.db.transaction() as conn:
            await self._auth(conn, ctx)
            await self._resource(conn, ctx, t.projects, project_id)
            return [
                dict(r)
                for r in (
                    await conn.execute(
                        select(
                            t.reports,
                            t.report_versions.c.content,
                            t.report_versions.c.version.label("report_version"),
                        )
                        .join(t.report_versions, t.report_versions.c.report_id == t.reports.c.id)
                        .where(
                            t.reports.c.workspace_id == ctx.workspace_id,
                            t.reports.c.project_id == project_id,
                        )
                        .order_by(t.reports.c.created_at.desc())
                        .limit(100)
                    )
                ).mappings()
            ]
