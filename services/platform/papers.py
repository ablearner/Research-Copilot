"""Workspace paper metadata independent of document ingestion.

Metadata strings, including academic identifiers and URLs, are inert data: this
service never fetches remote content or reads local paths.
"""

import json
from uuid import uuid4
from sqlalchemy import select, insert, delete
from adapters.storage.mysql import tables as t
from services.identity import require_membership, ResourceNotFound, PermissionDenied


class PaperService:
    def __init__(self, db):
        self.db = db

    async def _authorize(self, conn, ctx, *, write=False):
        role = await require_membership(conn, ctx)
        if write and role not in ("owner", "admin", "editor"):
            raise PermissionDenied("Write permission required")
        if write:
            # Serialize association insert/check with other workspace writers.
            await conn.execute(
                select(t.workspaces.c.id)
                .where(t.workspaces.c.id == ctx.workspace_id)
                .with_for_update()
            )

    async def _resource(self, conn, ctx, table, resource_id):
        row = (
            (
                await conn.execute(
                    select(table).where(
                        table.c.workspace_id == ctx.workspace_id,
                        table.c.id == resource_id,
                        table.c.status != "deleted",
                    )
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise ResourceNotFound("Resource not found")
        return dict(row)

    @staticmethod
    def _association(ctx, project_id, paper_id):
        return (
            t.project_papers.c.workspace_id == ctx.workspace_id,
            t.project_papers.c.project_id == project_id,
            t.project_papers.c.paper_id == paper_id,
        )

    async def _attach(self, conn, ctx, project_id, paper_id):
        existing = (
            await conn.execute(
                select(t.project_papers.c.id).where(*self._association(ctx, project_id, paper_id))
            )
        ).scalar_one_or_none()
        if existing is None:
            await conn.execute(
                insert(t.project_papers).values(
                    id=str(uuid4()),
                    workspace_id=ctx.workspace_id,
                    project_id=project_id,
                    paper_id=paper_id,
                )
            )

    async def save(self, ctx, title, metadata, *, project_id=None):
        """Save a metadata-only paper, optionally attaching it atomically."""
        if not isinstance(title, str) or not title.strip() or len(title) > 2048:
            raise ValueError("Paper title required (maximum 2048 characters)")
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        try:
            encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode("utf-8")) > 65536:
                raise ValueError("Metadata exceeds 64 KiB")
            metadata = json.loads(encoded)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Metadata must be a JSON object within 64 KiB") from exc
        paper_id = str(uuid4())
        async with self.db.transaction() as conn:
            await self._authorize(conn, ctx, write=True)
            if project_id is not None:
                await self._resource(conn, ctx, t.projects, project_id)
            await conn.execute(
                insert(t.papers).values(
                    id=paper_id,
                    workspace_id=ctx.workspace_id,
                    title=title.strip(),
                    metadata_json=metadata,
                    status="active",
                )
            )
            if project_id is not None:
                await self._attach(conn, ctx, project_id, paper_id)
            return await self._resource(conn, ctx, t.papers, paper_id)

    async def get(self, ctx, paper_id):
        async with self.db.transaction() as conn:
            await self._authorize(conn, ctx)
            return await self._resource(conn, ctx, t.papers, paper_id)

    async def list(self, ctx, *, project_id=None, limit=100, offset=0, paper_ids=None):
        async with self.db.transaction() as conn:
            await self._authorize(conn, ctx)
            query = select(t.papers).where(
                t.papers.c.workspace_id == ctx.workspace_id,
                t.papers.c.status != "deleted",
                t.papers.c.id.in_(paper_ids) if paper_ids is not None else True,
            )
            if project_id is not None:
                await self._resource(conn, ctx, t.projects, project_id)
                query = query.join(
                    t.project_papers,
                    (t.project_papers.c.workspace_id == t.papers.c.workspace_id)
                    & (t.project_papers.c.paper_id == t.papers.c.id),
                ).where(t.project_papers.c.project_id == project_id)
            rows = (
                (
                    await conn.execute(
                        query.order_by(t.papers.c.created_at, t.papers.c.id)
                        .limit(max(1, min(limit, 100)))
                        .offset(max(0, offset))
                    )
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in rows]

    async def attach(self, ctx, project_id, paper_id):
        async with self.db.transaction() as conn:
            await self._authorize(conn, ctx, write=True)
            await self._resource(conn, ctx, t.projects, project_id)
            paper = await self._resource(conn, ctx, t.papers, paper_id)
            await self._attach(conn, ctx, project_id, paper_id)
            return paper

    async def detach(self, ctx, project_id, paper_id):
        """Remove only the project association, preserving paper and document rows."""
        async with self.db.transaction() as conn:
            await self._authorize(conn, ctx, write=True)
            await self._resource(conn, ctx, t.projects, project_id)
            await self._resource(conn, ctx, t.papers, paper_id)
            await conn.execute(
                delete(t.project_papers).where(*self._association(ctx, project_id, paper_id))
            )
