"""Short-lived upload grants; only server-verified bytes become visible documents."""

import hashlib
import re
from uuid import uuid4
from sqlalchemy import insert, select, update, delete
from adapters.storage.mysql import tables as t
from services.identity import ResourceNotFound


class UploadService:
    def __init__(self, business):
        self.business, self.db, self.objects = business, business.db, business.objects

    async def begin(self, ctx, project_id, filename, size, sha256):
        if not 1 <= size <= self.business.max_file_bytes or not re.fullmatch(
            r"[a-f0-9]{64}", sha256
        ):
            raise ValueError("Invalid file size or SHA256")
        if not filename or len(filename) > 255:
            raise ValueError("Filename required (maximum 255 characters)")
        uid = str(uuid4())
        key = f"workspaces/{ctx.workspace_id}/uploads/{uid}.pdf"
        async with self.db.transaction() as c:
            await self.business._auth(c, ctx, True)
            await self.business._resource(c, ctx, t.projects, project_id)
            await c.execute(
                insert(t.upload_sessions).values(
                    id=uid,
                    workspace_id=ctx.workspace_id,
                    project_id=project_id,
                    filename=filename,
                    size=size,
                    sha256=sha256,
                    object_key=key,
                    expires_at=t.now_ms() + 300000,
                )
            )
        return {"id": uid, "expires_in": 300, **await self.objects.upload_form(key, size)}

    async def complete(self, ctx, upload_id):
        async with self.db.transaction() as c:
            await self.business._auth(c, ctx, True)
            row = (
                (
                    await c.execute(
                        select(t.upload_sessions).where(
                            t.upload_sessions.c.id == upload_id,
                            t.upload_sessions.c.workspace_id == ctx.workspace_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                raise ResourceNotFound()
            await self.business._resource(c, ctx, t.projects, row["project_id"])
            if row["document_id"]:
                await self.business._resource(c, ctx, t.documents, row["document_id"])
                return {
                    "document_id": row["document_id"],
                    "file_id": row["file_id"],
                    "duplicate": True,
                }
            if row["expires_at"] < t.now_ms():
                raise ValueError("Upload session expired")
        if await self.objects.size(row["object_key"]) != row["size"]:
            raise ValueError("Uploaded object size does not match")
        data = await self.objects.get(row["object_key"])
        if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Uploaded object checksum does not match")
        result = await self.business.upload(ctx, row["project_id"], row["filename"], data)
        async with self.db.transaction() as c:
            await self.business._auth(c, ctx, True)
            await c.execute(
                update(t.upload_sessions)
                .where(t.upload_sessions.c.id == upload_id)
                .values(
                    document_id=result["document_id"],
                    file_id=result["file_id"],
                )
            )
        # Grants remain writable until expiry. The verified document is a separate immutable copy.
        return result

    async def cleanup_expired(self):
        async with self.db.transaction() as c:
            rows = (
                (
                    await c.execute(
                        select(t.upload_sessions)
                        .where(t.upload_sessions.c.expires_at < t.now_ms())
                        .limit(100)
                    )
                )
                .mappings()
                .all()
            )
        for row in rows:
            await self.objects.delete(row["object_key"])
            async with self.db.transaction() as c:
                await c.execute(
                    delete(t.upload_sessions).where(t.upload_sessions.c.id == row["id"])
                )
        return len(rows)
