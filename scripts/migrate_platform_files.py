"""Migrate an explicit historical PDF manifest; default dry-run, never overwrite records.

Manifest: [{"paper_id": UUID, "project_id": UUID, "document_id": UUID, "path": "relative.pdf"}].
Run relational migration first. Missing files are reported, not fabricated. Reindex through platform import tasks.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL
from sqlalchemy import select, insert
from adapters.storage.mysql import Database, tables as t
from adapters.object_store.s3 import S3ObjectStore
from apps.platform.config import PlatformSettings
from domain.schemas.identity import RequestContext
from services.platform.business import BusinessService


async def migrate_files(db, objects, owner_id, root, manifest, *, apply=False):
    root = Path(root).resolve()
    business = BusinessService(db, objects)
    async with db.transaction() as c:
        wid = (
            await c.execute(
                select(t.workspace_members.c.workspace_id)
                .where(
                    t.workspace_members.c.user_id == owner_id, t.workspace_members.c.role == "owner"
                )
                .order_by(t.workspace_members.c.created_at)
                .limit(1)
            )
        ).scalar_one()
    ctx = RequestContext(owner_id, wid, "owner")
    results = []
    for entry in manifest:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes source root")
        if not path.is_file():
            results.append({"document_id": entry["document_id"], "status": "missing"})
            continue
        if path.stat().st_size > business.max_file_bytes:
            raise ValueError("Source exceeds file limit")
        data = await asyncio.to_thread(path.read_bytes)
        digest = hashlib.sha256(data).hexdigest()
        if not data.startswith(b"%PDF-") or (entry.get("sha256") and entry["sha256"] != digest):
            raise ValueError("Source PDF/checksum invalid")
        did = entry["document_id"]
        if len(did) > 36:
            raise ValueError("Historical ID exceeds schema capacity")
        fid = str(uuid5(NAMESPACE_URL, "platform-file:" + wid + did))
        vid = str(uuid5(NAMESPACE_URL, "platform-version:" + wid + did))
        key = f"workspaces/{wid}/migrated/{uuid4()}.pdf"
        async with db.transaction() as c:
            await business._auth(c, ctx, True)
            await business._resource(c, ctx, t.projects, entry["project_id"])
            await business._resource(c, ctx, t.papers, entry["paper_id"])
            existing = (
                await c.execute(
                    select(t.documents.c.paper_id, t.file_objects.c.sha256)
                    .join(
                        t.document_versions, t.document_versions.c.document_id == t.documents.c.id
                    )
                    .join(t.file_objects, t.file_objects.c.id == t.document_versions.c.file_id)
                    .where(t.documents.c.id == did, t.documents.c.workspace_id == wid)
                )
            ).first()
            if existing and (existing.paper_id != entry["paper_id"] or existing.sha256 != digest):
                raise ValueError("Target conflict; migration never overwrites")
        status = "exists" if existing else ("planned" if not apply else "copied")
        if apply and not existing:
            await objects.put(key, data, "application/pdf")
            if hashlib.sha256(await objects.get(key)).hexdigest() != digest:
                raise ValueError("Copied object checksum mismatch")
            async with db.transaction() as c:
                await business._auth(c, ctx, True)
                await c.execute(
                    select(t.workspaces.c.id).where(t.workspaces.c.id == wid).with_for_update()
                )
                await c.execute(
                    insert(t.file_objects).values(
                        id=fid,
                        workspace_id=wid,
                        object_key=key,
                        sha256=digest,
                        size=len(data),
                        mime="application/pdf",
                        status="available",
                    )
                )
                await c.execute(
                    insert(t.documents).values(
                        id=did, workspace_id=wid, paper_id=entry["paper_id"], status="pending"
                    )
                )
                await c.execute(
                    insert(t.document_versions).values(
                        id=vid,
                        workspace_id=wid,
                        document_id=did,
                        file_id=fid,
                        sha256=digest,
                        parser_version="legacy-rebuild",
                        embedding_version="legacy-rebuild",
                    )
                )
                link = (
                    await c.execute(
                        select(t.project_papers.c.id).where(
                            t.project_papers.c.workspace_id == wid,
                            t.project_papers.c.project_id == entry["project_id"],
                            t.project_papers.c.paper_id == entry["paper_id"],
                        )
                    )
                ).first()
                if not link:
                    await c.execute(
                        insert(t.project_papers).values(
                            id=str(uuid4()),
                            workspace_id=wid,
                            project_id=entry["project_id"],
                            paper_id=entry["paper_id"],
                        )
                    )
        results.append({"document_id": did, "sha256": digest, "size": len(data), "status": status})
    return {"mode": "apply" if apply else "dry-run", "files": results}


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner-id", required=True)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    settings = PlatformSettings()
    db = Database(settings.database_url)
    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    try:
        print(
            json.dumps(
                await migrate_files(
                    db,
                    objects,
                    args.owner_id,
                    args.root,
                    json.loads(args.manifest.read_text()),
                    apply=args.apply,
                ),
                indent=2,
            )
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
