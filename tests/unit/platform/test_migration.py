import json
import sqlite3
from uuid import uuid4
import pytest
from adapters.storage.mysql import Database, tables as t
from services.identity import IdentityService
from scripts.migrate_business_to_mysql import migrate
from sqlalchemy import select


@pytest.mark.asyncio
async def test_migration_dry_run_idempotence_and_conflict(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tasks(task_id TEXT,data_json TEXT)")
    tid = str(uuid4())
    data = {
        "task_id": tid,
        "topic": "Legacy",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    con.execute("INSERT INTO tasks VALUES (?,?)", (tid, json.dumps(data)))
    con.commit()
    con.close()
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    user = await IdentityService(db, require_email_verification=False).register(
        "m@example.org", "password123"
    )
    first = await migrate(db, path, user["user"]["id"], apply=False)
    assert first["counts"]["tasks"] == 1
    async with db.transaction() as c:
        assert (await c.execute(select(t.research_tasks))).first() is None
    await migrate(db, path, user["user"]["id"], apply=True)
    await migrate(db, path, user["user"]["id"], apply=True)
    async with db.transaction() as c:
        assert len((await c.execute(select(t.research_tasks))).all()) == 1
    con = sqlite3.connect(path)
    data["topic"] = "Changed"
    con.execute("UPDATE tasks SET data_json=?", (json.dumps(data),))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="conflict"):
        await migrate(db, path, user["user"]["id"], apply=True)
    await db.close()


@pytest.mark.asyncio
async def test_file_manifest_dry_run_copy_hash_repeat_and_missing(tmp_path):
    from adapters.object_store.local import LocalObjectStore
    from services.projects import ProjectService
    from services.platform.papers import PaperService
    from scripts.migrate_platform_files import migrate_files

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    await identity.register("files@example.test", "test-password")
    login = await identity.login("files@example.test", "test-password")
    ctx = await identity.resolve_workspace(login["token"], login["workspace"]["id"])
    project = await ProjectService(db).create(ctx, "History")
    paper = await PaperService(db).save(ctx, "Old paper", {}, project_id=project["id"])
    source = tmp_path / "source"
    source.mkdir()
    (source / "old.pdf").write_bytes(b"%PDF-1.4\nold file")
    manifest = [
        {
            "project_id": project["id"],
            "paper_id": paper["id"],
            "document_id": str(uuid4()),
            "path": "old.pdf",
        }
    ]
    objects = LocalObjectStore(tmp_path / "objects")
    result = await migrate_files(db, objects, ctx.user_id, source, manifest)
    assert result["files"][0]["status"] == "planned"
    assert not objects.root.exists()
    result = await migrate_files(db, objects, ctx.user_id, source, manifest, apply=True)
    assert result["files"][0]["status"] == "copied"
    assert (await migrate_files(db, objects, ctx.user_id, source, manifest, apply=True))["files"][
        0
    ]["status"] == "exists"
    (source / "old.pdf").write_bytes(b"%PDF-1.4\nchanged")
    with pytest.raises(ValueError, match="conflict"):
        await migrate_files(db, objects, ctx.user_id, source, manifest, apply=True)
    (source / "old.pdf").unlink()
    assert (await migrate_files(db, objects, ctx.user_id, source, manifest))["files"][0][
        "status"
    ] == "missing"
    await db.close()
