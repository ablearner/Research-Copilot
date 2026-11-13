import sqlite3
from uuid import uuid4
import pytest
from sqlalchemy import insert, update
from adapters.storage.mysql import Database, tables as t
from services.identity import IdentityService
from services.projects import ProjectService
from services.platform.memory_commit import MemoryCommitStore


@pytest.mark.asyncio
async def test_only_committed_attempt_can_publish_and_scope_versions_do_not_mix(tmp_path):
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    await identity.register("memory@example.test", "test-password")
    login = await identity.login("memory@example.test", "test-password")
    ctx = await identity.resolve_workspace(login["token"], login["workspace"]["id"])
    project = await ProjectService(db).create(ctx, "Memory")
    cid, tid = str(uuid4()), str(uuid4())
    task = dict(
        id=tid,
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        project_id=project["id"],
        conversation_id=cid,
        kind="research",
        execution_epoch=1,
        status="running",
    )
    async with db.transaction() as c:
        await c.execute(
            insert(t.conversations).values(
                id=cid, workspace_id=ctx.workspace_id, project_id=project["id"]
            )
        )
        await c.execute(insert(t.research_tasks).values(**task))
    memory = MemoryCommitStore(db, tmp_path)
    path = await memory.prepare(task, {})
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE facts(value TEXT)")
        c.execute("INSERT INTO facts VALUES ('committed evidence')")
    assert not await memory.commit(path)
    assert not (path.parents[2] / "current.sqlite").exists()
    async with db.transaction() as c:
        await c.execute(
            update(t.research_tasks).where(t.research_tasks.c.id == tid).values(status="succeeded")
        )
    assert await memory.commit(path)
    assert not await memory.commit(path)
    next_path = await memory.prepare(dict(task, id=str(uuid4())), {})
    with sqlite3.connect(next_path) as c:
        assert c.execute("SELECT value FROM facts").fetchone()[0] == "committed evidence"
    other_scope = await memory.prepare(dict(task, id=str(uuid4())), {"paper": "new-version"})
    assert not other_scope.exists()
    await memory.cleanup_document(ctx.workspace_id, "paper")
    assert not other_scope.parent.exists()
    assert path.exists()
    await db.close()


@pytest.mark.asyncio
async def test_corrupt_pending_manifest_does_not_block_other_recovery(
    tmp_path, monkeypatch, caplog
):
    import json

    memory = MemoryCommitStore(None, tmp_path)
    corrupt = tmp_path / "workspace/memory/scope/broken/1/pending.json"
    valid = tmp_path / "workspace/memory/scope/valid/1/pending.json"
    corrupt.parent.mkdir(parents=True)
    valid.parent.mkdir(parents=True)
    corrupt.write_text("{")
    valid.write_text(json.dumps({"conversation_id": "conversation"}))
    recovered = []

    async def commit(path):
        json.loads(path.with_name("pending.json").read_text())
        recovered.append(path)
        return True

    monkeypatch.setattr(memory, "commit", commit)
    await memory.retry_pending()
    assert recovered == [valid.with_name("memory.sqlite")]
    assert not corrupt.exists()
    assert corrupt.with_name("corrupt.manifest").exists()
    assert "Corrupt memory manifest" in caplog.text
