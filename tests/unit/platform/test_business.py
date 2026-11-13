import pytest
from adapters.storage.mysql import Database
from services.identity import IdentityService, ResourceNotFound
from services.projects import ProjectService
from services.platform.business import BusinessService
from adapters.object_store.local import LocalObjectStore


@pytest.fixture
async def env(tmp_path):
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    contexts = []
    for email in ["a@example.org", "b@example.org"]:
        await identity.register(email, "test-password-123")
        login = await identity.login(email, "test-password-123")
        contexts.append(await identity.resolve_workspace(login["token"], login["workspace"]["id"]))
    project = await ProjectService(db).create(contexts[0], "Research")
    yield BusinessService(db, LocalObjectStore(tmp_path)), contexts, project
    await db.close()


@pytest.mark.asyncio
async def test_private_upload_dedup_and_deletion(env):
    svc, (a, b), project = env
    doc = await svc.upload(a, project["id"], "paper.pdf", b"%PDF-1.4\nexample")
    duplicate = await svc.upload(a, project["id"], "another.pdf", b"%PDF-1.4\nexample")
    assert duplicate["document_id"] == doc["document_id"]
    with pytest.raises(ResourceNotFound):
        await svc.file(b, doc["file_id"])
    assert (await svc.file(a, doc["file_id"]))[0].startswith(b"%PDF")
    await svc.delete_document(a, doc["document_id"])
    with pytest.raises(ResourceNotFound):
        await svc.file(a, doc["file_id"])
    assert await svc.documents(a, project["id"]) == []


@pytest.mark.asyncio
async def test_conversation_scope_and_stable_pagination(env):
    svc, (a, b), project = env
    conv = await svc.create_conversation(a, project["id"], "Topic")
    for i in range(4):
        await svc.add_message(a, conv["id"], "user", {"text": str(i)})
    first = await svc.messages(a, conv["id"], after=0, limit=2)
    second = await svc.messages(a, conv["id"], after=first[-1]["seq"], limit=2)
    assert [r["content"]["text"] for r in first + second] == ["0", "1", "2", "3"]
    with pytest.raises(ResourceNotFound):
        await svc.messages(b, conv["id"])
    with pytest.raises(ResourceNotFound):
        await svc.create_conversation(b, project["id"], "bad")


@pytest.mark.asyncio
async def test_active_version_not_unpublished_latest_and_recent_messages(env):
    from sqlalchemy import insert, update
    from adapters.storage.mysql import tables as t
    from uuid import uuid4

    svc, (a, _), project = env
    uploaded = await svc.upload(a, project["id"], "version.pdf", b"%PDF-1.4\nversioned")
    rows = await svc.documents(a, project["id"])
    assert len(rows) == 1
    async with svc.db.transaction() as c:
        from sqlalchemy import select

        version = (
            (
                await c.execute(
                    select(t.document_versions).where(
                        t.document_versions.c.document_id == uploaded["document_id"]
                    )
                )
            )
            .mappings()
            .one()
        )
        await c.execute(
            update(t.documents)
            .where(t.documents.c.id == uploaded["document_id"])
            .values(active_version=version["id"])
        )
        newer = dict(version)
        newer.update(id=str(uuid4()), created_at=version["created_at"] + 1, parse_status="failed")
        await c.execute(insert(t.document_versions).values(**newer))
    rows = await svc.documents(a, project["id"])
    assert len(rows) == 1 and rows[0]["parse_status"] != "failed"
    conv = await svc.create_conversation(a, project["id"])
    for i in range(5):
        await svc.add_message(a, conv["id"], "user", str(i))
    messages = await svc.messages(a, conv["id"], limit=2, latest=True)
    assert [m["content"] for m in messages] == ["4", "3"]


@pytest.mark.asyncio
async def test_selected_document_beyond_first_page_still_authorized(env):
    svc, (a, b), project = env
    uploaded = []
    for i in range(101):
        uploaded.append(await svc.upload(a, project["id"], f"{i}.pdf", f"%PDF-1.4\n{i}".encode()))
    wanted = uploaded[-1]["document_id"]
    assert wanted not in {r["id"] for r in await svc.documents(a, project["id"])}
    assert [r["id"] for r in await svc.documents(a, project["id"], document_ids=[wanted])] == [
        wanted
    ]
    assert await svc.documents(a, project["id"], document_ids=[]) == []
    with pytest.raises(ResourceNotFound):
        await svc.documents(b, project["id"], document_ids=[wanted])
