import pytest
from sqlalchemy import select, func, update
from adapters.storage.mysql import Database, tables as t
from services.identity import IdentityService, ResourceNotFound, PermissionDenied
from services.projects import ProjectService
from services.platform.papers import PaperService


@pytest.fixture
async def env():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    contexts = []
    for email in ("paper-a@example.org", "paper-b@example.org"):
        await identity.register(email, "test-password-123")
        login = await identity.login(email, "test-password-123")
        contexts.append(await identity.resolve_workspace(login["token"], login["workspace"]["id"]))
    projects = [await ProjectService(db).create(ctx, "Project") for ctx in contexts]
    yield db, PaperService(db), contexts, projects
    await db.close()


@pytest.mark.asyncio
async def test_metadata_only_attach_detach_preserves_paper(env):
    db, svc, (a, _), (project, _) = env
    paper = await svc.save(a, "Metadata only", {"doi": "10.1000/example", "authors": ["A"]})
    assert (await svc.get(a, paper["id"]))["metadata_json"]["authors"] == ["A"]
    assert await svc.list(a, project_id=project["id"]) == []
    await svc.attach(a, project["id"], paper["id"])
    await svc.attach(a, project["id"], paper["id"])
    assert [r["id"] for r in await svc.list(a, project_id=project["id"])] == [paper["id"]]
    async with db.transaction() as conn:
        assert (await conn.execute(select(func.count()).select_from(t.documents))).scalar_one() == 0
        assert (
            await conn.execute(select(func.count()).select_from(t.project_papers))
        ).scalar_one() == 1
    await svc.detach(a, project["id"], paper["id"])
    await svc.detach(a, project["id"], paper["id"])
    assert await svc.list(a, project_id=project["id"]) == []
    assert len(await svc.list(a)) == 1


@pytest.mark.asyncio
async def test_cross_workspace_and_deleted_project_denied(env):
    db, svc, (a, b), (pa, pb) = env
    paper = await svc.save(a, "Title", {}, project_id=pa["id"])
    for method, args, kwargs in [
        (svc.get, (b, paper["id"]), {}),
        (svc.attach, (b, pb["id"], paper["id"]), {}),
        (svc.detach, (b, pa["id"], paper["id"]), {}),
        (svc.list, (b,), {"project_id": pa["id"]}),
        (svc.save, (b, "Title", {}), {"project_id": pa["id"]}),
    ]:
        with pytest.raises(ResourceNotFound):
            await method(*args, **kwargs)
    assert await svc.list(b) == []
    await ProjectService(db).delete(a, pa["id"], 1)
    with pytest.raises(ResourceNotFound):
        await svc.attach(a, pa["id"], paper["id"])


@pytest.mark.asyncio
async def test_membership_role_and_disable_rechecked(env):
    db, svc, (a, _), (pa, _) = env
    paper = await svc.save(a, "Title", {})
    async with db.transaction() as conn:
        await conn.execute(
            update(t.workspace_members)
            .where(t.workspace_members.c.user_id == a.user_id)
            .values(role="viewer")
        )
    assert len(await svc.list(a)) == 1
    with pytest.raises(PermissionDenied):
        await svc.attach(a, pa["id"], paper["id"])
    with pytest.raises(PermissionDenied):
        await svc.save(a, "No", {})
    async with db.transaction() as conn:
        await conn.execute(
            update(t.users).where(t.users.c.id == a.user_id).values(status="disabled")
        )
    with pytest.raises(ResourceNotFound):
        await svc.list(a)


@pytest.mark.asyncio
async def test_metadata_input_is_bounded_json(env):
    _, svc, (a, _), _ = env
    for title, metadata in [
        ("", {}),
        ("Title", {"bad": object()}),
        ("Title", {"large": "x" * 65537}),
    ]:
        with pytest.raises(ValueError):
            await svc.save(a, title, metadata)


@pytest.mark.asyncio
async def test_detach_preserves_uploaded_document_and_other_project(env, tmp_path):
    from adapters.object_store.local import LocalObjectStore
    from services.platform.business import BusinessService

    db, svc, (a, _), (project, _) = env
    business = BusinessService(db, LocalObjectStore(tmp_path))
    uploaded = await business.upload(a, project["id"], "source.pdf", b"%PDF-1.4\nsource")
    second = await ProjectService(db).create(a, "Second")
    async with db.transaction() as conn:
        paper_id = (
            await conn.execute(
                select(t.documents.c.paper_id).where(t.documents.c.id == uploaded["document_id"])
            )
        ).scalar_one()
    await svc.attach(a, second["id"], paper_id)
    await svc.detach(a, project["id"], paper_id)
    assert (await svc.get(a, paper_id))["id"] == paper_id
    assert (await business.file(a, uploaded["file_id"]))[0] == b"%PDF-1.4\nsource"
    assert len(await svc.list(a, project_id=second["id"])) == 1
    assert await svc.list(a, project_id=project["id"]) == []


@pytest.mark.asyncio
async def test_selected_paper_beyond_first_page(env):
    _, svc, (a, _), (project, _) = env
    papers = [await svc.save(a, f"Paper {i}", {}, project_id=project["id"]) for i in range(101)]
    wanted = papers[-1]["id"]
    assert wanted not in {r["id"] for r in await svc.list(a, project_id=project["id"])}
    assert [r["id"] for r in await svc.list(a, project_id=project["id"], paper_ids=[wanted])] == [
        wanted
    ]
