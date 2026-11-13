import pytest
import pytest_asyncio
from sqlalchemy import update, select
from adapters.storage.mysql import Database
from adapters.storage.mysql.tables import auth_sessions, users, projects
from services.identity import IdentityService, InvalidCredentials, DuplicateEmail, ResourceNotFound
from services.projects import ProjectService, VersionConflict


@pytest_asyncio.fixture
async def setup():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    yield db, identity, ProjectService(db)
    await db.close()


@pytest.mark.asyncio
async def test_duplicate_registration_and_personal_ownership(setup):
    db, identity, _ = setup
    registered = await identity.register("  A@Example.com ", "password123")
    with pytest.raises(DuplicateEmail):
        await identity.register("a@example.com", "password123")
    login = await identity.login("A@example.com", "password123")
    ctx = await identity.resolve_workspace(login["token"], registered["workspace"]["id"])
    assert ctx.role == "owner" and ctx.user_id == registered["user"]["id"]
    assert "password_hash" not in registered["user"]


@pytest.mark.asyncio
async def test_revoked_expired_disabled_and_wrong_credentials(setup):
    db, identity, _ = setup
    await identity.register("a@example.com", "password123")
    with pytest.raises(InvalidCredentials):
        await identity.login("a@example.com", "wrong")
    session = await identity.login("a@example.com", "password123")
    await identity.logout(session["token"])
    with pytest.raises(InvalidCredentials):
        await identity.authenticate(session["token"])
    session = await identity.login("a@example.com", "password123")
    async with db.transaction() as conn:
        await conn.execute(update(auth_sessions).values(expires_at=0))
    with pytest.raises(InvalidCredentials):
        await identity.authenticate(session["token"])
    session = await identity.login("a@example.com", "password123")
    async with db.transaction() as conn:
        await conn.execute(update(users).values(status="disabled"))
    with pytest.raises(InvalidCredentials):
        await identity.authenticate(session["token"])


@pytest.mark.asyncio
async def test_workspace_crud_isolation_version_and_rollback(setup):
    db, identity, service = setup
    contexts = []
    for email in ["a@example.com", "b@example.com"]:
        await identity.register(email, "password123")
        session = await identity.login(email, "password123")
        contexts.append(
            await identity.resolve_workspace(session["token"], session["workspace"]["id"])
        )
    a, b = contexts
    with pytest.raises(ResourceNotFound):
        await identity.resolve_workspace(session["token"], a.workspace_id)
    project = await service.create(a, "same")
    await service.create(b, "same")
    assert len(await service.list(a)) == 1
    for action in [
        service.get(b, project["id"]),
        service.update(b, project["id"], 1, name="bad"),
        service.delete(b, project["id"], 1),
    ]:
        with pytest.raises(ResourceNotFound):
            await action
    changed = await service.update(a, project["id"], 1, name="changed")
    assert changed["version"] == 2
    with pytest.raises(VersionConflict):
        await service.update(a, project["id"], 1, name="stale")
    with pytest.raises(RuntimeError):
        async with db.transaction() as conn:
            await conn.execute(update(projects).values(name="rolled back"))
            raise RuntimeError()
    assert (await service.get(a, project["id"]))["name"] == "changed"
    await service.delete(a, project["id"], 2)
    with pytest.raises(ResourceNotFound):
        await service.get(a, project["id"])


@pytest.mark.asyncio
async def test_verification_required_by_default_and_hashed_sessions(setup):
    from services.identity import token_hash

    db, identity, _ = setup
    strict = IdentityService(db)
    await strict.register("strict@example.com", "password123")
    with pytest.raises(InvalidCredentials):
        await strict.login("strict@example.com", "password123")
    await identity.register("local@example.com", "password123")
    session = await identity.login("local@example.com", "password123")
    async with db.transaction() as conn:
        stored = (await conn.execute(select(auth_sessions))).mappings().one()
        user = (
            (await conn.execute(select(users).where(users.c.email == "local@example.com")))
            .mappings()
            .one()
        )
    assert stored["token_hash"] == token_hash(session["token"])
    assert stored["token_hash"] != session["token"]
    assert user["password_hash"].startswith("$argon2id$")
