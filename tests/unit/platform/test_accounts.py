import pytest
from uuid import uuid4
from sqlalchemy import insert
from adapters.storage.mysql import Database, tables as t
from services.identity import IdentityService, InvalidCredentials, token_hash
from services.platform.accounts import AccountActions
from apps.platform.config import PlatformSettings


@pytest.mark.asyncio
async def test_reset_token_single_use_revokes_sessions():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    registered = await identity.register("u@example.org", "old-password")
    session = await identity.login("u@example.org", "old-password")
    async with db.transaction() as conn:
        await conn.execute(
            insert(t.email_action_tokens).values(
                id=str(uuid4()),
                user_id=registered["user"]["id"],
                kind="reset",
                token_hash=token_hash("reset-token"),
                expires_at=t.now_ms() + 10000,
            )
        )
    actions = AccountActions(db, identity, PlatformSettings())
    await actions.consume("reset-token", "reset", "new-password")
    with pytest.raises(InvalidCredentials):
        await identity.authenticate(session["token"])
    with pytest.raises(InvalidCredentials):
        await actions.consume("reset-token", "reset", "another-password")
    assert (await identity.login("u@example.org", "new-password"))["user"]["id"] == registered[
        "user"
    ]["id"]
    await db.close()


@pytest.mark.asyncio
async def test_login_rejects_password_changed_during_hash_verification(monkeypatch):
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    await identity.register("u@example.org", "old-password")
    import services.identity as module

    original = module.asyncio.to_thread

    async def intercept(fn, *args, **kwargs):
        result = await original(fn, *args, **kwargs)
        if fn == identity.hasher.verify:
            from sqlalchemy import update

            async with db.transaction() as conn:
                await conn.execute(update(t.users).values(password_hash="changed"))
        return result

    monkeypatch.setattr(module.asyncio, "to_thread", intercept)
    with pytest.raises(InvalidCredentials):
        await identity.login("u@example.org", "old-password")
    await db.close()
