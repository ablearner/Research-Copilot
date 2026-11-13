import pytest
from adapters.storage.mysql import Database
from services.identity import IdentityService, InvalidCredentials
from services.platform.credentials import CredentialService


@pytest.mark.asyncio
async def test_personal_token_scope_expiry_and_revocation():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    identity = IdentityService(db, require_email_verification=False)
    u = await identity.register("c@example.org", "password123")
    credentials = CredentialService(db)
    created = await credentials.create(u["user"]["id"], "CLI", ["read"], 3600)
    assert created["token"]
    assert (await credentials.authenticate(created["token"], "read"))["id"] == u["user"]["id"]
    with pytest.raises(InvalidCredentials):
        await credentials.authenticate(created["token"], "write")
    assert "token_hash" not in str(await credentials.list(u["user"]["id"]))
    await credentials.revoke(u["user"]["id"], created["id"])
    with pytest.raises(InvalidCredentials):
        await credentials.authenticate(created["token"], "read")
    await db.close()
