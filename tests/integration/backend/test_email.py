import re
from uuid import uuid4
import pytest
import httpx
from adapters.storage.mysql import Database
from apps.platform.config import PlatformSettings
from services.identity import IdentityService, InvalidCredentials
from services.platform.accounts import AccountActions


@pytest.mark.asyncio
async def test_real_smtp_verification_and_reset():
    settings = PlatformSettings()
    db = Database(settings.database_url)
    identity = IdentityService(db)
    actions = AccountActions(db, identity, settings)
    email = str(uuid4()) + "@mail.test"
    await identity.register(email, "test-password")
    with pytest.raises(InvalidCredentials):
        await identity.login(email, "test-password")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:58025") as client:

        async def token(kind):
            await actions.request(email, kind)
            messages = (await client.get("/api/v1/messages")).json()["messages"]
            message = next(m for m in messages if any(r["Address"] == email for r in m["To"]))
            content = (await client.get("/api/v1/message/" + message["ID"])).json()["Text"]
            return re.search(r"token=([A-Za-z0-9_-]+)", content).group(1)

        await actions.consume(await token("verify"), "verify")
        session = await identity.login(email, "test-password")
        reset = await token("reset")
        await actions.consume(reset, "reset", "new-password")
        with pytest.raises(InvalidCredentials):
            await identity.authenticate(session["token"])
        with pytest.raises(InvalidCredentials):
            await actions.consume(reset, "reset", "newer-password")
        assert await identity.login(email, "new-password")
    await db.close()
