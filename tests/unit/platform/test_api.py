import pytest
import httpx
from apps.platform.main import create_app
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database
from adapters.object_store.local import LocalObjectStore


@pytest.mark.asyncio
async def test_cookie_csrf_auth_and_workspace_scope(tmp_path):
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    app = create_app(
        PlatformSettings(require_email_verification=False),
        db=db,
        objects=LocalObjectStore(tmp_path),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get("/api/v1/me")).status_code == 401
        assert (
            await c.post(
                "/api/v1/auth/register",
                json={"email": "test@example.org", "password": "test-password"},
            )
        ).status_code == 201
        login = await c.post(
            "/api/v1/auth/login", json={"email": "test@example.org", "password": "test-password"}
        )
        assert login.status_code == 200
        wid = login.json()["workspace"]["id"]
        path = f"/api/v1/workspaces/{wid}/projects"
        assert (await c.post(path, json={"name": "P"})).status_code == 403
        headers = {"X-CSRF-Token": c.cookies["rc_csrf"]}
        assert (await c.post(path, json={"name": "P"}, headers=headers)).status_code == 201
        assert (await c.get("/api/v1/workspaces/foreign/projects")).status_code == 404
        assert (await c.post("/api/v1/auth/logout", headers=headers)).status_code == 204
        assert (await c.get("/api/v1/me")).status_code == 401
    await db.close()
