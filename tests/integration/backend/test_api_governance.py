from uuid import uuid4
import pytest
import httpx
from apps.platform.main import create_app
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database
from adapters.object_store.local import LocalObjectStore


@pytest.mark.asyncio
async def test_live_admin_metrics_and_redis_rate_limit(tmp_path):
    email = str(uuid4()) + "@admin.test"
    settings = PlatformSettings(
        require_email_verification=False, rate_limit_enabled=True, admin_emails=email
    )
    db = Database(settings.database_url)
    app = create_app(settings, db=db, objects=LocalObjectStore(tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(str(uuid4()), 1)), base_url="http://test"
    ) as c:
        assert (
            await c.post(
                "/api/v1/auth/register", json={"email": email, "password": "test-password"}
            )
        ).status_code == 201
        assert (
            await c.post("/api/v1/auth/login", json={"email": email, "password": "test-password"})
        ).status_code == 200
        metrics = await c.get("/api/v1/admin/metrics")
        assert metrics.status_code == 200
        assert "expired_running_leases" in metrics.json()
        codes = [(await c.get("/api/v1/me")).status_code for _ in range(121)]
        assert 200 in codes and codes[-1] == 429
    await db.close()


@pytest.mark.asyncio
async def test_redis_failure_closes_write_gate(tmp_path):
    settings = PlatformSettings(
        require_email_verification=False, rate_limit_enabled=True, redis_url="redis://127.0.0.1:1/0"
    )
    db = Database(settings.database_url)
    app = create_app(settings, db=db, objects=LocalObjectStore(tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/api/v1/auth/register",
            json={"email": "uncreated@test.local", "password": "test-password"},
        )
        assert response.status_code == 503
        assert response.json()["code"] == "rate_limit_unavailable"
    await db.close()
