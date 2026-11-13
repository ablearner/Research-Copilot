from uuid import uuid4
import pytest
import httpx
from apps.platform.main import create_app
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database
from adapters.object_store.local import LocalObjectStore


@pytest.mark.asyncio
async def test_sse_replay_resumes_after_last_event_and_rejects_invalid_cursor(tmp_path):
    db = Database(PlatformSettings().database_url)
    app = create_app(
        PlatformSettings(require_email_verification=False),
        db=db,
        objects=LocalObjectStore(tmp_path),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        credentials = {"email": f"events-{uuid4()}@example.org", "password": "test-password"}
        await c.post("/api/v1/auth/register", json=credentials)
        login = (await c.post("/api/v1/auth/login", json=credentials)).json()
        wp = f"/api/v1/workspaces/{login['workspace']['id']}"
        headers = {"X-CSRF-Token": c.cookies["rc_csrf"]}
        project = (await c.post(f"{wp}/projects", json={"name": "Events"}, headers=headers)).json()
        conv = (
            await c.post(f"{wp}/projects/{project['id']}/conversations", headers=headers)
        ).json()
        submitted = await c.post(
            f"{wp}/conversations/{conv['id']}/messages",
            json={"message": "Hello"},
            headers={**headers, "Idempotency-Key": "events-test"},
        )
        assert submitted.status_code == 202, submitted.text
        task = submitted.json()
        path = f"{wp}/tasks/{task['id']}"
        assert (await c.post(f"{path}/cancel", headers=headers)).status_code == 200
        replay = await c.get(f"{path}/events", headers={"Last-Event-ID": "1"})
        assert replay.status_code == 200
        assert "id: 1\n" not in replay.text
        assert "id: 2\n" in replay.text
        assert (
            await c.get(f"{path}/events", headers={"Last-Event-ID": "invalid"})
        ).status_code == 422
    await db.close()
