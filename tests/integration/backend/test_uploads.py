import hashlib
from uuid import uuid4
import httpx
import pytest
from adapters.storage.mysql import Database
from adapters.object_store.s3 import S3ObjectStore
from apps.platform.config import PlatformSettings
from services.identity import IdentityService
from services.projects import ProjectService
from services.platform.business import BusinessService
from services.platform.uploads import UploadService


@pytest.mark.asyncio
async def test_signed_upload_verifies_bytes_and_retries_completion():
    settings = PlatformSettings()
    db = Database(settings.database_url)
    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    await objects.initialize()
    identity = IdentityService(db, require_email_verification=False)
    email = str(uuid4()) + "@upload.test"
    await identity.register(email, "test-password")
    login = await identity.login(email, "test-password")
    ctx = await identity.resolve_workspace(login["token"], login["workspace"]["id"])
    project = await ProjectService(db).create(ctx, "Private upload")
    business = BusinessService(db, objects)
    uploads = UploadService(business)
    data = b"%PDF-1.4\nprivate test file"
    grant = await uploads.begin(
        ctx, project["id"], "test.pdf", len(data), hashlib.sha256(data).hexdigest()
    )
    async with httpx.AsyncClient() as c:
        response = await c.post(
            grant["url"],
            data=grant["fields"],
            files={"file": ("test.pdf", data, "application/pdf")},
        )
        assert response.status_code == 204, response.text
    result = await uploads.complete(ctx, grant["id"])
    assert result["document_id"] == (await uploads.complete(ctx, grant["id"]))["document_id"]
    assert (await business.file(ctx, result["file_id"]))[0] == data
    bad = await uploads.begin(ctx, project["id"], "test.pdf", len(data), "0" * 64)
    async with httpx.AsyncClient() as c:
        assert (
            await c.post(
                bad["url"],
                data=bad["fields"],
                files={"file": ("test.pdf", data, "application/pdf")},
            )
        ).status_code == 204
    with pytest.raises(ValueError, match="checksum"):
        await uploads.complete(ctx, bad["id"])
    assert len(await business.documents(ctx, project["id"])) == 1
    await db.close()
