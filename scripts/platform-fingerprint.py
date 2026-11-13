"""Read-only logical fingerprint for a stopped-writer backup/restore rehearsal."""
import asyncio
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from adapters.storage.mysql import Database, tables as t
from adapters.object_store.s3 import S3ObjectStore
from apps.platform.config import PlatformSettings


async def main():
    settings = PlatformSettings()
    db = Database(settings.database_url)
    objects = S3ObjectStore(settings.object_endpoint, settings.object_bucket, settings.object_access_key, settings.object_secret_key)
    result = {"tables": {}, "files": {}}
    try:
        async with db.transaction() as c:
            for table in t.metadata.tables.values():
                digest, count = hashlib.sha256(), 0
                stream = await c.stream(select(table).order_by(table.c.id))
                async for row in stream.mappings():
                    digest.update(json.dumps(dict(row), sort_keys=True, default=str, separators=(",", ":")).encode())
                    count += 1
                result["tables"][table.name] = {"count": count, "sha256": digest.hexdigest()}
            rows = (await c.execute(select(t.file_objects).where(t.file_objects.c.status == "available"))).mappings().all()
        for row in rows:
            if row["object_key"].startswith("benchmark/"):
                continue  # explicitly synthetic metadata, never an uploaded object
            digest = hashlib.sha256(await objects.get(row["object_key"])).hexdigest()
            if digest != row["sha256"]:
                raise ValueError("Object checksum mismatch: " + row["id"])
            result["files"][row["id"]] = digest
        print(json.dumps(result, sort_keys=True, indent=2))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
