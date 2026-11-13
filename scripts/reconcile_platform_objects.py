"""List orphan objects (default); --apply deletes only old, unreferenced objects."""

import argparse
import asyncio
import json
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database
from adapters.object_store.s3 import S3ObjectStore
from services.platform.reconciliation import ObjectReconciler


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    settings = PlatformSettings()
    db = Database(settings.database_url)
    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    try:
        result = await ObjectReconciler(
            db, objects, grace_seconds=max(86400, settings.task_timeout_seconds * 2)
        ).run(apply=args.apply)
        print(json.dumps(result, indent=2))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
