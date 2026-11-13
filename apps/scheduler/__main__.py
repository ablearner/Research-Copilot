import asyncio
import logging
from sqlalchemy import delete
from apps.platform.config import PlatformSettings
from adapters.storage.mysql import Database, tables as t
from adapters.queue import RabbitQueue, OutboxDispatcher
from services.task_execution import TaskService


async def main():
    settings = PlatformSettings()
    db = Database(settings.database_url)
    queue = await RabbitQueue(settings.rabbit_url).connect()
    from adapters.object_store.s3 import S3ObjectStore
    from services.platform.business import BusinessService
    from services.platform.uploads import UploadService

    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    uploads = UploadService(BusinessService(db, objects))
    dispatcher = OutboxDispatcher(db, queue)
    tasks = TaskService(db, max_running=settings.max_running, max_queued=settings.max_queued)
    try:
        while True:
            try:
                await uploads.cleanup_expired()
                await tasks.recover_expired()
                await dispatcher.dispatch_once()
                async with db.transaction() as c:
                    await c.execute(
                        delete(t.idempotency_records).where(
                            t.idempotency_records.c.expires_at < t.now_ms()
                        )
                    )
            except Exception:
                logging.getLogger(__name__).exception("Scheduler tick failed; retrying")
            await asyncio.sleep(1)
    finally:
        await queue.close()
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
