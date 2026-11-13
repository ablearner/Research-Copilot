import asyncio
import logging
import signal
from apps.platform.config import PlatformSettings
from core.config import Settings
from adapters.storage.mysql import Database
from adapters.object_store.s3 import S3ObjectStore
from adapters.queue import RabbitQueue
from services.task_execution import TaskService
from services.platform.business import BusinessService
from services.platform.runner import Worker


async def main():
    settings = PlatformSettings()
    db = Database(settings.database_url)
    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    await objects.initialize()
    # Production supplies model/provider configuration through normal Research-Copilot settings.
    worker = Worker(
        TaskService(db, max_running=settings.max_running, max_queued=settings.max_queued),
        BusinessService(db, objects),
        settings,
        Settings(),
    )
    await worker.memory.retry_pending()
    queue = await RabbitQueue(settings.rabbit_url).connect()
    consumer = asyncio.create_task(queue.consume(worker.consume))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    waiter = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait([consumer, waiter], return_when=asyncio.FIRST_COMPLETED)
        if consumer in done:
            await consumer
    finally:
        consumer.cancel()
        waiter.cancel()
        await asyncio.gather(consumer, waiter, return_exceptions=True)
        try:
            await asyncio.wait_for(worker.drain(), timeout=30)
        except asyncio.TimeoutError:
            pass  # durable leases recover interrupted tasks
        await queue.close()
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
