"""Publisher confirms and short manual acknowledgements after durable handling."""

import json
import asyncio
import logging
import aio_pika
from sqlalchemy import select, update, text
from adapters.storage.mysql.tables import outbox_events


def validate_envelope(envelope):
    """Validate routing fields before calling handlers; bool is not a generation."""
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise ValueError("Invalid message envelope")
    kind, payload = envelope.get("kind"), envelope["payload"]
    if kind == "task.dispatch":
        identifiers = ("task_id",)
        generation = payload.get("generation")
        if type(generation) is not int or generation < 1 or generation > 2147483647:
            raise ValueError("Invalid task generation")
    elif kind == "document.cleanup":
        identifiers = ("document_id", "workspace_id")
    else:
        raise ValueError("Unknown message kind")
    for field in identifiers:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 36:
            raise ValueError("Invalid message identifier")
    return envelope


class RabbitQueue:
    def __init__(self, url, queue_name="research.tasks", prefetch=2, publish_timeout=10):
        self.url, self.queue_name, self.prefetch = url, queue_name, prefetch
        self.connection = None
        self.publish_timeout = publish_timeout

    async def connect(self):
        self.connection = await aio_pika.connect_robust(self.url)
        self.channel = await self.connection.channel(publisher_confirms=True, on_return_raises=True)
        await self.channel.set_qos(prefetch_count=self.prefetch)
        self.queue = await self.channel.declare_queue(self.queue_name, durable=True)
        return self

    async def publish(self, payload):
        # Robust connections restore channels asynchronously after reconnect.
        await asyncio.wait_for(self.channel.ready(), timeout=self.publish_timeout)
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(payload).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=self.queue_name,
            mandatory=True,
            timeout=self.publish_timeout,
        )

    async def consume(self, handler):
        """Handler must return after durable claim. Exceptions requeue unacknowledged work."""
        async with self.queue.iterator() as iterator:
            async for message in iterator:
                try:
                    envelope = validate_envelope(json.loads(message.body))
                except (ValueError, UnicodeDecodeError):
                    await message.reject(requeue=False)
                    logging.getLogger(__name__).warning("Rejected malformed queue message")
                    continue
                try:
                    await handler(envelope)
                except asyncio.CancelledError:
                    await message.nack(requeue=True)
                    raise
                except Exception:
                    await message.nack(requeue=True)
                    logging.getLogger(__name__).exception("Queue handler failed, delivery requeued")
                    await asyncio.sleep(1)
                else:
                    await message.ack()

    async def close(self):
        if self.connection:
            await self.connection.close()


class OutboxDispatcher:
    def __init__(self, db, queue):
        self.db, self.queue = db, queue

    async def dispatch_once(self, limit=100):
        async with self.db.transaction() as c:
            now = int(
                (
                    await c.execute(text("SELECT UNIX_TIMESTAMP(CURRENT_TIMESTAMP(3))*1000"))
                ).scalar_one()
            )
            rows = (
                (
                    await c.execute(
                        select(outbox_events)
                        .where(
                            outbox_events.c.published_at.is_(None),
                            outbox_events.c.next_attempt_at <= now,
                        )
                        .order_by(outbox_events.c.created_at)
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
        count = 0
        for row in rows:
            try:
                await self.queue.publish(
                    {"outbox_id": row["id"], "kind": row["kind"], "payload": row["payload"]}
                )
            except Exception:
                async with self.db.transaction() as c:
                    await c.execute(
                        update(outbox_events)
                        .where(
                            outbox_events.c.id == row["id"], outbox_events.c.published_at.is_(None)
                        )
                        .values(
                            attempt_count=outbox_events.c.attempt_count + 1,
                            next_attempt_at=now
                            + min(60000, 1000 * 2 ** min(row["attempt_count"], 6)),
                        )
                    )
                continue
            async with self.db.transaction() as c:
                await c.execute(
                    update(outbox_events)
                    .where(outbox_events.c.id == row["id"], outbox_events.c.published_at.is_(None))
                    .values(published_at=now, attempt_count=outbox_events.c.attempt_count + 1)
                )
            count += 1
        return count
