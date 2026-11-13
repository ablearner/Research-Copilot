"""Live RabbitMQ/MySQL resilience checks; never consume the production queue."""

import asyncio
import contextlib
import json
from uuid import uuid4

import aio_pika
import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select, update

from adapters.queue.rabbitmq import OutboxDispatcher, RabbitQueue
from adapters.storage.mysql import Database
from adapters.storage.mysql.tables import outbox_events, workspaces

BROKER = "amqp://research:local-research-only@127.0.0.1:55672/"
DATABASE = "mysql+asyncmy://research:local-research-only@127.0.0.1:53306/research_platform"


@pytest_asyncio.fixture
async def queue():
    q = RabbitQueue(BROKER, queue_name="test.resilience." + str(uuid4()))
    await q.connect()
    yield q
    await q.queue.delete(if_unused=False, if_empty=False)
    await q.close()


async def stop(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_poison_payloads_are_rejected_without_calling_handler(queue):
    invalid = [
        None,
        [],
        {"kind": "unknown", "payload": {}},
        {"kind": "task.dispatch", "payload": {}},
        {"kind": "task.dispatch", "payload": {"task_id": "task"}},
        {"kind": "task.dispatch", "payload": {"generation": 1}},
        *(
            {"kind": "task.dispatch", "payload": {"task_id": "task", "generation": value}}
            for value in [None, True, False, 0, -1, 2147483648, "1", 1.5, [], {}]
        ),
        *(
            {"kind": "task.dispatch", "payload": {"task_id": value, "generation": 1}}
            for value in [None, "", " ", "x" * 37, 42, [], {}]
        ),
        {"kind": "document.cleanup", "payload": {"document_id": "doc"}},
        {"kind": "document.cleanup", "payload": {"workspace_id": "workspace"}},
        *(
            {
                "kind": "document.cleanup",
                "payload": {"document_id": value, "workspace_id": "workspace"},
            }
            for value in [None, "", " ", "x" * 37, 42, [], {}]
        ),
        *(
            {"kind": "document.cleanup", "payload": {"document_id": "doc", "workspace_id": value}}
            for value in [None, "", " ", "x" * 37, 42, [], {}]
        ),
    ]
    seen = []
    done = asyncio.Event()

    async def handle(value):
        seen.append(value)
        if value["payload"].get("task_id") == "sentinel":
            done.set()

    for value in invalid:
        await queue.publish(value)
    await queue.channel.default_exchange.publish(
        aio_pika.Message(b"not json"), routing_key=queue.queue_name
    )
    await queue.channel.default_exchange.publish(
        aio_pika.Message(b"\xff"), routing_key=queue.queue_name
    )
    valid = {"kind": "task.dispatch", "payload": {"task_id": "sentinel", "generation": 1}}
    await queue.publish(valid)
    consumer = asyncio.create_task(queue.consume(handle))
    try:
        await asyncio.wait_for(done.wait(), 5)
        assert seen == [valid]
    finally:
        await stop(consumer)


@pytest.mark.asyncio
async def test_transient_handler_failure_redelivers_and_cleanup_is_valid(queue):
    seen = []
    done = asyncio.Event()
    payload = {
        "kind": "document.cleanup",
        "payload": {"document_id": "doc", "workspace_id": "workspace"},
    }

    async def handle(value):
        seen.append(value)
        if len(seen) == 1:
            raise ConnectionError("transient database failure")
        done.set()

    await queue.publish(payload)
    consumer = asyncio.create_task(queue.consume(handle))
    try:
        await asyncio.wait_for(done.wait(), 5)
        assert seen == [payload, payload]
    finally:
        await stop(consumer)


@pytest.mark.asyncio
async def test_robust_connection_restores_declared_queue_and_consumer(queue):
    seen = []
    done = asyncio.Event()

    async def handle(value):
        seen.append(value)
        done.set()

    consumer = asyncio.create_task(queue.consume(handle))
    try:
        await queue.publish(
            {"kind": "task.dispatch", "payload": {"task_id": "before", "generation": 1}}
        )
        await asyncio.wait_for(done.wait(), 5)
        done.clear()
        await queue.connection.reconnect()
        await queue.publish(
            {"kind": "task.dispatch", "payload": {"task_id": "after", "generation": 1}}
        )
        await asyncio.wait_for(done.wait(), 10)
        assert [v["payload"]["task_id"] for v in seen] == ["before", "after"]
    finally:
        await stop(consumer)


@pytest.mark.asyncio
async def test_confirm_mark_gap_retains_outbox_and_permits_duplicate_delivery(queue):
    db = Database(DATABASE)
    wid, oid = str(uuid4()), str(uuid4())
    try:
        async with db.transaction() as c:
            await c.execute(insert(workspaces).values(id=wid, name="queue resilience"))
            # Earliest row and limit=1 isolate dispatch from unrelated developer outbox rows.
            await c.execute(
                insert(outbox_events).values(
                    id=oid,
                    workspace_id=wid,
                    kind="document.cleanup",
                    payload={"document_id": "doc", "workspace_id": wid},
                    created_at=-100,
                    next_attempt_at=0,
                )
            )

        class ConfirmThenLoseResponse:
            async def publish(self, payload):
                assert payload["outbox_id"] == oid
                async with db.transaction() as c:
                    await c.execute(
                        select(outbox_events)
                        .where(outbox_events.c.id == oid)
                        .with_for_update(nowait=True)
                    )
                await queue.publish(payload)
                raise ConnectionError("lost response after broker confirm")

        assert await OutboxDispatcher(db, ConfirmThenLoseResponse()).dispatch_once(limit=1) == 0
        async with db.transaction() as c:
            row = (
                (await c.execute(select(outbox_events).where(outbox_events.c.id == oid)))
                .mappings()
                .one()
            )
            assert row["published_at"] is None and row["attempt_count"] == 1
            assert row["next_attempt_at"] > 0
            await c.execute(
                update(outbox_events).where(outbox_events.c.id == oid).values(next_attempt_at=0)
            )
        assert await OutboxDispatcher(db, queue).dispatch_once(limit=1) == 1
        async with db.transaction() as c:
            row = (
                (await c.execute(select(outbox_events).where(outbox_events.c.id == oid)))
                .mappings()
                .one()
            )
            assert row["published_at"] is not None and row["attempt_count"] == 2
        deliveries = []
        for _ in range(2):
            msg = await queue.queue.get(timeout=5)
            deliveries.append(json.loads(msg.body))
            assert msg.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
            await msg.ack()
        assert deliveries[0] == deliveries[1]
        assert deliveries[0]["kind"] == "document.cleanup"
    finally:
        async with db.transaction() as c:
            await c.execute(delete(outbox_events).where(outbox_events.c.id == oid))
            await c.execute(delete(workspaces).where(workspaces.c.id == wid))
        await db.close()


@pytest.mark.asyncio
async def test_cancelled_handler_leaves_delivery_available_for_next_consumer(queue):
    entered = asyncio.Event()
    release = asyncio.Event()
    envelope = {"kind": "task.dispatch", "payload": {"task_id": "cancelled", "generation": 1}}

    async def handler(value):
        assert value == envelope
        entered.set()
        await release.wait()

    await queue.publish(envelope)
    consumer = asyncio.create_task(queue.consume(handler))
    await asyncio.wait_for(entered.wait(), 5)
    await stop(consumer)
    message = await queue.queue.get(timeout=5)
    assert message.redelivered
    assert json.loads(message.body) == envelope
    await message.ack()
