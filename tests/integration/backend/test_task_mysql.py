import asyncio
import os
from uuid import uuid4
import pytest
from sqlalchemy import insert, update
from adapters.storage.mysql import Database
from adapters.storage.mysql.tables import (
    users,
    workspaces,
    workspace_members,
    projects,
    research_tasks,
    task_attempts,
)
from domain.schemas.identity import RequestContext
from services.task_execution import TaskService, IdempotencyConflict, QuotaExceeded
from services.identity import ResourceNotFound

URL = os.getenv(
    "TEST_DATABASE_URL",
    "mysql+asyncmy://research:local-research-only@127.0.0.1:53306/research_platform",
)


async def setup():
    db = Database(URL)
    u, w, p = [str(uuid4()) for _ in range(3)]
    async with db.transaction() as c:
        await c.execute(insert(users).values(id=u, email=u + "@test.local", password_hash="unused"))
        await c.execute(insert(workspaces).values(id=w, name="task test"))
        await c.execute(
            insert(workspace_members).values(
                id=str(uuid4()), workspace_id=w, user_id=u, role="owner"
            )
        )
        await c.execute(insert(projects).values(id=p, workspace_id=w, name="task test"))
    return db, RequestContext(u, w, "owner"), p


@pytest.mark.asyncio
async def test_live_task_lifecycle():
    db, ctx, p = await setup()
    s = TaskService(db, max_queued=2, max_attempts=2)
    try:
        tasks = await asyncio.gather(
            *(s.submit(ctx, p, "research", {"x": 1}, idempotency_key="same") for _ in range(4))
        )
        t = tasks[0]
        assert len({x["id"] for x in tasks}) == 1
        with pytest.raises(IdempotencyConflict):
            await s.submit(ctx, p, "research", {"x": 2}, idempotency_key="same")
        other = RequestContext(ctx.user_id, str(uuid4()), "owner")
        with pytest.raises(ResourceNotFound):
            await s.get(other, t["id"])
        a = await s.claim(t["id"], 1, "worker")
        assert a["execution_epoch"] == 1
        assert await s.claim(t["id"], 1, "worker") is None
        assert await s.append_event(t["id"], 0, "bad", {}) is None
        assert await s.heartbeat(t["id"], 1)
        async with db.transaction() as c:
            await c.execute(
                update(task_attempts)
                .where(task_attempts.c.task_id == t["id"])
                .values(lease_expires_at=0)
            )
        assert await s.recover_expired() >= 1  # recovery also processes other expired test tenants
        assert not await s.finish(t["id"], 1, "succeeded")
        async with db.transaction() as c:
            await c.execute(
                update(research_tasks)
                .where(research_tasks.c.id == t["id"])
                .values(next_attempt_at=0)
            )
        assert await s.claim(t["id"], 1, "late") is None
        a = await s.claim(t["id"], 2, "worker")
        assert a["execution_epoch"] == 2
        await s.cancel(ctx, t["id"])
        assert not await s.finish(t["id"], 2, "succeeded")
        assert await s.finish(t["id"], 2, "cancelled")
        with pytest.raises(QuotaExceeded):
            await s.retry(ctx, t["id"])
        events = await s.events(ctx, t["id"])
        assert [e["event_seq"] for e in events] == list(range(1, len(events) + 1))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_atomic_quota_running_slots_and_revocation():
    db, ctx, p = await setup()
    s = TaskService(db, max_queued=1, max_running=1)
    try:
        results = await asyncio.gather(
            *(s.submit(ctx, p, "test", {}, idempotency_key=str(i)) for i in range(4)),
            return_exceptions=True,
        )
        tasks = [r for r in results if isinstance(r, dict)]
        assert len(tasks) == 1 and sum(isinstance(r, QuotaExceeded) for r in results) == 3
        a = await s.claim(tasks[0]["id"], 1, "one")
        assert a
        t = await s.submit(ctx, p, "test", {}, idempotency_key="next")
        assert await s.claim(t["id"], 1, "two") is None
        deferred = await s.get(ctx, t["id"])
        assert deferred["generation"] == 2
        assert await s.finish(a["id"], a["execution_epoch"], "succeeded")
        await s.cancel(ctx, a["id"])
        assert (await s.get(ctx, a["id"]))["status"] == "succeeded"
        async with db.transaction() as c:
            await c.execute(
                update(users).where(users.c.id == ctx.user_id).values(status="disabled")
            )
            await c.execute(
                update(research_tasks)
                .where(research_tasks.c.id == t["id"])
                .values(next_attempt_at=0)
            )
        assert await s.claim(t["id"], 2, "two") is None
        async with db.transaction() as c:
            from sqlalchemy import select

            assert (
                await c.execute(
                    select(research_tasks.c.status).where(research_tasks.c.id == t["id"])
                )
            ).scalar_one() == "cancelled"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_live_rabbit_duplicate_delivery_and_outbox_confirm():
    from adapters.queue import RabbitQueue, OutboxDispatcher
    from adapters.storage.mysql.tables import outbox_events
    from sqlalchemy import select

    db, ctx, p = await setup()
    s = TaskService(db)
    q = RabbitQueue(
        "amqp://research:local-research-only@127.0.0.1:55672/",
        queue_name="test.tasks." + str(uuid4()),
    )
    await q.connect()
    try:
        t = await s.submit(ctx, p, "test", {}, idempotency_key="rabbit")
        async with db.transaction() as c:
            row = (
                (
                    await c.execute(
                        select(outbox_events).where(
                            outbox_events.c.workspace_id == ctx.workspace_id
                        )
                    )
                )
                .mappings()
                .one()
            )
        payload = {"kind": row["kind"], "payload": row["payload"]}
        await q.publish(payload)
        await q.publish(payload)
        claims = []

        async def handler(envelope):
            if envelope["payload"].get("task_id") == t["id"]:
                claims.append(await s.claim(t["id"], envelope["payload"]["generation"], "rabbit"))

        consumer = asyncio.create_task(q.consume(handler))
        for _ in range(100):
            if len(claims) >= 2:
                break
            await asyncio.sleep(0.02)
        assert len(claims) == 2 and sum(x is not None for x in claims) == 1
        await OutboxDispatcher(db, q).dispatch_once()
        async with db.transaction() as c:
            assert (
                await c.execute(
                    select(outbox_events.c.published_at).where(outbox_events.c.id == row["id"])
                )
            ).scalar_one() is not None
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    finally:
        await q.queue.delete(if_unused=False, if_empty=False)
        await q.close()
        await db.close()


@pytest.mark.asyncio
async def test_conversation_serialization_stages_and_expiry_bound():
    from adapters.storage.mysql.tables import conversations, messages
    from sqlalchemy import select, func

    db, ctx, p = await setup()
    s = TaskService(db, max_attempts=1)
    cid = str(uuid4())
    try:
        async with db.transaction() as c:
            await c.execute(
                insert(conversations).values(
                    id=cid, workspace_id=ctx.workspace_id, project_id=p, title="test"
                )
            )
        t = await s.submit(
            ctx, p, "chat", {"message": "hello"}, idempotency_key="chat", conversation_id=cid
        )
        await s.submit(
            ctx, p, "chat", {"message": "hello"}, idempotency_key="chat", conversation_id=cid
        )
        t2 = await s.submit(ctx, p, "chat", {}, idempotency_key="chat2", conversation_id=cid)
        a = await s.claim(t["id"], 1, "one")
        assert a
        assert await s.claim(t2["id"], 1, "two") is None
        assert not await s.save_stage(t["id"], 0, "paper", "parse", "abc", "artifact")
        assert await s.save_stage(t["id"], 1, "paper", "parse", "abc", "artifact")
        assert (await s.get_stage(t["id"], "paper", "parse", "abc"))["artifact_ref"] == "artifact"
        async with db.transaction() as c:
            assert (
                await c.execute(
                    select(func.count()).select_from(messages).where(messages.c.task_id == t["id"])
                )
            ).scalar_one() == 1
            await c.execute(
                update(task_attempts)
                .where(task_attempts.c.task_id == t["id"])
                .values(lease_expires_at=0)
            )
        assert not await s.heartbeat(t["id"], 1)
        assert not await s.save_stage(t["id"], 1, "paper", "parse", "abc", "wrong")
        assert await s.recover_expired() >= 1  # recovery also processes other expired test tenants
        assert (await s.get(ctx, t["id"]))["status"] == "failed"
        assert await s.recover_expired() == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_global_user_quota_across_workspaces_and_deleted_project_event():
    db, ctx, p = await setup()
    service = TaskService(db, max_queued=1)
    w2, p2 = str(uuid4()), str(uuid4())
    try:
        async with db.transaction() as c:
            await c.execute(insert(workspaces).values(id=w2, name="second"))
            await c.execute(
                insert(workspace_members).values(
                    id=str(uuid4()), workspace_id=w2, user_id=ctx.user_id, role="owner"
                )
            )
            await c.execute(insert(projects).values(id=p2, workspace_id=w2, name="second"))
        first = await service.submit(ctx, p, "test", {}, idempotency_key="one")
        with pytest.raises(QuotaExceeded):
            await service.submit(
                RequestContext(ctx.user_id, w2, "owner"), p2, "test", {}, idempotency_key="two"
            )
        async with db.transaction() as c:
            await c.execute(update(projects).where(projects.c.id == p).values(status="deleted"))
        assert await service.claim(first["id"], 1, "test") is None
        events = await service.events(ctx, first["id"])
        assert events[-1]["event_type"] == "cancelled"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_partial_import_can_retry_with_durable_success_stages():
    db, ctx, project = await setup()
    service = TaskService(db)
    try:
        task = await service.submit(
            ctx, project, "import", {"document_ids": ["good", "bad"]}, idempotency_key="partial"
        )
        attempt = await service.claim(task["id"], 1, "test")
        assert await service.save_stage(
            task["id"], attempt["execution_epoch"], "good", "index", "fingerprint", "artifact"
        )
        assert await service.finish(task["id"], attempt["execution_epoch"], "partially_succeeded")
        retried = await service.retry(ctx, task["id"])
        assert retried["status"] == "retry_wait"
        assert await service.get_stage(task["id"], "good", "index", "fingerprint")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_admin_cancel_has_atomic_audit():
    from services.platform.administration import Administration
    from adapters.storage.mysql.tables import audit_events
    from sqlalchemy import select

    db, ctx, project = await setup()
    service = TaskService(db)
    try:
        task = await service.submit(ctx, project, "import", {}, idempotency_key="admin")
        result = await Administration(service).control(ctx.user_id, task["id"], "cancel", "request")
        assert result["status"] == "cancelled"
        async with db.transaction() as c:
            audit = (
                (
                    await c.execute(
                        select(audit_events).where(audit_events.c.resource_id == task["id"])
                    )
                )
                .mappings()
                .one()
            )
            assert audit["action"] == "task.cancel" and audit["request_id"] == "request"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_killed_claiming_process_recovers_without_duplicate_commit():
    import sys

    db, ctx, project = await setup()
    service = TaskService(db, lease_seconds=1)
    process = None
    try:
        task = await service.submit(ctx, project, "import", {}, idempotency_key="crash")
        code = """import asyncio,sys
from adapters.storage.mysql import Database
from services.task_execution import TaskService
async def main():
 db=Database(sys.argv[1])
 task=await TaskService(db,lease_seconds=1).claim(sys.argv[2],1,"crash-probe")
 print(task["execution_epoch"],flush=True)
 await asyncio.sleep(60)
asyncio.run(main())
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, URL, task["id"], stdout=asyncio.subprocess.PIPE
        )
        assert await asyncio.wait_for(process.stdout.readline(), 10) == b"1\n"
        process.kill()
        await process.wait()
        await asyncio.sleep(1.1)
        await service.recover_expired()
        assert (await service.get(ctx, task["id"]))["status"] == "retry_wait"
        assert not await service.finish(task["id"], 1, "succeeded")
        async with db.transaction() as c:
            await c.execute(
                update(research_tasks)
                .where(research_tasks.c.id == task["id"])
                .values(next_attempt_at=0)
            )
        assert await service.claim(task["id"], 1, "late") is None
        new = await service.claim(task["id"], 2, "replacement")
        assert new["execution_epoch"] == 2
        assert await service.finish(task["id"], 2, "succeeded")
    finally:
        if process and process.returncode is None:
            process.kill()
            await process.wait()
        await db.close()


@pytest.mark.asyncio
async def test_project_deletion_cancels_queued_and_requests_running_cancel():
    from services.projects import ProjectService

    db, ctx, project = await setup()
    service = TaskService(db)
    try:
        queued = await service.submit(ctx, project, "import", {}, idempotency_key="queued")
        running = await service.submit(ctx, project, "import", {}, idempotency_key="running")
        await service.claim(running["id"], 1, "worker")
        await ProjectService(db).delete(ctx, project, 1)
        assert (await service.get(ctx, queued["id"]))["status"] == "cancelled"
        assert (await service.get(ctx, running["id"]))["status"] == "cancel_requested"
        assert not await service.finish(running["id"], 1, "succeeded")
        assert (await service.events(ctx, queued["id"]))[-1]["event_type"] == "cancelled"
    finally:
        await db.close()
