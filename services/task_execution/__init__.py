"""Durable task state, workspace lock serialization and lease fencing."""

import hashlib
import json
import random
from uuid import uuid4
from sqlalchemy import select, insert, update, func, text
from adapters.storage.mysql.tables import (
    research_tasks as T,
    task_attempts as A,
    task_events as E,
    task_stages as S,
    outbox_events as O,
    idempotency_records as I,
    workspaces,
    projects,
    conversations,
    messages,
    users,
)
from services.identity import require_membership, ResourceNotFound, PermissionDenied
from domain.schemas.identity import RequestContext


class IdempotencyConflict(Exception):
    pass


class QuotaExceeded(Exception):
    pass


class TaskConflict(Exception):
    pass


def request_fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class TaskService:
    def __init__(self, db, max_running=2, max_queued=10, lease_seconds=60, max_attempts=3):
        self.db, self.max_running, self.max_queued, self.lease_seconds, self.max_attempts = (
            db,
            max_running,
            max_queued,
            lease_seconds,
            max_attempts,
        )

    async def _now(self, c):
        return int(
            (
                await c.execute(text("SELECT UNIX_TIMESTAMP(CURRENT_TIMESTAMP(3)) * 1000"))
            ).scalar_one()
        )

    async def _lock(self, c, wid):
        await c.execute(select(workspaces.c.id).where(workspaces.c.id == wid).with_for_update())

    async def _lock_user(self, c, uid):
        await c.execute(select(users.c.id).where(users.c.id == uid).with_for_update())

    async def _auth(self, c, ctx, write=False):
        role = await require_membership(c, ctx)
        if write and role not in ("owner", "editor", "admin"):
            raise PermissionDenied("Write permission required")

    async def _row(self, c, tid):
        row = (await c.execute(select(T).where(T.c.id == tid).with_for_update())).mappings().first()
        return dict(row) if row else None

    async def _scoped(self, c, ctx, tid, write=False):
        owner = (
            await c.execute(
                select(T.c.user_id).where(T.c.id == tid, T.c.workspace_id == ctx.workspace_id)
            )
        ).scalar_one_or_none()
        await self._lock_user(c, owner or ctx.user_id)
        await self._lock(c, ctx.workspace_id)
        await self._auth(c, ctx, write)
        t = await self._row(c, tid)
        if not t or t["workspace_id"] != ctx.workspace_id:
            raise ResourceNotFound("Resource not found")
        return t

    async def _internal(self, c, tid):
        scope = (
            await c.execute(select(T.c.workspace_id, T.c.user_id).where(T.c.id == tid))
        ).first()
        if scope is None:
            return None
        await self._lock_user(c, scope.user_id)
        await self._lock(c, scope.workspace_id)
        return await self._row(c, tid)

    async def _event(self, c, t, kind, payload):
        seq = t["event_seq"] + 1
        event = dict(
            id=str(uuid4()),
            workspace_id=t["workspace_id"],
            task_id=t["id"],
            event_seq=seq,
            event_type=kind,
            payload=payload,
        )
        await c.execute(insert(E).values(**event))
        await c.execute(update(T).where(T.c.id == t["id"]).values(event_seq=seq))
        t["event_seq"] = seq
        return event

    async def _outbox(self, c, t):
        await c.execute(
            insert(O).values(
                id=str(uuid4()),
                workspace_id=t["workspace_id"],
                kind="task.dispatch",
                payload={"task_id": t["id"], "generation": t["generation"]},
                next_attempt_at=t["next_attempt_at"],
            )
        )

    async def _queued(self, c, ctx):
        count = (
            await c.execute(
                select(func.count())
                .select_from(T)
                .where(
                    T.c.user_id == ctx.user_id,
                    T.c.status.in_(["queued", "retry_wait"]),
                )
            )
        ).scalar_one()
        if count >= self.max_queued:
            raise QuotaExceeded("Queued task quota exceeded")

    async def submit(
        self, ctx, project_id, kind, payload, *, idempotency_key, conversation_id=None
    ):
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("Idempotency key required (maximum 128 characters)")
        fingerprint = request_fingerprint([project_id, kind, payload, conversation_id])
        async with self.db.transaction() as c:
            await self._lock_user(c, ctx.user_id)
            await self._lock(c, ctx.workspace_id)
            await self._auth(c, ctx, True)
            record = (
                (
                    await c.execute(
                        select(I).where(
                            I.c.workspace_id == ctx.workspace_id,
                            I.c.actor_id == ctx.user_id,
                            I.c.operation == "task.submit",
                            I.c.key == idempotency_key,
                        )
                    )
                )
                .mappings()
                .first()
            )
            if record:
                if record["request_hash"] != fingerprint:
                    raise IdempotencyConflict("Idempotency key reused for different request")
                return await self._row(c, record["resource_id"])
            p = (
                await c.execute(
                    select(projects.c.id).where(
                        projects.c.id == project_id,
                        projects.c.workspace_id == ctx.workspace_id,
                        projects.c.status == "active",
                    )
                )
            ).first()
            if not p:
                raise ResourceNotFound("Project not found")
            if conversation_id:
                conv = (
                    await c.execute(
                        select(conversations.c.id)
                        .where(
                            conversations.c.id == conversation_id,
                            conversations.c.workspace_id == ctx.workspace_id,
                            conversations.c.project_id == project_id,
                            conversations.c.status == "active",
                        )
                        .with_for_update()
                    )
                ).first()
                if not conv:
                    raise ResourceNotFound("Conversation not found")
            await self._queued(c, ctx)
            now = await self._now(c)
            tid = str(uuid4())
            await c.execute(
                insert(T).values(
                    id=tid,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    kind=kind,
                    input_json=payload,
                    status="queued",
                )
            )
            await c.execute(
                insert(I).values(
                    id=str(uuid4()),
                    workspace_id=ctx.workspace_id,
                    actor_id=ctx.user_id,
                    operation="task.submit",
                    key=idempotency_key,
                    request_hash=fingerprint,
                    resource_id=tid,
                    expires_at=now + 86400000,
                )
            )
            if conversation_id and isinstance(payload, dict) and "message" in payload:
                seq = (
                    await c.execute(
                        select(func.coalesce(func.max(messages.c.seq), 0)).where(
                            messages.c.conversation_id == conversation_id
                        )
                    )
                ).scalar_one() + 1
                await c.execute(
                    insert(messages).values(
                        id=str(uuid4()),
                        workspace_id=ctx.workspace_id,
                        conversation_id=conversation_id,
                        seq=seq,
                        role="user",
                        content=payload["message"],
                        task_id=tid,
                    )
                )
            t = await self._row(c, tid)
            await self._outbox(c, t)
            await self._event(c, t, "queued", {})
            return t

    async def get(self, ctx, task_id):
        async with self.db.transaction() as c:
            return await self._scoped(c, ctx, task_id)

    async def list(self, ctx, project_id=None):
        async with self.db.transaction() as c:
            await self._auth(c, ctx)
            q = select(T).where(T.c.workspace_id == ctx.workspace_id)
            if project_id:
                q = q.where(T.c.project_id == project_id)
            return [
                dict(r)
                for r in (await c.execute(q.order_by(T.c.created_at, T.c.id).limit(100))).mappings()
            ]

    async def events(self, ctx, task_id, after=0):
        async with self.db.transaction() as c:
            await self._scoped(c, ctx, task_id)
            return [
                dict(r)
                for r in (
                    await c.execute(
                        select(E)
                        .where(E.c.task_id == task_id, E.c.event_seq > after)
                        .order_by(E.c.event_seq)
                        .limit(1000)
                    )
                ).mappings()
            ]

    async def claim(self, task_id, generation, owner):
        async with self.db.transaction() as c:
            t = await self._internal(c, task_id)
            now = await self._now(c)
            if (
                not t
                or t["status"] not in ("queued", "retry_wait")
                or t["generation"] != generation
                or t["next_attempt_at"] > now
            ):
                return None
            try:
                await self._auth(c, RequestContext(t["user_id"], t["workspace_id"], ""), True)
            except (ResourceNotFound, PermissionDenied):
                await c.execute(
                    update(T)
                    .where(T.c.id == task_id)
                    .values(status="cancelled", error_code="authorization_revoked")
                )
                await self._event(c, t, "cancelled", {"error_code": "authorization_revoked"})
                return None
            project = (
                await c.execute(
                    select(projects.c.id).where(
                        projects.c.id == t["project_id"], projects.c.status == "active"
                    )
                )
            ).first()
            if not project:
                await c.execute(
                    update(T)
                    .where(T.c.id == task_id)
                    .values(status="cancelled", error_code="project_unavailable")
                )
                await self._event(c, t, "cancelled", {"error_code": "project_unavailable"})
                return None
            running = (
                (
                    await c.execute(
                        select(T).where(
                            (
                                (T.c.user_id == t["user_id"])
                                | (
                                    (T.c.workspace_id == t["workspace_id"])
                                    & (T.c.conversation_id == t["conversation_id"])
                                )
                            )
                            if t["conversation_id"]
                            else (T.c.user_id == t["user_id"]),
                            T.c.status.in_(["running", "cancel_requested"]),
                        )
                    )
                )
                .mappings()
                .all()
            )
            blocked = sum(r["user_id"] == t["user_id"] for r in running) >= self.max_running or (
                t["conversation_id"]
                and any(r["conversation_id"] == t["conversation_id"] for r in running)
            )
            if blocked:
                t["next_attempt_at"] = now + 1000
                t["generation"] += 1
                await c.execute(
                    update(T)
                    .where(T.c.id == task_id)
                    .values(next_attempt_at=t["next_attempt_at"], generation=t["generation"])
                )
                await self._outbox(c, t)
                return None
            if t["attempt_count"] >= self.max_attempts:
                return None
            epoch = t["execution_epoch"] + 1
            attempt = t["attempt_count"] + 1
            await c.execute(
                update(T)
                .where(T.c.id == task_id)
                .values(
                    status="running", execution_epoch=epoch, attempt_count=attempt, updated_at=now
                )
            )
            await c.execute(
                insert(A).values(
                    id=str(uuid4()),
                    workspace_id=t["workspace_id"],
                    task_id=task_id,
                    attempt_no=attempt,
                    status="running",
                    execution_epoch=epoch,
                    lease_owner=owner,
                    lease_expires_at=now + self.lease_seconds * 1000,
                )
            )
            t.update(
                status="running",
                execution_epoch=epoch,
                attempt_count=attempt,
                attempt_no=attempt,
                lease_expires_at=now + self.lease_seconds * 1000,
            )
            await self._event(c, t, "running", {"execution_epoch": epoch})
            return t

    async def _valid(self, c, t, epoch, allow_cancel=False):
        if (
            not t
            or t["execution_epoch"] != epoch
            or t["status"]
            not in (("running", "cancel_requested") if allow_cancel else ("running",))
        ):
            return False
        expiry = (
            await c.execute(
                select(A.c.lease_expires_at).where(
                    A.c.task_id == t["id"], A.c.execution_epoch == epoch, A.c.status == "running"
                )
            )
        ).scalar_one_or_none()
        return expiry is not None and expiry > await self._now(c)

    async def heartbeat(self, task_id, epoch):
        async with self.db.transaction() as c:
            t = await self._internal(c, task_id)
            if not await self._valid(c, t, epoch):
                return False
            await c.execute(
                update(A)
                .where(A.c.task_id == task_id, A.c.execution_epoch == epoch)
                .values(lease_expires_at=await self._now(c) + self.lease_seconds * 1000)
            )
            return True

    async def finish(self, task_id, epoch, status, result_ref=None, error_code=None):
        if status not in ("succeeded", "partially_succeeded", "failed", "cancelled"):
            raise ValueError("Invalid terminal status")
        async with self.db.transaction() as c:
            t = await self._internal(c, task_id)
            if not await self._valid(c, t, epoch, allow_cancel=True) or (
                t["status"] == "cancel_requested" and status != "cancelled"
            ):
                return False
            await c.execute(
                update(T)
                .where(T.c.id == task_id)
                .values(
                    status=status,
                    result_ref=result_ref,
                    error_code=error_code,
                    updated_at=await self._now(c),
                )
            )
            await c.execute(
                update(A)
                .where(A.c.task_id == task_id, A.c.execution_epoch == epoch)
                .values(status=status, error_code=error_code)
            )
            await self._event(c, t, status, {"result_ref": result_ref, "error_code": error_code})
            return True

    async def cancel(self, ctx, task_id):
        async with self.db.transaction() as c:
            t = await self._scoped(c, ctx, task_id, True)
            if t["status"] in ("queued", "retry_wait", "running"):
                status = "cancel_requested" if t["status"] == "running" else "cancelled"
                await c.execute(
                    update(T)
                    .where(T.c.id == task_id)
                    .values(status=status, cancel_requested_at=await self._now(c))
                )
                await self._event(c, t, status, {})
                t["status"] = status
            return t

    async def _retry(self, c, t):
        t.update(
            generation=t["generation"] + 1,
            next_attempt_at=await self._now(c)
            + int((2 ** t["attempt_count"] + random.random()) * 1000),
            status="retry_wait",
        )
        await c.execute(
            update(T)
            .where(T.c.id == t["id"])
            .values(
                generation=t["generation"],
                next_attempt_at=t["next_attempt_at"],
                status="retry_wait",
                cancel_requested_at=None,
                error_code=None,
            )
        )
        await self._outbox(c, t)
        await self._event(c, t, "retry_wait", {"generation": t["generation"]})

    async def retry(self, ctx, task_id):
        async with self.db.transaction() as c:
            t = await self._scoped(c, ctx, task_id, True)
            if t["status"] not in ("failed", "cancelled") and not (
                t["kind"] == "import" and t["status"] == "partially_succeeded"
            ):
                raise TaskConflict("Task is not retryable")
            if t["attempt_count"] >= self.max_attempts:
                raise QuotaExceeded("Maximum attempts reached")
            await self._queued(c, RequestContext(t["user_id"], t["workspace_id"], ctx.role))
            await self._retry(c, t)
            return t

    async def recover_expired(self):
        async with self.db.transaction() as c:
            ids = (
                (
                    await c.execute(
                        select(A.c.task_id).where(
                            A.c.status == "running", A.c.lease_expires_at <= await self._now(c)
                        )
                    )
                )
                .scalars()
                .all()
            )
        recovered = 0
        for tid in ids:
            async with self.db.transaction() as c:
                t = await self._internal(c, tid)
                if t["status"] not in ("running", "cancel_requested"):
                    continue
                a = (
                    (
                        await c.execute(
                            select(A).where(
                                A.c.task_id == tid, A.c.execution_epoch == t["execution_epoch"]
                            )
                        )
                    )
                    .mappings()
                    .one()
                )
                if a["status"] != "running" or a["lease_expires_at"] > await self._now(c):
                    continue
                await c.execute(
                    update(A)
                    .where(A.c.id == a["id"])
                    .values(status="expired", error_code="lease_expired")
                )
                if t["status"] == "cancel_requested" or t["attempt_count"] >= self.max_attempts:
                    status = "cancelled" if t["status"] == "cancel_requested" else "failed"
                    await c.execute(
                        update(T)
                        .where(T.c.id == tid)
                        .values(status=status, error_code="lease_expired")
                    )
                    await self._event(c, t, status, {"error_code": "lease_expired"})
                else:
                    await self._retry(c, t)
                recovered += 1
        return recovered

    async def append_event(self, task_id, epoch, event_type, payload):
        async with self.db.transaction() as c:
            t = await self._internal(c, task_id)
            if not await self._valid(c, t, epoch):
                return None
            return await self._event(c, t, event_type, payload)

    async def save_stage(
        self, task_id, epoch, scope_key, stage, input_fingerprint, artifact_ref, status="succeeded"
    ):
        async with self.db.transaction() as c:
            t = await self._internal(c, task_id)
            if not await self._valid(c, t, epoch):
                return False
            q = [
                S.c.task_id == task_id,
                S.c.scope_key == scope_key,
                S.c.stage == stage,
                S.c.input_fingerprint == input_fingerprint,
            ]
            found = (await c.execute(select(S.c.id).where(*q))).scalar_one_or_none()
            if found:
                await c.execute(
                    update(S)
                    .where(S.c.id == found)
                    .values(artifact_ref=artifact_ref, status=status)
                )
            else:
                await c.execute(
                    insert(S).values(
                        id=str(uuid4()),
                        workspace_id=t["workspace_id"],
                        task_id=task_id,
                        scope_key=scope_key,
                        stage=stage,
                        input_fingerprint=input_fingerprint,
                        artifact_ref=artifact_ref,
                        status=status,
                    )
                )
            return True

    async def get_stage(self, task_id, scope_key, stage, input_fingerprint):
        async with self.db.transaction() as c:
            r = (
                (
                    await c.execute(
                        select(S).where(
                            S.c.task_id == task_id,
                            S.c.scope_key == scope_key,
                            S.c.stage == stage,
                            S.c.input_fingerprint == input_fingerprint,
                            S.c.status == "succeeded",
                        )
                    )
                )
                .mappings()
                .first()
            )
            return dict(r) if r else None
