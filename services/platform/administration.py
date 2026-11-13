"""Privileged operations; the HTTP boundary must authenticate the operator first."""

from uuid import uuid4
from sqlalchemy import insert, update
from adapters.storage.mysql import tables as t
from domain.schemas.identity import RequestContext
from services.identity import require_membership, ResourceNotFound
from services.task_execution import TaskConflict, QuotaExceeded


class Administration:
    def __init__(self, tasks):
        self.tasks, self.db = tasks, tasks.db

    async def control(self, actor_id, task_id, action, request_id):
        if action not in ("cancel", "retry"):
            raise ValueError("Unsupported administrative action")
        async with self.db.transaction() as c:
            row = await self.tasks._internal(c, task_id)
            if not row:
                raise ResourceNotFound()
            if action == "retry":
                context = RequestContext(row["user_id"], row["workspace_id"], "owner")
                await require_membership(c, context)
                if row["status"] not in ("failed", "cancelled") and not (
                    row["kind"] == "import" and row["status"] == "partially_succeeded"
                ):
                    raise TaskConflict("Task is not retryable")
                if row["attempt_count"] >= self.tasks.max_attempts:
                    raise QuotaExceeded("Maximum attempts reached")
                await self.tasks._queued(c, context)
                await self.tasks._retry(c, row)
            elif row["status"] in ("queued", "retry_wait", "running"):
                status = "cancel_requested" if row["status"] == "running" else "cancelled"
                await c.execute(
                    update(t.research_tasks)
                    .where(t.research_tasks.c.id == task_id)
                    .values(status=status, cancel_requested_at=await self.tasks._now(c))
                )
                await self.tasks._event(c, row, status, {})
                row["status"] = status
            await c.execute(
                insert(t.audit_events).values(
                    id=str(uuid4()),
                    workspace_id=row["workspace_id"],
                    actor_id=actor_id,
                    action="task." + action,
                    resource_id=task_id,
                    request_id=request_id,
                    outcome="success",
                )
            )
            return {"id": task_id, "status": row["status"]}
