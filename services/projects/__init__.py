from uuid import uuid4
from sqlalchemy import select, insert, update
from adapters.storage.mysql.tables import projects, now_ms, users, workspaces, research_tasks
from services.identity import ResourceNotFound, PermissionDenied, require_membership


class VersionConflict(Exception):
    pass


class ProjectService:
    def __init__(self, db):
        self.db = db

    async def _authorize(self, conn, context, write=False):
        role = await require_membership(conn, context)
        if write and role not in ("owner", "admin", "editor"):
            raise PermissionDenied("Write permission required")

    def _scope(self, context, project_id=None):
        conditions = [
            projects.c.workspace_id == context.workspace_id,
            projects.c.status != "deleted",
        ]
        if project_id is not None:
            conditions.append(projects.c.id == project_id)
        return conditions

    async def create(self, context, name, description=""):
        if not name.strip() or len(name) > 255:
            raise ValueError("Project name required (maximum 255 characters)")
        project_id = str(uuid4())
        async with self.db.transaction() as conn:
            await self._authorize(conn, context, True)
            await conn.execute(
                insert(projects).values(
                    id=project_id,
                    workspace_id=context.workspace_id,
                    name=name,
                    description=description,
                )
            )
            return dict(
                (await conn.execute(select(projects).where(*self._scope(context, project_id))))
                .mappings()
                .one()
            )

    async def list(self, context, *, limit=100, offset=0):
        async with self.db.transaction() as conn:
            await self._authorize(conn, context)
            rows = (
                (
                    await conn.execute(
                        select(projects)
                        .where(*self._scope(context))
                        .order_by(projects.c.created_at, projects.c.id)
                        .limit(max(1, min(limit, 100)))
                        .offset(max(0, offset))
                    )
                )
                .mappings()
                .all()
            )
            return [dict(row) for row in rows]

    async def get(self, context, project_id):
        async with self.db.transaction() as conn:
            await self._authorize(conn, context)
            row = (
                (await conn.execute(select(projects).where(*self._scope(context, project_id))))
                .mappings()
                .first()
            )
            if row is None:
                raise ResourceNotFound("Resource not found")
            return dict(row)

    async def _change(self, context, project_id, expected_version, values):
        async with self.db.transaction() as conn:
            await conn.execute(
                select(users.c.id).where(users.c.id == context.user_id).with_for_update()
            )
            await conn.execute(
                select(workspaces.c.id)
                .where(workspaces.c.id == context.workspace_id)
                .with_for_update()
            )
            await self._authorize(conn, context, True)
            result = await conn.execute(
                update(projects)
                .where(*self._scope(context, project_id), projects.c.version == expected_version)
                .values(**values, version=projects.c.version + 1, updated_at=now_ms())
            )
            if not result.rowcount:
                exists = (
                    await conn.execute(
                        select(projects.c.id).where(*self._scope(context, project_id))
                    )
                ).first()
                if not exists:
                    raise ResourceNotFound("Resource not found")
                raise VersionConflict("Project version changed")
            if values.get("status") == "deleted":
                from services.task_execution import TaskService

                service = TaskService(self.db)
                pending = (
                    (
                        await conn.execute(
                            select(research_tasks)
                            .where(
                                research_tasks.c.workspace_id == context.workspace_id,
                                research_tasks.c.project_id == project_id,
                                research_tasks.c.status.in_(["queued", "retry_wait", "running"]),
                            )
                            .order_by(research_tasks.c.id)
                            .with_for_update()
                        )
                    )
                    .mappings()
                    .all()
                )
                for row in pending:
                    status = "cancel_requested" if row["status"] == "running" else "cancelled"
                    await conn.execute(
                        update(research_tasks)
                        .where(research_tasks.c.id == row["id"])
                        .values(status=status, cancel_requested_at=now_ms())
                    )
                    await service._event(conn, dict(row), status, {"reason": "project_deleted"})
            return dict(
                (
                    await conn.execute(
                        select(projects).where(
                            projects.c.id == project_id,
                            projects.c.workspace_id == context.workspace_id,
                        )
                    )
                )
                .mappings()
                .one()
            )

    async def update(self, context, project_id, expected_version, *, name=None, description=None):
        values = {}
        if name is not None:
            if not name.strip() or len(name) > 255:
                raise ValueError("Project name required (maximum 255 characters)")
            values["name"] = name
        if description is not None:
            values["description"] = description
        return await self._change(context, project_id, expected_version, values)

    async def delete(self, context, project_id, expected_version):
        await self._change(context, project_id, expected_version, {"status": "deleted"})
