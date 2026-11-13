import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from adapters.storage.mysql import Database
from adapters.storage.mysql.tables import metadata, workspaces, projects, conversations


@pytest.mark.asyncio
async def test_schema_and_composite_tenant_foreign_keys():
    required = {
        "users",
        "auth_sessions",
        "email_action_tokens",
        "workspaces",
        "workspace_members",
        "projects",
        "papers",
        "project_papers",
        "documents",
        "document_versions",
        "file_objects",
        "conversations",
        "messages",
        "research_tasks",
        "task_attempts",
        "task_stages",
        "task_events",
        "reports",
        "report_versions",
        "outbox_events",
        "idempotency_records",
        "audit_events",
    }
    assert required <= set(metadata.tables)
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    try:
        async with db.transaction() as conn:
            await conn.execute(
                insert(workspaces), [{"id": "a", "name": "a"}, {"id": "b", "name": "b"}]
            )
            await conn.execute(insert(projects).values(id="p", workspace_id="a", name="p"))
        with pytest.raises(IntegrityError):
            async with db.transaction() as conn:
                await conn.execute(
                    insert(conversations).values(id="c", workspace_id="b", project_id="p")
                )
    finally:
        await db.close()
