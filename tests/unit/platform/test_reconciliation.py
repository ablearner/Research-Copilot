import time
import pytest
from sqlalchemy import insert
from adapters.storage.mysql import Database, tables as t
from services.platform.reconciliation import ObjectReconciler


@pytest.mark.asyncio
async def test_reconciliation_retains_referenced_and_recent_objects():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()

    class Objects:
        removed = []

        async def list_objects(self):
            return [
                {"key": k, "modified_at": age}
                for k, age in [("old-orphan", 0), ("referenced", 0), ("new-orphan", time.time())]
            ]

        async def delete(self, key):
            self.removed.append(key)

    objects = Objects()
    async with db.transaction() as c:
        await c.execute(insert(t.workspaces).values(id="w", name="Workspace"))
        await c.execute(
            insert(t.file_objects).values(
                id="f", workspace_id="w", object_key="referenced", size=1, sha256="a" * 64
            )
        )
    reconciler = ObjectReconciler(db, objects)
    assert (await reconciler.run())["orphaned"] == ["old-orphan"]
    assert objects.removed == []
    await reconciler.run(apply=True)
    assert objects.removed == ["old-orphan"]
    await db.close()
