"""Age-bounded orphan cleanup; committed references and active uploads are retained."""

import time
from sqlalchemy import select
from adapters.storage.mysql import tables as t


class ObjectReconciler:
    def __init__(self, db, objects, *, grace_seconds=86400):
        self.db, self.objects, self.grace_seconds = db, objects, grace_seconds

    async def run(self, *, apply=False):
        # Enumerate first: objects created while collecting DB references cannot be candidates.
        candidates = [
            r["key"]
            for r in await self.objects.list_objects()
            if r["modified_at"] < time.time() - self.grace_seconds
        ]
        references = set()
        async with self.db.transaction() as c:
            for column in [
                t.file_objects.c.object_key,
                t.document_versions.c.parsed_ref,
                t.research_tasks.c.result_ref,
                t.task_stages.c.artifact_ref,
                t.upload_sessions.c.object_key,
            ]:
                references.update(
                    (await c.execute(select(column).where(column.is_not(None)))).scalars()
                )
        orphaned = sorted(set(candidates) - references)
        if apply:
            for key in orphaned:
                await self.objects.delete(key)
        return {"mode": "apply" if apply else "dry-run", "orphaned": orphaned}
