"""Offline, read-only source migration. Default is dry-run; conflicts never overwrite data.

Run as a module from the repository: python -m scripts.migrate_business_to_mysql --help.
Agent memory tables are deliberately untouched. Missing source files remain unavailable.
"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from sqlalchemy import select, insert
from adapters.storage.mysql import Database, tables as t
from apps.platform.config import PlatformSettings


def ident(value):
    return str(uuid5(NAMESPACE_URL, "research-copilot-migration:" + str(value)))


def stamp(value):
    if not value:
        return 0
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def read_source(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        return {
            name: [dict(r) for r in conn.execute("SELECT * FROM " + name)] if name in tables else []
            for name in ["tasks", "papers", "reports", "conversations", "messages", "jobs"]
        }
    finally:
        conn.close()


async def migrate(db, path, owner_id, *, apply=False):
    raw = await asyncio.to_thread(read_source, path)
    report = {
        "mode": "apply" if apply else "dry-run",
        "counts": {k: len(v) for k, v in raw.items()},
        "owner_id": owner_id,
        "warnings": [],
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    async with db.transaction() as c:
        member = (
            (
                await c.execute(
                    select(t.workspace_members)
                    .where(
                        t.workspace_members.c.user_id == owner_id,
                        t.workspace_members.c.role == "owner",
                    )
                    .order_by(t.workspace_members.c.created_at)
                    .limit(1)
                )
            )
            .mappings()
            .first()
        )
        if not member:
            raise ValueError("Explicit existing owner with personal workspace required")
        wid = member["workspace_id"]
        project_map = {}
        task_ids = set()
        conversations = {}
        plan = []

        def add(table, record):
            if len(record["id"]) > 36:
                raise ValueError("Legacy ID exceeds schema capacity: " + record["id"])
            plan.append((table, dict(workspace_id=wid, **record)))

        def project(tid, topic):
            pid = ident("project:" + tid)
            if tid not in project_map:
                project_map[tid] = pid
                add(
                    t.projects,
                    {
                        "id": pid,
                        "name": str(topic or "历史研究")[:255],
                        "description": "Imported historical research",
                        "status": "active",
                        "created_at": 0,
                        "updated_at": 0,
                    },
                )
            return pid

        for row in raw["tasks"]:
            data = json.loads(row["data_json"])
            tid = data["task_id"]
            task_ids.add(tid)
            pid = project(tid, data.get("topic"))
            add(
                t.research_tasks,
                {
                    "id": tid,
                    "user_id": owner_id,
                    "project_id": pid,
                    "kind": "legacy_research",
                    "status": {
                        "completed": "succeeded",
                        "failed": "failed",
                        "running": "failed",
                    }.get(data.get("status"), "succeeded"),
                    "input_json": {"legacy": data},
                    "created_at": stamp(data.get("created_at")),
                    "updated_at": stamp(data.get("updated_at")),
                },
            )
        fallback = project("unassigned", "历史研究（未关联）")
        for row in raw["conversations"]:
            data = json.loads(row.get("snapshot_json") or row.get("data_json") or "{}")
            cid = data.get("conversation_id", row.get("conversation_id"))
            tid = data.get("task_id") or row.get("task_id")
            conversations[cid] = data
            add(
                t.conversations,
                {
                    "id": cid,
                    "project_id": project_map.get(tid, fallback),
                    "title": data.get("title", "历史会话"),
                    "status": "active",
                    "created_at": stamp(data.get("created_at") or row.get("created_at")),
                    "updated_at": stamp(data.get("updated_at") or row.get("updated_at")),
                },
            )
        seen_papers = {}
        for row in raw["papers"]:
            pid = project_map.get(row["task_id"], fallback)
            for data in json.loads(row["data_json"]):
                paper_id = data["paper_id"]
                if paper_id not in seen_papers:
                    seen_papers[paper_id] = data
                    add(
                        t.papers,
                        {
                            "id": paper_id,
                            "title": data.get("title", ""),
                            "metadata_json": data,
                            "status": "active",
                            "created_at": 0,
                            "updated_at": 0,
                        },
                    )
                    if data.get("ingest_status") == "ingested":
                        report["warnings"].append(
                            {
                                "paper_id": paper_id,
                                "reason": "Historical file/index requires explicit reimport; not exposed as available",
                            }
                        )
                add(
                    t.project_papers,
                    {
                        "id": ident(pid + ":" + paper_id),
                        "project_id": pid,
                        "paper_id": paper_id,
                        "created_at": 0,
                        "updated_at": 0,
                    },
                )
        for row in raw["messages"]:
            cid = row["conversation_id"]
            if cid not in conversations:
                raise ValueError("Orphan message collection: " + cid)
            for seq, data in enumerate(json.loads(row["data_json"]), 1):
                add(
                    t.messages,
                    {
                        "id": data["message_id"],
                        "conversation_id": cid,
                        "seq": seq,
                        "role": data["role"],
                        "content": {"text": data.get("content", ""), "legacy": data},
                        "created_at": stamp(data.get("created_at")),
                        "updated_at": stamp(data.get("created_at")),
                    },
                )
        for row in raw["reports"]:
            data = json.loads(row["data_json"])
            tid = data.get("task_id") or row.get("task_id")
            if tid not in task_ids:
                raise ValueError("Orphan report task: " + str(tid))
            rid = data["report_id"]
            add(
                t.reports,
                {
                    "id": rid,
                    "project_id": project_map[tid],
                    "task_id": tid,
                    "output_key": "legacy:" + rid,
                    "title": data.get("topic", "历史报告"),
                    "created_at": stamp(data.get("generated_at")),
                    "updated_at": stamp(data.get("generated_at")),
                },
            )
            add(
                t.report_versions,
                {
                    "id": ident("report-version:" + rid),
                    "report_id": rid,
                    "version": 1,
                    "content": data,
                    "created_at": stamp(data.get("generated_at")),
                    "updated_at": stamp(data.get("generated_at")),
                },
            )
        for row in raw["jobs"]:
            data = json.loads(row["data_json"])
            jid = data["job_id"]
            if jid in task_ids:
                raise ValueError("Legacy job/task ID collision")
            add(
                t.research_tasks,
                {
                    "id": jid,
                    "user_id": owner_id,
                    "project_id": project_map.get(data.get("task_id"), fallback),
                    "kind": "legacy_import",
                    "status": {"completed": "succeeded", "failed": "failed"}.get(
                        data.get("status"), "failed"
                    ),
                    "input_json": {"legacy": data},
                    "error_code": "migration_interrupted"
                    if data.get("status") in ("running", "queued")
                    else None,
                    "created_at": stamp(data.get("created_at")),
                    "updated_at": stamp(data.get("updated_at")),
                },
            )
        counts = {}
        for table, record in plan:
            existing = (
                (await c.execute(select(table).where(table.c.id == record["id"])))
                .mappings()
                .first()
            )
            if existing:
                for field, value in record.items():
                    if existing[field] != value:
                        raise ValueError(f"Migration conflict: {table.name}/{record['id']}/{field}")
            elif apply:
                await c.execute(insert(table).values(**record))
            counts[table.name] = counts.get(table.name, 0) + 1
        report["target_records"] = counts
    return report


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--owner-id", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    db = Database(PlatformSettings().database_url)
    try:
        result = await migrate(db, args.source, args.owner_id, apply=args.apply)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.report:
            args.report.write_text(payload)
        print(payload)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
