"""Synthetic service/database baseline. No HTTP, queue dispatch, or model calls."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
from uuid import uuid4

from sqlalchemy import insert, select, text, func
from adapters.storage.mysql import Database, tables as t
from apps.platform.config import PlatformSettings
from domain.schemas.identity import RequestContext
from services.platform.business import BusinessService
from services.task_execution import TaskService
from adapters.object_store.local import LocalObjectStore


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(durations, pool_waits, elapsed, errors):
    return {
        "operations": len(durations) + len(errors),
        "elapsed_seconds": elapsed,
        "successful_operations_per_second": len(durations) / elapsed,
        "p50_ms": percentile(durations, 0.50),
        "p95_ms": percentile(durations, 0.95),
        "p99_ms": percentile(durations, 0.99),
        "errors": len(errors),
        "error_rate": len(errors) / max(1, len(durations) + len(errors)),
        "error_samples": errors[:10],
        "latency_samples_ms": durations,
        "pool_acquisition_p50_ms": percentile(pool_waits, 0.50),
        "pool_acquisition_p95_ms": percentile(pool_waits, 0.95),
        "pool_acquisition_samples_ms": pool_waits,
    }


class MeasuredDatabase(Database):
    """Measure connect/checkout/pre-ping together; not pure pool queue waiting."""

    def __init__(self, url):
        super().__init__(url)
        self.acquisitions = []

    @asynccontextmanager
    async def transaction(self):
        start = time.perf_counter()
        async with self.engine.connect() as connection:
            self.acquisitions.append((time.perf_counter() - start) * 1000)
            async with connection.begin():
                yield connection


def host_information():
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    memory = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu.splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        "logical_cpu_count": os.cpu_count(),
        "visible_memory_kib": next(
            (int(line.split()[1]) for line in memory.splitlines() if line.startswith("MemTotal:")),
            None,
        ),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ["SQLAlchemy", "asyncmy", "aio-pika"]
        },
        "resource_scope": "Linux/WSL-visible host and benchmark Python process; not dedicated hardware or database-container CPU/RSS",
    }


async def seed(db, args):
    uid, wid, pid, cid = [str(uuid4()) for _ in range(4)]
    ctx = RequestContext(uid, wid, "owner", "benchmark")
    async with db.transaction() as c:
        await c.execute(
            insert(t.users).values(
                id=uid,
                email=uid + "@benchmark.local",
                password_hash="synthetic-no-login",
                email_verified_at=t.now_ms(),
            )
        )
        await c.execute(insert(t.workspaces).values(id=wid, name="Isolated synthetic benchmark"))
        await c.execute(
            insert(t.workspace_members).values(
                id=str(uuid4()), workspace_id=wid, user_id=uid, role="owner"
            )
        )
        await c.execute(
            insert(t.projects).values(id=pid, workspace_id=wid, name="Synthetic metadata baseline")
        )
        await c.execute(
            insert(t.conversations).values(
                id=cid, workspace_id=wid, project_id=pid, title="100k synthetic messages"
            )
        )
    for start in range(0, args.messages, 1000):
        async with db.transaction() as c:
            await c.execute(
                insert(t.messages),
                [
                    dict(
                        id=str(uuid4()),
                        workspace_id=wid,
                        conversation_id=cid,
                        seq=i + 1,
                        role="user",
                        content={"text": "synthetic benchmark message"},
                    )
                    for i in range(start, min(start + 1000, args.messages))
                ],
            )
    for start in range(0, args.documents, 500):
        count = min(500, args.documents - start)
        papers, docs, files, versions, links = [], [], [], [], []
        for i in range(count):
            paper, doc, file, version = [str(uuid4()) for _ in range(4)]
            papers.append(
                dict(
                    id=paper,
                    workspace_id=wid,
                    title=f"Synthetic paper {start + i}",
                    metadata_json={"synthetic": True},
                )
            )
            links.append(dict(id=str(uuid4()), workspace_id=wid, project_id=pid, paper_id=paper))
            docs.append(
                dict(
                    id=doc,
                    workspace_id=wid,
                    paper_id=paper,
                    status="ready",
                    active_version=None,
                    created_at=1700000000000 + start + i,
                )
            )
            files.append(
                dict(
                    id=file,
                    workspace_id=wid,
                    object_key="benchmark/" + file,
                    sha256=f"{start + i:064x}",
                    size=1024,
                    mime="application/pdf",
                    status="available",
                )
            )
            versions.append(
                dict(
                    id=version,
                    workspace_id=wid,
                    document_id=doc,
                    file_id=file,
                    parse_status="succeeded",
                    index_status="succeeded",
                    parser_version="synthetic-v1",
                    embedding_version="none",
                )
            )
        async with db.transaction() as c:
            for table, rows in [
                (t.papers, papers),
                (t.project_papers, links),
                (t.file_objects, files),
                (t.documents, docs),
                (t.document_versions, versions),
            ]:
                await c.execute(insert(table), rows)
            await c.execute(
                text(
                    "UPDATE documents d JOIN document_versions v ON v.document_id=d.id AND v.workspace_id=d.workspace_id SET d.active_version=v.id WHERE d.workspace_id=:w"
                ),
                {"w": wid},
            )
    tids = []
    for start in range(0, args.tasks, 1000):
        rows = []
        for i in range(start, min(start + 1000, args.tasks)):
            tid = str(uuid4())
            tids.append(tid)
            rows.append(
                dict(
                    id=tid,
                    workspace_id=wid,
                    user_id=uid,
                    project_id=pid,
                    kind="synthetic_benchmark",
                    status="succeeded" if i % 2 == 0 else "failed",
                    input_json={"synthetic": True},
                    attempt_count=1,
                )
            )
        async with db.transaction() as c:
            await c.execute(insert(t.research_tasks), rows)
    async with db.transaction() as c:
        counts = {
            table.name: (
                await c.execute(
                    select(func.count()).select_from(table).where(table.c.workspace_id == wid)
                )
            ).scalar_one()
            for table in [
                t.messages,
                t.documents,
                t.document_versions,
                t.research_tasks,
                t.outbox_events,
            ]
        }
    assert counts["outbox_events"] == 0
    return ctx, pid, cid, tids, counts


async def measure(db, operation, requests, concurrency, warmup):
    for i in range(warmup):
        await operation(i)
    db.acquisitions.clear()
    durations, errors = [], []
    before = resource.getrusage(resource.RUSAGE_SELF)
    begin = time.perf_counter()

    async def worker(offset):
        for i in range(offset, requests, concurrency):
            start = time.perf_counter()
            try:
                await operation(i)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
            else:
                durations.append((time.perf_counter() - start) * 1000)

    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    elapsed = time.perf_counter() - begin
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    result = summarize(durations, list(db.acquisitions), elapsed, errors)
    result.update(
        concurrency=concurrency,
        warmup_operations=warmup,
        client_cpu_seconds=cpu,
        client_cpu_percent_of_one_core=cpu / elapsed * 100,
        client_peak_rss_kib=after.ru_maxrss,
    )
    return result


async def query_plans(db, ctx, pid, cid, tids, args):
    queries = {
        "message_cursor": (
            "SELECT * FROM messages WHERE workspace_id=:w AND conversation_id=:c AND seq>:s ORDER BY seq LIMIT 50",
            {"w": ctx.workspace_id, "c": cid, "s": args.messages // 2},
        ),
        "document_list": (
            """SELECT d.*, p.title, v.file_id, v.parse_status, v.index_status FROM documents d JOIN project_papers pp ON pp.paper_id=d.paper_id JOIN papers p ON p.id=d.paper_id JOIN document_versions v ON v.id=COALESCE(d.active_version,(SELECT lv.id FROM document_versions lv WHERE lv.document_id=d.id AND lv.workspace_id=:w ORDER BY lv.created_at DESC,lv.id DESC LIMIT 1)) WHERE d.workspace_id=:w AND pp.workspace_id=:w AND pp.project_id=:p AND d.deleted_at IS NULL ORDER BY d.created_at,d.id LIMIT 100""",
            {"w": ctx.workspace_id, "p": pid},
        ),
        "task_status": (
            "SELECT * FROM research_tasks WHERE id=:t FOR UPDATE",
            {"t": tids[len(tids) // 2]},
        ),
    }
    plans = {}
    async with db.transaction() as c:
        for name, (sql, params) in queries.items():
            plans[name] = {
                "sql": sql,
                "parameters": params,
                "explain": [
                    dict(row)
                    for row in (await c.execute(text("EXPLAIN " + sql), params)).mappings()
                ],
            }
    return plans


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--messages", type=int, default=100000)
    p.add_argument("--documents", type=int, default=10000)
    p.add_argument("--tasks", type=int, default=10000)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reuse", help="Reuse a previous result dataset for a controlled comparison")
    p.add_argument("--output", default="benchmarks/backend/latest.json")
    args = p.parse_args()
    if (
        min(args.messages, args.documents, args.tasks, args.requests, *args.concurrency) < 1
        or args.warmup < 0
    ):
        p.error("Counts/concurrency must be positive; warmup must be nonnegative")
    db = MeasuredDatabase(PlatformSettings().database_url)
    try:
        started = datetime.now(timezone.utc).isoformat()
        if args.reuse:
            prior = json.loads(Path(args.reuse).read_text())["dataset"]
            wid, pid, cid = prior["workspace_id"], prior["project_id"], prior["conversation_id"]
            async with db.transaction() as c:
                uid = (
                    await c.execute(
                        select(t.workspace_members.c.user_id).where(
                            t.workspace_members.c.workspace_id == wid
                        )
                    )
                ).scalar_one()
                tids = list(
                    (
                        await c.execute(
                            select(t.research_tasks.c.id)
                            .where(t.research_tasks.c.workspace_id == wid)
                            .order_by(t.research_tasks.c.id)
                        )
                    ).scalars()
                )
            ctx, counts = RequestContext(uid, wid, "owner"), prior["row_counts"]
        else:
            ctx, pid, cid, tids, counts = await seed(db, args)
        svc = BusinessService(db, LocalObjectStore("/tmp/research-benchmark-objects"))
        tasks = TaskService(db)

        async def messages(i):
            rows = await svc.messages(
                ctx, cid, after=(i * 997) % max(1, args.messages - 50), limit=50
            )
            assert len(rows) == min(50, args.messages)

        async def documents(i):
            rows = await svc.documents(ctx, pid)
            assert len(rows) == min(100, args.documents)

        async def task_status(i):
            row = await tasks.get(ctx, tids[(i * 97) % len(tids)])
            assert row["status"] in ("succeeded", "failed")

        async with db.transaction() as c:
            mysql = dict(
                (
                    await c.execute(
                        text(
                            "SELECT VERSION() AS version, @@transaction_isolation AS isolation_level, @@innodb_buffer_pool_size AS innodb_buffer_pool_size, @@max_connections AS max_connections"
                        )
                    )
                )
                .mappings()
                .one()
            )
        result = {
            "started_at_utc": started,
            "finished_at_utc": None,
            "scope": "Python service + MySQL, not HTTP throughput or model execution",
            "model_calls": 0,
            "mock_dependencies": [],
            "synthetic": True,
            "hardware": host_information(),
            "mysql": mysql,
            "config": vars(args),
            "dataset": {
                "workspace_id": ctx.workspace_id,
                "project_id": pid,
                "conversation_id": cid,
                "row_counts": counts,
                "task_statuses": ["succeeded", "failed"],
                "file_bytes_created": 0,
            },
            "pool": {"size": 5, "max_overflow": 10, "pool_timeout_seconds": 30, "pre_ping": True},
            "sampling": "Closed-loop bounded coroutine workers; latency includes service authorization, transaction, database, row materialization and connection acquisition; excludes seeding and warmup. Acquisition includes connect/pre_ping, not pure queue wait. Peak RSS is cumulative process high-water mark.",
            "scenarios": {},
            "query_plans": await query_plans(db, ctx, pid, cid, tids, args),
        }
        for name, operation in [
            ("message_cursor", messages),
            ("document_list", documents),
            ("task_status", task_status),
        ]:
            result["scenarios"][name] = []
            for concurrency in args.concurrency:
                sample = await measure(db, operation, args.requests, concurrency, args.warmup)
                result["scenarios"][name].append(sample)
                print(
                    json.dumps(
                        {
                            "scenario": name,
                            **{k: v for k, v in sample.items() if not k.endswith("_samples_ms")},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Raw evidence: {output}")
        if any(sample["errors"] for samples in result["scenarios"].values() for sample in samples):
            raise SystemExit("Benchmark completed with errors; inspect raw evidence")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
