"""Real HTTP benchmark against an explicitly supplied, disposable platform fixture.

Set PLATFORM_BENCH_TOKEN (read/write PAT), PLATFORM_BENCH_WORKSPACE_ID,
PLATFORM_BENCH_PROJECT_ID, PLATFORM_BENCH_CONVERSATION_ID and PLATFORM_BENCH_TASK_ID.
Stop Worker and Scheduler before using --workers-stopped: cancellation cannot prevent
an already running Worker from calling a model. User messages and cancelled tasks
remain in the dedicated benchmark project. This script creates no accounts.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


def latency_summary(samples):
    ordered = sorted(samples)
    return {
        f"p{percent}_ms": ordered[math.ceil(len(ordered) * percent / 100) - 1] if ordered else None
        for percent in (50, 95, 99)
    }


def failure(exc):
    # Never serialize request headers, response bodies, credentials or arbitrary URLs.
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP_{exc.response.status_code}"
    return type(exc).__name__


async def measure(client, path, count, concurrency, *, submit=False, cancel_root=None):
    samples, submit_samples, errors, cleanup_failures, unresolved = [], [], [], [], []
    stop = asyncio.Event()
    counter = iter(range(count))
    started = time.perf_counter()

    async def worker():
        for _ in counter:
            if submit and stop.is_set():
                return
            task_id, key = None, str(uuid4())
            begin = time.perf_counter()
            error = None
            try:
                if submit:
                    response = await client.post(
                        path,
                        headers={"Idempotency-Key": key},
                        json={"message": "HTTP benchmark fixture query", "document_ids": []},
                    )
                    response.raise_for_status()
                    task_id = response.json()["id"]
                    submit_samples.append((time.perf_counter() - begin) * 1000)
                else:
                    response = await client.get(path)
                    response.raise_for_status()
                    response.json()
            except Exception as exc:
                error = failure(exc)
                if submit:
                    # A transport/server error may hide a committed submission. Preserve
                    # its key for operator reconciliation; do not silently create more.
                    stop.set()
                    if (
                        not isinstance(exc, httpx.HTTPStatusError)
                        or exc.response.status_code >= 500
                    ):
                        unresolved.append({"idempotency_key": key, "error": error})
            finally:
                if task_id is not None:
                    try:
                        response = await client.post(f"{cancel_root}/{task_id}/cancel")
                        response.raise_for_status()
                        if response.json().get("status") != "cancelled":
                            raise RuntimeError("Task was claimed; stop workers before benchmarking")
                    except Exception as exc:
                        error = "cleanup_" + failure(exc)
                        cleanup_failures.append({"task_id": task_id, "error": error})
                        stop.set()
            elapsed = (time.perf_counter() - begin) * 1000
            if error:
                errors.append({"error": error, "latency_ms": elapsed})
            else:
                samples.append(elapsed)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    completed = len(samples) + len(errors)
    result = {
        "concurrency": concurrency,
        "requested_operations": count,
        "completed_operations": completed,
        "elapsed_seconds": elapsed,
        "successful_operations_per_second": len(samples) / elapsed,
        **latency_summary(samples),
        "latency_samples_ms": samples,
        "errors": len(errors),
        "error_rate": len(errors) / completed if completed else None,
        "error_samples": errors,
    }
    if submit:
        result.update(
            submit_only={**latency_summary(submit_samples), "latency_samples_ms": submit_samples},
            cleanup_failures=cleanup_failures,
            unresolved_submissions=unresolved,
            aborted=stop.is_set(),
        )
    return result


async def run(args, fixture, token):
    workspace = f"workspaces/{fixture['workspace_id']}"
    scenarios = {
        "message_cursor": f"{workspace}/conversations/{fixture['conversation_id']}/messages?after=0&limit=50",
        "document_list": f"{workspace}/projects/{fixture['project_id']}/documents",
        "task_status": f"{workspace}/tasks/{fixture['task_id']}",
        "submit_cancel": f"{workspace}/conversations/{fixture['conversation_id']}/messages",
    }
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real HTTP client -> configured API/proxy -> authentication, services and MySQL",
        "model_execution": "Worker and Scheduler stopped by operator; no model benchmark",
        "mutation": "Creates user messages, cancelled research tasks and Outbox rows in dedicated fixture",
        "measurement": "Closed-loop, no warmup; percentiles use successful operations only. submit_cancel latency and throughput include cancellation; submit_only contains POST latency. Client connection reuse enabled.",
        "base_url": args.base_url,
        "fixture": fixture,
        "python": platform.python_version(),
        "httpx": httpx.__version__,
        "requests_per_scenario_per_concurrency": args.requests,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "scenarios": {},
    }
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=args.timeout,
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=max(args.concurrency), max_keepalive_connections=max(args.concurrency)
        ),
    ) as client:
        for name, path in scenarios.items():
            report["scenarios"][name] = []
            for concurrency in args.concurrency:
                result = await measure(
                    client,
                    path,
                    args.requests,
                    concurrency,
                    submit=name == "submit_cancel",
                    cancel_root=f"{workspace}/tasks",
                )
                report["scenarios"][name].append(result)
                print(
                    f"{name} concurrency={concurrency}: completed={result['completed_operations']} errors={result['errors']}"
                )
                if result.get("aborted"):
                    break
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return int(any(row["errors"] for rows in report["scenarios"].values() for row in rows))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="API root; web proxy alternative: http://localhost:3000/api/backend/api/v1",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="Operations per scenario and concurrency (default 5 keeps default run below 120 requests/minute)",
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", default="benchmarks/backend/http-latest.json")
    parser.add_argument(
        "--workers-stopped",
        action="store_true",
        help="Required acknowledgement: dedicated fixture, Worker and Scheduler stopped",
    )
    args = parser.parse_args()
    if not args.workers_stopped:
        parser.error(
            "Stop Worker and Scheduler and supply --workers-stopped; submissions must not execute models"
        )
    if args.requests < 1 or min(args.concurrency) < 1 or args.timeout <= 0:
        parser.error("requests, concurrency and timeout must be positive")
    base = urlsplit(args.base_url)
    if (
        base.scheme not in ("http", "https")
        or not base.netloc
        or base.username
        or base.password
        or base.query
        or base.fragment
    ):
        parser.error("base-url must be an HTTP(S) API root without credentials, query or fragment")
    fixture = {}
    for field in ("workspace_id", "project_id", "conversation_id", "task_id"):
        name = "PLATFORM_BENCH_" + field.upper()
        value = os.getenv(name, "").strip()
        if not value or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in value
        ):
            parser.error(f"Set {name} to a dedicated fixture resource ID")
        fixture[field] = value
    token = os.getenv("PLATFORM_BENCH_TOKEN", "").strip()
    if not token:
        parser.error("Set PLATFORM_BENCH_TOKEN to a read/write PAT; its value will not be logged")
    return asyncio.run(run(args, fixture, token))


if __name__ == "__main__":
    raise SystemExit(main())
