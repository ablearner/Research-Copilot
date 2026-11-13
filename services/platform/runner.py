"""Lease-fenced workers and atomic publication of existing Supervisor results."""

import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import httpx
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update, insert, func
from adapters.storage.mysql import tables as t
from domain.schemas.identity import RequestContext
from domain.schemas.task_execution import ExecutionContext
from domain.schemas.document import ParsedDocument
from services.identity import require_membership, ResourceNotFound, PermissionDenied
from services.platform.agent_bridge import AgentBridge
from services.platform.memory_commit import MemoryCommitStore


def transient_failure(exc):
    while exc is not None:
        if isinstance(exc, (TimeoutError, ConnectionError, OSError, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError) and (
            exc.response.status_code == 429 or exc.response.status_code >= 500
        ):
            return True
        exc = exc.__cause__
    return False


class StaleExecution(Exception):
    pass


class Worker:
    def __init__(self, tasks, business, settings, agent_settings, *, slots=2, owner=None):
        self.tasks, self.business, self.db = tasks, business, tasks.db
        self.settings, self.agent_settings = settings, agent_settings
        self.slots = asyncio.Semaphore(slots)
        self.owner = owner or str(uuid4())
        self.running = set()
        self.bridges = {}
        self.memory = MemoryCommitStore(self.db, settings.worker_root)
        self.conversation_locks = {}

    async def consume(self, envelope):
        if envelope["kind"] == "document.cleanup":
            await self.cleanup(envelope["payload"])
            return
        if envelope["kind"] != "task.dispatch":
            raise ValueError("Unsupported queue envelope")
        await self.slots.acquire()
        try:
            payload = envelope["payload"]
            task = await self.tasks.claim(payload["task_id"], payload["generation"], self.owner)
            if task is None:
                self.slots.release()
                return
            operation = asyncio.create_task(self.execute(task))
            self.running.add(operation)
            operation.add_done_callback(self.running.discard)
        except BaseException:
            self.slots.release()
            raise

    async def drain(self):
        if self.running:
            await asyncio.gather(*list(self.running))

    async def execute(self, task):
        current = asyncio.current_task()

        async def heartbeat():
            while True:
                await asyncio.sleep(max(0.1, self.tasks.lease_seconds / 3))
                try:
                    valid = await self.tasks.heartbeat(task["id"], task["execution_epoch"])
                except Exception:
                    valid = False
                if not valid:
                    current.cancel()
                    return

        pulse = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(self.settings.task_timeout_seconds):
                await self.tasks.append_event(
                    task["id"],
                    task["execution_epoch"],
                    "worker.started",
                    {
                        "llm_provider": self.agent_settings.llm_provider,
                        "llm_model": self.agent_settings.llm_model,
                        "embedding_provider": self.agent_settings.embedding_provider,
                        "embedding_model": self.agent_settings.embedding_model,
                        "timeout_seconds": self.settings.task_timeout_seconds,
                    },
                )
                if task["kind"] == "import":
                    await self.import_documents(task)
                elif task["kind"] == "research":
                    lock = self.conversation_locks.setdefault(
                        task["conversation_id"], asyncio.Lock()
                    )
                    async with lock:
                        await self.research(task)
                else:
                    raise ValueError("Unsupported task kind")
        except asyncio.CancelledError:
            await self.tasks.finish(task["id"], task["execution_epoch"], "cancelled")
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Task execution failed",
                extra={"task_id": task["id"], "execution_epoch": task["execution_epoch"]},
            )
            completed = await self.tasks.finish(
                task["id"], task["execution_epoch"], "failed", error_code=type(exc).__name__
            )
            if completed and transient_failure(exc):
                from services.task_execution import QuotaExceeded

                with contextlib.suppress(QuotaExceeded, ResourceNotFound):
                    await self.tasks.retry(self.context(task), task["id"])
        finally:
            pulse.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pulse
            bridge = self.bridges.pop(task["id"], None)
            if bridge:
                with contextlib.suppress(Exception):
                    await bridge.close()
            self.slots.release()

    async def bridge(self, task, allowed, memory_path=None):
        bridge = await asyncio.to_thread(
            AgentBridge,
            self.agent_settings,
            self.root(task),
            task["workspace_id"],
            allowed,
            memory_path,
        )
        self.bridges[task["id"]] = bridge
        await bridge.initialize()
        return bridge

    def context(self, task):
        return RequestContext(task["user_id"], task["workspace_id"], "editor", task["id"])

    def root(self, task):
        return (
            Path(self.settings.worker_root).resolve()
            / task["workspace_id"]
            / (task["conversation_id"] or "imports")
            / task["id"]
            / str(task["execution_epoch"])
        )

    async def documents(self, task, *, active=False):
        requested = task["input_json"].get("document_ids")
        rows = await self.business.documents(
            self.context(task), task["project_id"], document_ids=requested or None
        )
        if requested:
            if set(requested) - {r["id"] for r in rows}:
                raise ResourceNotFound("Document not available in project")
            rows = [r for r in rows if r["id"] in requested]
        if active:
            if any(r["active_version"] is None or r["index_status"] != "succeeded" for r in rows):
                raise ValueError("Document indexing is incomplete")
        return rows

    async def versions(self, task, document_id, version_id=None):
        async with self.db.transaction() as c:
            return dict(
                (
                    await c.execute(
                        select(t.document_versions)
                        .where(
                            t.document_versions.c.workspace_id == task["workspace_id"],
                            t.document_versions.c.document_id == document_id,
                            *([t.document_versions.c.id == version_id] if version_id else []),
                        )
                        .order_by(t.document_versions.c.created_at.desc())
                        .limit(1)
                    )
                )
                .mappings()
                .one()
            )

    async def processing_version(self, task, doc):
        import importlib.metadata

        parser = "pymupdf-" + importlib.metadata.version("pymupdf")
        config = self.agent_settings
        embedding = hashlib.sha256(
            json.dumps(
                {
                    "provider": config.embedding_provider,
                    "model": config.embedding_model,
                    "dimension": config.milvus_dimension,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        version = await self.versions(task, doc["id"])
        if version["parser_version"] == parser and version["embedding_version"] == embedding:
            return version
        async with self.db.transaction() as c:
            await self.guarded(c, task)
            await self.check_documents(c, task, [doc])
            version = dict(
                version,
                id=str(uuid4()),
                parser_version=parser,
                embedding_version=embedding,
                parsed_ref=None,
                parse_status="pending",
                index_status="pending",
                created_at=await self.tasks._now(c),
                updated_at=await self.tasks._now(c),
            )
            await c.execute(insert(t.document_versions).values(**version))
        return version

    async def guarded(self, c, task):
        row = await self.tasks._internal(c, task["id"])
        if not await self.tasks._valid(c, row, task["execution_epoch"]):
            raise StaleExecution()
        role = await require_membership(c, self.context(task))
        if role not in ("owner", "admin", "editor"):
            raise PermissionDenied("Write permission revoked")
        project = (
            await c.execute(
                select(t.projects.c.id)
                .where(
                    t.projects.c.id == task["project_id"],
                    t.projects.c.workspace_id == task["workspace_id"],
                    t.projects.c.status == "active",
                )
                .with_for_update()
            )
        ).first()
        if not project:
            raise StaleExecution("Project no longer available")
        return row

    async def check_documents(self, c, task, documents):
        for doc in sorted(documents, key=lambda d: d["id"]):
            row = (
                (
                    await c.execute(
                        select(t.documents)
                        .where(
                            t.documents.c.workspace_id == task["workspace_id"],
                            t.documents.c.id == doc["id"],
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .first()
            )
            if (
                not row
                or row["deleted_at"] is not None
                or row["active_version"] != doc["active_version"]
            ):
                raise StaleExecution("Document changed during execution")

    def parsed_key(self, task, doc, version):
        return f"workspaces/{task['workspace_id']}/documents/{doc['id']}/{version['id']}/{task['id']}/{task['execution_epoch']}/parsed.json"

    async def import_public(self, task):
        from services.platform.papers import PaperService
        from services.platform.public_import import download_arxiv

        ids, failures, transient = [], 0, False
        for paper_id in task["input_json"].get("paper_ids", []):
            paper = await PaperService(self.db).get(self.context(task), paper_id)
            async with self.db.transaction() as c:
                await self.guarded(c, task)
                linked = (
                    await c.execute(
                        select(t.project_papers.c.id).where(
                            t.project_papers.c.workspace_id == task["workspace_id"],
                            t.project_papers.c.project_id == task["project_id"],
                            t.project_papers.c.paper_id == paper_id,
                        )
                    )
                ).first()
                if not linked:
                    raise ResourceNotFound()
            identifier = (paper["metadata_json"] or {}).get("arxiv_id", "")
            fingerprint = hashlib.sha256(identifier.encode()).hexdigest()
            try:
                stage = await self.tasks.get_stage(task["id"], paper_id, "download", fingerprint)
                if stage:
                    ids.append(stage["artifact_ref"])
                    continue
                data = await download_arxiv(identifier, self.business.max_file_bytes)
                async with self.db.transaction() as c:
                    await self.guarded(c, task)
                uploaded = await self.business.upload(
                    self.context(task), task["project_id"], paper["title"][:250] + ".pdf", data
                )
                if not await self.tasks.save_stage(
                    task["id"],
                    task["execution_epoch"],
                    paper_id,
                    "download",
                    fingerprint,
                    uploaded["document_id"],
                ):
                    raise StaleExecution()
                ids.append(uploaded["document_id"])
            except StaleExecution:
                raise
            except Exception as exc:
                failures += 1
                transient = transient or transient_failure(exc)
                await self.tasks.append_event(
                    task["id"],
                    task["execution_epoch"],
                    "document.download_failed",
                    {"paper_id": paper_id, "error_code": type(exc).__name__},
                )
        return ids, failures, transient

    async def import_documents(self, task):
        external_ids, external_failures, transient = await self.import_public(task)
        if task["input_json"].get("paper_ids"):
            task = dict(
                task,
                input_json=dict(
                    task["input_json"],
                    document_ids=list(
                        dict.fromkeys(task["input_json"].get("document_ids", []) + external_ids)
                    ),
                ),
            )
            if not task["input_json"]["document_ids"]:
                if transient:
                    raise ConnectionError("Public paper provider unavailable")
                raise ValueError("No public PDF could be imported")
        documents = await self.documents(task)
        if not documents:
            raise ValueError("No documents selected")
        bridge = await self.bridge(task, [])
        failures = external_failures
        for doc in documents:
            version = await self.processing_version(task, doc)
            fingerprint = hashlib.sha256(
                (
                    version["sha256"] + version["parser_version"] + version["embedding_version"]
                ).encode()
            ).hexdigest()
            key = self.parsed_key(task, doc, version)
            # Successful documents in a partial batch are not processed or published again.
            completed = await self.tasks.get_stage(task["id"], doc["id"], "index", fingerprint)
            if (
                completed
                and doc["active_version"] == version["id"]
                and doc["index_status"] == "succeeded"
            ):
                continue
            try:
                stage = await self.tasks.get_stage(task["id"], doc["id"], "parse", fingerprint)
                if stage:
                    key = stage["artifact_ref"]
                    parsed = ParsedDocument.model_validate_json(
                        await self.business.objects.get(stage["artifact_ref"])
                    )
                else:
                    data, _ = await self.business.file(self.context(task), version["file_id"])
                    path = self.root(task) / "uploads" / (doc["id"] + ".pdf")
                    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
                    await asyncio.to_thread(path.write_bytes, data)
                    parsed = await bridge.parse(path, doc["id"])
                    await self.business.objects.put(
                        key, parsed.model_dump_json().encode(), "application/json"
                    )
                    if not await self.tasks.save_stage(
                        task["id"], task["execution_epoch"], doc["id"], "parse", fingerprint, key
                    ):
                        raise StaleExecution()
                stage = await self.tasks.get_stage(task["id"], doc["id"], "index", fingerprint)
                if not stage:
                    await bridge.index(parsed)
                async with self.db.transaction() as c:
                    await self.guarded(c, task)
                    await self.check_documents(c, task, [doc])
                    await c.execute(
                        update(t.document_versions)
                        .where(t.document_versions.c.id == version["id"])
                        .values(parse_status="succeeded", index_status="succeeded", parsed_ref=key)
                    )
                    await c.execute(
                        update(t.documents)
                        .where(t.documents.c.id == doc["id"])
                        .values(active_version=version["id"], status="ready")
                    )
                if not await self.tasks.save_stage(
                    task["id"], task["execution_epoch"], doc["id"], "index", fingerprint, key
                ):
                    raise StaleExecution()
                await self.tasks.append_event(
                    task["id"],
                    task["execution_epoch"],
                    "document.indexed",
                    {"document_id": doc["id"]},
                )
            except StaleExecution:
                raise
            except Exception as exc:
                failures += 1
                transient = transient or transient_failure(exc)
                async with self.db.transaction() as c:
                    await self.guarded(c, task)
                    await c.execute(
                        update(t.document_versions)
                        .where(t.document_versions.c.id == version["id"])
                        .values(
                            index_status="failed",
                            parse_status="succeeded"
                            if await self.tasks.get_stage(
                                task["id"], doc["id"], "parse", fingerprint
                            )
                            else "failed",
                        )
                    )
                await self.tasks.append_event(
                    task["id"],
                    task["execution_epoch"],
                    "document.failed",
                    {"document_id": doc["id"], "error_code": type(exc).__name__},
                )
        status = (
            "succeeded"
            if not failures
            else (
                "failed"
                if failures == len(documents) + external_failures
                else "partially_succeeded"
            )
        )
        completed = await self.tasks.finish(task["id"], task["execution_epoch"], status)
        if completed and failures and transient:
            from services.task_execution import QuotaExceeded

            with contextlib.suppress(QuotaExceeded, ResourceNotFound):
                await self.tasks.retry(self.context(task), task["id"])

    async def research(self, task):
        if not task["conversation_id"]:
            raise ValueError("Research requires conversation")
        docs = await self.documents(task, active=True)
        messages = await self.business.messages(
            self.context(task), task["conversation_id"], limit=100, latest=True
        )
        messages.reverse()
        await self.memory.retry_pending(conversation_id=task["conversation_id"])
        memory_path = await self.memory.prepare(task, {d["id"]: d["active_version"] for d in docs})
        bridge = await self.bridge(task, [d["id"] for d in docs], memory_path)

        async def validate_scope():
            async with self.db.transaction() as c:
                await self.guarded(c, task)
                await self.check_documents(c, task, docs)

        bridge.runtime.retrieval_tools.hybrid_retriever.validate = validate_scope
        # Fresh graph/memory and local vector providers are hydrated only from active authorized data.
        for doc in docs:
            version = await self.versions(task, doc["id"], doc["active_version"])
            parsed = ParsedDocument.model_validate_json(
                await self.business.objects.get(version["parsed_ref"])
            )
            await bridge.index(parsed)

        async def progress(*args, **kwargs):
            payload = {"detail": json.loads(json.dumps({"args": args, **kwargs}, default=str))}
            if not await self.tasks.append_event(
                task["id"], task["execution_epoch"], "research.progress", payload
            ):
                raise StaleExecution()

        context = ExecutionContext(
            task["workspace_id"],
            task["user_id"],
            task["project_id"],
            task["conversation_id"],
            task["id"],
            task["execution_epoch"],
            tuple(d["id"] for d in docs),
            self.settings.task_timeout_seconds,
        )
        result = await bridge.run(task, docs, messages, progress, context)
        output = result.model_dump(mode="json")
        if output["status"] == "failed":
            await self.tasks.finish(
                task["id"], task["execution_epoch"], "failed", error_code="agent_failed"
            )
            return
        key = f"workspaces/{task['workspace_id']}/tasks/{task['id']}/{task['execution_epoch']}/result.json"
        await self.business.objects.put(key, json.dumps(output).encode(), "application/json")
        await self.publish(task, docs, output, key)
        try:
            await self.memory.commit(memory_path)
        except Exception:
            logging.getLogger(__name__).exception(
                "Memory publication deferred", extra={"task_id": task["id"]}
            )

    async def publish(self, task, docs, output, key):
        status = {"partial": "partially_succeeded", "failed": "failed"}.get(
            output["status"], "succeeded"
        )
        async with self.db.transaction() as c:
            row = await self.guarded(c, task)
            await self.check_documents(c, task, docs)
            conv = (
                (
                    await c.execute(
                        select(t.conversations)
                        .where(t.conversations.c.id == task["conversation_id"])
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            if conv["status"] == "deleted":
                raise StaleExecution()
            exists = (
                await c.execute(
                    select(t.messages.c.id).where(
                        t.messages.c.task_id == task["id"], t.messages.c.role == "assistant"
                    )
                )
            ).first()
            if exists:
                raise StaleExecution("Already published")
            seq = (
                await c.execute(
                    select(func.coalesce(func.max(t.messages.c.seq), 0)).where(
                        t.messages.c.conversation_id == task["conversation_id"]
                    )
                )
            ).scalar_one() + 1
            text = (output.get("qa") or {}).get("answer") or "\n".join(
                m.get("content", "")
                for m in output.get("messages", [])
                if m.get("role") == "assistant"
            )
            await c.execute(
                insert(t.messages).values(
                    id=str(uuid4()),
                    workspace_id=task["workspace_id"],
                    conversation_id=task["conversation_id"],
                    seq=seq,
                    role="assistant",
                    task_id=task["id"],
                    content={"text": text, "result_ref": key, "result": output},
                )
            )
            # Preserve discovered public metadata in the durable project library.
            for candidate in output.get("papers", []):
                if candidate.get("paper_id") in {d["paper_id"] for d in docs}:
                    continue
                external = (
                    candidate.get("paper_id") or candidate.get("doi") or candidate.get("title", "")
                )
                from uuid import uuid5, NAMESPACE_URL

                paper_id = str(
                    uuid5(NAMESPACE_URL, task["workspace_id"] + ":candidate:" + external)
                )
                existing = (
                    await c.execute(select(t.papers.c.id).where(t.papers.c.id == paper_id))
                ).first()
                if not existing:
                    await c.execute(
                        insert(t.papers).values(
                            id=paper_id,
                            workspace_id=task["workspace_id"],
                            title=candidate.get("title", "Untitled"),
                            metadata_json=candidate,
                        )
                    )
                linked = (
                    await c.execute(
                        select(t.project_papers.c.id).where(
                            t.project_papers.c.project_id == task["project_id"],
                            t.project_papers.c.paper_id == paper_id,
                        )
                    )
                ).first()
                if not linked:
                    await c.execute(
                        insert(t.project_papers).values(
                            id=str(uuid4()),
                            workspace_id=task["workspace_id"],
                            project_id=task["project_id"],
                            paper_id=paper_id,
                        )
                    )
            if output.get("report"):
                rid = str(uuid4())
                await c.execute(
                    insert(t.reports).values(
                        id=rid,
                        workspace_id=task["workspace_id"],
                        project_id=task["project_id"],
                        task_id=task["id"],
                        output_key="primary",
                        title=output["report"].get("title", "Research report"),
                    )
                )
                await c.execute(
                    insert(t.report_versions).values(
                        id=str(uuid4()),
                        workspace_id=task["workspace_id"],
                        report_id=rid,
                        content=output["report"],
                    )
                )
            await c.execute(
                update(t.conversations)
                .where(t.conversations.c.id == task["conversation_id"])
                .values(updated_at=await self.tasks._now(c))
            )
            await c.execute(
                update(t.research_tasks)
                .where(t.research_tasks.c.id == task["id"])
                .values(status=status, result_ref=key, updated_at=await self.tasks._now(c))
            )
            await c.execute(
                update(t.task_attempts)
                .where(
                    t.task_attempts.c.task_id == task["id"],
                    t.task_attempts.c.execution_epoch == task["execution_epoch"],
                )
                .values(status=status)
            )
            await self.tasks._event(c, row, status, {"result_ref": key})

    async def cleanup(self, payload):
        wid, did = payload["workspace_id"], payload["document_id"]
        async with self.db.transaction() as c:
            doc = (
                (
                    await c.execute(
                        select(t.documents).where(
                            t.documents.c.workspace_id == wid, t.documents.c.id == did
                        )
                    )
                )
                .mappings()
                .one()
            )
            if doc["deleted_at"] is None:
                raise ValueError("Cleanup requires tombstone")
            rows = (
                (
                    await c.execute(
                        select(t.document_versions, t.file_objects.c.object_key)
                        .join(t.file_objects, t.file_objects.c.id == t.document_versions.c.file_id)
                        .where(
                            t.document_versions.c.document_id == did,
                            t.document_versions.c.workspace_id == wid,
                        )
                    )
                )
                .mappings()
                .all()
            )
        bridge = await asyncio.to_thread(
            AgentBridge,
            self.agent_settings,
            Path(self.settings.worker_root) / wid / "cleanup" / did,
            wid,
            [],
        )
        await bridge.initialize()
        vector = bridge.runtime.embedding_index_service.vector_store
        await vector.delete_by_doc_id(did)
        for row in rows:
            await self.business.objects.delete(row["object_key"])
            if row["parsed_ref"]:
                await self.business.objects.delete(row["parsed_ref"])
        await self.memory.cleanup_document(wid, did)
        async with self.db.transaction() as c:
            for row in rows:
                await c.execute(
                    update(t.file_objects)
                    .where(t.file_objects.c.id == row["file_id"])
                    .values(status="deleted")
                )
        await asyncio.to_thread(
            shutil.rmtree, Path(self.settings.worker_root) / wid / "cleanup" / did, True
        )
        await bridge.close()
