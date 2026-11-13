"""Isolated adapter from durable platform records to the existing Supervisor runtime."""

import asyncio
import hashlib
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps.api.runtime import build_rag_runtime, initialize_rag_runtime, close_rag_runtime
from apps.api.research_runtime import build_literature_research_service
from adapters.local_runtime import InMemoryGraphStore
from domain.schemas.research import (
    PaperCandidate,
    ResearchTask,
    ResearchConversation,
    ResearchMessage,
    ResearchAgentRunRequest,
)
from domain.schemas.retrieval import HybridRetrievalResult


class ScopedRetriever:
    """Enforce scope below the tool registry, including worker knowledge-port calls."""

    def __init__(self, delegate, allowed):
        self.delegate, self.allowed = delegate, frozenset(allowed)
        self.validate = None

    async def retrieve(self, query):
        if self.validate:
            await self.validate()
        requested = set(query.document_ids) if query.document_ids else self.allowed
        if not requested <= self.allowed:
            raise PermissionError("Document outside authorized scope")
        if not requested:
            return HybridRetrievalResult(query=query)
        query = query.model_copy(update={"document_ids": sorted(requested)})
        result = await self.delegate.retrieve(query)
        # Fail closed even if a storage provider ignores its document filter.
        if any(hit.document_id not in requested for hit in result.hits):
            raise PermissionError("Retrieval provider returned unauthorized evidence")
        if any(item.document_id not in requested for item in result.evidence_bundle.evidences):
            raise PermissionError("Retrieval provider returned unauthorized evidence")
        for hit in result.hits:
            graph_items = [*hit.graph_nodes, *hit.graph_edges]
            for triple in hit.graph_triples:
                graph_items.extend((triple.subject, triple.predicate, triple.object))
            if any(item.source_reference.document_id not in requested for item in graph_items):
                raise PermissionError("Retrieval provider returned unauthorized graph source")
        if self.validate:
            await self.validate()
        return result


def scoped_settings(settings, root, workspace_id):
    root = Path(root).resolve()
    return settings.model_copy(
        update={
            "research_storage_root": str(root / "snapshot"),
            "research_sqlite_db_path": str(root / "memory.sqlite"),
            "research_session_memory_dir": str(root / "sessions"),
            "research_long_term_memory_dir": str(root / "long-term"),
            "research_paper_knowledge_dir": str(root / "papers"),
            "local_storage_root": str(root / "artifacts"),
            "upload_dir": str(root / "uploads"),
            "storage_provider": "json",
            "mcp_servers": {},
            "mcp_allowed_tools": "",
            "local_code_execution_enabled": False,
            "zotero_api_key": None,
            "graph_store_provider": "memory",
            "milvus_collection_name": "rc_attempt_"
            + hashlib.sha256((workspace_id + str(root)).encode()).hexdigest()[:40],
        }
    )


class AgentBridge:
    def __init__(self, settings, root, workspace_id, allowed, memory_path=None):
        self.settings = scoped_settings(settings, root, workspace_id)
        if memory_path is not None:
            self.settings = self.settings.model_copy(
                update={"research_sqlite_db_path": str(memory_path)}
            )
        # This graph belongs to one authorized attempt, including with remote models.
        self.runtime = build_rag_runtime(self.settings, graph_store=InMemoryGraphStore())
        retrieval = self.runtime.retrieval_tools
        retrieval.hybrid_retriever = ScopedRetriever(retrieval.hybrid_retriever, allowed)
        self.service = build_literature_research_service(self.settings, graph_runtime=self.runtime)
        self.allowed = list(allowed)
        self.pdf_paths: dict[str, str] = {}

        # Public academic search remains available; server-local personal libraries do not.
        self.service.paper_search_service.zotero_tool = None
        self.runtime.mcp_client_registry = None

    async def initialize(self):
        await initialize_rag_runtime(self.runtime)

    async def materialize_document(self, document_id, data):
        if document_id not in self.allowed:
            raise PermissionError("Document outside authorized scope")
        # Only server-authorized bytes become local inputs to figure tools.
        path = Path(self.settings.upload_dir) / (
            hashlib.sha256(document_id.encode()).hexdigest() + ".pdf"
        )
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        self.pdf_paths[document_id] = str(path)
        return path

    async def close(self):
        vector = self.runtime.embedding_index_service.vector_store
        try:
            # This bridge owns its attempt collection exclusively, including on stale completion.
            if hasattr(vector, "collection_name") and vector.client is not None:
                await vector._call_sync(
                    vector.client.drop_collection, collection_name=vector.collection_name
                )
        finally:
            await close_rag_runtime(self.runtime)

    def bind_identity(self, user_id):
        manager = self.service.memory_manager
        for name in dir(manager):
            method = getattr(manager, name)
            if (
                callable(method)
                and not name.startswith("_")
                and "user_id" in inspect.signature(method).parameters
            ):

                def bound(*args, _method=method, **kwargs):
                    kwargs["user_id"] = user_id
                    return _method(*args, **kwargs)

                setattr(manager, name, bound)

    async def parse(self, path, document_id):
        from domain.schemas.document import ParsedDocument

        output = Path(path).with_suffix(".parsed.json")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "apps.worker.parse_pdf",
            str(path),
            document_id,
            str(self.settings.local_storage_root),
            str(output),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            code = await asyncio.wait_for(process.wait(), timeout=150)
            if code:
                raise ValueError("PDF parsing failed or exceeded processing limits")
            return ParsedDocument.model_validate_json(await asyncio.to_thread(output.read_text))
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            await asyncio.to_thread(output.unlink, missing_ok=True)

    async def index(self, parsed):
        result = await self.runtime.handle_index_document(parsed, include_graph=True)
        if result.status != "succeeded":
            raise RuntimeError("Document indexing failed")
        return result

    async def run(self, task, documents, messages, on_progress, context):
        if context.workspace_id != task["workspace_id"] or set(context.allowed_document_ids) != set(
            self.allowed
        ):
            raise PermissionError("Execution scope mismatch")
        self.bind_identity(context.user_id)
        now = datetime.now(timezone.utc).isoformat()
        store = self.service.report_service
        store.save_task(
            ResearchTask(
                task_id=task["id"],
                topic=task["input_json"]["message"][:300],
                created_at=now,
                updated_at=now,
                imported_document_ids=self.allowed,
                sources=[],
                paper_count=len(documents),
            )
        )
        store.save_papers(
            task["id"],
            [
                PaperCandidate(
                    paper_id=d["paper_id"],
                    title=d["title"],
                    source="zotero",
                    ingest_status="ingested",
                    metadata={
                        "document_id": d["id"],
                        "document_ids": [d["id"]],
                        "platform_upload": True,
                        "storage_uri": self.pdf_paths[d["id"]],
                    },
                )
                for d in documents
            ],
        )
        store.save_conversation(
            ResearchConversation(
                conversation_id=task["conversation_id"],
                title="Research",
                created_at=now,
                updated_at=now,
                task_id=task["id"],
                message_count=len(messages),
            )
        )
        store.save_messages(
            task["conversation_id"],
            [
                ResearchMessage(
                    message_id=m["id"],
                    role=m["role"],
                    kind="question" if m["role"] == "user" else "answer",
                    title="",
                    content=(
                        m["content"].get("text", str(m["content"]))
                        if isinstance(m["content"], dict)
                        else str(m["content"])
                    ),
                    created_at=now,
                )
                for m in messages
            ],
        )
        return await self.service.run_agent(
            ResearchAgentRunRequest(
                message=task["input_json"]["message"],
                mode="qa" if documents else "research",
                task_id=task["id"],
                conversation_id=task["conversation_id"],
                selected_document_ids=self.allowed,
                selected_paper_ids=[d["paper_id"] for d in documents],
                sources=["arxiv", "openalex"] if not documents else [],
                auto_import=False,
                include_graph=True,
                metadata={"user_id": task["user_id"], "workspace_id": task["workspace_id"]},
            ),
            graph_runtime=self.runtime,
            on_progress=on_progress,
        )
