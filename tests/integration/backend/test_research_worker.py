from uuid import uuid4
import pytest
import fitz
from adapters.storage.mysql import Database
from adapters.object_store.s3 import S3ObjectStore
from services.identity import IdentityService
from services.projects import ProjectService
from services.platform.business import BusinessService
from services.platform.runner import Worker
from services.task_execution import TaskService
from apps.platform.config import PlatformSettings
from core.config import Settings


@pytest.mark.asyncio
async def test_actual_pdf_import_and_supervisor_answer(tmp_path, monkeypatch):
    settings = PlatformSettings(worker_root=tmp_path, require_email_verification=False)
    db = Database(settings.database_url)
    objects = S3ObjectStore(
        settings.object_endpoint,
        settings.object_bucket,
        settings.object_access_key,
        settings.object_secret_key,
    )
    await objects.initialize()
    ident = IdentityService(db, require_email_verification=False)
    email = str(uuid4()) + "@worker.test"
    await ident.register(email, "test-password")
    login = await ident.login(email, "test-password")
    ctx = await ident.resolve_workspace(login["token"], login["workspace"]["id"])
    project = await ProjectService(db).create(ctx, "Actual research")
    business = BusinessService(db, objects)
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (50, 50), "Research evidence: method Alpha achieved accuracy 92 percent on dataset TestSet."
    )
    data = pdf.tobytes()
    pdf.close()
    uploaded = await business.upload(ctx, project["id"], "evidence.pdf", data)
    ts = TaskService(db)
    config = Settings(
        _env_file=None,
        runtime_backend="local",
        llm_provider="local",
        embedding_provider="local",
        vector_store_provider="memory",
        graph_store_provider="memory",
        reranker_unavailable_policy="heuristic",
        reranker_allow_download=False,
    )
    worker = Worker(ts, business, settings, config)
    task = await ts.submit(
        ctx,
        project["id"],
        "import",
        {"document_ids": [uploaded["document_id"]]},
        idempotency_key=str(uuid4()),
    )
    await worker.consume(
        {"kind": "task.dispatch", "payload": {"task_id": task["id"], "generation": 1}}
    )
    await worker.drain()
    outcome = await ts.get(ctx, task["id"])
    events = await ts.events(ctx, task["id"])
    assert outcome["status"] == "succeeded", (outcome, events)
    conv = await business.create_conversation(ctx, project["id"])
    task = await ts.submit(
        ctx,
        project["id"],
        "research",
        {
            "message": "What accuracy does method Alpha achieve?",
            "document_ids": [uploaded["document_id"]],
        },
        idempotency_key=str(uuid4()),
        conversation_id=conv["id"],
    )
    await worker.consume(
        {"kind": "task.dispatch", "payload": {"task_id": task["id"], "generation": 1}}
    )
    await worker.drain()
    outcome = await ts.get(ctx, task["id"])
    events = await ts.events(ctx, task["id"])
    assert outcome["status"] in ("succeeded", "partially_succeeded"), (outcome, events)
    messages = await business.messages(ctx, conv["id"])
    assert (
        len(messages) == 2
        and messages[-1]["role"] == "assistant"
        and messages[-1]["content"]["text"]
    )
    # Public providers are the unstable external dependency; parsing/indexing and DB remain real.
    from services.platform.papers import PaperService
    from services.platform import public_import
    from sqlalchemy import update
    from adapters.storage.mysql import tables as tables

    papers = PaperService(db)
    good = await papers.save(
        ctx, "Public good", {"arxiv_id": "2401.12345"}, project_id=project["id"]
    )
    bad = await papers.save(
        ctx, "Public unavailable", {"arxiv_id": "2401.12346"}, project_id=project["id"]
    )
    calls = []
    recovered = False

    async def download(identifier, max_bytes):
        calls.append(identifier)
        if identifier == "2401.12346" and not recovered:
            raise ValueError("Unavailable PDF")
        return data

    monkeypatch.setattr(public_import, "download_arxiv", download)
    batch = await ts.submit(
        ctx,
        project["id"],
        "import",
        {"paper_ids": [good["id"], bad["id"]]},
        idempotency_key=str(uuid4()),
    )
    await worker.consume(
        {"kind": "task.dispatch", "payload": {"task_id": batch["id"], "generation": 1}}
    )
    await worker.drain()
    assert (await ts.get(ctx, batch["id"]))["status"] == "partially_succeeded"
    recovered = True
    await ts.retry(ctx, batch["id"])
    async with db.transaction() as c:
        await c.execute(
            update(tables.research_tasks)
            .where(tables.research_tasks.c.id == batch["id"])
            .values(next_attempt_at=0)
        )
    await worker.consume(
        {"kind": "task.dispatch", "payload": {"task_id": batch["id"], "generation": 2}}
    )
    await worker.drain()
    assert (await ts.get(ctx, batch["id"]))["status"] == "succeeded"
    assert calls.count("2401.12345") == 1  # completed download not repeated on batch retry
    await business.delete_document(ctx, uploaded["document_id"])
    await worker.cleanup({"workspace_id": ctx.workspace_id, "document_id": uploaded["document_id"]})
    assert await business.documents(ctx, project["id"]) == []
    await db.close()
