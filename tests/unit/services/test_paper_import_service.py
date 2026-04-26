from pathlib import Path

import httpx

from domain.schemas.document import ParsedDocument
from domain.schemas.research import ImportPapersRequest, PaperCandidate, ResearchTask
from services.research.paper_import_service import PaperImportService
from services.research.literature_research_service import LiteratureResearchService
from services.research.research_report_service import ResearchReportService


class PaperSearchServiceStub:
    async def search(self, **kwargs):  # pragma: no cover - not used in this test
        raise NotImplementedError


class PaperImportServiceStub:
    async def download_paper(self, paper):
        target = Path("/tmp/test_imported_paper.pdf")
        return type(
            "Artifact",
            (),
            {
                "paper": paper,
                "document_id": "paper_doc_1",
                "storage_uri": str(target),
                "filename": target.name,
            },
        )()


class GraphRuntimeStub:
    async def handle_parse_document(self, **kwargs):
        return ParsedDocument(
            id=kwargs.get("document_id") or "paper_doc_1",
            filename="paper.pdf",
            content_type="application/pdf",
            status="parsed",
            pages=[],
            metadata=kwargs.get("metadata") or {},
        )

    async def handle_index_document(self, **kwargs):
        return type("IndexResult", (), {"status": "succeeded"})()


class ResearchFunctionServiceStub:
    def __init__(self) -> None:
        self.calls: list[PaperCandidate] = []

    async def sync_paper_to_zotero(self, paper: PaperCandidate):
        self.calls.append(paper)
        return {
            "status": "imported",
            "action": "imported",
            "zotero_item_key": "ZOT123",
            "matched_by": None,
            "collection_name": "Research-Copilot",
            "attachment_count": 1,
            "warnings": [],
        }


class GraphRuntimeWithZoteroSyncStub(GraphRuntimeStub):
    def __init__(self) -> None:
        self.research_function_service = ResearchFunctionServiceStub()


async def test_literature_research_service_imports_and_persists_ingest_status(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_1",
        topic="无人机",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
    )
    report_service.save_task(task)
    report_service.save_papers(
        "task_1",
        [
            PaperCandidate(
                paper_id="arxiv:1",
                title="UAV Survey",
                authors=["Alice"],
                abstract="A survey.",
                source="arxiv",
                pdf_url="https://arxiv.org/pdf/1.pdf",
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )

    response = await service.import_papers(
        ImportPapersRequest(task_id="task_1", paper_ids=["arxiv:1"]),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.imported_count == 1
    persisted = report_service.load_papers("task_1")
    assert persisted[0].ingest_status == "ingested"
    assert persisted[0].metadata["document_id"] == "paper_doc_1"


async def test_literature_research_service_syncs_imported_paper_to_zotero(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_sync",
        topic="本地文献库同步",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["zotero", "arxiv"],
        imported_document_ids=[],
    )
    report_service.save_task(task)
    report_service.save_papers(
        "task_sync",
        [
            PaperCandidate(
                paper_id="arxiv:sync-1",
                title="Unified Import",
                authors=["Alice"],
                abstract="A survey.",
                source="arxiv",
                pdf_url="https://arxiv.org/pdf/sync-1.pdf",
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )
    graph_runtime = GraphRuntimeWithZoteroSyncStub()

    response = await service.import_papers(
        ImportPapersRequest(task_id="task_sync", paper_ids=["arxiv:sync-1"]),
        graph_runtime=graph_runtime,
    )

    assert response.imported_count == 1
    assert response.results[0].metadata["zotero_sync"]["zotero_item_key"] == "ZOT123"
    persisted = report_service.load_papers("task_sync")
    assert persisted[0].metadata["zotero_sync"]["collection_name"] == "Research-Copilot"
    assert graph_runtime.research_function_service.calls[0].paper_id == "arxiv:sync-1"


class FakeAsyncClient:
    def __init__(self, responses: dict[str, httpx.Response], calls: list[tuple[str, dict | None]]) -> None:
        self.responses = responses
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict | None = None):
        self.calls.append((url, headers))
        response = self.responses[url]
        response.request = httpx.Request("GET", url, headers=headers)
        return response


async def test_paper_import_service_retries_with_landing_page_pdf_discovery(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []
    landing_page = "https://www.mdpi.com/2504-446X/10/4/275"
    blocked_pdf = "https://www.mdpi.com/2504-446X/10/4/275/pdf?version=1775809941"
    discovered_pdf = "https://www.mdpi.com/2504-446X/10/4/275/pdf"
    responses = {
        blocked_pdf: httpx.Response(status_code=403, text="forbidden"),
        landing_page: httpx.Response(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f'<html><head><meta name="citation_pdf_url" content="{discovered_pdf}"></head></html>',
        ),
        discovered_pdf: httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 fake",
        ),
    }

    monkeypatch.setattr(
        "services.research.paper_import_service.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    service = PaperImportService(upload_dir=tmp_path)
    artifact = await service.download_paper(
        PaperCandidate(
            paper_id="openalex:mdpi1",
            title="MDPI UAV Survey",
            authors=["Alice"],
            abstract="Survey abstract",
            source="openalex",
            pdf_url=blocked_pdf,
            url=landing_page,
        )
    )

    assert Path(artifact.storage_uri).exists()
    assert Path(artifact.storage_uri).read_bytes().startswith(b"%PDF")
    assert any(url == landing_page for url, _headers in calls)
    assert any(url == discovered_pdf for url, _headers in calls)


async def test_paper_import_service_discovers_pdf_from_landing_page_without_pdf_url(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []
    landing_page = "https://example.org/paper/123"
    discovered_pdf = "https://example.org/paper/123.pdf"
    responses = {
        landing_page: httpx.Response(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f'<html><head><meta name="citation_pdf_url" content="{discovered_pdf}"></head></html>',
        ),
        discovered_pdf: httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 fake",
        ),
    }

    monkeypatch.setattr(
        "services.research.paper_import_service.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    service = PaperImportService(upload_dir=tmp_path)
    artifact = await service.download_paper(
        PaperCandidate(
            paper_id="semantic:1",
            title="Landing Page Discovery",
            authors=["Alice"],
            abstract="Abstract",
            source="semantic_scholar",
            url=landing_page,
        )
    )

    assert Path(artifact.storage_uri).exists()
    assert Path(artifact.storage_uri).read_bytes().startswith(b"%PDF")
    assert [url for url, _headers in calls] == [landing_page, discovered_pdf]


async def test_paper_import_service_retries_queryless_pdf_variant(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []
    versioned_pdf = "https://www.mdpi.com/2504-446X/10/4/275/pdf?version=1775809941"
    queryless_pdf = "https://www.mdpi.com/2504-446X/10/4/275/pdf"
    responses = {
        versioned_pdf: httpx.Response(status_code=403, text="forbidden"),
        queryless_pdf: httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 fake",
        ),
    }

    monkeypatch.setattr(
        "services.research.paper_import_service.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    service = PaperImportService(upload_dir=tmp_path)
    artifact = await service.download_paper(
        PaperCandidate(
            paper_id="openalex:mdpi-query",
            title="Versioned PDF Retry",
            authors=["Alice"],
            abstract="Abstract",
            source="openalex",
            pdf_url=versioned_pdf,
            url=versioned_pdf,
        )
    )

    assert Path(artifact.storage_uri).exists()
    assert Path(artifact.storage_uri).read_bytes().startswith(b"%PDF")
    assert [url for url, _headers in calls] == [versioned_pdf, queryless_pdf]


async def test_paper_import_service_uses_local_zotero_pdf_path(tmp_path) -> None:
    local_pdf = tmp_path / "zotero-library-paper.pdf"
    local_pdf.write_bytes(b"%PDF-1.7 local")

    service = PaperImportService(upload_dir=tmp_path / "uploads")
    artifact = await service.download_paper(
        PaperCandidate(
            paper_id="zotero:ITEM123",
            title="Local Zotero PDF",
            authors=["Alice"],
            abstract="Abstract",
            source="zotero",
            metadata={"zotero_local_path": str(local_pdf)},
        )
    )

    assert Path(artifact.storage_uri).exists()
    assert Path(artifact.storage_uri).read_bytes() == b"%PDF-1.7 local"
