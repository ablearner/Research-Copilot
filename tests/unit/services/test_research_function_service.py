from types import SimpleNamespace

import httpx
import pytest

from adapters.llm.base import BaseLLMAdapter
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import ImportPapersResponse, ImportedPaperResult, PaperCandidate, ResearchReport, ResearchTopicPlan
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from domain.schemas.research_context import ResearchContext
from memory.memory_manager import MemoryManager
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from retrieval.evidence_builder import build_evidence_bundle
from services.research.paper_search_service import SearchResultBundle
from services.research.research_function_service import ResearchFunctionService
from services.research.research_report_service import ResearchReportService
from skills.research import PaperReadingSkill, ResearchEvaluationSkill, ReviewWritingSkill
from tools.retrieval_toolkit import RetrievalAgentResult


class PaperSearchServiceStub:
    async def search(self, *, topic: str, days_back: int, max_papers: int, sources: list[str], task_id=None):
        del days_back, max_papers, sources, task_id
        papers = [
            PaperCandidate(
                paper_id="paper-a",
                title=f"{topic} Paper A",
                abstract="This paper proposes a structured review pipeline with grounded evidence.",
                source="arxiv",
                citations=8,
                url="https://arxiv.org/abs/1",
            ),
            PaperCandidate(
                paper_id="paper-b",
                title=f"{topic} Paper B",
                abstract="This paper studies citation-heavy scientific question answering.",
                source="openalex",
                citations=12,
                url="https://openalex.org/W2",
            ),
        ]
        return SearchResultBundle(
            plan=ResearchTopicPlan(
                topic=topic,
                normalized_topic=topic.lower(),
                queries=[topic],
                sources=["arxiv", "openalex"],
            ),
            papers=papers,
            report=ResearchReport(
                report_id="report-search",
                task_id=None,
                topic=topic,
                generated_at="2026-04-20T00:00:00+00:00",
                markdown="# report",
                paper_count=len(papers),
            ),
            warnings=[],
        )


class ResearchServiceStub(SimpleNamespace):
    async def import_papers(self, request, *, graph_runtime):
        del graph_runtime
        paper = request.papers[0]
        return ImportPapersResponse(
            results=[
                ImportedPaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    status="imported",
                    document_id=f"doc_{paper.paper_id}",
                    storage_uri=f"/tmp/{paper.paper_id}.pdf",
                    parsed=True,
                    indexed=True,
                )
            ],
            imported_count=1,
            skipped_count=0,
            failed_count=0,
        )


class ManagerDecisionLLMStub(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        return response_model.model_validate(
            {
                "action_name": "search_literature",
                "worker_agent": "LiteratureScoutAgent",
                "instruction": "Search recent papers relevant to the user request.",
                "thought": "Need to gather evidence before any deeper operation.",
                "rationale": "Literature search is the highest-value first step.",
                "phase": "plan",
                "payload": {"goal": input_data["state"]["goal"], "mode": input_data["state"]["mode"]},
            }
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class RetrievalToolsStub:
    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        hits = [
            RetrievalHit(
                id=f"{document_ids[0]}:block_1",
                source_type="text_block",
                source_id=f"{document_ids[0]}:block_1",
                document_id=document_ids[0],
                content="The method uses planner-guided retrieval and grounded synthesis with explicit evidence control.",
                merged_score=0.93,
            )
        ]
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query=question, document_ids=document_ids, top_k=top_k),
            hits=hits,
            evidence_bundle=build_evidence_bundle(hits),
        )
        return RetrievalAgentResult(
            question=question,
            document_ids=document_ids,
            evidence_bundle=retrieval_result.evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=kwargs,
        )


class GraphSummaryStub:
    async def __call__(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            hits=[
                RetrievalHit(
                    id="summary:paper-a",
                    source_type="graph_summary",
                    source_id="summary:paper-a",
                    document_id="doc_paper-a",
                    content="Graph summary: Paper A emphasizes grounded retrieval orchestration.",
                    merged_score=0.74,
                )
            ],
            metadata={"mode": "graph_summary"},
        )


def _build_service(tmp_path, *, code_execution_enabled: bool = False):
    report_service = ResearchReportService(tmp_path / "research")
    papers = [
        PaperCandidate(
            paper_id="paper-a",
            title="Paper A",
            abstract=(
                "This paper proposes a planner-guided review system. "
                "The method coordinates retrieval and synthesis. "
                "Experiments show improved grounding."
            ),
            source="arxiv",
            citations=4,
            url="https://arxiv.org/abs/1",
            metadata={"document_id": "doc_paper-a"},
        ),
        PaperCandidate(
            paper_id="paper-b",
            title="Paper B",
            abstract="This paper studies evidence control for scientific QA and compares citation behaviors.",
            source="openalex",
            citations=11,
            url="https://openalex.org/W2",
            metadata={"document_id": "doc_paper-b"},
        ),
    ]
    report_service.save_papers("task-1", papers)
    research_service = ResearchServiceStub(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        memory_manager=MemoryManager(
            paper_knowledge_memory=PaperKnowledgeMemory(
                JsonPaperKnowledgeStore(tmp_path / "paper_knowledge")
            )
        ),
        paper_reading_skill=PaperReadingSkill(),
        review_writing_skill=ReviewWritingSkill(),
        evaluation_skill=ResearchEvaluationSkill(),
    )
    return ResearchFunctionService(
        research_service=research_service,
        graph_runtime=SimpleNamespace(
            plan_and_solve_reasoning_agent=SimpleNamespace(llm_adapter=ManagerDecisionLLMStub()),
            retrieval_tools=RetrievalToolsStub(),
            query_graph_summary=GraphSummaryStub(),
            resolve_skill_context=lambda **kwargs: {"task_type": kwargs.get("task_type", "analyze_papers")},
        ),
        allowed_file_roots=[report_service.storage_root],
        code_execution_enabled=code_execution_enabled,
    )


class AsyncClientStub:
    def __init__(self, responses: list[httpx.Response]):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        del url, params
        return self._responses.pop(0)

    async def post(self, url, json=None):
        del url, json
        return self._responses.pop(0)


class MCPClientRegistryStub:
    def __init__(self, *, search_output: dict | None = None, import_output: dict | None = None) -> None:
        self.search_output = search_output or {"items": []}
        self.import_output = import_output or {
            "imported_item_key": "ITEM123",
            "attachment_title": "Paper A PDF",
            "selected_collection": {"collection_name": "Research-Copilot"},
            "warnings": [],
        }
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, *, tool_name: str, arguments=None, server_name=None, call_id=None):
        del server_name, call_id
        payload = dict(arguments or {})
        self.calls.append((tool_name, payload))
        if tool_name == "zotero_search_items":
            return SimpleNamespace(status="succeeded", output=self.search_output, error_message=None)
        if tool_name == "zotero_attach_pdf_to_item":
            return SimpleNamespace(
                status="succeeded",
                output={"status": "attached", "attachment_count": 1, "warnings": []},
                error_message=None,
            )
        if tool_name == "zotero_import_paper":
            return SimpleNamespace(status="succeeded", output=self.import_output, error_message=None)
        return SimpleNamespace(status="not_found", output=None, error_message="missing tool")


@pytest.mark.asyncio
async def test_research_function_service_search_extract_and_generate_review(tmp_path) -> None:
    service = _build_service(tmp_path)

    search_output = await service.search_papers(
        query="agentic research",
        source=["arxiv", "openalex"],
        max_results=5,
        sort_by="citations",
    )
    structure_output = await service.extract_paper_structure(paper_id="paper-a")
    review_output = await service.generate_review(
        paper_ids=["paper-a", "paper-b"],
        style="academic",
        min_length=800,
        include_citations=True,
    )

    assert len(search_output.papers) == 2
    assert search_output.papers[0].id == "paper-b"
    assert structure_output.knowledge_card is not None
    assert structure_output.knowledge_card.paper_id == "paper-a"
    assert "## 方法对比" in review_output.review_text
    assert review_output.word_count >= 100


@pytest.mark.asyncio
async def test_research_function_service_local_file_code_execution_and_notification(tmp_path) -> None:
    service = _build_service(tmp_path)

    await service.local_file(operation="write", path="notes/test.md", content="hello")
    read_output = await service.local_file(operation="read", path="notes/test.md")
    exec_output = await service.code_execution(code="print('ok')", timeout_seconds=5)
    notify_output = await service.notification(operation="enqueue", message="read paper-a", metadata={})
    list_output = await service.notification(operation="list", metadata={})

    assert read_output.content == "hello"
    assert exec_output.success is False
    assert exec_output.return_code == 126
    assert "disabled" in exec_output.stderr
    assert notify_output.queue_size == 1
    assert list_output.items[0].message == "read paper-a"


@pytest.mark.asyncio
async def test_research_function_service_code_execution_requires_explicit_enable(tmp_path) -> None:
    service = _build_service(tmp_path, code_execution_enabled=True)

    exec_output = await service.code_execution(code="print('ok')", timeout_seconds=5)

    assert exec_output.success is True
    assert exec_output.stdout.strip() == "ok"


@pytest.mark.asyncio
async def test_research_function_service_library_sync_to_zotero(tmp_path, monkeypatch) -> None:
    service = _build_service(tmp_path)
    service.zotero_api_key = "token"
    service.zotero_library_type = "users"
    service.zotero_library_id = "12345"

    responses = [
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://api.zotero.org/users/12345/collections"),
            json=[],
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.zotero.org/users/12345/collections"),
            json={
                "successful": {
                    "0": {"key": "COLL1234", "data": {"name": "Reading List"}}
                },
                "failed": {},
            },
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.zotero.org/users/12345/items"),
            json={
                "successful": {
                    "0": {"key": "ITEM0001"},
                    "1": {"key": "ITEM0002"},
                },
                "failed": {},
            },
        ),
    ]
    monkeypatch.setattr(service, "_build_async_client", lambda **kwargs: AsyncClientStub(responses))

    output = await service.library_sync(
        provider="zotero",
        operation="sync",
        paper_ids=["paper-a", "paper-b"],
        target_collection="Reading List",
    )

    assert output.provider == "zotero"
    assert output.status == "sync"
    assert output.exported_count == 2
    assert output.output_path == "https://api.zotero.org/users/12345/collections/COLL1234"
    assert output.warnings == []


@pytest.mark.asyncio
async def test_research_function_service_library_sync_to_zotero_requires_config(tmp_path) -> None:
    service = _build_service(tmp_path)

    output = await service.library_sync(
        provider="zotero",
        operation="sync",
        paper_ids=["paper-a"],
        target_collection="Reading List",
    )

    assert output.status == "not_configured"
    assert output.exported_count == 0
    assert output.warnings


@pytest.mark.asyncio
async def test_research_function_service_decompose_task_uses_manager_decision(tmp_path) -> None:
    service = _build_service(tmp_path)

    output = await service.decompose_task(
        user_request="调研 agentic research",
        context=ResearchContext(research_topic="agentic research"),
    )

    assert len(output.task_plan) == 1
    assert output.task_plan[0].assigned_to == "LiteratureScoutAgent"
    assert output.task_plan[0].task_type == "search_literature"


@pytest.mark.asyncio
async def test_search_or_import_paper_reuses_existing_zotero_item(tmp_path) -> None:
    registry = MCPClientRegistryStub(
        search_output={
            "items": [
                {
                    "key": "ITEM_EXISTING",
                    "title": "agentic research Paper A",
                    "doi": None,
                    "url": "https://arxiv.org/abs/1",
                    "attachments": [{"url": "https://arxiv.org/pdf/1.pdf"}],
                }
            ]
        }
    )
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        collection_name="Research-Copilot",
    )

    assert output.status == "reused"
    assert output.action == "reused"
    assert output.candidate_index == 0
    assert len(output.candidates) == 2
    assert output.zotero_item_key == "ITEM_EXISTING"
    assert output.matched_by in {"title", "url"}
    assert [call[0] for call in registry.calls] == ["zotero_search_items"]


@pytest.mark.asyncio
async def test_search_or_import_paper_attaches_pdf_when_existing_item_has_no_attachment(tmp_path) -> None:
    registry = MCPClientRegistryStub(
        search_output={
            "items": [
                {
                    "key": "ITEM_EXISTING",
                    "title": "agentic research Paper A",
                    "doi": None,
                    "url": "https://arxiv.org/abs/1",
                    "attachments": [],
                }
            ]
        }
    )
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        collection_name="Research-Copilot",
    )

    assert output.status == "reused"
    assert output.attachment_count == 1
    assert [call[0] for call in registry.calls] == ["zotero_search_items", "zotero_attach_pdf_to_item"]


@pytest.mark.asyncio
async def test_search_or_import_paper_imports_when_missing(tmp_path) -> None:
    registry = MCPClientRegistryStub()
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        collection_name="Research-Copilot",
    )

    assert output.status == "imported"
    assert output.action == "imported"
    assert output.candidate_index == 0
    assert len(output.candidates) == 2
    assert output.zotero_item_key == "ITEM123"
    assert output.collection_name == "Research-Copilot"
    assert registry.calls[0][0] == "zotero_search_items"
    assert registry.calls[-1][0] == "zotero_import_paper"


@pytest.mark.asyncio
async def test_search_or_import_paper_supports_candidate_selection(tmp_path) -> None:
    registry = MCPClientRegistryStub()
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        candidate_index=1,
        collection_name="Research-Copilot",
    )

    assert output.status == "imported"
    assert output.candidate_index == 1
    assert output.selected_paper_id == "paper-b"
    import_call = registry.calls[-1]
    assert import_call[0] == "zotero_import_paper"
    assert import_call[1]["title"] == "agentic research Paper B"


@pytest.mark.asyncio
async def test_search_or_import_paper_rejects_out_of_range_candidate_index(tmp_path) -> None:
    registry = MCPClientRegistryStub()
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        candidate_index=5,
    )

    assert output.status == "failed"
    assert output.action == "none"
    assert len(output.candidates) == 2
    assert "out of range" in output.warnings[0]
    assert registry.calls == []


@pytest.mark.asyncio
async def test_search_or_import_paper_can_ingest_to_workspace(tmp_path) -> None:
    registry = MCPClientRegistryStub()
    service = _build_service(tmp_path)
    service.graph_runtime.mcp_client_registry = registry

    output = await service.search_or_import_paper(
        query="agentic research",
        source=["arxiv", "openalex"],
        collection_name="Research-Copilot",
        ingest_to_workspace=True,
    )

    assert output.status == "imported"
    assert output.workspace_status == "imported"
    assert output.workspace_document_id == "doc_paper-a"


@pytest.mark.asyncio
async def test_research_function_service_analyze_and_compatibility_wrappers(tmp_path) -> None:
    service = _build_service(tmp_path)

    analysis = await service.analyze_papers(
        question="这两篇论文有什么差异，哪篇更值得先读？",
        paper_ids=["paper-a", "paper-b"],
    )
    ask_output = await service.ask_paper(
        question="基于这两篇论文，帮我解释它们各自解决了什么问题。",
        paper_ids=["paper-a", "paper-b"],
        return_citations=True,
        min_length=200,
    )
    compare_output = await service.compare_papers(
        paper_ids=["paper-a", "paper-b"],
        dimensions=["method", "limitation"],
    )
    recommend_output = await service.recommend_papers(
        based_on_context="优先推荐值得先精读的论文",
        based_on_history=[],
        top_k=2,
    )

    assert analysis.answer
    assert analysis.paper_notes
    assert "evidence_hit_count" in analysis.metadata
    assert ask_output.answer
    assert "Paper A" in ask_output.answer
    assert compare_output.summary
    assert "Paper A" in compare_output.summary
    assert compare_output.metadata["delegated_to"] == "analyze_papers"
    assert recommend_output.metadata["delegated_to"] == "analyze_papers"
