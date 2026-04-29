import pytest
from tempfile import SpooledTemporaryFile
from unittest.mock import AsyncMock
from fastapi import HTTPException
from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.datastructures import UploadFile
from types import SimpleNamespace

from apps.api.main import create_app
from apps.api.routers.ask import AskDocumentRequest, AskFusedRequest, ask_document, ask_fused
from apps.api.routers.charts import AskChartRequest, UnderstandChartRequest, ask_chart, understand_chart
from apps.api.routers.health import health_check
from apps.api.routers.index import IndexDocumentRequest, index_document
from apps.api.routers.mcp import MCPToolCallRequest, call_tool as call_mcp_tool
from apps.api.routers.parse import ParseDocumentRequest, parse_document
from apps.api.routers.research import (
    create_task,
    get_job,
    get_task,
    import_task_papers_job,
    import_from_todo,
    import_papers,
    reset_research_workspace,
    rerun_todo_search,
    run_research_agent,
    search_papers,
    update_todo_status,
)
from apps.api.routers.upload import upload_document
from apps.api.security import build_quota_context, require_api_key
from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import DocumentPage, ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    CreateResearchTaskRequest,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchAgentTraceStep,
    ResearchReport,
    ResearchTask,
    ResearchTaskResponse,
    ResearchTodoItem,
    SearchPapersRequest,
    SearchPapersResponse,
    ResearchTopicPlan,
    ImportPapersRequest,
    ImportPapersResponse,
    ImportedPaperResult,
    ResearchTodoActionRequest,
    ResearchTodoActionResponse,
    ResearchJob,
    UpdateResearchTodoRequest,
)
from rag_runtime.schemas import FusedAskResult
from mcp.schemas import MCPToolCallResult, MCPToolSpec
from services.research.research_context import ResearchExecutionContext


class GraphRuntimeStub:
    def __init__(self):
        self.session_memory = SimpleNamespace(
            chart_history=lambda session_id, image_path=None: [],
            append_chart_turn=lambda **kwargs: None,
        )
        self.chart_tools = SimpleNamespace(
            ask_chart=self._ask_chart,
        )
        self.last_ask_document_kwargs: dict | None = None
        self.last_ask_fused_kwargs: dict | None = None
        self.last_invoke_state: dict | None = None
        self.tool_registry = SimpleNamespace(
            get_tool=lambda name, include_disabled=False: SimpleNamespace(
                input_schema=SimpleNamespace(model_fields={})
            )
        )

    async def _ask_chart(self, **kwargs):
        return "chart answer"

    async def handle_ask_fused(self, **kwargs):
        self.last_ask_fused_kwargs = dict(kwargs)
        question = kwargs.get("question", "")
        return FusedAskResult(
            qa=QAResponse(answer="fused answer", question=question, evidence_bundle=EvidenceBundle(), confidence=0.7, metadata={}),
            chart_answer="chart answer",
            chart_confidence=0.78,
            metadata={},
        )

    async def handle_ask_document(self, **kwargs):
        self.last_ask_document_kwargs = dict(kwargs)
        question = kwargs.get("question", "")
        return QAResponse(answer="证据不足", question=question, evidence_bundle=EvidenceBundle(), confidence=0, metadata={})

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
        return SimpleNamespace(status="succeeded")

    async def invoke(self, state):
        self.last_invoke_state = dict(state)
        task_type = state["task_type"]
        if task_type == "parse":
            parsed = ParsedDocument(
                id=state.get("document_id") or "doc1",
                filename=state["file_path"],
                content_type="application/pdf",
                status="parsed",
                pages=[DocumentPage(id="p1", document_id=state.get("document_id") or "doc1", page_number=1)],
            )
            return {"parsed_document": parsed, "errors": [], "tool_traces": [], "metadata": {}}
        if task_type == "index":
            return {
                "graph_extraction_result": None,
                "errors": [],
                "tool_traces": [],
                "metadata": {},
            }
        if task_type == "chart_understand":
            chart = ChartSchema(
                id=state["chart_id"],
                document_id=state["document_id"],
                page_id=state["page_id"],
                page_number=state["page_number"],
                chart_type="bar",
            )
            return {"chart_result": {"chart": chart, "graph_text": "chart graph text", "metadata": {}}, "errors": [], "tool_traces": []}
        return {
            "final_answer": QAResponse(answer="证据不足", question=state["user_input"], evidence_bundle=EvidenceBundle(), confidence=0),
            "errors": [],
            "tool_traces": [],
            "metadata": {},
        }

    def _model_from_payload(self, payload):
        return payload

class MCPServerStub:
    def __init__(self) -> None:
        self.list_tools_kwargs: dict | None = None
        self.call_tool_kwargs: dict | None = None

    def list_tools(self, **kwargs):
        self.list_tools_kwargs = dict(kwargs)
        return [
            MCPToolSpec(
                name="hybrid_retrieve",
                description="Retrieve evidence",
                input_schema={"type": "object"},
                enabled=True,
                source="local",
                server_name="stub",
            )
        ]

    async def call_tool(self, **kwargs):
        self.call_tool_kwargs = dict(kwargs)
        return MCPToolCallResult(
            call_id=str(kwargs.get("call_id") or "call_1"),
            tool_name=str(kwargs.get("tool_name") or "tool"),
            status="succeeded",
            output={"ok": True},
            server_name="stub",
        )


class LiteratureResearchServiceStub:
    def __init__(self) -> None:
        self.last_search_request = None
        self.last_create_request = None
        self.last_persist_runtime_state = None
        self.last_record_agent_turn = None
        self.reset_called = False

    async def search_papers(self, request, *, graph_runtime=None):
        self.last_search_request = request
        return SearchPapersResponse(
            plan=ResearchTopicPlan(
                topic=request.topic,
                normalized_topic=request.topic,
                queries=[request.topic],
                days_back=request.days_back,
                max_papers=request.max_papers,
                sources=request.sources,
                metadata={},
            ),
            papers=[
                PaperCandidate(
                    paper_id="arxiv:1",
                    title="UAV Survey",
                    authors=["Alice"],
                    abstract="A survey.",
                    source="arxiv",
                )
            ],
            report=ResearchReport(
                report_id="report_1",
                topic=request.topic,
                generated_at="2026-04-17T00:00:00+00:00",
                markdown="# report",
                paper_count=1,
            ),
            warnings=[],
        )

    async def create_task(self, request, *, graph_runtime=None):
        self.last_create_request = request
        return self.get_task("research_1")

    async def run_agent(self, request, *, graph_runtime):
        self.last_create_request = request
        task_response = self.get_task(request.task_id or "research_1")
        return ResearchAgentRunResponse(
            status="succeeded",
            task=task_response.task,
            papers=task_response.papers,
            report=task_response.report,
            qa=QAResponse(
                answer="基于研究集合，当前最值得优先阅读的是 UAV Survey。",
                question=request.message,
                evidence_bundle=EvidenceBundle(),
                confidence=0.8,
                metadata={},
            ),
            trace=[
                ResearchAgentTraceStep(
                    step_index=1,
                    thought="stub",
                    action_name="answer_question",
                    status="succeeded",
                    observation="stub",
                    rationale="stub",
                )
            ],
            warnings=[],
            workspace=task_response.task.workspace,
        )

    def build_execution_context(self, **kwargs):
        return ResearchExecutionContext()

    def persist_runtime_state(
        self,
        *,
        task_response,
        workspace,
        conversation_id=None,
        advanced_strategy=None,
    ):
        self.last_persist_runtime_state = {
            "task_response": task_response,
            "workspace": workspace,
            "conversation_id": conversation_id,
            "advanced_strategy": advanced_strategy,
        }
        if task_response is not None:
            task_response.task.workspace = workspace
            if task_response.report is not None:
                task_response.report.workspace = workspace
        return task_response

    def record_agent_turn(self, conversation_id: str, *, request, response):
        self.last_record_agent_turn = {
            "conversation_id": conversation_id,
            "request": request,
            "response": response,
        }
        return None

    async def import_papers(self, request, *, graph_runtime):
        return ImportPapersResponse(
            results=[
                ImportedPaperResult(
                    paper_id="arxiv:1",
                    title="UAV Survey",
                    status="imported",
                    document_id="paper_doc_1",
                    storage_uri="/tmp/paper.pdf",
                    parsed=True,
                    indexed=True,
                )
            ],
            imported_count=1,
            skipped_count=0,
            failed_count=0,
        )

    def update_todo_status(self, task_id: str, todo_id: str, status: str):
        response = self.get_task(task_id)
        response.task.todo_items = [
            ResearchTodoItem(
                todo_id=todo_id,
                content="围绕该问题整理对比表。",
                created_at="2026-04-17T00:00:00+00:00",
                source="qa_follow_up",
                status=status,
            )
        ]
        return response

    async def rerun_todo_search(self, task_id: str, todo_id: str, request):
        response = self.get_task(task_id)
        response.task.todo_items = [
            ResearchTodoItem(
                todo_id=todo_id,
                content="围绕该问题整理对比表。",
                created_at="2026-04-17T00:00:00+00:00",
                source="qa_follow_up",
                metadata={"last_action_type": "search"},
            )
        ]
        return ResearchTodoActionResponse(
            task=response.task,
            todo=response.task.todo_items[0],
            papers=response.papers,
            report=response.report,
            warnings=[],
        )

    async def import_from_todo(self, task_id: str, todo_id: str, request, *, graph_runtime):
        response = self.get_task(task_id)
        response.task.todo_items = [
            ResearchTodoItem(
                todo_id=todo_id,
                content="围绕该问题整理对比表。",
                created_at="2026-04-17T00:00:00+00:00",
                source="qa_follow_up",
                metadata={"last_action_type": "import"},
            )
        ]
        return ResearchTodoActionResponse(
            task=response.task,
            todo=response.task.todo_items[0],
            papers=response.papers,
            report=response.report,
            warnings=[],
            import_result=ImportPapersResponse(
                results=[
                    ImportedPaperResult(
                        paper_id="arxiv:1",
                        title="UAV Survey",
                        status="imported",
                        document_id="paper_doc_1",
                        storage_uri="/tmp/paper.pdf",
                        parsed=True,
                        indexed=True,
                    )
                ],
                imported_count=1,
                skipped_count=0,
                failed_count=0,
            ),
        )

    async def start_import_job(self, request, *, graph_runtime):
        return ResearchJob(
            job_id="job_1",
            kind="paper_import",
            status="queued",
            created_at="2026-04-17T00:00:00+00:00",
            updated_at="2026-04-17T00:00:00+00:00",
            task_id=request.task_id,
            conversation_id=request.conversation_id,
            progress_message="导入任务已创建，等待后台执行。",
            progress_current=0,
            progress_total=max(len(request.paper_ids), 1),
            metadata={},
            output={},
        )

    def get_job(self, job_id: str):
        return ResearchJob(
            job_id=job_id,
            kind="paper_import",
            status="running",
            created_at="2026-04-17T00:00:00+00:00",
            updated_at="2026-04-17T00:00:00+00:00",
            task_id="research_1",
            progress_message="后台正在导入。",
            progress_current=1,
            progress_total=2,
            metadata={},
            output={},
        )

    async def reset_state(self):
        self.reset_called = True

    def get_task(self, task_id: str):
        return ResearchTaskResponse(
            task=ResearchTask(
                task_id=task_id,
                topic="无人机",
                status="completed",
                created_at="2026-04-17T00:00:00+00:00",
                updated_at="2026-04-17T00:00:00+00:00",
                days_back=30,
                max_papers=5,
                sources=["arxiv"],
                paper_count=1,
                imported_document_ids=["paper_doc_1"],
                todo_items=[
                    ResearchTodoItem(
                        todo_id="todo_1",
                        content="围绕该问题整理对比表。",
                        created_at="2026-04-17T00:00:00+00:00",
                        source="qa_follow_up",
                    )
                ],
                report_id="report_1",
            ),
            papers=[
                PaperCandidate(
                    paper_id="arxiv:1",
                    title="UAV Survey",
                    authors=["Alice"],
                    abstract="A survey.",
                    source="arxiv",
                )
            ],
            report=ResearchReport(
                report_id="report_1",
                task_id=task_id,
                topic="无人机",
                generated_at="2026-04-17T00:00:00+00:00",
                markdown="# report",
                paper_count=1,
            ),
            warnings=[],
        )


def make_request() -> Request:
    app = SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(audit_log_enabled=False, api_key_enabled=False)))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "app": app}
    request = Request(scope)
    return request


def make_upload_file(
    content: bytes,
    *,
    filename: str = "resume.pdf",
    content_type: str = "application/pdf",
) -> UploadFile:
    file_obj = SpooledTemporaryFile()
    file_obj.write(content)
    file_obj.seek(0)
    return UploadFile(
        filename=filename,
        file=file_obj,
        headers=Headers({"content-type": content_type}),
    )


def make_upload_settings(tmp_path, **overrides):
    values = {
        "app_name": "Research-Copilot",
        "app_env": "local",
        "log_level": "INFO",
        "runtime_backend": "local",
        "llm_provider": "local",
        "llm_model": "stub",
        "chart_vision_provider": None,
        "chart_vision_model": None,
        "vision_model": None,
        "embedding_provider": "local",
        "embedding_model": "stub",
        "vector_store_provider": "memory",
        "graph_store_provider": "memory",
        "research_reset_on_startup": False,
        "upload_dir": "uploads",
        "upload_max_bytes": 25 * 1024 * 1024,
        "api_key_enabled": False,
        "api_key": None,
        "cors_allow_origins": "http://localhost:3000",
        "rate_limit_max_requests": 60,
        "rate_limit_window_seconds": 60,
        "json_log_format": False,
    }
    values.update(overrides)
    settings = SimpleNamespace(**values)
    settings.resolve_path = lambda _: tmp_path
    return settings


def test_app_registers_api_routes() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/documents/upload" in paths
    assert "/documents/parse" in paths
    assert "/documents/index" in paths
    assert "/documents/ask" in paths
    assert "/documents/ask/fused" in paths
    assert "/uploads" in paths
    assert "/upload/documents" in paths
    assert "/parse/documents" in paths
    assert "/index/documents" in paths
    assert "/charts/understand" in paths
    assert "/ask/documents" in paths
    assert "/research/papers/search" in paths
    assert "/research/agent" in paths
    assert "/research/agent/run" in paths
    assert "/research/reset" in paths
    assert "/research/tasks/{task_id}/papers/import" in paths
    assert "/research/tasks/{task_id}/papers/import/jobs" in paths
    assert "/mcp/discovery" in paths
    assert app.openapi()


def test_app_disables_upload_preview_route_outside_local(tmp_path, monkeypatch) -> None:
    settings = make_upload_settings(tmp_path, app_env="production")
    graph_runtime = SimpleNamespace(
        external_tool_registry=SimpleNamespace(register_server=lambda *args, **kwargs: None)
    )
    research_service = SimpleNamespace(reset_state=AsyncMock())
    monkeypatch.setattr("apps.api.main.get_settings", lambda: settings)
    monkeypatch.setattr("apps.api.main.build_graph_runtime", lambda settings: graph_runtime)
    monkeypatch.setattr(
        "apps.api.main.build_literature_research_service",
        lambda settings, graph_runtime: research_service,
    )
    monkeypatch.setattr(
        "apps.api.main.register_research_runtime_extensions",
        lambda settings, graph_runtime, research_service: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "apps.api.main.build_academic_search_mcp_dependencies",
        lambda settings: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "apps.api.main.build_academic_search_mcp_client",
        lambda deps: SimpleNamespace(),
    )
    monkeypatch.setattr("apps.api.main.MCPServerApp.from_graph_runtime", lambda graph_runtime: SimpleNamespace())

    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/uploads" not in paths


def test_upload_routes_require_api_key_dependencies() -> None:
    app = create_app()
    upload_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path in {"/documents/upload", "/upload/documents"}
    ]

    assert {route.path for route in upload_routes} == {"/documents/upload", "/upload/documents"}
    for route in upload_routes:
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        assert require_api_key in dependency_calls
        assert build_quota_context in dependency_calls


@pytest.mark.asyncio
async def test_parse_router_handler() -> None:
    runtime = GraphRuntimeStub()
    response = await parse_document(
        ParseDocumentRequest(file_path="sample.pdf", document_id="doc1"),
        http_request=make_request(),
        graph_runtime=runtime,
        quota_context={},
    )

    assert response.parsed_document.id == "doc1"
    assert runtime.last_invoke_state is not None


@pytest.mark.asyncio
async def test_upload_router_handler_persists_document(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.api.routers.upload.get_settings",
        lambda: make_upload_settings(tmp_path),
    )

    upload = make_upload_file(b"%PDF-1.4 mock")
    response = await upload_document(
        request=make_request(),
        file=upload,
    )

    assert response.status == "uploaded"
    assert response.filename == "resume.pdf"
    assert response.metadata["preview_url"].startswith("/uploads/")
    stored_path = tmp_path / f"{response.document_id}.pdf"
    assert stored_path.exists()
    assert stored_path.read_bytes() == b"%PDF-1.4 mock"


@pytest.mark.asyncio
async def test_upload_router_handler_rejects_oversized_document(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.api.routers.upload.get_settings",
        lambda: make_upload_settings(tmp_path, upload_max_bytes=4),
    )

    with pytest.raises(HTTPException) as excinfo:
        await upload_document(
            request=make_request(),
            file=make_upload_file(b"12345"),
        )

    assert excinfo.value.status_code == 413
    assert "max size" in str(excinfo.value.detail)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_index_router_handler() -> None:
    runtime = GraphRuntimeStub()
    parsed = ParsedDocument(
        id="doc1",
        filename="sample.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[],
    )
    response = await index_document(
        IndexDocumentRequest(parsed_document=parsed),
        http_request=make_request(),
        graph_runtime=runtime,
        quota_context={},
    )

    assert response.result.document_id == "doc1"
    assert runtime.last_invoke_state is not None


@pytest.mark.asyncio
async def test_charts_router_handler() -> None:
    runtime = GraphRuntimeStub()
    response = await understand_chart(
        UnderstandChartRequest(
            image_path="/tmp/chart.png",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_id="chart1",
        ),
        http_request=make_request(),
        graph_runtime=runtime,
        quota_context={},
    )

    assert response.result.chart.id == "chart1"
    assert runtime.last_invoke_state is not None


@pytest.mark.asyncio
async def test_ask_chart_router_handler() -> None:
    response = await ask_chart(
        AskChartRequest(
            image_path="/tmp/chart.png",
            question="What does the blue line show?",
            session_id="s1",
            document_id="doc1",
            page_id="p1",
            chart_id="chart1",
        ),
        http_request=make_request(),
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.answer == "chart answer"
    assert response.session_id == "s1"
    assert response.evidence["source"] == "vision_model"
    assert response.confidence is not None


@pytest.mark.asyncio
async def test_ask_router_handler() -> None:
    runtime = GraphRuntimeStub()
    response = await ask_document(
        AskDocumentRequest(
            question="What happened?",
            doc_id="doc1",
            reasoning_style="react",
        ),
        http_request=make_request(),
        graph_runtime=runtime,
        quota_context={},
    )

    assert response.qa.answer == "证据不足"
    assert runtime.last_ask_document_kwargs is not None
    assert runtime.last_ask_document_kwargs["reasoning_style"] == "react"


@pytest.mark.asyncio
async def test_ask_fused_router_handler() -> None:
    runtime = GraphRuntimeStub()
    response = await ask_fused(
        AskFusedRequest(
            question="Does the chart agree with the document conclusion?",
            image_path="/tmp/chart.png",
            doc_id="doc1",
            document_ids=["doc1"],
            page_id="p1",
            chart_id="chart1",
            session_id="fused1",
            reasoning_style="react",
        ),
        http_request=make_request(),
        graph_runtime=runtime,
        quota_context={},
    )

    assert response.chart_answer == "chart answer"
    assert response.qa.answer == "fused answer"
    assert runtime.last_ask_fused_kwargs is not None
    assert runtime.last_ask_fused_kwargs["reasoning_style"] == "react"


@pytest.mark.asyncio
async def test_research_search_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await search_papers(
        SearchPapersRequest(topic="无人机", days_back=90, max_papers=10, sources=["arxiv", "openalex"]),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.report.paper_count == 1
    assert service.last_search_request.topic == "无人机"


@pytest.mark.asyncio
async def test_research_task_router_handlers() -> None:
    service = LiteratureResearchServiceStub()
    created = await create_task(
        CreateResearchTaskRequest(topic="无人机", days_back=30, max_papers=5, sources=["arxiv"], run_immediately=True),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )
    fetched = await get_task(
        "research_1",
        research_service=service,
    )

    assert created.task.task_id == "research_1"
    assert fetched.report is not None


@pytest.mark.asyncio
async def test_research_import_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await import_papers(
        ImportPapersRequest(
            papers=[
                PaperCandidate(
                    paper_id="arxiv:1",
                    title="UAV Survey",
                    authors=["Alice"],
                    abstract="A survey.",
                    source="arxiv",
                    pdf_url="https://arxiv.org/pdf/1.pdf",
                )
            ],
            paper_ids=["arxiv:1"],
        ),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.imported_count == 1
    assert response.results[0].document_id == "paper_doc_1"


@pytest.mark.asyncio
async def test_research_import_job_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await import_task_papers_job(
        "research_1",
        ImportPapersRequest(
            task_id="research_1",
            paper_ids=["arxiv:1"],
            question="当前最值得优先阅读的论文是哪篇？",
            top_k=8,
        ),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.job_id == "job_1"
    assert response.task_id == "research_1"


@pytest.mark.asyncio
async def test_research_job_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await get_job(
        "job_1",
        research_service=service,
    )

    assert response.job_id == "job_1"
    assert response.progress_current == 1


@pytest.mark.asyncio
async def test_research_reset_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await reset_research_workspace(
        http_request=make_request(),
        research_service=service,
        quota_context={},
    )

    assert response == {"status": "ok"}
    assert service.reset_called is True


@pytest.mark.asyncio
async def test_research_agent_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await run_research_agent(
        ResearchAgentRunRequest(
            message="这组论文最值得优先读哪篇？",
            mode="qa",
            task_id="research_1",
            sources=["arxiv"],
            auto_import=False,
            import_top_k=0,
        ),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.task is not None
    assert response.task.task_id == "research_1"
    assert response.qa is not None
    assert "UAV Survey" in response.qa.answer
    assert any(step.action_name == "answer_question" for step in response.trace)


@pytest.mark.asyncio
async def test_research_todo_update_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await update_todo_status(
        "research_1",
        "todo_1",
        UpdateResearchTodoRequest(status="done"),
        http_request=make_request(),
        research_service=service,
        quota_context={},
    )

    assert response.task.todo_items[0].status == "done"


@pytest.mark.asyncio
async def test_research_todo_search_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await rerun_todo_search(
        "research_1",
        "todo_1",
        ResearchTodoActionRequest(max_papers=5),
        http_request=make_request(),
        research_service=service,
        quota_context={},
    )

    assert response.todo.metadata["last_action_type"] == "search"
    assert response.report is not None


@pytest.mark.asyncio
async def test_research_todo_import_router_handler() -> None:
    service = LiteratureResearchServiceStub()
    response = await import_from_todo(
        "research_1",
        "todo_1",
        ResearchTodoActionRequest(max_papers=3),
        http_request=make_request(),
        research_service=service,
        graph_runtime=GraphRuntimeStub(),
        quota_context={},
    )

    assert response.import_result is not None
    assert response.import_result.imported_count == 1


@pytest.mark.asyncio
async def test_mcp_call_tool_route_forwards_arguments() -> None:
    runtime = GraphRuntimeStub()
    mcp_server = MCPServerStub()

    response = await call_mcp_tool(
        MCPToolCallRequest(
            tool_name="hybrid_retrieve",
            arguments={"question": "What happened?"},
            call_id="call-x",
        ),
        graph_runtime=runtime,
        mcp_server=mcp_server,
    )

    assert response.status == "succeeded"
    assert mcp_server.call_tool_kwargs == {
        "tool_name": "hybrid_retrieve",
        "arguments": {
            "question": "What happened?",
        },
        "call_id": "call-x",
    }


@pytest.mark.asyncio
async def test_health_router_exposes_effective_model_configuration() -> None:
    class MemorySaver:
        pass

    class SQLiteSessionMemoryStore:
        pass

    graph_runtime = SimpleNamespace(
        checkpointer=MemorySaver(),
        session_memory=SimpleNamespace(store=SQLiteSessionMemoryStore()),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(
                app_name="Research-Copilot",
                app_env="local",
                runtime_backend="business",
                llm_provider="dashscope",
                llm_model="qwen-plus-2025-07-28",
                chart_vision_provider="openai",
                chart_vision_model="gpt-4o-mini",
                embedding_provider="dashscope",
                embedding_model="text-embedding-v3",
                vector_store_provider="milvus",
                graph_store_provider="neo4j",
            ),
            graph_runtime=graph_runtime,
        )
    )
    scope = {"type": "http", "method": "GET", "path": "/health", "headers": [], "app": app}

    response = await health_check(Request(scope))

    assert response.llm_provider == "dashscope"
    assert response.llm_model == "qwen-plus-2025-07-28"
    assert response.chart_vision_provider == "openai"
    assert response.chart_vision_model == "gpt-4o-mini"
    assert response.embedding_model == "text-embedding-v3"
