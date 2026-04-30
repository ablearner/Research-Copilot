import asyncio

import pytest

from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import BoundingBox, DocumentPage, ParsedDocument, TextBlock
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research_context import ResearchContext
from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    CreateResearchConversationRequest,
    CreateResearchTaskRequest,
    ImportPapersRequest,
    ImportPapersResponse,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchJob,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskResponse,
    ResearchTaskAskRequest,
    ResearchTaskResponse,
    ResearchTodoActionRequest,
    ResearchTodoItem,
    ResearchWorkspaceState,
)
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from rag_runtime.memory import GraphSessionMemory
from retrieval.evidence_builder import build_evidence_bundle
from services.research.paper_search_service import PaperSearchService
from services.research.literature_research_service import LiteratureResearchService
from services.research.research_report_service import ResearchReportService
from services.research.research_supervisor_graph_runtime_core import ResearchSupervisorGraphRuntime
from services.research.research_workspace import build_workspace_state
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.agent_message import AgentResultMessage
from services.research.capabilities.qa_routing import ResearchQARouter
from services.research.capabilities.user_intent import ResearchIntentResolver
from services.research.capabilities.visual_anchor import VisualAnchor
from tooling.schemas import GraphSummaryToolOutput
from tools.retrieval_toolkit import RetrievalAgentResult


class ArxivToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id="arxiv:1234.5678",
                title="UAV Path Planning with Multi-Agent Reinforcement Learning",
                authors=["Alice"],
                abstract="We study UAV path planning with a multi-agent reinforcement learning policy.",
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id="1234.5678",
                pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
                url="https://arxiv.org/abs/1234.5678",
                is_open_access=True,
                published_at="2026-04-01T00:00:00+00:00",
            )
        ]


class OpenAlexToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id="https://openalex.org/W123",
                title="UAV Path Planning with Multi-Agent Reinforcement Learning",
                authors=["Alice", "Bob"],
                abstract="We study UAV path planning with a multi-agent reinforcement learning policy.",
                year=2026,
                venue="Conference on UAV Intelligence",
                source="openalex",
                doi="10.1000/example",
                url="https://openalex.org/W123",
                citations=12,
                is_open_access=True,
                published_at="2026-04-02",
            ),
            PaperCandidate(
                paper_id="https://openalex.org/W456",
                title="Remote Sensing Perception for Drone Navigation",
                authors=["Carol"],
                abstract="This paper surveys remote sensing perception for drone navigation.",
                year=2025,
                venue="OpenAlex Venue",
                source="openalex",
                url="https://openalex.org/W456",
                citations=5,
                published_at="2025-12-01",
            ),
        ]


async def test_paper_search_service_deduplicates_and_generates_report() -> None:
    service = PaperSearchService(
        arxiv_tool=ArxivToolStub(),
        openalex_tool=OpenAlexToolStub(),
    )

    result = await service.search(
        topic="无人机路径规划",
        days_back=365,
        max_papers=10,
        sources=["arxiv", "openalex"],
        task_id="task_1",
    )

    assert result.plan.queries
    assert len(result.papers) == 2
    assert result.report.task_id == "task_1"
    assert "文献调研报告" in result.report.markdown
    assert "## 研究背景" in result.report.markdown
    assert "## 方法对比" in result.report.markdown
    assert "[P1]" in result.report.markdown
    assert len(result.report.markdown) >= 800
    assert any(paper.relevance_score is not None for paper in result.papers)


class PaperSearchServiceStub:
    async def search(self, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError


class FigureSelectionLLMStub:
    def __init__(self, *, selected_figure_id: str) -> None:
        self.selected_figure_id = selected_figure_id
        self.calls: list[dict] = []

    async def generate_structured(self, prompt: str, input_data: dict, response_model):
        self.calls.append({"prompt": prompt, "input_data": dict(input_data)})
        return response_model(figure_id=self.selected_figure_id, rationale="LLM reranked the candidates.")


class QARoutingLLMStub:
    def __init__(self, *, route: str, confidence: float = 0.91, rationale: str = "LLM judged this as a chart question.") -> None:
        self.route = route
        self.confidence = confidence
        self.rationale = rationale
        self.calls: list[dict] = []

    async def generate_structured(self, prompt: str, input_data: dict, response_model):
        self.calls.append({"prompt": prompt, "input_data": dict(input_data)})
        if "selected_figure_id" in response_model.model_fields:
            raise AssertionError("QARoutingLLMStub should not be used for figure reranking.")
        return response_model(route=self.route, confidence=self.confidence, rationale=self.rationale)


class PaperSearchServiceWithLLMStub(PaperSearchServiceStub):
    def __init__(self, *, llm_adapter) -> None:
        self.llm_adapter = llm_adapter


class UserIntentLLMStub:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def generate_structured(self, prompt: str, input_data: dict, response_model):
        self.calls.append({"prompt": prompt, "input_data": dict(input_data)})
        return response_model.model_validate(self.payload)


@pytest.mark.asyncio
async def test_visual_anchor_prefers_explicit_figure_number_before_llm() -> None:
    skill = VisualAnchor(llm_adapter=FigureSelectionLLMStub(selected_figure_id="paper-b:chart-1"))
    paper = PaperCandidate(
        paper_id="paper-b",
        title="Paper B",
        abstract="B",
        source="arxiv",
        ingest_status="ingested",
        metadata={
            "document_id": "doc_2",
            "paper_figure_cache": {
                "document_id": "doc_2",
                "figures": [
                    {
                        "figure_id": "paper-b:chart-1",
                        "paper_id": "paper-b",
                        "document_id": "doc_2",
                        "page_id": "page-1",
                        "page_number": 1,
                        "chart_id": "chart-1",
                        "image_path": "/tmp/overview.png",
                        "title": "Figure 1. Overview",
                        "caption": "System overview",
                        "source": "chart_candidate",
                        "metadata": {},
                    },
                    {
                        "figure_id": "paper-b:chart-2",
                        "paper_id": "paper-b",
                        "document_id": "doc_2",
                        "page_id": "page-2",
                        "page_number": 2,
                        "chart_id": "chart-2",
                        "image_path": "/tmp/results.png",
                        "title": "Figure 2. Main results",
                        "caption": "Evaluation results",
                        "source": "chart_candidate",
                        "metadata": {},
                    },
                ],
                "warnings": [],
            },
        },
    )

    anchor = await skill.infer_cached_visual_anchor(
        papers=[paper],
        document_ids=["doc_2"],
        question="Figure 2 的实验效果怎么样？",
        load_cached_figure_payload=lambda *, paper: paper.metadata.get("paper_figure_cache"),
    )

    assert anchor is not None
    assert anchor["chart_id"] == "chart-2"
    assert anchor["anchor_selection"] == "deterministic_figure_reference"


@pytest.mark.asyncio
async def test_user_intent_prefers_llm_over_marker_hint_when_available() -> None:
    llm = UserIntentLLMStub(
        {
            "intent": "general_answer",
            "confidence": 0.92,
            "target_kind": "none",
            "needs_clarification": False,
            "rationale": "The user is asking a general programming question.",
            "markers": ["heuristic_hint_used"],
            "source": "llm",
        }
    )
    skill = ResearchIntentResolver(llm_adapter=llm)

    result = await skill.resolve_async(
        message="什么是 Python 生成器？",
        has_task=False,
        candidate_paper_count=0,
        active_paper_ids=[],
        selected_paper_ids=[],
        has_visual_anchor=False,
        has_document_input=False,
    )

    assert result.intent == "general_answer"
    assert result.source == "llm"
    assert llm.calls
    assert "heuristic_hint" in llm.calls[0]["input_data"]


def test_user_intent_resolves_p1_style_reference_from_candidate_pool() -> None:
    skill = ResearchIntentResolver()

    result = skill.resolve(
        message="导入 p1 到 zotero",
        has_task=True,
        candidate_paper_count=2,
        candidate_papers=[
            {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
            {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
        ],
        active_paper_ids=[],
        selected_paper_ids=[],
        has_visual_anchor=False,
        has_document_input=False,
    )

    assert result.resolved_paper_ids == ["paper-1"]
    assert result.reference_type == "ordinal"


def test_user_intent_recognizes_zotero_sync_request_from_candidate_scope() -> None:
    skill = ResearchIntentResolver()

    result = skill.resolve(
        message="导入第一篇论文到 Zotero",
        has_task=True,
        candidate_paper_count=2,
        candidate_papers=[
            {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
            {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
        ],
        active_paper_ids=[],
        selected_paper_ids=[],
        has_visual_anchor=False,
        has_document_input=False,
    )

    assert result.intent == "sync_to_zotero"
    assert result.resolved_paper_ids == ["paper-1"]
    assert result.reference_type == "ordinal"


def test_user_intent_recognizes_workspace_import_request_for_grounded_follow_up() -> None:
    skill = ResearchIntentResolver()

    result = skill.resolve(
        message="把第一篇导入工作区供后续问答",
        has_task=True,
        candidate_paper_count=2,
        candidate_papers=[
            {"index": 1, "paper_id": "paper-1", "title": "VLN Paper One"},
            {"index": 2, "paper_id": "paper-2", "title": "VLN Paper Two"},
        ],
        active_paper_ids=[],
        selected_paper_ids=[],
        has_visual_anchor=False,
        has_document_input=False,
    )

    assert result.intent == "paper_import"
    assert result.resolved_paper_ids == ["paper-1"]
    assert result.reference_type == "ordinal"


def test_user_intent_keeps_resolved_paper_scope_for_figure_question() -> None:
    skill = ResearchIntentResolver()

    result = skill.resolve(
        message="第二篇论文的系统框图",
        has_task=True,
        candidate_paper_count=3,
        candidate_papers=[
            {"index": 1, "paper_id": "paper-1", "title": "Paper One"},
            {"index": 2, "paper_id": "paper-2", "title": "Paper Two"},
            {"index": 3, "paper_id": "paper-3", "title": "Paper Three"},
        ],
        active_paper_ids=[],
        selected_paper_ids=[],
        has_visual_anchor=False,
        has_document_input=False,
    )

    assert result.intent == "figure_qa"
    assert result.resolved_paper_ids == ["paper-2"]
    assert result.reference_type == "ordinal"


@pytest.mark.asyncio
async def test_qa_routing_prefers_structured_visual_anchor_over_marker_logic() -> None:
    skill = ResearchQARouter(llm_adapter=QARoutingLLMStub(route="collection_qa"))

    result = await skill.classify_async(
        question="请解释这张图表达了什么。",
        scope_mode="selected_documents",
        paper_ids=["paper-1"],
        document_ids=["doc-1"],
        has_visual_anchor=True,
    )

    assert result.route == "chart_drilldown"
    assert result.source == "heuristic"


class RetrievalAgentSuccessStub:
    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        hits = [
            RetrievalHit(
                id="hit_doc_1",
                source_type="text_block",
                source_id="doc_1:block_1",
                document_id="doc_1",
                content="UAV Survey systematically compares path planning families, benchmark settings, and representative experiments.",
                merged_score=0.88,
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
            metadata={},
        )


class RetrievalAgentInsufficientStub:
    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query=question, document_ids=document_ids, top_k=top_k),
            hits=[],
            evidence_bundle=EvidenceBundle(),
        )
        return RetrievalAgentResult(
            question=question,
            document_ids=document_ids,
            evidence_bundle=retrieval_result.evidence_bundle,
            retrieval_result=retrieval_result,
            metadata={},
        )


class RetrievalAgentScopedStub:
    def __init__(self) -> None:
        self.last_document_ids: list[str] = []
        self.last_filters: dict | None = None

    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        self.last_document_ids = list(document_ids)
        self.last_filters = dict(kwargs.get("filters") or {})
        hits = [
            RetrievalHit(
                id="hit_scope_doc_2",
                source_type="text_block",
                source_id="doc_2:block_1",
                document_id="doc_2",
                content="Paper B argues for document-scoped evidence control in agentic scientific QA.",
                merged_score=0.93,
                metadata={"page_number": 3, "section": "Method"},
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
            metadata={},
        )


class AnswerAgentSuccessStub:
    async def answer_with_evidence(self, *, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, **kwargs):
        return QAResponse(
            answer="当前最值得优先阅读的是 UAV Survey，因为它系统梳理了路径规划方法谱系与 benchmark 设定。",
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.82,
            metadata={"mode": "collection"},
        )


class AnswerAgentInsufficientStub:
    async def answer_with_evidence(self, *, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, **kwargs):
        return QAResponse(
            answer="证据不足，当前研究集合无法稳定确认这个问题。",
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.1,
            metadata={"mode": "collection"},
        )


class AnswerAgentScopedStub:
    async def answer_with_evidence(self, *, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, **kwargs):
        return QAResponse(
            answer="优先阅读 Paper B，因为它最直接回答了 document-scoped evidence control 这个问题。",
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.88,
            metadata={"mode": "collection"},
        )


class GraphRuntimeSuccessStub:
    retrieval_tools = RetrievalAgentSuccessStub()
    answer_tools = AnswerAgentSuccessStub()
    react_reasoning_agent = None

    def __init__(self) -> None:
        self.session_memory = GraphSessionMemory()

    async def query_graph_summary(self, **kwargs):
        return GraphSummaryToolOutput(
            hits=[
                RetrievalHit(
                    id="summary_doc_2",
                    source_type="graph_summary",
                    source_id="doc_2:summary",
                    document_id="doc_2",
                    content="Graph summary highlights that survey papers cover method taxonomy and comparative evaluation settings.",
                    merged_score=0.67,
                )
            ],
            metadata={"mode": "graph_summary"},
        )



def test_create_conversation_initializes_context_summary_and_runtime_event(tmp_path) -> None:
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=ResearchReportService(tmp_path / "research_storage"),
        paper_import_service=None,
    )

    response = service.create_conversation(
        CreateResearchConversationRequest(
            topic="GraphRAG",
            days_back=30,
            max_papers=8,
            sources=["arxiv", "openalex"],
        )
    )

    summary = response.conversation.snapshot.context_summary
    assert summary.topic == "GraphRAG"
    assert summary.current_stage == "discover"
    assert response.conversation.snapshot.recent_events
    assert response.conversation.snapshot.recent_events[0].event_type == "agent_started"
    assert response.conversation.status_metadata.lifecycle_status == "waiting_input"


def test_delete_conversation_clears_session_memory_and_artifacts(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=None,
    )
    response = service.create_conversation(
        CreateResearchConversationRequest(
            topic="GraphRAG",
            days_back=30,
            max_papers=8,
            sources=["arxiv"],
        )
    )
    conversation = response.conversation.model_copy(update={"task_id": "task-clear"})
    report_service.save_conversation(conversation)
    report_service.save_task(
        ResearchTask(
            task_id="task-clear",
            topic="GraphRAG",
            created_at="2026-04-25T00:00:00+00:00",
            updated_at="2026-04-25T00:00:00+00:00",
        )
    )
    report_service.save_papers(
        "task-clear",
        [
            PaperCandidate(
                paper_id="paper-1",
                title="GraphRAG Systems",
                abstract="A survey.",
                source="arxiv",
            )
        ],
    )
    report_service.save_report(
        ResearchReport(
            report_id="report-clear",
            task_id="task-clear",
            topic="GraphRAG",
            generated_at="2026-04-25T00:00:00+00:00",
            markdown="# report",
        )
    )
    report_service.save_job(
        ResearchJob(
            job_id="job-clear",
            kind="paper_import",
            status="completed",
            created_at="2026-04-25T00:00:00+00:00",
            updated_at="2026-04-25T00:00:00+00:00",
            task_id="task-clear",
            conversation_id=conversation.conversation_id,
        )
    )
    service.memory_manager.save_context(
        conversation.conversation_id,
        ResearchContext(research_topic="GraphRAG"),
    )

    service.delete_conversation(conversation.conversation_id)

    assert report_service.load_conversation(conversation.conversation_id) is None
    assert report_service.load_messages(conversation.conversation_id) == []
    assert report_service.load_task("task-clear") is None
    assert report_service.load_papers("task-clear") == []
    assert report_service.load_report("task-clear") is None
    assert report_service.list_jobs(conversation_id=conversation.conversation_id) == []
    assert service.memory_manager.session_memory.load(conversation.conversation_id).context.research_topic == ""


@pytest.mark.asyncio
async def test_create_and_run_task_updates_status_metadata(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    response = await service.create_task(
        CreateResearchTaskRequest.model_validate(
            {
                "topic": "GraphRAG",
                "days_back": 30,
                "max_papers": 5,
                "sources": ["arxiv"],
                "run_immediately": False,
            }
        )
    )

    assert response.task.status == "created"
    assert response.task.status_metadata.lifecycle_status == "queued"
    assert response.task.status_metadata.correlation_id is not None


@pytest.mark.asyncio
async def test_start_import_job_initializes_status_metadata(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )

    job = await service.start_import_job(
        ImportPapersRequest(task_id="task_1", paper_ids=["paper_1"]),
        graph_runtime=GraphRuntimeSuccessStub(),
    )
    created_job = service.get_job(job.job_id)

    assert created_job.status == "queued"
    assert created_job.status_metadata.lifecycle_status == "queued"
    assert created_job.status_metadata.correlation_id is not None


def test_record_qa_turn_appends_runtime_event(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(CreateResearchConversationRequest(topic="GraphRAG"))
    task = ResearchTask(
        task_id="task_qa_evt_1",
        topic="GraphRAG",
        status="completed",
        created_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        report_id="report_1",
        status_metadata={"lifecycle_status": "completed", "correlation_id": "corr_qa_evt"},
    )
    task_response = ResearchTaskResponse(task=task, papers=[], report=None, warnings=[])
    ask_response = ResearchTaskAskResponse(
        task_id=task.task_id,
        qa=QAResponse(
            question="有什么证据缺口？",
            answer="目前长文档 benchmark 证据不足。",
            confidence=0.76,
            evidence_bundle=EvidenceBundle(),
            metadata={"qa_route": "collection_qa"},
        ),
    )

    service.record_qa_turn(
        conversation.conversation.conversation_id,
        task_response=task_response,
        ask_response=ask_response,
    )
    persisted = service.get_conversation(conversation.conversation.conversation_id).conversation

    assert persisted.snapshot.recent_events[-1].event_type == "memory_updated"
    assert any(event.payload.get("tool_name") == "collection_qa" for event in persisted.snapshot.recent_events)


def test_record_import_turn_appends_tool_event(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(CreateResearchConversationRequest(topic="GraphRAG"))
    task = ResearchTask(
        task_id="task_import_evt_1",
        topic="GraphRAG",
        status="completed",
        created_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        report_id="report_1",
        status_metadata={"lifecycle_status": "completed", "correlation_id": "corr_import_evt"},
    )
    task_response = ResearchTaskResponse(task=task, papers=[], report=None, warnings=[])

    service.record_import_turn(
        conversation.conversation.conversation_id,
        task_response=task_response,
        import_response=ImportPapersResponse(imported_count=1, results=[]),
        selected_paper_ids=["paper-1"],
    )
    persisted = service.get_conversation(conversation.conversation.conversation_id).conversation

    assert any(event.payload.get("tool_name") == "paper_import" for event in persisted.snapshot.recent_events)
    assert persisted.snapshot.selected_paper_ids == ["paper-1"]
    assert persisted.snapshot.active_paper_ids == ["paper-1"]


def test_create_conversation_updates_user_profile_and_observability(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    service.create_conversation(
        CreateResearchConversationRequest(
            topic="GraphRAG",
            sources=["arxiv", "openalex"],
        )
    )

    profile = service.memory_manager.load_user_profile()
    metrics_path = report_service.storage_root / "observability" / "metrics.jsonl"

    assert profile.last_active_topic == "GraphRAG"
    assert profile.preferred_sources == []
    assert metrics_path.exists() is False


def test_create_unnamed_conversation_does_not_pollute_user_profile_topic(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    service.create_conversation(CreateResearchConversationRequest())

    profile = service.memory_manager.load_user_profile()

    assert profile.last_active_topic is None
    assert profile.interest_topics == []


def test_create_title_only_conversation_does_not_treat_title_as_profile_topic(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    service.create_conversation(
        CreateResearchConversationRequest(
            title="科研助手里的长期记忆方向，有没有比较新的论文？",
        )
    )

    profile = service.memory_manager.load_user_profile()

    assert profile.last_active_topic is None
    assert profile.interest_topics == []


def test_record_agent_turn_promotes_context_summary_version_when_history_is_long(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="replayable agents")
    )
    conversation_id = conversation.conversation.conversation_id
    history_messages = [
        ResearchMessage(
            message_id=f"msg_hist_{index}",
            role="assistant" if index % 2 == 0 else "user",
            kind="notice",
            title=f"历史消息 {index}",
            content=f"这是第 {index} 条较长历史消息，用于验证上下文压缩摘要版本升级。",
            created_at=f"2026-04-23T00:00:{index:02d}+00:00",
        )
        for index in range(1, 9)
    ]
    report_service.save_messages(conversation_id, history_messages)
    task = ResearchTask(
        task_id="task_ctx_summary_1",
        topic="replayable agents",
        status="completed",
        created_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        report_id="report_1",
        workspace=ResearchWorkspaceState(
            objective="replayable agents",
            metadata={"context_compression": {"paper_count": 2, "summary_count": 3}},
        ),
        status_metadata={"lifecycle_status": "completed", "correlation_id": "corr_ctx_summary"},
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        task=task,
        messages=[],
        papers=[],
        warnings=[],
        next_actions=[],
        workspace=task.workspace,
        metadata={},
    )

    service.record_agent_turn(
        conversation_id=conversation_id,
        request=ResearchAgentRunRequest(message="继续总结", conversation_id=conversation_id),
        response=response,
    )
    persisted = service.get_conversation(conversation_id).conversation

    assert persisted.snapshot.context_summary.summary_version == 2
    assert "已启用上下文压缩" in persisted.snapshot.context_summary.status_summary


def test_build_execution_context_sets_answer_language_from_user_message(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="GraphRAG")
    )
    service.memory_manager.update_user_profile(answer_language="zh-CN")
    context = service.build_execution_context(
        graph_runtime=object(),
        conversation_id=conversation.conversation.conversation_id,
        skill_name="research_report",
        reasoning_style="cot",
        metadata={"user_message": "What are the latest UAV papers?", "context": {}},
    )

    assert context.preference_context["answer_language"] == "en-US"
    assert context.preference_context["follow_user_language"] is True


@pytest.mark.asyncio
async def test_start_import_job_records_tool_called_event(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="GraphRAG")
    )
    task = ResearchTask(
        task_id="task_evt_called_1",
        topic="GraphRAG",
        status="completed",
        created_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        sources=["arxiv"],
    )
    report_service.save_task(task)
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper_1",
                title="GraphRAG Survey",
                authors=["Alice"],
                abstract="A survey.",
                source="arxiv",
                pdf_url="https://arxiv.org/pdf/1.pdf",
            )
        ],
    )
    job = await service.start_import_job(
        ImportPapersRequest(
            task_id=task.task_id,
            conversation_id=conversation.conversation.conversation_id,
            paper_ids=["paper_1"],
        ),
        graph_runtime=GraphRuntimeImportAndQaStub(),
    )
    await service._job_tasks[job.job_id]
    persisted = service.get_conversation(conversation.conversation.conversation_id).conversation

    assert any(
        event.event_type == "tool_called" and event.payload.get("tool_name") == "paper_import_job"
        for event in persisted.snapshot.recent_events
    )


class RecordingRetrievalAgentStub(RetrievalAgentSuccessStub):
    def __init__(self) -> None:
        self.last_question = ""

    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        self.last_question = question
        return await super().retrieve(question=question, document_ids=document_ids, top_k=top_k, **kwargs)


class RecordingAnswerAgentStub(AnswerAgentSuccessStub):
    def __init__(self) -> None:
        self.last_question = ""

    async def answer_with_evidence(self, *, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, **kwargs):
        self.last_question = question
        return await super().answer_with_evidence(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            **kwargs,
        )


class GraphRuntimeVagueQuestionStub(GraphRuntimeSuccessStub):
    def __init__(self) -> None:
        super().__init__()
        self.retrieval_tools = RecordingRetrievalAgentStub()
        self.answer_tools = RecordingAnswerAgentStub()


class GraphRuntimeInsufficientStub:
    retrieval_tools = RetrievalAgentInsufficientStub()
    answer_tools = AnswerAgentInsufficientStub()
    react_reasoning_agent = None

    async def query_graph_summary(self, **kwargs):
        return GraphSummaryToolOutput(hits=[], metadata={"mode": "graph_summary"})



class GraphRuntimeScopedSelectionStub:
    react_reasoning_agent = None

    def __init__(self) -> None:
        self.session_memory = GraphSessionMemory()
        self.retrieval_tools = RetrievalAgentScopedStub()
        self.answer_tools = AnswerAgentScopedStub()

    async def query_graph_summary(self, **kwargs):
        return GraphSummaryToolOutput(hits=[], metadata={"mode": "graph_summary"})



class GraphRuntimeDocumentDrilldownStub:
    react_reasoning_agent = None

    def __init__(self) -> None:
        self.session_memory = GraphSessionMemory()
        self.last_handle_ask_document_kwargs: dict | None = None
        self.last_handle_ask_fused_kwargs: dict | None = None

    async def handle_ask_document(self, **kwargs):
        self.last_handle_ask_document_kwargs = dict(kwargs)
        hit = RetrievalHit(
            id="doc_drilldown_hit_1",
            source_type="text_block",
            source_id="doc_2:block_3",
            document_id="doc_2",
            content="Paper B explicitly defines document-scoped evidence control in Section 3.",
            merged_score=0.95,
        )
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query=kwargs["question"], document_ids=kwargs.get("document_ids") or [], top_k=kwargs.get("top_k", 10)),
            hits=[hit],
            evidence_bundle=build_evidence_bundle([hit]),
        )
        return QAResponse(
            answer="Paper B 在正文第 3 节明确提出了 document-scoped evidence control。",
            question=kwargs["question"],
            evidence_bundle=retrieval_result.evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.91,
            metadata={"mode": "document"},
        )

    async def handle_ask_fused(self, **kwargs):
        self.last_handle_ask_fused_kwargs = dict(kwargs)
        hit = RetrievalHit(
            id="chart_drilldown_hit_1",
            source_type="chart",
            source_id=kwargs.get("chart_id") or "chart_auto",
            document_id=(kwargs.get("document_ids") or [None])[0],
            content="The chart shows x-axis as recall@k and y-axis as answer accuracy.",
            merged_score=0.97,
        )
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query=kwargs["question"], document_ids=kwargs.get("document_ids") or [], top_k=kwargs.get("top_k", 10)),
            hits=[hit],
            evidence_bundle=build_evidence_bundle([hit]),
        )
        class _FusedResult:
            def __init__(self, qa):
                self.qa = qa
        return _FusedResult(
            QAResponse(
                answer="图中 x-axis 是 recall@k，y-axis 是 answer accuracy。",
                question=kwargs["question"],
                evidence_bundle=retrieval_result.evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=0.93,
                metadata={"mode": "fused"},
            )
        )


class ResearchQARuntimeLowConfidenceStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(
        self,
        *,
        graph_runtime,
        task,
        request,
        report,
        papers,
        document_ids,
        execution_context=None,
    ):
        del graph_runtime, task, report, papers, document_ids, execution_context
        self.calls.append({"question": request.question, "metadata": dict(request.metadata or {})})

        class _Result:
            def __init__(self):
                self.qa = QAResponse(
                    answer="我暂时无法确认这篇论文正文里的精确定义，当前证据不足。",
                    question=request.question,
                    evidence_bundle=EvidenceBundle(),
                    confidence=0.22,
                    metadata={"mode": "collection"},
                )

        return _Result()



class GraphRuntimeImportStub:
    def __init__(self) -> None:
        self.session_memory = GraphSessionMemory()
        self.last_parse_kwargs: dict | None = None
        self.last_index_kwargs: dict | None = None
        self.last_backfill_kwargs: dict | None = None

    async def handle_parse_document(self, **kwargs):
        self.last_parse_kwargs = dict(kwargs)
        from domain.schemas.document import ParsedDocument

        return ParsedDocument(
            id=kwargs.get("document_id") or "paper_doc_1",
            filename="paper.pdf",
            content_type="application/pdf",
            status="parsed",
            pages=[],
            metadata=kwargs.get("metadata") or {},
        )

    async def handle_index_document(self, **kwargs):
        self.last_index_kwargs = dict(kwargs)
        return type("IndexResult", (), {"status": "succeeded"})()

    async def handle_graph_backfill_document(self, **kwargs):
        self.last_backfill_kwargs = dict(kwargs)
        return type("IndexResult", (), {"status": "succeeded"})()


class GraphRuntimeImportAndQaStub(GraphRuntimeSuccessStub):
    def __init__(self) -> None:
        super().__init__()
        self.last_parse_kwargs: dict | None = None
        self.last_index_kwargs: dict | None = None
        self.last_backfill_kwargs: dict | None = None

    async def handle_parse_document(self, **kwargs):
        self.last_parse_kwargs = dict(kwargs)
        from domain.schemas.document import ParsedDocument

        return ParsedDocument(
            id=kwargs.get("document_id") or "paper_doc_1",
            filename="paper.pdf",
            content_type="application/pdf",
            status="parsed",
            pages=[],
            metadata=kwargs.get("metadata") or {},
        )

    async def handle_index_document(self, **kwargs):
        self.last_index_kwargs = dict(kwargs)
        return type("IndexResult", (), {"status": "succeeded"})()

    async def handle_graph_backfill_document(self, **kwargs):
        self.last_backfill_kwargs = dict(kwargs)
        return type("IndexResult", (), {"status": "succeeded"})()


class FigureDocumentToolsStub:
    def __init__(self, candidates_by_page: dict[str, list] | None = None) -> None:
        self.candidates_by_page = candidates_by_page or {}
        self.locate_calls = 0

    async def locate_chart_candidates(self, page: DocumentPage):
        from tools.document_toolkit import normalize_chart_candidate

        self.locate_calls += 1
        return [
            item
            if hasattr(item, "page_id")
            else normalize_chart_candidate(page=page, raw_candidate=item, index=index)
            for index, item in enumerate(self.candidates_by_page.get(page.id, []))
        ]


class GraphRuntimePaperFigureStub:
    react_reasoning_agent = None

    def __init__(self, parsed_document: ParsedDocument, *, candidates_by_page: dict[str, list] | None = None) -> None:
        self.session_memory = GraphSessionMemory()
        self.parsed_document = parsed_document
        self.document_tools = FigureDocumentToolsStub(candidates_by_page)
        self.parse_calls = 0
        self.understand_calls = 0

    async def handle_parse_document(self, **kwargs):
        self.parse_calls += 1
        return self.parsed_document

    async def handle_understand_chart(self, **kwargs):
        self.understand_calls += 1
        chart = ChartSchema(
            id=str(kwargs.get("chart_id") or "chart_1"),
            document_id=str(kwargs.get("document_id") or "doc_1"),
            page_id=str(kwargs.get("page_id") or "page_1"),
            page_number=int(kwargs.get("page_number") or 1),
            title="Figure 1",
            chart_type="line",
            x_axis={"label": "Epoch"},
            y_axis={"label": "Accuracy"},
            series=[{"name": "Model A", "values": [{"x": "1", "y": "0.7"}]}],
            metadata={"source": "paper_figure_test"},
        )
        return type(
            "UnderstandChartResult",
            (),
            {
                "chart": chart,
                "graph_text": "Accuracy rises over epochs.",
                "metadata": {"provider": "stub"},
            },
        )()


class GraphRuntimeChartDiscoveryStub(GraphRuntimeDocumentDrilldownStub):
    react_reasoning_agent = None

    def __init__(self, parsed_document: ParsedDocument, *, candidates_by_page: dict[str, list] | None = None) -> None:
        super().__init__()
        self.parsed_document = parsed_document
        self.document_tools = FigureDocumentToolsStub(candidates_by_page)
        self.parse_calls = 0

    async def handle_parse_document(self, **kwargs):
        self.parse_calls += 1
        return self.parsed_document


class PaperImportServiceStub:
    async def download_paper(self, paper):
        return type(
            "Artifact",
            (),
            {
                "paper": paper,
                "document_id": "paper_doc_1",
                "storage_uri": "/tmp/paper.pdf",
                "filename": "paper.pdf",
            },
        )()


class ConcurrentPaperImportServiceStub(PaperImportServiceStub):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def download_paper(self, paper):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return await super().download_paper(paper)
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_ask_task_collection_updates_report_and_todo(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_qa_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=2,
        imported_document_ids=["doc_1", "doc_2"],
    )
    report_service.save_task(task)
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper_1",
                title="UAV Survey A",
                authors=["Alice"],
                abstract="Survey A.",
                source="arxiv",
            ),
            PaperCandidate(
                paper_id="paper_2",
                title="UAV Survey B",
                authors=["Bob"],
                abstract="Survey B.",
                source="arxiv",
            ),
        ],
    )
    report_service.save_report(
        ResearchReport(
            report_id="report_1",
            task_id="task_qa_1",
            topic="无人机路径规划",
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机路径规划",
            paper_count=2,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    response = await service.ask_task_collection(
        "task_qa_1",
        ResearchTaskAskRequest(question="当前最值得优先阅读的论文是哪篇？", top_k=10),
        graph_runtime=GraphRuntimeSuccessStub(),
    )

    assert response.report is not None
    assert "研究集合问答补充" in response.report.markdown
    assert "当前最值得优先阅读的论文是哪篇？" in response.report.markdown
    assert response.qa.answer == "当前最值得优先阅读的是 UAV Survey，因为它系统梳理了路径规划方法谱系与 benchmark 设定。"
    assert "## 直接回答" not in response.qa.answer
    assert "证据依据" not in response.report.markdown
    assert "自动待办" not in response.report.markdown
    assert response.qa.metadata["autonomy_mode"] == "lead_agent_loop"
    assert response.todo_items
    assert response.todo_items[0].source == "qa_follow_up"
    assert response.todo_items[0].priority == "medium"
    assert response.qa.metadata["agent_architecture"] == "main_agents_only"
    assert "ResearchKnowledgeAgent" in response.qa.metadata["primary_agents"]
    assert "ResearchWriterAgent" in response.qa.metadata["primary_agents"]
    assert response.report.workspace.current_stage == "qa"

    persisted_task = report_service.load_task("task_qa_1")
    assert persisted_task is not None
    assert len(persisted_task.todo_items) == 1
    assert persisted_task.todo_items[0].question == "当前最值得优先阅读的论文是哪篇？"
    assert persisted_task.workspace.current_stage == "qa"
    assert "papers=2" in persisted_task.workspace.status_summary

    persisted_report = report_service.load_report("task_qa_1", "report_1")
    assert persisted_report is not None
    assert persisted_report.highlights
    assert "问答补充" in persisted_report.highlights[0]
    assert persisted_report.workspace.current_stage == "qa"


@pytest.mark.asyncio
async def test_ask_task_collection_rewrites_vague_effect_question_to_collection_scope(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_qa_effect",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=2,
        imported_document_ids=["doc_1", "doc_2"],
        report_id="report_effect",
    )
    report_service.save_task(task)
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper_1",
                title="UAV Survey A",
                abstract="Survey A compares planning methods and experiments.",
                source="arxiv",
            ),
            PaperCandidate(
                paper_id="paper_2",
                title="UAV Survey B",
                abstract="Survey B evaluates planning performance and limitations.",
                source="arxiv",
            ),
        ],
    )
    report_service.save_report(
        ResearchReport(
            report_id="report_effect",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机路径规划",
            paper_count=2,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeVagueQuestionStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(question="效果怎么样", top_k=10),
        graph_runtime=graph_runtime,
    )

    assert response.qa.question == "效果怎么样"
    assert "综合评价" in graph_runtime.answer_tools.last_question
    assert "不要只回答单篇论文" in graph_runtime.answer_tools.last_question
    assert "无人机路径规划" in graph_runtime.retrieval_tools.last_question
    assert response.qa.metadata["original_question"] == "效果怎么样"
    assert response.qa.metadata["resolved_question"] == graph_runtime.answer_tools.last_question


@pytest.mark.asyncio
async def test_ask_task_collection_adds_gap_todo_when_evidence_is_insufficient(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_qa_2",
        topic="无人机感知",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_1"],
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_2",
            task_id="task_qa_2",
            topic="无人机感知",
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机感知",
            paper_count=1,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    response = await service.ask_task_collection(
        "task_qa_2",
        ResearchTaskAskRequest(question="这些论文是否已经证明了低空复杂天气下的鲁棒性？", top_k=10),
        graph_runtime=GraphRuntimeInsufficientStub(),
    )

    assert response.report is not None
    assert response.qa.metadata["autonomy_mode"] == "lead_agent_loop"
    assert response.qa.metadata["agent_architecture"] == "main_agents_only"
    assert response.todo_items
    assert response.todo_items[0].source == "evidence_gap"
    assert response.todo_items[0].priority == "high"
    assert "补充与" in response.todo_items[0].content
    assert response.report is not None
    assert response.report.workspace.current_stage == "qa"
    assert response.report.workspace.stop_reason is not None

    persisted_report = report_service.load_report("task_qa_2", "report_2")
    assert persisted_report is not None
    assert persisted_report.gaps
    assert "证据仍不足" in persisted_report.gaps[0]
    assert persisted_report.workspace.current_stage == "qa"


@pytest.mark.asyncio
async def test_ask_task_collection_uses_memory_context_and_persists_research_history(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_mem_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_1"],
        report_id="report_mem_1",
    )
    report = ResearchReport(
        report_id="report_mem_1",
        task_id="task_mem_1",
        topic="无人机路径规划",
        generated_at="2026-04-17T00:00:00+00:00",
        markdown="# 文献调研报告：无人机路径规划",
        paper_count=1,
    )
    paper = PaperCandidate(
        paper_id="arxiv:mem1",
        title="UAV Survey",
        authors=["Alice"],
        abstract="A survey.",
        source="arxiv",
    )
    report_service.save_task(task)
    report_service.save_report(report)
    report_service.save_papers(task.task_id, [paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(CreateResearchConversationRequest(topic=task.topic))
    conversation_id = conversation.conversation.conversation_id
    service.record_task_turn(
        conversation_id,
        response=ResearchTaskResponse(task=task, papers=[paper], report=report, warnings=[]),
    )
    graph_runtime = GraphRuntimeSuccessStub()
    graph_runtime.session_memory.update_research_context(
        session_id=conversation_id,
        current_task_intent="research_seed",
    )

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="当前最值得优先阅读的论文是哪篇？",
            top_k=10,
            conversation_id=conversation_id,
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["memory_enabled"] is True
    assert response.qa.metadata["session_id"] == conversation_id
    history = graph_runtime.session_memory.research_history(conversation_id)
    assert history
    assert history[-1]["question"] == "当前最值得优先阅读的论文是哪篇？"
    assert "UAV Survey" in history[-1]["answer"]
    layered_record = service.memory_manager.session_memory.load(conversation_id)
    assert layered_record.context.research_topic == "无人机路径规划"
    assert layered_record.context.session_history[-1].question == "当前最值得优先阅读的论文是哪篇？"


def test_record_agent_turn_persists_messages_and_snapshot(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="无人机路径规划")
    )
    conversation_id = conversation.conversation.conversation_id
    workspace = build_workspace_state(
        objective="无人机路径规划",
        stage="complete",
        stop_reason="Comparison completed.",
        metadata={
            "advanced_strategy": {
                "action": "compare",
                "comparison_dimensions": ["method", "experiment"],
                "recommendation_goal": None,
                "recommendation_top_k": 3,
                "force_context_compression": True,
            }
        },
    )
    task = ResearchTask(
        task_id="task_agent_turn_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        sources=["arxiv", "openalex"],
        workspace=workspace,
    )
    report = ResearchReport(
        report_id="report_agent_turn_1",
        task_id=task.task_id,
        topic=task.topic,
        generated_at="2026-04-20T00:00:00+00:00",
        markdown="# report",
        paper_count=0,
        workspace=workspace,
    )
    paper = PaperCandidate(
        paper_id="paper-replay-1",
        title="Replayable Research Agent Messages",
        authors=["Alice"],
        abstract="A paper used to verify replayable conversation payloads.",
        source="arxiv",
        year=2026,
        pdf_url="https://arxiv.org/pdf/replay.pdf",
        url="https://arxiv.org/abs/replay",
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        task=task,
        papers=[paper],
        report=report,
        messages=[
            ResearchMessage(
                message_id="msg_user_1",
                role="user",
                kind="topic",
                title="用户研究目标",
                content="请对比这些论文的方法和实验。",
                created_at="2026-04-20T00:00:00+00:00",
            ),
            ResearchMessage(
                message_id="msg_report_1",
                role="assistant",
                kind="report",
                title="文献综述结果",
                content=report.markdown,
                meta="候选论文 1 篇",
                created_at="2026-04-20T00:00:01+00:00",
                payload={"report": report.model_dump(mode="json")},
            ),
            ResearchMessage(
                message_id="msg_candidates_1",
                role="assistant",
                kind="candidates",
                title="候选论文池",
                meta="当前共 1 篇，可勾选后导入",
                created_at="2026-04-20T00:00:02+00:00",
                payload={"papers": [paper.model_dump(mode="json")]},
            ),
            ResearchMessage(
                message_id="msg_comparison_1",
                role="assistant",
                kind="notice",
                title="多论文对比结果",
                content="已生成结构化多论文对比结果。",
                created_at="2026-04-20T00:00:03+00:00",
                payload={
                    "comparison": {
                        "summary": "Replay comparison summary.",
                        "table": [
                            {
                                "dimension": "method",
                                "values": {
                                    paper.paper_id: "Structured replay pipeline."
                                },
                            }
                        ],
                    }
                },
            ),
            ResearchMessage(
                message_id="msg_recommendation_1",
                role="assistant",
                kind="notice",
                title="长期兴趣论文推荐",
                content="- Replayable Research Agent Messages：最适合验证消息恢复。",
                meta="recommended=1",
                created_at="2026-04-20T00:00:04+00:00",
                payload={
                    "recommendations": {
                        "recommendations": [
                            {
                                "paper_id": paper.paper_id,
                                "title": paper.title,
                                "reason": "最适合验证消息恢复。",
                                "source": paper.source,
                                "year": paper.year,
                                "url": paper.url,
                            }
                        ]
                    }
                },
            ),
            ResearchMessage(
                message_id="msg_compression_1",
                role="assistant",
                kind="notice",
                title="上下文压缩摘要",
                content="当前研究上下文已经压缩为更短的论文摘要视图。",
                meta="papers=1 · summaries=2",
                created_at="2026-04-20T00:00:05+00:00",
                payload={
                    "context_compression": {
                        "paper_count": 1,
                        "summary_count": 2,
                        "levels": ["paper", "cluster"],
                        "compressed_paper_ids": [paper.paper_id],
                    }
                },
            ),
            ResearchMessage(
                message_id="msg_trace_1",
                role="assistant",
                kind="notice",
                title="Agent 决策轨迹",
                content="1. ComparisonAgent · act:compare_papers · succeeded",
                meta="1 step(s)",
                created_at="2026-04-20T00:00:06+00:00",
                payload={
                    "trace": [
                        {
                            "step_index": 1,
                            "agent": "ComparisonAgent",
                            "thought": "Need replayable comparison.",
                            "action_name": "compare_papers",
                            "phase": "act",
                            "action_input": {"dimensions": ["method"]},
                            "status": "succeeded",
                            "observation": "comparison complete",
                            "rationale": "verify persisted payloads",
                            "stop_signal": False,
                            "workspace_summary": "complete",
                            "metadata": {},
                        }
                    ]
                },
            ),
        ],
        trace=[],
        warnings=[],
        next_actions=[],
        workspace=workspace,
        metadata={
            "advanced_strategy": {
                "action": "compare",
                "comparison_dimensions": ["method", "experiment"],
                "recommendation_goal": None,
                "recommendation_top_k": 3,
                "force_context_compression": True,
            }
        },
    )

    result = service.record_agent_turn(
        conversation_id,
        request=ResearchAgentRunRequest(
            message="请对比这些论文的方法和实验。",
            mode="research",
            conversation_id=conversation_id,
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            advanced_action="compare",
            comparison_dimensions=["method", "experiment"],
            force_context_compression=True,
        ),
        response=response,
    )

    assert result.conversation.snapshot.advanced_strategy.action == "compare"
    assert result.conversation.snapshot.advanced_strategy.comparison_dimensions == [
        "method",
        "experiment",
    ]
    assert result.conversation.snapshot.task_result is not None
    assert result.conversation.snapshot.task_result.task.task_id == task.task_id
    assert len(result.messages) == 7
    assert result.messages[-1].title == "上下文压缩摘要"

    saved = service.get_conversation(conversation_id)
    messages_by_title = {message.title: message for message in saved.messages}
    assert saved.conversation.message_count == len(saved.messages)
    assert saved.conversation.last_message_preview == "当前研究上下文已经压缩为更短的论文摘要视图。"
    assert messages_by_title["候选论文池"].payload["papers"][0]["paper_id"] == paper.paper_id
    assert (
        messages_by_title["多论文对比结果"]
        .payload["comparison"]["table"][0]["values"][paper.paper_id]
        == "Structured replay pipeline."
    )
    assert (
        messages_by_title["长期兴趣论文推荐"]
        .payload["recommendations"]["recommendations"][0]["paper_id"]
        == paper.paper_id
    )
    assert messages_by_title["上下文压缩摘要"].payload["context_compression"]["levels"] == [
        "paper",
        "cluster",
    ]
    assert "Agent 决策轨迹" not in messages_by_title


def test_record_agent_turn_preserves_scoped_qa_snapshot(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="agentic scientific QA")
    )
    conversation_id = conversation.conversation.conversation_id
    workspace = build_workspace_state(
        objective="agentic scientific QA",
        stage="qa",
        stop_reason="Scoped QA completed.",
    )
    task = ResearchTask(
        task_id="task_agent_qa_scope_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=2,
        imported_document_ids=["doc_1", "doc_2"],
        report_id="report_agent_qa_scope_1",
        workspace=workspace,
    )
    report = ResearchReport(
        report_id="report_agent_qa_scope_1",
        task_id=task.task_id,
        topic=task.topic,
        generated_at="2026-04-20T00:00:00+00:00",
        markdown="# report",
        paper_count=2,
        workspace=workspace,
    )
    papers = [
        PaperCandidate(
            paper_id="paper-a",
            title="Paper A",
            abstract="A",
            source="arxiv",
            ingest_status="ingested",
            metadata={"document_id": "doc_1"},
        ),
        PaperCandidate(
            paper_id="paper-b",
            title="Paper B",
            abstract="B",
            source="arxiv",
            ingest_status="ingested",
            metadata={"document_id": "doc_2"},
        ),
    ]

    result = service.record_agent_turn(
        conversation_id,
        request=ResearchAgentRunRequest(
            message="请解释这篇论文的方法细节。",
            mode="qa",
            task_id=task.task_id,
            conversation_id=conversation_id,
            sources=["arxiv"],
            selected_paper_ids=["paper-b"],
            selected_document_ids=["doc_2"],
        ),
        response=ResearchAgentRunResponse(
            status="succeeded",
            task=task,
            papers=papers,
            report=report,
            qa=QAResponse(
                answer="Paper B 的方法细节主要体现在分阶段证据路由。",
                question="请解释这篇论文的方法细节。",
                evidence_bundle=EvidenceBundle(),
                confidence=0.84,
                metadata={
                    "selected_paper_ids": ["paper-b"],
                    "selected_document_ids": ["doc_2"],
                    "qa_scope_mode": "selected_documents",
                    "selection_warnings": ["仅对选中文档执行正文问答。"],
                },
            ),
            messages=[],
            trace=[],
            warnings=[],
            next_actions=[],
            workspace=workspace,
        ),
    )

    assert result.conversation.snapshot.ask_result is not None
    assert result.conversation.snapshot.ask_result.paper_ids == ["paper-b"]
    assert result.conversation.snapshot.ask_result.document_ids == ["doc_2"]
    assert result.conversation.snapshot.ask_result.scope_mode == "selected_documents"
    assert result.conversation.snapshot.ask_result.warnings == [
        "仅对选中文档执行正文问答。"
    ]


@pytest.mark.asyncio
async def test_ask_task_collection_respects_selected_paper_scope(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_scope_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv", "openalex"],
        paper_count=2,
        imported_document_ids=["doc_1", "doc_2"],
        report_id="report_scope_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_scope_1",
            task_id="task_scope_1",
            topic="agentic scientific QA",
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=2,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-a",
                title="Paper A",
                abstract="A",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_1"},
            ),
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="openalex",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            ),
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeScopedSelectionStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="哪篇论文最适合回答 document-scoped evidence control？",
            top_k=10,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.paper_ids == ["paper-b"]
    assert response.document_ids == ["doc_2"]
    assert response.scope_mode == "selected_papers"
    assert graph_runtime.retrieval_tools.last_document_ids == ["doc_2"]
    assert graph_runtime.retrieval_tools.last_filters is not None
    assert graph_runtime.retrieval_tools.last_filters["qa_scope_mode"] == "selected_papers"
    assert graph_runtime.retrieval_tools.last_filters["selected_paper_ids"] == ["paper-b"]
    assert graph_runtime.retrieval_tools.last_filters["selected_document_ids"] == ["doc_2"]
    assert response.qa.answer == "优先阅读 Paper B，因为它最直接回答了 document-scoped evidence control 这个问题。"
    assert "## 论文级证据" not in response.qa.answer
    assert "## 综合判断" not in response.qa.answer
    assert "## 局限与注意事项" not in response.qa.answer
    assert response.qa.metadata["paper_scope"]["paper_titles"] == ["Paper B"]
    assert response.qa.metadata["selection_summary"] is not None
    assert response.qa.metadata["scope_statistics"]["paper_count"] == 1


@pytest.mark.asyncio
async def test_ask_task_collection_routes_single_selected_paper_to_document_drilldown(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_doc_route_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_doc_route_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_doc_route_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这篇论文正文里如何定义 document-scoped evidence control？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.answer == "Paper B 在正文第 3 节明确提出了 document-scoped evidence control。"
    assert response.qa.metadata["qa_route"] == "document_drilldown"
    assert response.qa.metadata["qa_route_confidence"] >= 0.8
    assert response.qa.metadata["autonomy_mode"] == "task_scoped_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["document_ids"] == ["doc_2"]
    assert graph_runtime.last_handle_ask_document_kwargs["task_intent"] == "research_document_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs["filters"]["qa_route"] == "document_drilldown"


@pytest.mark.asyncio
async def test_ask_task_collection_routes_chart_like_question_to_chart_drilldown(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里 x-axis 和 y-axis 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["task_intent"] == "research_chart_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs["filters"]["qa_route"] == "chart_drilldown"


@pytest.mark.asyncio
async def test_ask_task_collection_can_conservatively_recover_from_collection_qa_to_document_drilldown(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_route_recovery_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_route_recovery_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_route_recovery_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    llm_stub = QARoutingLLMStub(route="collection_qa", confidence=0.88, rationale="LLM kept this broad.")
    collection_runtime = ResearchQARuntimeLowConfidenceStub()
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceWithLLMStub(llm_adapter=llm_stub),
        report_service=report_service,
        paper_import_service=object(),
        research_qa_runtime=collection_runtime,
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这篇论文正文里如何定义 document-scoped evidence control？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.answer == "Paper B 在正文第 3 节明确提出了 document-scoped evidence control。"
    assert response.qa.metadata["qa_route"] == "document_drilldown"
    assert response.qa.metadata["qa_route_recovery_count"] == 1
    assert response.qa.metadata["qa_route_recovered_from"] == "collection_qa"
    assert response.qa.metadata["answer_quality_check"]["route"] == "document_drilldown"
    assert len(collection_runtime.calls) == 1
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["filters"]["qa_route"] == "document_drilldown"


@pytest.mark.asyncio
async def test_ask_task_collection_routes_system_architecture_question_to_chart_drilldown(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_arch_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_arch_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_arch_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这篇论文的系统结构图长什么样？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["task_intent"] == "research_chart_drilldown"
    assert graph_runtime.last_handle_ask_document_kwargs["filters"]["qa_route"] == "chart_drilldown"


@pytest.mark.asyncio
async def test_ask_task_collection_uses_qa_routing_skill_to_route_chart_question(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_llm_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_llm_1",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_llm_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    llm_stub = QARoutingLLMStub(route="chart_drilldown")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceWithLLMStub(llm_adapter=llm_stub),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="能解释一下这篇论文里的整体模块组织吗？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["qa_route_confidence"] == pytest.approx(0.91)
    assert "chart question" in response.qa.metadata["qa_route_rationale"]
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["filters"]["qa_route"] == "chart_drilldown"
    assert llm_stub.calls


@pytest.mark.asyncio
async def test_ask_task_collection_routes_explicit_visual_anchor_to_fused_chart_runtime(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_2",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_2",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_2",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里 x-axis 和 y-axis 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
            image_path="/tmp/chart.png",
            page_id="page-2",
            page_number=2,
            chart_id="chart-2",
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.answer == "图中 x-axis 是 recall@k，y-axis 是 answer accuracy。"
    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-2"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/chart.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-2"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_uses_metadata_visual_anchor_for_fused_chart_runtime(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_2b",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_2b",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_2b",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里 x-axis 和 y-axis 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
            metadata={
                "image_path": "/tmp/chart-from-metadata.png",
                "page_id": "page-3",
                "page_number": 3,
                "chart_id": "chart-3",
            },
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["image_path"] == "/tmp/chart-from-metadata.png"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-3"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/chart-from-metadata.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_id"] == "page-3"
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-3"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_restores_workspace_visual_anchor_for_fused_chart_runtime(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    workspace = ResearchWorkspaceState(
        metadata={
            "last_visual_anchor": {
                "image_path": "/tmp/workspace-chart.png",
                "page_id": "page-5",
                "page_number": 5,
                "chart_id": "chart-5",
                "anchor_source": "workspace_memory",
            }
        }
    )
    task = ResearchTask(
        task_id="task_chart_route_2c",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_2c",
        workspace=workspace,
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_2c",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
            workspace=workspace,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里 x-axis 和 y-axis 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["image_path"] == "/tmp/workspace-chart.png"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-5"
    assert response.qa.metadata["visual_anchor"]["anchor_source"] == "workspace_memory"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/workspace-chart.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_id"] == "page-5"
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-5"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_auto_uses_cached_paper_figure_for_chart_drilldown(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_3",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_3",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_3",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_2",
                    "storage_uri": "/tmp/paper-b.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_2",
                        "storage_uri": "/tmp/paper-b.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-b:chart-1",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-2",
                                "page_number": 2,
                                "chart_id": "chart-1",
                                "image_path": "/tmp/cached-chart.png",
                                "source": "chart_candidate",
                                "metadata": {},
                            }
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里 x-axis 和 y-axis 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.answer == "图中 x-axis 是 recall@k，y-axis 是 answer accuracy。"
    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-1"
    assert response.qa.metadata["visual_anchor"]["anchor_source"] == "paper_figure_cache"
    assert response.qa.metadata["visual_anchor_figure"]["figure_id"] == "paper-b:chart-1"
    assert response.qa.metadata["visual_anchor_figure"]["image_path"] == "/tmp/cached-chart.png"
    assert response.report is not None
    assert response.report.workspace.metadata["last_visual_anchor_figure_id"] == "paper-b:chart-1"
    assert response.report.workspace.metadata["last_visual_anchor_figure"]["chart_id"] == "chart-1"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/cached-chart.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_id"] == "page-2"
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-1"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_discovers_paper_figure_and_uses_fused_chart_runtime(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_3b",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_3b",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_3b",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    page_image_path = tmp_path / "page_2.png"
    page_image_path.write_bytes(b"fake-image")
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_2",
                    "storage_uri": "/tmp/paper-b.pdf",
                },
            )
        ],
    )
    parsed_document = ParsedDocument(
        id="doc_2",
        filename="paper-b.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[
            DocumentPage(
                id="page-2",
                document_id="doc_2",
                page_number=2,
                image_uri=str(page_image_path),
                text_blocks=[
                    TextBlock(
                        id="tb_2",
                        document_id="doc_2",
                        page_id="page-2",
                        page_number=2,
                        text="Figure 2 shows the overall system architecture.",
                    )
                ],
            )
        ],
        metadata={"source_path": ""},
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeChartDiscoveryStub(
        parsed_document,
        candidates_by_page={
            "page-2": [
                {
                    "id": "chart-arch-1",
                    "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.8, "unit": "relative"},
                    "image_uri": str(page_image_path),
                    "title": "System Architecture",
                    "caption": "Overall pipeline and module orchestration.",
                }
            ]
        },
    )

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这篇论文的系统结构图长什么样？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.answer == "图中 x-axis 是 recall@k，y-axis 是 answer accuracy。"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-arch-1"
    assert graph_runtime.parse_calls == 1
    assert graph_runtime.document_tools.locate_calls == 1
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-arch-1"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_prefers_cached_figure_matching_question_terms(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_4",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_4",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_4",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_2",
                    "storage_uri": "/tmp/paper-b.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_2",
                        "storage_uri": "/tmp/paper-b.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-b:chart-1",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-1",
                                "page_number": 1,
                                "chart_id": "chart-1",
                                "image_path": "/tmp/overview-chart.png",
                                "title": "System Overview",
                                "caption": "Overall architecture pipeline",
                                "source": "chart_candidate",
                                "metadata": {},
                            },
                            {
                                "figure_id": "paper-b:chart-2",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-4",
                                "page_number": 4,
                                "chart_id": "chart-2",
                                "image_path": "/tmp/recall-chart.png",
                                "title": "Recall Curve",
                                "caption": "Recall@k versus answer accuracy",
                                "source": "chart_candidate",
                                "metadata": {},
                            },
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图里的 recall 和 answer accuracy 分别表示什么？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-2"
    assert response.report is not None
    assert response.report.workspace.metadata["last_visual_anchor_figure_id"] == "paper-b:chart-2"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/recall-chart.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_id"] == "page-4"


@pytest.mark.asyncio
async def test_ask_task_collection_selects_cached_figure_across_selected_papers(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_4b",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=2,
        imported_document_ids=["doc_2", "doc_3"],
        report_id="report_chart_route_4b",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_4b",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=2,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="System Paper",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_2",
                    "storage_uri": "/tmp/paper-b.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_2",
                        "storage_uri": "/tmp/paper-b.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-b:chart-1",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-1",
                                "page_number": 1,
                                "chart_id": "chart-1",
                                "image_path": "/tmp/system-chart.png",
                                "title": "System Overview",
                                "caption": "Overall architecture pipeline",
                                "source": "chart_candidate",
                                "metadata": {},
                            }
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            ),
            PaperCandidate(
                paper_id="paper-c",
                title="Navigation Results Paper",
                abstract="C",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_3",
                    "storage_uri": "/tmp/paper-c.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_3",
                        "storage_uri": "/tmp/paper-c.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-c:chart-9",
                                "paper_id": "paper-c",
                                "document_id": "doc_3",
                                "page_id": "page-6",
                                "page_number": 6,
                                "chart_id": "chart-9",
                                "image_path": "/tmp/navigation-results.png",
                                "title": "Navigation Experiment Results",
                                "caption": "Success rate and path efficiency across navigation samples.",
                                "source": "chart_candidate",
                                "metadata": {},
                            }
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            ),
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="选中的论文里，导航实验结果图里 success rate 和 path efficiency 展示了什么？",
            top_k=8,
            paper_ids=["paper-b", "paper-c"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-9"
    assert response.qa.metadata["visual_anchor"]["paper_id"] == "paper-c"
    assert response.qa.metadata["visual_anchor_figure"]["figure_id"] == "paper-c:chart-9"
    assert response.qa.metadata["visual_anchor_figure"]["paper_id"] == "paper-c"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-9"
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/navigation-results.png"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_avoids_first_page_fallback_when_real_result_chart_exists(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_4c",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_4"],
        report_id="report_chart_route_4c",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_4c",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-d",
                title="Navigation Benchmark Paper",
                abstract="D",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_4",
                    "storage_uri": "/tmp/paper-d.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_4",
                        "storage_uri": "/tmp/paper-d.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-d:page-1-fallback",
                                "paper_id": "paper-d",
                                "document_id": "doc_4",
                                "page_id": "page-1",
                                "page_number": 1,
                                "chart_id": "page-1-fallback",
                                "image_path": "/tmp/page-1.png",
                                "title": None,
                                "caption": None,
                                "source": "page_fallback",
                                "metadata": {},
                            },
                            {
                                "figure_id": "paper-d:chart-6",
                                "paper_id": "paper-d",
                                "document_id": "doc_4",
                                "page_id": "page-6",
                                "page_number": 6,
                                "chart_id": "chart-6",
                                "image_path": "/tmp/nav-results.png",
                                "title": "Navigation Results",
                                "caption": "Success rate and path efficiency across navigation benchmarks.",
                                "source": "chart_candidate",
                                "metadata": {},
                            },
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这篇论文的导航实验结果图里 success rate 和 path efficiency 怎么样？",
            top_k=8,
            paper_ids=["paper-d"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.metadata["qa_route"] == "chart_drilldown"
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-6"
    assert response.qa.metadata["visual_anchor_figure"]["figure_id"] == "paper-d:chart-6"
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/nav-results.png"
    assert graph_runtime.last_handle_ask_document_kwargs is None


@pytest.mark.asyncio
async def test_ask_task_collection_uses_llm_to_rerank_cached_figures(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_chart_route_5",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_chart_route_5",
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_chart_route_5",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={
                    "document_id": "doc_2",
                    "storage_uri": "/tmp/paper-b.pdf",
                    "paper_figure_cache": {
                        "document_id": "doc_2",
                        "storage_uri": "/tmp/paper-b.pdf",
                        "figures": [
                            {
                                "figure_id": "paper-b:chart-1",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-1",
                                "page_number": 1,
                                "chart_id": "chart-1",
                                "image_path": "/tmp/overview-chart.png",
                                "title": "Overview Figure",
                                "caption": "System overview",
                                "source": "chart_candidate",
                                "metadata": {},
                            },
                            {
                                "figure_id": "paper-b:chart-2",
                                "paper_id": "paper-b",
                                "document_id": "doc_2",
                                "page_id": "page-2",
                                "page_number": 2,
                                "chart_id": "chart-2",
                                "image_path": "/tmp/metrics-chart.png",
                                "title": "Metrics Figure",
                                "caption": "Evaluation metrics",
                                "source": "chart_candidate",
                                "metadata": {},
                            },
                        ],
                        "analyze_targets": {},
                        "warnings": [],
                    },
                },
            )
        ],
    )
    llm_stub = FigureSelectionLLMStub(selected_figure_id="paper-b:chart-2")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceWithLLMStub(llm_adapter=llm_stub),
        report_service=report_service,
        paper_import_service=object(),
    )
    graph_runtime = GraphRuntimeDocumentDrilldownStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="这张图主要想表达什么？",
            top_k=8,
            paper_ids=["paper-b"],
        ),
        graph_runtime=graph_runtime,
    )

    assert llm_stub.calls
    assert response.qa.metadata["visual_anchor"]["chart_id"] == "chart-2"
    assert response.qa.metadata["visual_anchor"]["anchor_selection"] == "llm_rerank"
    assert response.qa.metadata["visual_anchor"]["anchor_rationale"] == "LLM reranked the candidates."
    assert response.report is not None
    assert response.report.workspace.metadata["last_visual_anchor_figure"]["metadata"]["anchor_rationale"] == "LLM reranked the candidates."
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-2"


@pytest.mark.asyncio
async def test_import_papers_passes_session_id_and_updates_research_memory(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_import_mem_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
    )
    paper = PaperCandidate(
        paper_id="arxiv:1234.5678",
        title="UAV Path Planning with Multi-Agent Reinforcement Learning",
        authors=["Alice"],
        abstract="We study UAV path planning.",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )
    conversation = service.create_conversation(CreateResearchConversationRequest(topic=task.topic))
    conversation_id = conversation.conversation.conversation_id
    service.record_task_turn(
        conversation_id,
        response=ResearchTaskResponse(task=task, papers=[paper], report=None, warnings=[]),
    )
    graph_runtime = GraphRuntimeImportStub()

    response = await service.import_papers(
        ImportPapersRequest(
            task_id=task.task_id,
            paper_ids=[paper.paper_id],
            conversation_id=conversation_id,
        ),
        graph_runtime=graph_runtime,
    )

    assert response.imported_count == 1
    assert graph_runtime.last_parse_kwargs is not None
    assert graph_runtime.last_index_kwargs is not None
    assert graph_runtime.last_parse_kwargs["session_id"] == conversation_id
    assert graph_runtime.last_index_kwargs["session_id"] == conversation_id
    snapshot = graph_runtime.session_memory.load(conversation_id)
    assert snapshot is not None
    assert snapshot.metadata["imported_count"] == 1
    assert snapshot.metadata["task_id"] == task.task_id
    persisted_task = report_service.load_task(task.task_id)
    assert persisted_task is not None
    assert persisted_task.workspace.current_stage == "qa"


@pytest.mark.asyncio
async def test_import_papers_uses_small_parallelism_for_multiple_papers(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_import_parallel_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
    )
    papers = [
        PaperCandidate(
            paper_id=f"arxiv:{index}",
            title=f"paper-{index}",
            authors=["Alice"],
            abstract="We study UAV path planning.",
            source="arxiv",
            pdf_url=f"https://arxiv.org/pdf/{index}.pdf",
        )
        for index in range(3)
    ]
    report_service.save_task(task)
    report_service.save_papers(task.task_id, papers)
    paper_import_service = ConcurrentPaperImportServiceStub()
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=paper_import_service,
        import_concurrency=2,
    )

    response = await service.import_papers(
        ImportPapersRequest(
            task_id=task.task_id,
            paper_ids=[paper.paper_id for paper in papers],
        ),
        graph_runtime=GraphRuntimeImportStub(),
    )

    assert response.imported_count == 3
    assert paper_import_service.max_active == 2


@pytest.mark.asyncio
async def test_import_papers_fast_mode_indexes_embeddings_first_then_backfills_graph(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_import_fast_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
    )
    paper = PaperCandidate(
        paper_id="arxiv:fast-1",
        title="Fast UAV Survey",
        authors=["Alice"],
        abstract="We study UAV path planning.",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/fast-1.pdf",
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )
    graph_runtime = GraphRuntimeImportStub()

    response = await service.import_papers(
        ImportPapersRequest(
            task_id=task.task_id,
            paper_ids=[paper.paper_id],
            fast_mode=True,
            include_graph=True,
        ),
        graph_runtime=graph_runtime,
    )

    assert response.imported_count == 1
    assert response.results[0].graph_pending is True
    assert graph_runtime.last_index_kwargs is not None
    assert graph_runtime.last_index_kwargs["include_graph"] is False


@pytest.mark.asyncio
async def test_import_papers_auto_completes_ingest_priority_todo(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    paper = PaperCandidate(
        paper_id="arxiv:ingest-1",
        title="Priority UAV Survey",
        authors=["Alice"],
        abstract="We study UAV path planning.",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/ingest-1.pdf",
    )
    extra_paper = PaperCandidate(
        paper_id="arxiv:ingest-2",
        title="Next UAV Survey",
        authors=["Bob"],
        abstract="We study UAV sensing.",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/ingest-2.pdf",
    )
    todo = ResearchTodoItem(
        todo_id="todo_ingest_priority",
        content="优先导入并精读这些开放论文：Priority UAV Survey。",
        created_at="2026-04-17T00:00:00+00:00",
        priority="high",
        source="qa_follow_up",
        metadata={"todo_kind": "ingest_priority", "paper_ids": [paper.paper_id]},
    )
    task = ResearchTask(
        task_id="task_import_priority_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
        todo_items=[todo],
        workspace=build_workspace_state(
            objective="无人机路径规划",
            stage="ingest",
            papers=[paper, extra_paper],
            imported_document_ids=[],
            todo_items=[todo],
            must_read_ids=[paper.paper_id, extra_paper.paper_id],
            ingest_candidate_ids=[paper.paper_id, extra_paper.paper_id],
        ),
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper, extra_paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )

    response = await service.import_papers(
        ImportPapersRequest(
            task_id=task.task_id,
            paper_ids=[paper.paper_id],
        ),
        graph_runtime=GraphRuntimeImportStub(),
    )

    assert response.imported_count == 1
    persisted_task = report_service.load_task(task.task_id)
    assert persisted_task is not None
    assert persisted_task.todo_items[0].status == "done"
    assert persisted_task.todo_items[0].metadata["auto_completed_by"] == "import_papers"
    assert persisted_task.todo_items[0].metadata["completed_paper_ids"] == [paper.paper_id]
    assert persisted_task.workspace.must_read_paper_ids == [paper.paper_id, extra_paper.paper_id]
    assert persisted_task.workspace.ingest_candidate_ids == [extra_paper.paper_id]
    assert "open_todos=0" in persisted_task.workspace.status_summary


@pytest.mark.asyncio
async def test_list_paper_figures_persists_cache_and_reuses_it(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_fig_cache_1",
        topic="论文图表分析",
        status="completed",
        created_at="2026-04-21T00:00:00+00:00",
        updated_at="2026-04-21T00:00:00+00:00",
        sources=["arxiv"],
    )
    paper = PaperCandidate(
        paper_id="paper_fig_1",
        title="Figure Paper",
        source="arxiv",
        ingest_status="ingested",
        metadata={
            "document_id": "doc_fig_1",
            "storage_uri": "/tmp/paper_fig_1.pdf",
        },
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    parsed_document = ParsedDocument(
        id="doc_fig_1",
        filename="paper_fig_1.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[
            DocumentPage(
                id="page_1",
                document_id="doc_fig_1",
                page_number=1,
                image_uri=str(tmp_path / "page_1.png"),
                text_blocks=[
                    TextBlock(
                        id="tb_1",
                        document_id="doc_fig_1",
                        page_id="page_1",
                        page_number=1,
                        text="Figure 1 shows the main trend.",
                    )
                ],
            )
        ],
        metadata={"source_path": ""},
    )
    runtime = GraphRuntimePaperFigureStub(
        parsed_document,
        candidates_by_page={
            "page_1": [
                {
                    "id": "chart_1",
                    "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.8, "y1": 0.9, "unit": "relative"},
                    "image_uri": str(tmp_path / "page_1.png"),
                    "title": "Main Result",
                    "caption": "Accuracy comparison",
                }
            ]
        },
    )

    first = await service.list_paper_figures(task.task_id, paper.paper_id, graph_runtime=runtime)
    second = await service.list_paper_figures(task.task_id, paper.paper_id, graph_runtime=runtime)

    assert len(first.figures) == 1
    assert first.figures[0].metadata["analyze_target"]["figure_id"] == first.figures[0].figure_id
    assert second.figures[0].figure_id == first.figures[0].figure_id
    assert runtime.document_tools.locate_calls == 1
    persisted = report_service.load_papers(task.task_id)
    cache = persisted[0].metadata.get("paper_figure_cache")
    assert isinstance(cache, dict)
    assert cache["document_id"] == "doc_fig_1"
    assert cache["analyze_targets"][first.figures[0].figure_id]["chart_id"] == "chart_1"


@pytest.mark.asyncio
async def test_analyze_paper_figure_uses_cached_target_without_reparsing(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_fig_cache_2",
        topic="论文图表分析",
        status="completed",
        created_at="2026-04-21T00:00:00+00:00",
        updated_at="2026-04-21T00:00:00+00:00",
        sources=["arxiv"],
    )
    figure_id = "paper_fig_2:chart_1"
    paper = PaperCandidate(
        paper_id="paper_fig_2",
        title="Figure Paper 2",
        source="arxiv",
        ingest_status="ingested",
        metadata={
            "document_id": "doc_fig_2",
            "storage_uri": "/tmp/paper_fig_2.pdf",
            "paper_figure_cache": {
                "document_id": "doc_fig_2",
                "storage_uri": "/tmp/paper_fig_2.pdf",
                "figures": [
                    {
                        "figure_id": figure_id,
                        "paper_id": "paper_fig_2",
                        "document_id": "doc_fig_2",
                        "page_id": "page_2",
                        "page_number": 2,
                        "chart_id": "chart_1",
                        "source": "chart_candidate",
                        "image_path": str(tmp_path / "cached_chart.png"),
                        "metadata": {},
                    }
                ],
                "analyze_targets": {
                    figure_id: {
                        "figure_id": figure_id,
                        "paper_id": "paper_fig_2",
                        "document_id": "doc_fig_2",
                        "page_id": "page_2",
                        "page_number": 2,
                        "chart_id": "chart_1",
                        "image_path": str(tmp_path / "cached_chart.png"),
                        "source": "chart_candidate",
                        "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4, "unit": "point"},
                        "metadata": {"title": "Cached Figure"},
                    }
                },
                "warnings": [],
            },
        },
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper])
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    parsed_document = ParsedDocument(
        id="doc_fig_2",
        filename="paper_fig_2.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[],
    )
    runtime = GraphRuntimePaperFigureStub(parsed_document)

    response = await service.analyze_paper_figure(
        task.task_id,
        paper.paper_id,
        AnalyzeResearchPaperFigureRequest(
            figure_id=figure_id,
            page_id="page_2",
            chart_id="chart_1",
            question="这张图主要说明什么？",
        ),
        graph_runtime=runtime,
    )

    assert runtime.parse_calls == 0
    assert runtime.understand_calls == 1
    assert response.figure.figure_id == figure_id
    assert response.figure.bbox == BoundingBox(x0=1, y0=2, x1=3, y1=4, unit="point")
    assert response.metadata["figure_source"] == "chart_candidate"


@pytest.mark.asyncio
async def test_start_import_job_can_continue_with_collection_qa(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_import_job_1",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        imported_document_ids=[],
        report_id="report_import_job_1",
    )
    paper = PaperCandidate(
        paper_id="arxiv:job1",
        title="UAV Survey",
        authors=["Alice"],
        abstract="A survey.",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/job1.pdf",
    )
    report_service.save_task(task)
    report_service.save_papers(task.task_id, [paper])
    report_service.save_report(
        ResearchReport(
            report_id="report_import_job_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机路径规划",
            paper_count=1,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )

    job = await service.start_import_job(
        ImportPapersRequest(
            task_id=task.task_id,
            paper_ids=[paper.paper_id],
            question="当前最值得优先阅读的论文是哪篇？",
            top_k=8,
            reasoning_style="react",
        ),
        graph_runtime=GraphRuntimeImportAndQaStub(),
    )
    await service._job_tasks[job.job_id]
    persisted_job = service.get_job(job.job_id)

    assert persisted_job.status == "completed"
    assert persisted_job.progress_total == 2
    assert persisted_job.output["import_result"]["imported_count"] == 1
    assert persisted_job.output["import_result"]["results"][0]["metadata"]["graph_backfill_status"] == "succeeded"
    assert persisted_job.output["ask_result"]["qa"]["question"] == "当前最值得优先阅读的论文是哪篇？"
    assert persisted_job.output["task_result"]["task"]["imported_document_ids"] == ["paper_doc_1"]


def test_update_todo_status_persists_change(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_todo_1",
        topic="无人机感知",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        todo_items=[
            ResearchTodoItem(
                todo_id="todo_1",
                content="补充相关论文。",
                created_at="2026-04-17T00:00:00+00:00",
            )
        ],
    )
    report_service.save_task(task)
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )

    response = service.update_todo_status("task_todo_1", "todo_1", "done")

    assert response.task.todo_items[0].status == "done"
    assert response.task.todo_items[0].metadata["last_status_change_at"]


@pytest.mark.asyncio
async def test_rerun_todo_search_updates_task_report_and_candidates(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_todo_2",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv", "openalex"],
        todo_items=[
            ResearchTodoItem(
                todo_id="todo_2",
                content="补充路径规划方向论文。",
                question="有哪些更新的多智能体路径规划论文？",
                created_at="2026-04-17T00:00:00+00:00",
                source="evidence_gap",
            )
        ],
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_todo_2",
            task_id="task_todo_2",
            topic="无人机路径规划",
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机路径规划",
            paper_count=0,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=report_service,
        paper_import_service=object(),
    )

    response = await service.rerun_todo_search(
        "task_todo_2",
        "todo_2",
        ResearchTodoActionRequest(max_papers=5),
        graph_runtime=GraphRuntimeSuccessStub(),
    )

    assert response.papers
    assert response.todo.metadata["last_action_type"] == "search"
    assert response.report is not None
    assert "TODO 执行记录" in response.report.markdown
    assert "重新检索" in response.report.markdown
    persisted_task = report_service.load_task("task_todo_2")
    assert persisted_task is not None
    assert persisted_task.paper_count == len(response.papers)


@pytest.mark.asyncio
async def test_import_from_todo_imports_additional_papers(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_todo_3",
        topic="无人机路径规划",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv", "openalex"],
        todo_items=[
            ResearchTodoItem(
                todo_id="todo_3",
                content="补充可入库的开放论文。",
                question="有哪些可以直接下载 PDF 的路径规划论文？",
                created_at="2026-04-17T00:00:00+00:00",
                source="evidence_gap",
            )
        ],
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_todo_3",
            task_id="task_todo_3",
            topic="无人机路径规划",
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：无人机路径规划",
            paper_count=0,
        )
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=report_service,
        paper_import_service=PaperImportServiceStub(),
    )

    response = await service.import_from_todo(
        "task_todo_3",
        "todo_3",
        ResearchTodoActionRequest(max_papers=1),
        graph_runtime=GraphRuntimeImportStub(),
    )

    assert response.import_result is not None
    assert response.import_result.imported_count == 1
    assert response.todo.metadata["last_action_type"] == "import"
    assert response.task.imported_document_ids == ["paper_doc_1"]
    assert response.report is not None
    assert "补充导入" in response.report.markdown


class LegacyDiscoveryRuntimeShouldNotRun:
    async def run(self, **kwargs):
        raise AssertionError("legacy research_runtime.run should not be called")


class LegacyCollectionQARuntimeShouldNotRun:
    async def run(self, **kwargs):
        raise AssertionError("legacy research_qa_runtime.run should not be called")


@pytest.mark.asyncio
async def test_search_papers_does_not_call_legacy_research_runtime(tmp_path) -> None:
    service = LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=object(),
        research_runtime=LegacyDiscoveryRuntimeShouldNotRun(),
    )

    response = await service.search_papers(
        CreateResearchTaskRequest(
            topic="无人机路径规划",
            days_back=365,
            max_papers=5,
            sources=["arxiv", "openalex"],
            run_immediately=False,
        ),
        graph_runtime=GraphRuntimeSuccessStub(),
    )

    assert response.papers
    assert response.report is not None
    assert response.report.metadata["decision_model"] == "supervisor_direct_execution"


@pytest.mark.asyncio
async def test_ask_task_collection_does_not_call_legacy_qa_runtime_for_collection_qa(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    task = ResearchTask(
        task_id="task_direct_qa_1",
        topic="agentic scientific QA",
        status="completed",
        created_at="2026-04-17T00:00:00+00:00",
        updated_at="2026-04-17T00:00:00+00:00",
        sources=["arxiv"],
        paper_count=1,
        imported_document_ids=["doc_2"],
        report_id="report_direct_qa_1",
        workspace=build_workspace_state(
            objective="agentic scientific QA",
            stage="qa",
            imported_document_ids=["doc_2"],
        ),
    )
    report_service.save_task(task)
    report_service.save_report(
        ResearchReport(
            report_id="report_direct_qa_1",
            task_id=task.task_id,
            topic=task.topic,
            generated_at="2026-04-17T00:00:00+00:00",
            markdown="# 文献调研报告：agentic scientific QA",
            paper_count=1,
        )
    )
    report_service.save_papers(
        task.task_id,
        [
            PaperCandidate(
                paper_id="paper-b",
                title="Paper B",
                abstract="B",
                source="arxiv",
                ingest_status="ingested",
                metadata={"document_id": "doc_2"},
            )
        ],
    )
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceWithLLMStub(
            llm_adapter=QARoutingLLMStub(route="collection_qa", confidence=0.88, rationale="LLM kept this broad.")
        ),
        report_service=report_service,
        paper_import_service=object(),
        research_qa_runtime=LegacyCollectionQARuntimeShouldNotRun(),
    )
    graph_runtime = GraphRuntimeSuccessStub()

    response = await service.ask_task_collection(
        task.task_id,
        ResearchTaskAskRequest(
            question="当前最值得优先阅读的是哪篇？",
            top_k=6,
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa.answer
    assert response.qa.metadata["qa_execution_path"] == "research_supervisor_direct"


def test_record_agent_turn_general_chat_clears_active_research_scope(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(title="Test conversation", topic="无人机路径规划")
    ).conversation
    updated = conversation.model_copy(
        update={
            "snapshot": conversation.snapshot.model_copy(
                update={
                    "active_route_mode": "paper_follow_up",
                    "selected_paper_ids": ["paper_1"],
                    "active_paper_ids": ["paper_1"],
                }
            )
        }
    )
    report_service.save_conversation(updated)

    request = ResearchAgentRunRequest(
        message="你好",
        conversation_id=conversation.conversation_id,
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        messages=[
            ResearchMessage(
                message_id="msg_user",
                role="user",
                kind="question",
                title="普通聊天",
                content="你好",
                created_at="2026-04-24T00:00:00+00:00",
            ),
            ResearchMessage(
                message_id="msg_assistant",
                role="assistant",
                kind="answer",
                title="General answer",
                content="你好，我在。",
                created_at="2026-04-24T00:00:01+00:00",
            ),
        ],
        workspace=ResearchWorkspaceState(
            objective="普通聊天",
            current_stage="complete",
            status_summary="general chat turn",
            stop_reason="Handled as general chat.",
        ),
        metadata={"has_general_answer": True, "route_mode": "general_chat"},
    )

    service.record_agent_turn(conversation.conversation_id, request=request, response=response)
    saved = report_service.load_conversation(conversation.conversation_id)

    assert saved is not None
    assert saved.snapshot.active_route_mode == "general_chat"
    assert saved.snapshot.selected_paper_ids == []
    assert saved.snapshot.active_paper_ids == []


def test_supervisor_hydration_does_not_inherit_paper_scope_for_new_topic_discovery(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(title="Threaded conversation", topic="无人机路径规划")
    ).conversation
    updated = conversation.model_copy(
        update={
            "snapshot": conversation.snapshot.model_copy(
                update={
                    "active_route_mode": "paper_follow_up",
                    "selected_paper_ids": ["paper_1"],
                    "active_paper_ids": ["paper_1"],
                }
            )
        }
    )
    report_service.save_conversation(updated)

    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    hydrated_request, _ = runtime._hydrate_request_from_conversation(
        request=ResearchAgentRunRequest(
            message="量子纠错文献调研",
            conversation_id=conversation.conversation_id,
        )
    )

    assert hydrated_request.selected_paper_ids == []
    assert hydrated_request.metadata["context"]["route_mode"] == "research_discovery"
    assert hydrated_request.metadata["context"].get("active_paper_ids", []) == []


def test_record_agent_turn_persists_thread_workspace_summary(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    conversation = service.create_conversation(
        CreateResearchConversationRequest(title="Thread summary", topic="无人机路径规划")
    ).conversation

    request = ResearchAgentRunRequest(
        message="无人机路径规划文献调研",
        conversation_id=conversation.conversation_id,
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        task=ResearchTask(
            task_id="task_summary_1",
            topic="无人机路径规划",
            status="completed",
            created_at="2026-04-24T00:00:00+00:00",
            updated_at="2026-04-24T00:00:00+00:00",
        ),
        workspace=ResearchWorkspaceState(
            objective="无人机路径规划",
            current_stage="complete",
            status_summary="papers=12; imported_docs=0",
            next_actions=["继续追问方法对比", "导入高价值论文"],
            stop_reason="Discovery finished.",
        ),
        metadata={"route_mode": "research_discovery"},
    )

    service.record_agent_turn(conversation.conversation_id, request=request, response=response)
    saved = report_service.load_conversation(conversation.conversation_id)

    assert saved is not None
    assert saved.snapshot.thread_history
    latest = saved.snapshot.thread_history[-1]
    assert latest.metadata["workspace_summary"] == "papers=12; imported_docs=0"
    assert latest.metadata["next_actions"] == ["继续追问方法对比", "导入高价值论文"]
    assert latest.metadata["stop_reason"] == "Discovery finished."


def test_supervisor_uses_explicit_clarify_request_action_for_ambiguous_turn() -> None:
    agent = ResearchSupervisorAgent()
    decision = agent._intent_guardrail_decision(
        state=ResearchSupervisorState(
            goal="这篇文章怎么样",
            has_task=True,
            route_mode="research_follow_up",
            user_intent={
                "intent": "general_follow_up",
                "needs_clarification": True,
                "clarification_question": "你想问哪篇论文？请给我标题、序号或先选中论文。",
            },
        ),
        all_messages=[],
        results=[],
        planner_runs=0,
        replan_count=0,
    )

    assert decision is not None
    assert decision.action_name == "clarify_request"
    assert decision.stop_reason == "你想问哪篇论文？请给我标题、序号或先选中论文。"


def test_supervisor_standardizes_observation_envelope_for_new_topic_discovery(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    context = type(
        "Ctx",
        (),
        {
            "workspace": ResearchWorkspaceState(next_actions=["write_review", "import_papers"]),
            "qa_result": None,
            "task": None,
            "papers": [],
        },
    )()

    metadata = runtime._with_standardized_observation(
        action_name="search_literature",
        context=context,
        status="succeeded",
        metadata={"paper_count": 8},
    )

    envelope = metadata["observation_envelope"]
    assert envelope["progress_made"] is True
    assert envelope["confidence"] == 0.82
    assert envelope["suggested_next_actions"] == ["write_review", "import_papers", "answer_question"]


def test_supervisor_standardizes_observation_envelope_for_missing_task(tmp_path) -> None:
    report_service = ResearchReportService(tmp_path / "research")
    service = LiteratureResearchService(
        paper_search_service=PaperSearchServiceStub(),
        report_service=report_service,
        paper_import_service=object(),
    )
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    context = type(
        "Ctx",
        (),
        {
            "workspace": ResearchWorkspaceState(),
            "qa_result": None,
            "task": None,
            "papers": [],
        },
    )()

    metadata = runtime._with_standardized_observation(
        action_name="answer_question",
        context=context,
        status="skipped",
        metadata={"reason": "missing_task"},
    )

    envelope = metadata["observation_envelope"]
    assert envelope["progress_made"] is False
    assert envelope["missing_inputs"] == ["task"]
    assert envelope["suggested_next_actions"] == ["clarify_request", "search_literature"]


def test_supervisor_consumes_missing_input_observation_as_clarify_request() -> None:
    agent = ResearchSupervisorAgent()
    decision = agent._guardrail_decision(
        state=ResearchSupervisorState(
            goal="这篇文章讲了什么",
            has_task=True,
            route_mode="paper_follow_up",
        ),
        all_messages=[],
        results=[
            AgentResultMessage(
                task_id="r1",
                agent_from="ResearchKnowledgeAgent",
                agent_to="ResearchSupervisorAgent",
                task_type="answer_question",
                status="skipped",
                instruction="answer",
                payload={
                    "reason": "no_target_papers",
                    "observation_envelope": {
                        "progress_made": False,
                        "missing_inputs": ["paper_scope"],
                        "suggested_next_actions": ["clarify_request", "search_literature"],
                    },
                },
                metadata={},
            )
        ],
        planner_runs=1,
        replan_count=0,
    )

    assert decision is not None
    assert decision.action_name == "clarify_request"
    assert "哪篇论文" in (decision.stop_reason or "")


def test_supervisor_consumes_suggested_next_action_from_observation() -> None:
    agent = ResearchSupervisorAgent()
    decision = agent._guardrail_decision(
        state=ResearchSupervisorState(
            goal="无人机路径规划文献调研",
            has_task=True,
            route_mode="research_follow_up",
            has_import_candidates=True,
        ),
        all_messages=[],
        results=[
            AgentResultMessage(
                task_id="r2",
                agent_from="LiteratureScoutAgent",
                agent_to="ResearchSupervisorAgent",
                task_type="search_literature",
                status="succeeded",
                instruction="search",
                payload={
                    "observation_envelope": {
                        "progress_made": True,
                        "suggested_next_actions": ["write_review", "import_papers"],
                    },
                },
                metadata={},
            )
        ],
        planner_runs=1,
        replan_count=0,
    )

    assert decision is not None
    assert decision.action_name == "write_review"


def test_supervisor_action_priority_uses_latest_observation_signal() -> None:
    agent = ResearchSupervisorAgent()
    state = ResearchSupervisorState(
        goal="无人机路径规划文献调研",
        has_task=True,
        route_mode="research_follow_up",
        latest_suggested_next_actions=["write_review"],
    )

    assert agent._action_priority_score("write_review", state) > agent._action_priority_score("sync_to_zotero", state)


def test_supervisor_state_snapshot_exposes_latest_observation_signal() -> None:
    agent = ResearchSupervisorAgent()
    snapshot = agent._state_snapshot(
        ResearchSupervisorState(
            goal="无人机路径规划文献调研",
            latest_result_task_type="search_literature",
            latest_result_status="succeeded",
            latest_progress_made=True,
            latest_result_confidence=0.82,
            latest_missing_inputs=["paper_scope"],
            latest_suggested_next_actions=["write_review", "import_papers"],
        )
    )

    assert snapshot["latest_result_task_type"] == "search_literature"
    assert snapshot["latest_progress_made"] is True
    assert snapshot["latest_result_confidence"] == 0.82
    assert snapshot["latest_suggested_next_actions"] == ["write_review", "import_papers"]
