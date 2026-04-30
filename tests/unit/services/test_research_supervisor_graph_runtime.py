import asyncio

import pytest

from adapters.llm.base import BaseLLMAdapter
from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    CreateResearchConversationRequest,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchTask,
    ResearchTaskResponse,
)
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from rag_runtime.memory import GraphSessionMemory
from retrieval.evidence_builder import build_evidence_bundle
from services.research.literature_research_service import LiteratureResearchService
from services.research.paper_search_service import PaperSearchService
from services.research.research_supervisor_graph_runtime import ResearchSupervisorGraphRuntime
from services.research.research_report_service import ResearchReportService
from rag_runtime.schemas import ChartUnderstandingResult
from tooling.schemas import GraphSummaryToolOutput
from tools.retrieval_toolkit import RetrievalAgentResult


class GeneralAnswerLLMStub(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        question = str(input_data.get("question") or input_data.get("message") or "")
        if response_model.__name__ == "ResearchUserIntentResult":
            return response_model.model_validate(
                {
                    "intent": "general_answer",
                    "confidence": 0.9,
                    "target_kind": "none",
                    "needs_clarification": False,
                    "rationale": "General question",
                    "markers": ["what is"],
                    "source": "llm",
                }
            )
        if response_model.__name__ == "ResearchSupervisorLLMDecision":
            return response_model.model_validate(
                {
                    "action_name": "general_answer",
                    "worker_agent": "GeneralAnswerAgent",
                    "instruction": "Answer the general question directly.",
                    "thought": "No literature workflow is needed.",
                    "rationale": "Use the new lightweight general-answer branch.",
                    "phase": "act",
                    "payload": {"goal": question or "什么是 Python 生成器？", "mode": "auto"},
                }
            )
        return response_model.model_validate(
            {
                "answer": "生成器是一种按需逐步产出值的可迭代对象，通常通过 yield 定义。",
                "confidence": 0.87,
                "key_points": ["惰性生成", "节省内存", "可迭代"],
                "answer_type": "general",
                "warnings": [],
            }
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class GeneralAnswerRerouteLLMStub(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        if response_model.__name__ == "ResearchUserIntentResult":
            return response_model.model_validate(
                {
                    "intent": "general_answer",
                    "confidence": 0.72,
                    "target_kind": "none",
                    "needs_clarification": False,
                    "rationale": "Initial lightweight intent guess.",
                    "markers": [],
                    "source": "llm",
                }
            )
        if response_model.__name__ == "ResearchSupervisorLLMDecision":
            message = str(input_data.get("state", {}).get("goal") or "")
            recent_results = list(input_data.get("recent_results") or [])
            if recent_results and recent_results[-1].get("task_type") == "general_answer":
                return response_model.model_validate(
                    {
                        "action_name": "answer_question",
                        "worker_agent": "ResearchKnowledgeAgent",
                        "instruction": "Answer the research question using the current imported evidence and workspace.",
                        "thought": "The prior general answer branch indicated this is actually a research question.",
                        "rationale": "Reroute to the grounded research QA worker.",
                        "phase": "act",
                        "payload": {"goal": message, "mode": "qa"},
                    }
                )
            return response_model.model_validate(
                {
                    "action_name": "general_answer",
                    "worker_agent": "GeneralAnswerAgent",
                    "instruction": "Answer the question directly first.",
                    "thought": "Try the lightweight general-answer branch first.",
                    "rationale": "This may be a general question, but the worker can request rerouting if needed.",
                    "phase": "act",
                    "payload": {"goal": message, "mode": "qa"},
                }
            )
        return response_model.model_validate(
            {
                "answer": "这更像一条论文问答请求，建议切回研究问答链路。",
                "confidence": 0.31,
                "key_points": ["已有 task", "涉及论文内容"],
                "answer_type": "reroute_hint",
                "warnings": ["route_mismatch"],
            }
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class ArxivToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id="arxiv:2601.00001",
                title="Autonomous Literature Agents for UAV Path Planning",
                authors=["Alice"],
                abstract="This paper studies agentic literature review for UAV path planning.",
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id="2601.00001",
                pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
                url="https://arxiv.org/abs/2601.00001",
                is_open_access=True,
                published_at="2026-04-01T00:00:00+00:00",
            )
        ]


class OpenAlexToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        return []


class PaperImportServiceStub:
    async def download_paper(self, paper):
        return type(
            "Artifact",
            (),
            {
                "paper": paper,
                "document_id": "paper_doc_agent_1",
                "storage_uri": "/tmp/agent-paper.pdf",
                "filename": "agent-paper.pdf",
            },
        )()


class PaperImportServiceFailureStub:
    async def download_paper(self, paper):
        raise RuntimeError(f"download failed for {paper.paper_id}")


class RetrievalAgentStub:
    async def retrieve(self, *, question: str, document_ids: list[str], top_k: int, **kwargs):
        hits = [
            RetrievalHit(
                id="hit_agent_doc_1",
                source_type="text_block",
                source_id="paper_doc_agent_1:block_1",
                document_id="paper_doc_agent_1",
                content="The imported paper explains agentic literature review for UAV path planning.",
                merged_score=0.91,
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


class AnswerAgentStub:
    async def answer_with_evidence(self, *, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, **kwargs):
        return QAResponse(
            answer="优先阅读 Autonomous Literature Agents for UAV Path Planning，因为它直接覆盖 agentic literature review。",
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.84,
        )


class RuntimeStageTimeoutStub(asyncio.TimeoutError):
    pass


class GraphRuntimeStub:
    retrieval_tools = RetrievalAgentStub()
    answer_tools = AnswerAgentStub()
    react_reasoning_agent = None

    def __init__(self) -> None:
        self.session_memory = GraphSessionMemory()
        self.last_handle_ask_document_kwargs = None
        self.last_handle_ask_fused_kwargs = None

    async def handle_parse_document(self, **kwargs):
        return ParsedDocument(
            id=kwargs.get("document_id") or "paper_doc_agent_1",
            filename="agent-paper.pdf",
            content_type="application/pdf",
            status="parsed",
            pages=[],
            metadata=kwargs.get("metadata") or {},
        )

    async def handle_index_document(self, **kwargs):
        return type("IndexResult", (), {"status": "succeeded"})()

    async def handle_understand_chart(self, **kwargs):
        chart = ChartSchema(
            id=kwargs.get("chart_id") or "chart_1",
            document_id=kwargs.get("document_id") or "doc_1",
            page_id=kwargs.get("page_id") or "page-1",
            page_number=kwargs.get("page_number") or 1,
            chart_type="line",
            summary="The chart shows an increasing trend.",
            confidence=0.82,
            metadata={"image_path": kwargs.get("image_path")},
        )
        return ChartUnderstandingResult(
            chart=chart,
            graph_text="chart_type: line\nsummary: The chart shows an increasing trend.",
            metadata={},
        )

    async def query_graph_summary(self, **kwargs):
        return GraphSummaryToolOutput(hits=[], metadata={"mode": "graph_summary"})

    async def handle_ask_document(self, **kwargs):
        self.last_handle_ask_document_kwargs = kwargs
        filters = kwargs.get("filters") or {}
        return QAResponse(
            answer="这篇论文的方法细节主要集中在 agentic literature review pipeline。",
            question=kwargs["question"],
            evidence_bundle=EvidenceBundle(),
            confidence=0.83,
            metadata={
                "selected_paper_ids": list(filters.get("selected_paper_ids") or []),
                "selected_document_ids": list(filters.get("selected_document_ids") or []),
                "qa_scope_mode": filters.get("qa_scope_mode"),
                "paper_scope": {
                    "paper_ids": list(filters.get("selected_paper_ids") or []),
                    "scope_mode": filters.get("qa_scope_mode") or "selected_papers",
                },
            },
        )

    async def handle_ask_fused(self, **kwargs):
        self.last_handle_ask_fused_kwargs = kwargs
        filters = kwargs.get("filters") or {}
        qa = QAResponse(
            answer="这张图展示了 agentic literature review pipeline 的阶段化结构。",
            question=kwargs["question"],
            evidence_bundle=EvidenceBundle(),
            confidence=0.88,
            metadata={
                "selected_paper_ids": list(filters.get("selected_paper_ids") or []),
                "selected_document_ids": list(filters.get("selected_document_ids") or []),
                "qa_scope_mode": filters.get("qa_scope_mode"),
                "paper_scope": {
                    "paper_ids": list(filters.get("selected_paper_ids") or []),
                    "scope_mode": filters.get("qa_scope_mode") or "selected_papers",
                },
                "visual_anchor": {
                    "image_path": kwargs.get("image_path"),
                    "page_id": kwargs.get("page_id"),
                    "page_number": kwargs.get("page_number"),
                    "chart_id": kwargs.get("chart_id"),
                },
            },
        )
        return type("FusedAskResult", (), {"qa": qa})()



class AdvancedArxivToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        del query, max_results, days_back
        return [
            PaperCandidate(
                paper_id="arxiv:adv-1",
                title="Agentic Survey for UAV Path Planning",
                authors=["Alice"],
                abstract=(
                    "This survey proposes a structured comparison pipeline for UAV path planning research. "
                    "The method organizes retrieval and synthesis. "
                    "Experiments compare benchmark settings."
                ),
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id="adv-1",
                pdf_url="https://arxiv.org/pdf/adv-1.pdf",
                url="https://arxiv.org/abs/adv-1",
                citations=18,
                is_open_access=True,
                published_at="2026-04-01T00:00:00+00:00",
            ),
            PaperCandidate(
                paper_id="arxiv:adv-2",
                title="Benchmarking Multi-Agent UAV Planning",
                authors=["Bob"],
                abstract=(
                    "This paper compares multi-agent UAV planning methods and evaluates them on shared benchmarks."
                ),
                year=2025,
                venue="arXiv",
                source="arxiv",
                arxiv_id="adv-2",
                pdf_url="https://arxiv.org/pdf/adv-2.pdf",
                url="https://arxiv.org/abs/adv-2",
                citations=11,
                is_open_access=True,
                published_at="2025-12-10T00:00:00+00:00",
            ),
            PaperCandidate(
                paper_id="arxiv:adv-3",
                title="Grounded Evidence Control for Research QA",
                authors=["Carol"],
                abstract=(
                    "This work studies grounded evidence control for research QA and analyzes failure modes."
                ),
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id="adv-3",
                pdf_url="https://arxiv.org/pdf/adv-3.pdf",
                url="https://arxiv.org/abs/adv-3",
                citations=7,
                is_open_access=True,
                published_at="2026-03-15T00:00:00+00:00",
            ),
        ]


class AdvancedOpenAlexToolStub:
    async def search(self, *, query: str, max_results: int, days_back: int):
        del query, max_results, days_back
        return [
            PaperCandidate(
                paper_id="openalex:adv-4",
                title="OpenAlex Reproducibility Study for UAV Research Agents",
                authors=["Dave"],
                abstract="This paper studies reproducibility and reading priorities for research-agent workflows.",
                year=2024,
                venue="OpenAlex Venue",
                source="openalex",
                pdf_url="https://openalex.org/adv-4.pdf",
                url="https://openalex.org/adv-4",
                citations=22,
                is_open_access=True,
                published_at="2024-10-01",
            )
        ]


def build_service(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=PaperImportServiceStub(),
    )


def test_research_supervisor_graph_hydration_prefers_latest_conversation_task(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="UAV path planning")
    ).conversation
    stale_task = ResearchTask(
        task_id="task-stale",
        topic="UAV path planning",
        status="completed",
        created_at="2026-04-25T00:00:00+00:00",
        updated_at="2026-04-25T00:00:00+00:00",
    )
    fresh_task = ResearchTask(
        task_id="task-fresh",
        topic="UAV path planning",
        status="completed",
        created_at="2026-04-25T00:00:00+00:00",
        updated_at="2026-04-25T00:00:00+00:00",
        imported_document_ids=["paper_doc_agent_1"],
    )
    fresh_paper = PaperCandidate(
        paper_id="arxiv:2601.00001",
        title="Autonomous Literature Agents for UAV Path Planning",
        abstract="This paper studies agentic literature review for UAV path planning.",
        source="arxiv",
        metadata={"document_id": "paper_doc_agent_1"},
    )
    service.report_service.save_task(stale_task)
    service.report_service.save_task(fresh_task)
    service.report_service.save_papers("task-fresh", [fresh_paper])
    service.report_service.save_conversation(
        conversation.model_copy(
            update={
                "task_id": "task-fresh",
                "snapshot": conversation.snapshot.model_copy(
                    update={
                        "task_result": ResearchTaskResponse(
                            task=stale_task,
                            papers=[],
                            report=None,
                            warnings=[],
                        ),
                        "selected_paper_ids": ["arxiv:2601.00001"],
                        "active_paper_ids": ["arxiv:2601.00001"],
                    }
                ),
            }
        )
    )

    hydrated_request, restored_task_response = runtime._hydrate_request_from_conversation(
        request=ResearchAgentRunRequest(
            message="请解释这篇论文的方法细节。",
            mode="qa",
            conversation_id=conversation.conversation_id,
            selected_paper_ids=["arxiv:2601.00001"],
        )
    )

    assert restored_task_response is not None
    assert restored_task_response.task.task_id == "task-fresh"
    assert hydrated_request.task_id == "task-fresh"
    assert hydrated_request.selected_document_ids == ["paper_doc_agent_1"]


def test_research_supervisor_context_builder_resolves_skill_context(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    context = runtime._build_tool_context(
        request=ResearchAgentRunRequest(
            message="请对比这些论文的方法差异",
            mode="qa",
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert context.skill_context is not None
    assert "Paper Comparison Skill" in context.skill_context
    assert context.skill_selection is not None
    assert context.skill_selection.active_skill_names == ["paper-comparison"]
    assert context.skill_selection.match_reasons["paper-comparison"].startswith("trigger:")
    assert context.knowledge_access is not None


def build_advanced_service(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=AdvancedArxivToolStub(),
            openalex_tool=AdvancedOpenAlexToolStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=PaperImportServiceStub(),
    )


def build_general_answer_service(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
            llm_adapter=GeneralAnswerLLMStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=PaperImportServiceStub(),
    )


def build_general_answer_reroute_service(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
            llm_adapter=GeneralAnswerRerouteLLMStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=PaperImportServiceStub(),
    )


def build_service_with_import_failure(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivToolStub(),
            openalex_tool=OpenAlexToolStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=PaperImportServiceFailureStub(),
    )


def _find_unified_result(response, task_type: str) -> dict:
    for item in response.metadata.get("unified_agent_results", []):
        if item.get("task_type") == task_type:
            return item
    raise AssertionError(f"missing unified_agent_result for task_type={task_type}")


def _find_unified_result_metadata(response, task_type: str) -> dict:
    return _find_unified_result(response, task_type).get("metadata", {})


def _find_unified_result_action_output(response, task_type: str) -> dict:
    return _find_unified_result(response, task_type).get("action_output", {})


@pytest.mark.asyncio
async def test_research_supervisor_graph_creates_task_and_imports_relevant_papers(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv", "openalex"],
            import_top_k=1,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    assert response.report is not None
    assert response.import_result is not None
    assert response.import_result.imported_count == 1
    assert response.task.imported_document_ids == ["paper_doc_agent_1"]
    assert response.workspace.current_stage == "qa"
    assert response.workspace.stop_reason is not None
    assert response.metadata["routing_engine"] == "langgraph_supervisor_specialists"
    assert response.metadata["manager_engine"] == "langgraph_supervisor"
    assert response.metadata["supervisor_action_tool_engine"] == "tool_executor"
    assert response.metadata["supervisor_worker_execution_engine"] == "unified_agent_registry"
    assert response.metadata["supervisor_action_trace_count"] >= 3
    assert response.metadata["unified_supervisor_mode"] == "pure_supervisor"
    assert response.metadata["unified_agent_registry"]
    assert response.metadata["unified_delegation_plan"]
    assert response.metadata["unified_agent_messages"]
    assert response.metadata["unified_agent_results"]
    assert response.metadata["unified_agent_messages"][0]["preferred_skill_name"] == "research_report"
    assert response.metadata["unified_delegation_plan"][0]["preferred_skill_name"] == "research_report"
    assert response.metadata["unified_delegation_plan"][0]["action_output"]["unified_input_adapter"] == "literature_search_input"
    assert response.metadata["unified_agent_results"][0]["execution_mode"] in {
        "tool_native",
        "service_native",
        "hybrid",
    }
    assert response.metadata["unified_agent_results"][0]["metadata"]["execution_engine"] == "unified_agent_registry"
    assert _find_unified_result_action_output(response, "search_literature")["unified_input_adapter"] == "literature_search_input"
    assert _find_unified_result_metadata(response, "search_literature")["unified_input_adapter"] == "literature_search_input"
    assert _find_unified_result_action_output(response, "write_review")["unified_input_adapter"] == "review_draft_input"
    assert _find_unified_result_metadata(response, "write_review")["unified_input_adapter"] == "review_draft_input"
    assert _find_unified_result_action_output(response, "import_papers")["unified_input_adapter"] == "paper_import_input"
    assert _find_unified_result_metadata(response, "import_papers")["unified_input_adapter"] == "paper_import_input"
    assert response.metadata["agent_message_count"] >= 3
    assert response.trace[0].phase == "plan"
    assert response.trace[1].phase == "reflect"
    assert response.trace[2].phase == "act"
    assert response.trace[0].agent == "LiteratureScoutAgent"
    assert response.trace[1].agent == "ResearchWriterAgent"
    assert response.trace[2].agent == "ResearchKnowledgeAgent"
    assert [step.action_name for step in response.trace][:3] == [
        "search_literature",
        "write_review",
        "import_papers",
    ]
    assert response.trace[-1].action_name == "finalize"
    assert response.trace[-1].stop_signal is True


@pytest.mark.asyncio
async def test_research_supervisor_graph_answers_existing_research_task(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    initial = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=GraphRuntimeStub(),
    )
    assert initial.task is not None

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="哪篇论文最值得优先阅读，为什么？",
            mode="qa",
            task_id=initial.task.task_id,
            sources=["arxiv"],
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.qa is not None
    assert "优先阅读" in response.qa.answer
    assert response.workspace.current_stage == "qa"
    assert response.trace[0].phase in {"reflect", "plan"}
    assert response.trace[0].agent == "ResearchKnowledgeAgent"
    assert any(step.action_name == "answer_question" for step in response.trace)
    assert _find_unified_result_action_output(response, "answer_question")["unified_input_adapter"] == "collection_qa_input"
    assert _find_unified_result_metadata(response, "answer_question")["unified_input_adapter"] == "collection_qa_input"


@pytest.mark.asyncio
async def test_research_supervisor_graph_finalizes_after_discovery_without_auto_import(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="帮我调研 vln 相关的论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    assert response.report is not None
    assert response.import_result is None
    assert [step.action_name for step in response.trace] == [
        "search_literature",
        "write_review",
        "finalize",
    ]
    assert response.trace[-1].stop_signal is True


@pytest.mark.asyncio
async def test_research_supervisor_graph_preserves_document_scope_for_main_chain_qa(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    graph_runtime = GraphRuntimeStub()
    initial = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=graph_runtime,
    )
    assert initial.task is not None

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="请解释这篇论文的方法细节。",
            mode="qa",
            task_id=initial.task.task_id,
            sources=["arxiv"],
            selected_paper_ids=["arxiv:2601.00001"],
            selected_document_ids=["paper_doc_agent_1"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa is not None
    assert graph_runtime.last_handle_ask_document_kwargs is not None
    assert graph_runtime.last_handle_ask_document_kwargs["document_ids"] == [
        "paper_doc_agent_1"
    ]
    assert graph_runtime.last_handle_ask_document_kwargs["filters"][
        "selected_paper_ids"
    ] == ["arxiv:2601.00001"]
    assert graph_runtime.last_handle_ask_document_kwargs["filters"][
        "selected_document_ids"
    ] == ["paper_doc_agent_1"]
    assert response.qa.metadata["selected_document_ids"] == ["paper_doc_agent_1"]
    assert response.qa.metadata["qa_scope_mode"] == "selected_documents"


@pytest.mark.asyncio
async def test_research_supervisor_graph_preserves_visual_anchor_for_main_chain_qa(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    graph_runtime = GraphRuntimeStub()
    initial = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=graph_runtime,
    )
    assert initial.task is not None

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="请解释这张图表达了什么。",
            mode="qa",
            task_id=initial.task.task_id,
            sources=["arxiv"],
            selected_paper_ids=["arxiv:2601.00001"],
            selected_document_ids=["paper_doc_agent_1"],
            chart_image_path="/tmp/chart-agent.png",
            page_id="page-2",
            page_number=2,
            chart_id="chart-agent-1",
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa is not None
    assert graph_runtime.last_handle_ask_fused_kwargs is not None
    assert graph_runtime.last_handle_ask_fused_kwargs["document_ids"] == [
        "paper_doc_agent_1"
    ]
    assert graph_runtime.last_handle_ask_fused_kwargs["image_path"] == "/tmp/chart-agent.png"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_id"] == "page-2"
    assert graph_runtime.last_handle_ask_fused_kwargs["page_number"] == 2
    assert graph_runtime.last_handle_ask_fused_kwargs["chart_id"] == "chart-agent-1"
    assert response.qa.metadata["visual_anchor"]["image_path"] == "/tmp/chart-agent.png"


@pytest.mark.asyncio
async def test_research_supervisor_graph_import_mode_answers_original_question_after_ingest(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    initial = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )
    assert initial.task is not None
    assert initial.task.imported_document_ids == []

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="哪篇论文最值得优先阅读，为什么？",
            mode="import",
            task_id=initial.task.task_id,
            sources=["arxiv"],
            selected_paper_ids=["arxiv:2601.00001"],
            import_top_k=1,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.import_result is not None
    assert response.import_result.imported_count == 1
    assert response.qa is not None
    assert response.qa.question == "哪篇论文最值得优先阅读，为什么？"
    assert "优先阅读" in response.qa.answer
    assert response.workspace.current_stage == "qa"
    assert response.trace[0].agent == "ResearchKnowledgeAgent"
    assert response.trace[1].agent == "ResearchKnowledgeAgent"
    assert [step.action_name for step in response.trace[:3]] == [
        "import_papers",
        "answer_question",
        "finalize",
    ]


@pytest.mark.asyncio
async def test_research_supervisor_graph_requests_clarification_for_broad_topic(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="AI",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is None
    assert response.metadata["clarification_requested"] is True
    assert any("当前研究目标还比较宽泛" in warning for warning in response.warnings)
    assert response.trace[-1].action_name == "finalize"
    assert response.trace[-1].agent == "ResearchSupervisorAgent"
    assert response.status == "partial"


@pytest.mark.asyncio
async def test_research_supervisor_graph_replans_after_import_failure(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service_with_import_failure(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    assert response.import_result is not None
    assert response.import_result.failed_count == 1
    assert response.metadata["recovery_decision_count"] == 1
    assert response.status == "partial"
    assert [step.action_name for step in response.trace[:4]] == [
        "search_literature",
        "write_review",
        "import_papers",
        "finalize",
    ]
    assert response.trace[2].status == "failed"


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_use_document_understanding_tool(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="理解这份文档并建立可检索证据。",
            mode="document",
            document_file_path="/tmp/example.pdf",
            document_id="doc_tool_1",
            include_graph=True,
            include_embeddings=True,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.parsed_document is not None
    assert response.parsed_document.id == "doc_tool_1"
    assert response.document_index_result is not None
    assert _find_unified_result_action_output(response, "understand_document")["unified_input_adapter"] == "document_understanding_input"
    assert _find_unified_result_metadata(response, "understand_document")["unified_input_adapter"] == "document_understanding_input"
    assert any(
        step.action_name == "understand_document" and step.agent == "DocumentTools"
        for step in response.trace
    )
    assert "DocumentRuntime" not in response.metadata["primary_agents"]
    assert "DocumentRuntime" not in response.metadata["primary_runtime_workers"]


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_analyze_selected_papers_with_context_compression(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_advanced_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="对比无人机路径规划论文的方法和实验",
            mode="research",
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    assert response.workspace.current_stage == "complete"
    assert "latest_paper_analysis" in response.workspace.metadata
    assert "context_compression" in response.workspace.metadata
    analysis_payload = response.workspace.metadata["latest_paper_analysis"]
    assert "evidence_hit_count" in analysis_payload["metadata"]
    assert "evidence_backed_paper_count" in analysis_payload["metadata"]
    assert any(message.title == "论文分析结果" for message in response.messages)
    assert any(message.title == "上下文压缩摘要" for message in response.messages)
    assert [step.action_name for step in response.trace[:4]] == [
        "search_literature",
        "compress_context",
        "analyze_papers",
        "finalize",
    ]
    assert _find_unified_result_action_output(response, "compress_context")["unified_input_adapter"] == "context_compression_input"
    assert _find_unified_result_metadata(response, "compress_context")["unified_input_adapter"] == "context_compression_input"
    assert _find_unified_result_action_output(response, "analyze_papers")["unified_input_adapter"] == "paper_analysis_input"
    assert _find_unified_result_metadata(response, "analyze_papers")["unified_input_adapter"] == "paper_analysis_input"
    assert any(
        step.action_name == "analyze_papers" and step.agent == "PaperAnalysisAgent"
        for step in response.trace
    )


@pytest.mark.asyncio
async def test_research_supervisor_graph_honors_explicit_advanced_comparison_dimensions(tmp_path) -> None:
    service = build_advanced_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="继续当前研究任务",
            mode="research",
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            import_top_k=0,
            advanced_action="analyze",
            comparison_dimensions=["method", "experiment", "year"],
            force_context_compression=True,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    analysis_payload = response.workspace.metadata["latest_paper_analysis"]
    assert analysis_payload["focus"] in {"compare", "analysis"}
    assert "evidence_hit_count" in analysis_payload["metadata"]
    assert response.workspace.metadata["context_compression"]["paper_count"] >= 1
    persisted = service.get_task(response.task.task_id)
    assert persisted.task.workspace.metadata["advanced_strategy"]["action"] == "analyze"
    assert persisted.task.workspace.metadata["advanced_strategy"]["comparison_dimensions"] == [
        "method",
        "experiment",
        "year",
    ]


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_provide_reading_advice_with_context_compression(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_advanced_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="推荐接下来值得精读的无人机路径规划论文",
            mode="research",
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    assert response.workspace.current_stage == "complete"
    assert "latest_paper_analysis" in response.workspace.metadata
    analysis_payload = response.workspace.metadata["latest_paper_analysis"]
    assert "evidence_hit_count" in analysis_payload["metadata"]
    assert response.workspace.must_read_paper_ids
    assert any(message.title == "论文分析结果" for message in response.messages)
    assert any(message.title == "上下文压缩摘要" for message in response.messages)
    assert [step.action_name for step in response.trace[:4]] == [
        "search_literature",
        "compress_context",
        "analyze_papers",
        "finalize",
    ]
    assert any(
        step.action_name == "analyze_papers" and step.agent == "PaperAnalysisAgent"
        for step in response.trace
    )


@pytest.mark.asyncio
async def test_research_supervisor_graph_honors_explicit_recommendation_goal_and_top_k(tmp_path) -> None:
    service = build_advanced_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="继续当前研究任务",
            mode="research",
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            import_top_k=0,
            advanced_action="analyze",
            recommendation_goal="优先推荐开放 PDF 且适合继续入库的论文",
            recommendation_top_k=2,
            force_context_compression=True,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    analysis_payload = response.workspace.metadata["latest_paper_analysis"]
    assert analysis_payload["recommended_paper_ids"]
    assert set(analysis_payload["recommended_paper_ids"]).issubset(
        set(response.workspace.must_read_paper_ids)
    )
    persisted = service.get_task(response.task.task_id)
    assert persisted.task.workspace.metadata["advanced_strategy"]["action"] == "analyze"
    assert (
        persisted.task.workspace.metadata["advanced_strategy"]["recommendation_goal"]
        == "优先推荐开放 PDF 且适合继续入库的论文"
    )
    assert persisted.task.workspace.metadata["advanced_strategy"]["recommendation_top_k"] == 2


@pytest.mark.asyncio
async def test_research_supervisor_graph_persists_advanced_strategy_into_conversation_snapshot(tmp_path) -> None:
    service = build_advanced_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="无人机路径规划")
    )
    conversation_id = conversation.conversation.conversation_id

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="继续当前研究任务",
            mode="research",
            conversation_id=conversation_id,
            days_back=365,
            max_papers=4,
            sources=["arxiv", "openalex"],
            import_top_k=0,
            advanced_action="recommend",
            recommendation_goal="优先筛出最值得精读的代表论文",
            recommendation_top_k=2,
            force_context_compression=True,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.task is not None
    saved = service.get_conversation(conversation_id).conversation.snapshot
    assert saved.advanced_strategy.action == "recommend"
    assert saved.advanced_strategy.recommendation_top_k == 2
    assert saved.workspace.metadata["advanced_strategy"]["action"] == "recommend"


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_use_chart_understanding_tool(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="理解这张科研图表。",
            mode="chart",
            chart_image_path="/tmp/chart.png",
            document_id="doc_chart_1",
            page_id="page-1",
            chart_id="chart_tool_1",
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.chart is not None
    assert response.chart.id == "chart_tool_1"
    assert response.chart_graph_text
    assert response.task is None
    assert response.workspace.current_stage == "chart"
    assert response.trace[0].phase == "act"
    assert [step.action_name for step in response.trace] == ["understand_chart", "finalize"]
    assert _find_unified_result_action_output(response, "understand_chart")["unified_input_adapter"] == "chart_understanding_input"
    assert _find_unified_result_metadata(response, "understand_chart")["unified_input_adapter"] == "chart_understanding_input"


@pytest.mark.asyncio
async def test_research_supervisor_graph_response_exposes_session_memory_context(tmp_path) -> None:
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    conversation = service.create_conversation(
        CreateResearchConversationRequest(topic="无人机路径规划")
    )
    conversation_id = conversation.conversation.conversation_id
    graph_runtime = GraphRuntimeStub()
    graph_runtime.session_memory.update_research_context(
        session_id=conversation_id,
        current_task_intent="research_seed",
    )

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            conversation_id=conversation_id,
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=graph_runtime,
    )

    assert response.metadata["session_id"] == conversation_id
    assert response.metadata["memory_enabled"] is True


@pytest.mark.asyncio
async def test_research_supervisor_graph_falls_back_on_timeout_only(tmp_path, monkeypatch) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    async def timeout_synthesize(_state):
        raise RuntimeStageTimeoutStub()

    monkeypatch.setattr(runtime.research_writer_agent, "synthesize_async", timeout_synthesize)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="帮我调研 vln 相关的论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.report is not None
    assert response.trace[-1].action_name == "finalize"


@pytest.mark.asyncio
async def test_research_supervisor_graph_does_not_swallow_non_retryable_stage_errors(tmp_path, monkeypatch) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))

    async def broken_synthesize(_state):
        raise RuntimeError("writer schema bug")

    monkeypatch.setattr(runtime.research_writer_agent, "synthesize_async", broken_synthesize)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="帮我调研 vln 相关的论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.report is None
    assert response.status in {"failed", "partial"}
    assert any(step.status == "failed" for step in response.trace)
    assert any("tool_execution_failed" == step.metadata.get("reason") for step in response.trace if step.status == "failed")


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_answer_general_question_without_research_chain(tmp_path) -> None:
    runtime = ResearchSupervisorGraphRuntime(research_service=build_general_answer_service(tmp_path))

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="什么是 Python 生成器？",
            mode="auto",
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert response.qa is None
    assert response.task is None
    assert response.metadata["has_general_answer"] is True
    assert response.metadata["general_answer"]
    assert response.metadata["general_answer_metadata"]["answer_type"] == "general"
    assert any(step.action_name == "general_answer" for step in response.trace)
    assert any(message.title == "通用回答" for message in response.messages)
    assert any(item.get("task_type") == "general_answer" for item in response.metadata.get("unified_agent_results", []))


@pytest.mark.asyncio
async def test_research_supervisor_graph_can_reroute_from_general_answer_to_grounded_research_qa(tmp_path) -> None:
    service = build_general_answer_reroute_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    graph_runtime = GraphRuntimeStub()
    initial = await runtime.run(
        ResearchAgentRunRequest(
            message="调研无人机路径规划方向的自主文献助手论文",
            mode="research",
            days_back=365,
            max_papers=3,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=graph_runtime,
    )
    assert initial.task is not None

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="这篇论文的方法是什么？",
            mode="qa",
            task_id=initial.task.task_id,
            selected_paper_ids=["arxiv:2601.00001"],
            selected_document_ids=["paper_doc_agent_1"],
        ),
        graph_runtime=graph_runtime,
    )

    assert response.qa is not None
    assert any(step.action_name == "general_answer" and step.status == "skipped" for step in response.trace)
    assert any(step.action_name == "answer_question" and step.status == "succeeded" for step in response.trace)
    assert response.metadata["recovery_decision_count"] >= 1


@pytest.mark.asyncio
async def test_research_supervisor_graph_routes_recent_generic_recommendation_to_preference_memory_agent(tmp_path) -> None:
    service = build_service(tmp_path)
    service.memory_manager.observe_user_query(
        topics=["GraphRAG", "agent memory"],
        sources=["arxiv"],
        keywords=["groundedness", "retrieval"],
    )
    runtime = ResearchSupervisorGraphRuntime(research_service=service)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="推荐一些论文",
            mode="auto",
            days_back=30,
            recommendation_top_k=2,
            sources=["arxiv"],
            selected_paper_ids=["arxiv:2601.00001"],
        ),
        graph_runtime=GraphRuntimeStub(),
    )

    assert any(
        step.action_name == "recommend_from_preferences" and step.agent == "PreferenceMemoryAgent"
        for step in response.trace
    )
    assert response.metadata["preference_recommendations"]["recommendations"]
    recommendation_message = next(message for message in response.messages if message.title == "长期兴趣论文推荐")
    assert "sources=" in (recommendation_message.meta or "")
    assert "主题：" in (recommendation_message.content or "")
    assert "链接：" in (recommendation_message.content or "")
    assert "论文讲解：" in (recommendation_message.content or "")
