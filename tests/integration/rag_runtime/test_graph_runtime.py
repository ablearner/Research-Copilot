import pytest
import json

from adapters.llm.base import BaseLLMAdapter
from domain.schemas.api import QAResponse
from domain.schemas.chart import AxisSchema, ChartSchema, SeriesSchema
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock
from domain.schemas.evidence import Evidence
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph import GraphExtractionResult
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from rag_runtime.runtime import GraphRuntime
from rag_runtime.schemas import GraphTaskRequest
from reasoning import ReActReasoningAgent
from rag_runtime.services.embedding_index_service import EmbeddingIndexResult
from rag_runtime.services.graph_index_service import GraphIndexStats
from tooling.schemas import AnswerWithEvidenceToolInput, ToolSpec
from tools.answer_toolkit import AnswerAgent as RealAnswerAgent
from tools.retrieval_toolkit import RetrievalAgentError, RetrievalAgentResult


class StructuredPlannerLLM(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        self.calls.append(response_model.__name__)
        messages = input_data.get("messages", [])
        content = messages[-1]["content"] if messages else "{}"
        payload = json.loads(content) if isinstance(content, str) else content
        if response_model.__name__ == "RetrievalPlan":
            heuristic_plan = payload.get("heuristic_plan", {})
            return response_model.model_validate(
                {
                    "modes": heuristic_plan.get("modes", ["vector", "graph"]),
                    "query": heuristic_plan.get("query") or payload.get("question", ""),
                    "retrieval_focus": heuristic_plan.get("retrieval_focus"),
                    "reasoning_summary": "Structured retrieval planner selected a GraphRAG evidence plan.",
                    "react_trace": [
                        {"step": 1, "agent": "retrieval_planner", "action": "plan", "observation": "structured_output"}
                    ],
                    "max_steps": heuristic_plan.get("max_steps", 3),
                }
            )
        if response_model.__name__ == "ValidationDecision":
            answer_payload = payload.get("answer", {})
            answer_text = answer_payload.get("answer", "")
            evidence_count = payload.get("evidence_count", 0)
            decision = "retry_retrieval" if evidence_count == 0 and "证据不足" in answer_text else "finalize"
            return response_model.model_validate(
                {
                    "decision": decision,
                    "confidence": 0.15 if decision == "retry_retrieval" else 0.82,
                    "warnings": ["structured_validation_retry"] if decision == "retry_retrieval" else [],
                    "critique_summary": "Structured validation evaluated the answer.",
                    "retrieval_focus": payload.get("question"),
                    "reasoning_summary": "Structured validation decided whether another retrieval pass is needed.",
                    "react_trace": [
                        {"step": 1, "agent": "validation", "action": "validate", "observation": decision}
                    ],
                }
            )
        return response_model.model_validate(payload)

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class StructuredReactLLM(StructuredPlannerLLM):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        if response_model.__name__ == "ReActDecision":
            raise AssertionError("ReActDecision should not be requested when evidence is already available")
        if response_model.__name__ == "QAResponse":
            evidence_bundle = EvidenceBundle.model_validate(input_data["evidence_bundle"])
            return QAResponse(
                answer="根据文档证据，这篇文章主要讲的是一种基于学习的四旋翼轨迹生成方法，面向动态环境中的快速自主感知规划。",
                question=input_data["question"],
                evidence_bundle=evidence_bundle,
                confidence=0.86,
                metadata={"source": "react_mock"},
            )
        if response_model.__name__ == "ReActFinalDraft":
            return response_model.model_validate(
                {
                    "answer": "根据文档证据，这篇文章主要讲的是一种基于学习的四旋翼轨迹生成方法，面向动态环境中的快速自主感知规划。",
                    "confidence": 0.86,
                    "warnings": [],
                }
            )
        return await super()._generate_structured(prompt, input_data, response_model)


class DocumentAgent:
    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        block = TextBlock(id="tb1", document_id="doc1", page_id="p1", page_number=1, text="hello")
        page = DocumentPage(id="p1", document_id="doc1", page_number=1, text_blocks=[block])
        return ParsedDocument(id="doc1", filename=file_path, content_type="application/pdf", status="parsed", pages=[page])

    async def summarize_page(self, page: DocumentPage):
        from tools.document_toolkit import PageSummary

        return PageSummary(document_id=page.document_id, page_id=page.id, page_number=1, summary="hello")


class ChartAgent:
    def __init__(self) -> None:
        self.ask_calls: list[dict] = []
        self.ocr_calls: list[dict] = []

    async def parse_chart(self, image_path: str, document_id: str, page_id: str, page_number: int, chart_id: str, context=None) -> ChartSchema:
        return ChartSchema(
            id=chart_id,
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
            chart_type="bar",
            title="Revenue by quarter",
            caption="Figure 1. Quarterly revenue pipeline.",
            x_axis=AxisSchema(label="quarter", categories=["Q1", "Q2", "Q3", "Q4"]),
            y_axis=AxisSchema(label="revenue", unit="USD"),
            series=[SeriesSchema(name="revenue", chart_role="bar")],
            summary="A bar chart comparing quarterly revenue.",
        )

    def to_graph_text(self, chart: ChartSchema) -> str:
        return "chart graph text"

    def explain_chart(self, chart: ChartSchema) -> str:
        return f"Chart type: {chart.chart_type}.\nTitle: {chart.title}.\nSummary: {chart.summary}"

    async def extract_visible_text(self, image_path: str, context=None, chart=None) -> str:
        self.ocr_calls.append({"image_path": image_path, "context": context or {}, "chart": chart.id if chart else None})
        return "title: Revenue by quarter\nx_axis: quarter\ny_axis: revenue"

    async def ask_chart(self, image_path: str, question: str, context=None, history=None) -> str:
        self.ask_calls.append({"image_path": image_path, "question": question, "context": context or {}, "history": history or []})
        return "chart answer"


class GraphExtractionAgent:
    async def extract_from_text_blocks(self, document_id: str, text_blocks: list[TextBlock], page_summaries=None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=document_id)

    async def extract_from_chart(self, chart: ChartSchema, chart_summary: str | None = None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=chart.document_id)

    def merge_graph_candidates(self, document_id: str, candidates: list[GraphExtractionResult]) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=document_id)


class RetrievalAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
    ) -> RetrievalAgentResult:
        self.calls.append(
            {
                "question": question,
                "filters": filters or {},
                "session_id": session_id,
                "task_id": task_id,
            }
        )
        retrieval_result = HybridRetrievalResult(query=RetrievalQuery(query=question))
        return RetrievalAgentResult(
            question=question,
            document_ids=[doc_id] if doc_id else [],
            evidence_bundle=EvidenceBundle(),
            retrieval_result=retrieval_result,
        )


class RetrievalAgentWithEvidence(RetrievalAgent):
    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
    ) -> RetrievalAgentResult:
        self.calls.append(
            {
                "question": question,
                "filters": filters or {},
                "session_id": session_id,
                "task_id": task_id,
            }
        )
        evidence = Evidence(
            id="ev1",
            document_id=doc_id or "doc1",
            source_type="text_block",
            source_id="tb1",
            snippet="项目目标：构建文档问答系统。",
        )
        hit = RetrievalHit(
            id="hit1",
            source_type="text_block",
            source_id="tb1",
            document_id=doc_id or "doc1",
            content=evidence.snippet,
            merged_score=0.92,
            evidence=EvidenceBundle(evidences=[evidence]),
        )
        evidence_bundle = EvidenceBundle(
            evidences=[evidence],
            summary="检索命中了文档中的关键描述。",
        )
        retrieval_result = HybridRetrievalResult(query=RetrievalQuery(query=question), hits=[hit], evidence_bundle=evidence_bundle)
        return RetrievalAgentResult(
            question=question,
            document_ids=[doc_id] if doc_id else [],
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
        )


class RetrievalAgentWithVectorHitsOnly(RetrievalAgent):
    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
    ) -> RetrievalAgentResult:
        self.calls.append(
            {
                "question": question,
                "filters": filters or {},
                "session_id": session_id,
                "task_id": task_id,
            }
        )
        hit = RetrievalHit(
            id="vector_hit_1",
            source_type="text_block",
            source_id="tb1",
            document_id=doc_id or "doc1",
            content="项目目标：构建文档问答系统。",
            vector_score=0.91,
            merged_score=0.91,
        )
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(query=question),
            hits=[hit],
            evidence_bundle=EvidenceBundle(),
        )
        return RetrievalAgentResult(
            question=question,
            document_ids=[doc_id] if doc_id else [],
            evidence_bundle=retrieval_result.evidence_bundle,
            retrieval_result=retrieval_result,
        )


class RetrievalAgentWithVisualAnchor(RetrievalAgentWithEvidence):
    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
    ) -> RetrievalAgentResult:
        self.calls.append(
            {
                "question": question,
                "filters": filters or {},
                "session_id": session_id,
                "task_id": task_id,
            }
        )
        modalities = set((filters or {}).get("modalities") or [])
        if modalities and modalities <= {"page", "chart"}:
            hit = RetrievalHit(
                id="page_hit1",
                source_type="page",
                source_id="p1",
                document_id=doc_id or "doc1",
                content="Page with a chart and supporting text.",
                merged_score=0.88,
                metadata={"page_id": "p1", "page_number": 1, "uri": "/tmp/page-1.png"},
            )
            retrieval_result = HybridRetrievalResult(
                query=RetrievalQuery(query=question),
                hits=[hit],
                evidence_bundle=EvidenceBundle(),
            )
            return RetrievalAgentResult(
                question=question,
                document_ids=[doc_id] if doc_id else [],
                evidence_bundle=retrieval_result.evidence_bundle,
                retrieval_result=retrieval_result,
            )
        return await super().retrieve(
            question=question,
            doc_id=doc_id,
            document_ids=document_ids,
            top_k=top_k,
            filters=filters,
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints,
        )


class RetrievalAgentWithGraphFailure(RetrievalAgentWithEvidence):
    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
    ) -> RetrievalAgentResult:
        self.calls.append(
            {
                "question": question,
                "filters": filters or {},
                "session_id": session_id,
                "task_id": task_id,
            }
        )
        if (filters or {}).get("retrieval_mode") == "graph":
            raise RetrievalAgentError("graph backend unavailable")
        return await super().retrieve(
            question=question,
            doc_id=doc_id,
            document_ids=document_ids,
            top_k=top_k,
            filters=filters,
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints,
        )


class AnswerAgent:
    def __init__(self, llm_adapter=None) -> None:
        self.llm_adapter = llm_adapter
        self.calls: list[dict] = []

    async def answer(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result=None,
        metadata=None,
        session_context=None,
        task_context=None,
        **kwargs,
    ) -> QAResponse:
        self.calls.append(
            {
                "question": question,
                "task_context": task_context or {},
                "metadata": metadata or {},
            }
        )
        answer_text = "chart-aware answer" if (task_context or {}).get("chart_answer") else "证据不足"
        confidence = 0.74 if answer_text == "chart-aware answer" else 0
        return QAResponse(answer=answer_text, question=question, evidence_bundle=evidence_bundle, retrieval_result=retrieval_result, confidence=confidence)


class InsufficientAnswerAgent(AnswerAgent):
    async def answer(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result=None,
        metadata=None,
        session_context=None,
        task_context=None,
        **kwargs,
    ) -> QAResponse:
        self.calls.append(
            {
                "question": question,
                "task_context": task_context or {},
                "metadata": metadata or {},
            }
        )
        return QAResponse(answer="证据不足", question=question, evidence_bundle=evidence_bundle, retrieval_result=retrieval_result, confidence=0.12)


class GraphIndexService:
    async def index_graph_result(self, graph_result: GraphExtractionResult) -> GraphIndexStats:
        return GraphIndexStats(document_id=graph_result.document_id, status="indexed")


class EmbeddingIndexService:
    async def index_text_blocks(self, document_id: str, text_blocks: list[TextBlock]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(text_blocks))

    async def index_pages(self, document_id: str, pages: list[DocumentPage]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="skipped")

    async def index_charts(self, document_id: str, charts: list[ChartSchema]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="skipped")


class FailingSessionMemory:
    def load(self, session_id: str | None):
        return None

    def as_prompt_context(self, snapshot):
        return {"memory_enabled": False}

    def update_from_state(self, state):
        raise RuntimeError("session memory update failed")


@pytest.fixture
def graph_runtime() -> GraphRuntime:
    planner_llm = StructuredPlannerLLM()
    chart_agent = ChartAgent()
    answer_agent = AnswerAgent(llm_adapter=planner_llm)
    return GraphRuntime(
        DocumentAgent(),
        chart_agent,
        GraphExtractionAgent(),
        RetrievalAgent(),
        answer_agent,
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )


@pytest.mark.asyncio
async def test_graph_runtime_dynamic_dispatch_parse(graph_runtime: GraphRuntime) -> None:
    result = await graph_runtime.handle(
        GraphTaskRequest(task_type="parse_document", params={"file_path": "sample.pdf", "document_id": "doc1"})
    )

    assert result.status == "succeeded"
    assert result.output.id == "doc1"


@pytest.mark.asyncio
async def test_graph_runtime_index_and_ask(graph_runtime: GraphRuntime) -> None:
    parsed = await graph_runtime.handle_parse_document("sample.pdf", "doc1")
    index_result = await graph_runtime.handle_index_document(parsed)
    qa = await graph_runtime.handle_ask_document("What happened?", doc_id="doc1")

    assert index_result.document_id == "doc1"
    assert qa.answer == "证据不足"
    assert "RetrievalPlan" in graph_runtime.llm_adapter.calls
    assert "ValidationDecision" in graph_runtime.llm_adapter.calls


@pytest.mark.asyncio
async def test_graph_runtime_ask_survives_session_memory_failure() -> None:
    planner_llm = StructuredPlannerLLM()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        ChartAgent(),
        GraphExtractionAgent(),
        RetrievalAgent(),
        AnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
        session_memory=FailingSessionMemory(),
    )

    qa = await graph_runtime.handle_ask_document("What happened?", doc_id="doc1", session_id="session-1")

    assert qa.answer == "证据不足"


@pytest.mark.asyncio
async def test_graph_runtime_rewrites_insufficient_answer_when_evidence_exists() -> None:
    planner_llm = StructuredPlannerLLM()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        ChartAgent(),
        GraphExtractionAgent(),
        RetrievalAgentWithEvidence(),
        InsufficientAnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    qa = await graph_runtime.handle_ask_document("这份文档讲了什么？", doc_id="doc1")

    assert qa.answer != "证据不足"
    assert "项目目标：构建文档问答系统。" in qa.answer


@pytest.mark.asyncio
async def test_graph_runtime_react_path_uses_seeded_evidence_bundle() -> None:
    react_llm = StructuredReactLLM()
    answer_agent = RealAnswerAgent(llm_adapter=react_llm)
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        ChartAgent(),
        GraphExtractionAgent(),
        RetrievalAgentWithEvidence(),
        answer_agent,
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=react_llm,
    )
    graph_runtime.tool_registry.register(
        ToolSpec(
            name="answer_with_evidence",
            description="Generate a grounded answer from the supplied evidence.",
            input_schema=AnswerWithEvidenceToolInput,
            output_schema=QAResponse,
            handler=answer_agent.answer_with_evidence,
            tags=["answer", "generation"],
        ),
    )
    graph_runtime.react_reasoning_agent = ReActReasoningAgent(
        llm_adapter=react_llm,
        tool_registry=graph_runtime.tool_registry,
        tool_executor=graph_runtime.tool_executor,
    )

    qa = await graph_runtime.handle_ask_document(
        "这篇文章在讲什么？",
        doc_id="doc1",
        reasoning_style="react",
    )

    assert qa.answer.startswith("根据文档证据")
    assert qa.metadata["reasoning_style"] == "react"
    assert "ReActDecision" not in react_llm.calls


@pytest.mark.asyncio
async def test_graph_runtime_builds_evidence_from_vector_hits_without_embedded_evidence() -> None:
    planner_llm = StructuredPlannerLLM()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        ChartAgent(),
        GraphExtractionAgent(),
        RetrievalAgentWithVectorHitsOnly(),
        InsufficientAnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    qa = await graph_runtime.handle_ask_document("这份文档讲了什么？", doc_id="doc1")

    assert qa.answer != "证据不足"
    assert "项目目标：构建文档问答系统。" in qa.answer
    assert any(evidence.source_type == "text_block" for evidence in qa.evidence_bundle.evidences)


@pytest.mark.asyncio
async def test_graph_runtime_chart_path(graph_runtime: GraphRuntime) -> None:
    result = await graph_runtime.handle_understand_chart(
        image_path="/tmp/chart.png",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        chart_id="chart1",
        context={"extract_chart_graph": True},
    )

    assert result.chart.id == "chart1"


@pytest.mark.asyncio
async def test_graph_runtime_fused_ask_path(graph_runtime: GraphRuntime) -> None:
    result = await graph_runtime.handle_ask_fused(
        question="Does the chart agree with the document conclusion?",
        image_path="/tmp/chart.png",
        doc_id="doc1",
        document_ids=["doc1"],
        page_id="p1",
        page_number=1,
        chart_id="chart1",
        session_id="fused-session",
    )

    assert result.chart_answer == "chart answer"
    assert result.chart_confidence is not None
    assert result.qa.answer == "chart-aware answer"
    assert result.qa.metadata["fused"] is True
    assert any(evidence.source_type == "chart" for evidence in result.qa.evidence_bundle.evidences)
    assert any(evidence.source_type == "page_image" for evidence in result.qa.evidence_bundle.evidences)
    chart_evidence = next(evidence for evidence in result.qa.evidence_bundle.evidences if evidence.source_type == "chart")
    assert "Visual answer: chart answer" in chart_evidence.snippet
    assert "Title: Revenue by quarter" in chart_evidence.snippet
    assert "Caption: Figure 1. Quarterly revenue pipeline." in chart_evidence.snippet
    assert "legend: revenue" in chart_evidence.snippet
    assert "x_axis_categories: Q1, Q2, Q3, Q4" in chart_evidence.snippet
    assert graph_runtime.chart_tools.ask_calls
    assert graph_runtime.chart_tools.ocr_calls


@pytest.mark.asyncio
async def test_graph_runtime_compacts_chart_question_context() -> None:
    planner_llm = StructuredPlannerLLM()
    chart_agent = ChartAgent()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        chart_agent,
        GraphExtractionAgent(),
        RetrievalAgentWithEvidence(),
        RealAnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    oversized_metadata = {
        "research_task_id": "task-1",
        "research_topic": "topic-" + ("x" * 500),
        "selection_summary": "summary-" + ("y" * 500),
        "selected_paper_ids": [f"paper-{index}" for index in range(10)],
        "figure_context": {
            "title": "figure title",
            "caption": "caption-" + ("z" * 500),
            "extra": "ignored",
            "deep": {"nested": "value", "more": "content"},
        },
        "task_payload": {"report_markdown": "should not be forwarded"},
        "workspace": {"huge": "should not be forwarded"},
    }

    await graph_runtime.handle_ask_fused(
        question="Explain the system diagram.",
        image_path="/tmp/chart.png",
        doc_id="doc1",
        document_ids=["doc1"],
        page_id="p1",
        page_number=1,
        chart_id="chart1",
        metadata=oversized_metadata,
    )

    ask_context = chart_agent.ask_calls[-1]["context"]
    ocr_context = chart_agent.ocr_calls[-1]["context"]

    assert ask_context == ocr_context
    assert ask_context["research_task_id"] == "task-1"
    assert len(ask_context["research_topic"]) < len(oversized_metadata["research_topic"])
    assert len(ask_context["selection_summary"]) < len(oversized_metadata["selection_summary"])
    assert ask_context["selected_paper_ids"][-1] == "...(+4 more)"
    assert "task_payload" not in ask_context
    assert "workspace" not in ask_context
    assert ask_context["figure_context"]["caption"].endswith("...")


@pytest.mark.asyncio
async def test_graph_runtime_merges_visual_anchor_figure_into_chart_evidence() -> None:
    planner_llm = StructuredPlannerLLM()
    chart_agent = ChartAgent()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        chart_agent,
        GraphExtractionAgent(),
        RetrievalAgentWithEvidence(),
        RealAnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    result = await graph_runtime.handle_ask_fused(
        question="Explain the second paper system figure.",
        image_path="/tmp/chart.png",
        doc_id="doc1",
        document_ids=["doc1"],
        page_id="p2",
        page_number=2,
        chart_id="chart1",
        metadata={
            "visual_anchor_figure": {
                "figure_id": "paper1:fig2",
                "title": "Overall system architecture",
                "caption": "Figure 2. The system consists of encoder, planner, and executor stages.",
                "source": "page_fallback",
                "page_id": "p2",
                "page_number": 2,
            }
        },
    )

    chart_evidence = next(evidence for evidence in result.qa.evidence_bundle.evidences if evidence.source_type == "chart")
    assert "Figure title: Overall system architecture" in chart_evidence.snippet
    assert "Figure caption: Figure 2. The system consists of encoder, planner, and executor stages." in chart_evidence.snippet
    assert chart_evidence.metadata["figure_figure_id"] == "paper1:fig2"
    assert chart_evidence.metadata["figure_source"] == "page_fallback"


@pytest.mark.asyncio
async def test_graph_runtime_auto_routes_chart_question_to_fused_path() -> None:
    planner_llm = StructuredPlannerLLM()
    chart_agent = ChartAgent()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        chart_agent,
        GraphExtractionAgent(),
        RetrievalAgentWithVisualAnchor(),
        AnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    qa = await graph_runtime.handle_ask_document("What is the main focus of this graph?", doc_id="doc1")

    assert qa.answer == "chart-aware answer"
    assert qa.metadata["fused"] is True
    assert qa.metadata["auto_fused"] is True
    assert any(
        call["filters"].get("retrieval_mode") == "vector"
        for call in graph_runtime.retrieval_tools.calls
        if (call["filters"] or {}).get("modalities")
    )
    assert chart_agent.ask_calls


@pytest.mark.asyncio
async def test_graph_runtime_continues_when_graph_retrieval_fails() -> None:
    planner_llm = StructuredPlannerLLM()
    graph_runtime = GraphRuntime(
        DocumentAgent(),
        ChartAgent(),
        GraphExtractionAgent(),
        RetrievalAgentWithGraphFailure(),
        InsufficientAnswerAgent(llm_adapter=planner_llm),
        GraphIndexService(),
        EmbeddingIndexService(),
        llm_adapter=planner_llm,
    )

    qa = await graph_runtime.handle_ask_document("这份文档讲了什么？", doc_id="doc1")

    assert qa.answer != "证据不足"
    assert "项目目标：构建文档问答系统。" in qa.answer
    assert qa.metadata["runtime_engine"] == "tool_runtime"
