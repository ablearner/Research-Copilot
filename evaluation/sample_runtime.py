from __future__ import annotations

import asyncio
from pathlib import Path

from domain.schemas.api import QAResponse
from domain.schemas.chart import AxisSchema, ChartSchema, SeriesPoint, SeriesSchema
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock
from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from rag_runtime.runtime import RagRuntime
from retrieval.evidence_builder import build_evidence_bundle
from tools.retrieval_toolkit import RetrievalAgentResult


DOCUMENT_ID = "portfolio-doc-001"
PAGE_ID = "page-1"
CHART_ID = "chart-portfolio-1"
IMAGE_PATH = "synthetic-chart.png"


def build_sample_runtime() -> RagRuntime:
    document = sample_document()
    chart = sample_chart()
    return RagRuntime(
        document_tools=SampleDocumentAgent(document),
        chart_tools=SampleChartAgent(chart),
        graph_extraction_tools=SampleGraphExtractionAgent(),
        retrieval_tools=SampleRetrievalAgent(document=document, chart=chart),
        answer_tools=SampleAnswerAgent(chart=chart),
        graph_index_service=SampleGraphIndexService(),
        embedding_index_service=SampleEmbeddingIndexService(),
        llm_adapter=None,
    )


def sample_document() -> ParsedDocument:
    blocks = [
        TextBlock(
            id="tb-1",
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=1,
            text="In 2025 Q1, revenue increased by 18% year over year while operating margin improved to 24%.",
            block_type="paragraph",
        ),
        TextBlock(
            id="tb-2",
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=1,
            text="Management attributed the improvement to cloud subscriptions and stronger enterprise retention.",
            block_type="paragraph",
        ),
        TextBlock(
            id="tb-3",
            document_id=DOCUMENT_ID,
            page_id=PAGE_ID,
            page_number=1,
            text="A companion chart on the same page shows year-over-year revenue growth reaching 18% in 2025 Q1.",
            block_type="caption",
        ),
    ]
    page = DocumentPage(
        id=PAGE_ID,
        document_id=DOCUMENT_ID,
        page_number=1,
        image_uri=IMAGE_PATH,
        text_blocks=blocks,
        metadata={},
    )
    return ParsedDocument(
        id=DOCUMENT_ID,
        filename="quarterly_report.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[page],
        metadata={},
    )


def sample_chart() -> ChartSchema:
    return ChartSchema(
        id=CHART_ID,
        document_id=DOCUMENT_ID,
        page_id=PAGE_ID,
        page_number=1,
        chart_type="bar",
        title="Quarterly Revenue Growth",
        caption="Year-over-year revenue growth by quarter.",
        x_axis=AxisSchema(label="Quarter", categories=["2025 Q1"]),
        y_axis=AxisSchema(label="Growth", unit="%"),
        series=[
            SeriesSchema(
                name="Revenue Growth",
                chart_role="bar",
                points=[SeriesPoint(label="2025 Q1", value=18)],
            )
        ],
        summary="The chart shows year-over-year revenue growth reaching 18% in 2025 Q1.",
        confidence=0.92,
        metadata={"image_path": IMAGE_PATH, "image_uri": IMAGE_PATH},
    )


class SampleDocumentAgent:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document

    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        return self.document.model_copy(update={"id": document_id or self.document.id, "filename": Path(file_path).name})

    async def summarize_page(self, page: DocumentPage):
        from tools.document_toolkit import PageSummary

        joined = " ".join(block.text for block in page.text_blocks)
        return PageSummary(
            document_id=page.document_id,
            page_id=page.id,
            page_number=page.page_number,
            summary=joined[:240],
        )


class SampleChartAgent:
    def __init__(self, chart: ChartSchema) -> None:
        self.chart = chart

    async def parse_chart(self, image_path: str, document_id: str, page_id: str, page_number: int, chart_id: str, context=None) -> ChartSchema:
        await asyncio.sleep(0.002)
        return self.chart.model_copy(
            update={
                "id": chart_id,
                "document_id": document_id,
                "page_id": page_id,
                "page_number": page_number,
                "metadata": {**self.chart.metadata, "image_path": image_path},
            }
        )

    def to_graph_text(self, chart: ChartSchema) -> str:
        return (
            f"title: {chart.title}\n"
            f"chart_type: {chart.chart_type}\n"
            f"summary: {chart.summary}\n"
            f"series: Revenue Growth=18%"
        )

    def explain_chart(self, chart: ChartSchema) -> str:
        return (
            f"Chart type: {chart.chart_type}.\n"
            f"Title: {chart.title}.\n"
            f"Summary: {chart.summary}"
        )

    async def extract_visible_text(self, image_path: str, context=None, chart=None) -> str:
        await asyncio.sleep(0.001)
        return "title: Quarterly Revenue Growth\nx_axis: Quarter\ny_axis: Growth (%)\nlabel: 2025 Q1 = 18%"

    async def ask_chart(self, image_path: str, question: str, context=None, history=None) -> str:
        await asyncio.sleep(0.002)
        return "The chart shows year-over-year revenue growth reaching 18% in 2025 Q1."


class SampleGraphExtractionAgent:
    async def extract_from_text_blocks(self, document_id: str, text_blocks: list[TextBlock], page_summaries=None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=document_id, status="succeeded")

    async def extract_from_chart(self, chart: ChartSchema, chart_summary: str | None = None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=chart.document_id, status="succeeded")

    def merge_graph_candidates(self, document_id: str, candidates: list[GraphExtractionResult]) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=document_id, status="succeeded")


class SampleRetrievalAgent:
    def __init__(self, *, document: ParsedDocument, chart: ChartSchema) -> None:
        self.document = document
        self.chart = chart

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
        skill_context=None,
    ) -> RetrievalAgentResult:
        await asyncio.sleep(0.002)
        filters = filters or {}
        retrieval_mode = filters.get("retrieval_mode", "hybrid")
        modalities = set(filters.get("modalities") or [])
        resolved_document_ids = list(document_ids or ([doc_id] if doc_id else [self.document.id]))
        hits = self._select_hits(
            question=question,
            retrieval_mode=retrieval_mode,
            modalities=modalities,
        )[:top_k]
        evidence_bundle = build_evidence_bundle(hits)
        retrieval_result = HybridRetrievalResult(
            query=RetrievalQuery(
                query=question,
                document_ids=resolved_document_ids,
                mode="hybrid",
                top_k=top_k,
                filters=filters,
            ),
            hits=hits,
            evidence_bundle=evidence_bundle,
            metadata={"retrieval_mode": retrieval_mode, "skill_name": skill_context.get("name") if isinstance(skill_context, dict) else None},
        )
        return RetrievalAgentResult(
            question=question,
            document_ids=resolved_document_ids,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
        )

    def _select_hits(self, *, question: str, retrieval_mode: str, modalities: set[str]) -> list[RetrievalHit]:
        lowered = question.lower()
        if "attrition" in lowered:
            return []
        if retrieval_mode == "vector" and modalities & {"chart", "page"}:
            return [self._chart_hit(), self._page_hit()]
        if retrieval_mode == "vector":
            hits = [self._text_hit("tb-1"), self._text_hit("tb-2")]
            if "chart" in lowered or "graph" in lowered or "figure" in lowered:
                hits.insert(0, self._chart_hit())
            return hits
        if retrieval_mode == "graph":
            return [self._graph_edge_hit()]
        if retrieval_mode == "graphrag_summary":
            return [self._graph_summary_hit()]
        return [self._text_hit("tb-1"), self._text_hit("tb-2"), self._graph_edge_hit(), self._graph_summary_hit()]

    def _text_hit(self, block_id: str) -> RetrievalHit:
        block = next(block for block in self.document.pages[0].text_blocks if block.id == block_id)
        evidence = Evidence(
            id=f"ev_{block.id}",
            document_id=block.document_id,
            page_id=block.page_id,
            page_number=block.page_number,
            source_type="text_block",
            source_id=block.id,
            snippet=block.text,
            score=0.88,
        )
        return RetrievalHit(
            id=f"hit_{block.id}",
            source_type="text_block",
            source_id=block.id,
            document_id=block.document_id,
            content=block.text,
            vector_score=0.89,
            merged_score=0.89,
            evidence=EvidenceBundle(evidences=[evidence]),
            metadata={"page_id": block.page_id, "page_number": block.page_number},
        )

    def _chart_hit(self) -> RetrievalHit:
        evidence = Evidence(
            id="ev_chart",
            document_id=self.chart.document_id,
            page_id=self.chart.page_id,
            page_number=self.chart.page_number,
            source_type="chart",
            source_id=self.chart.id,
            snippet=self.chart.summary,
            score=0.91,
            metadata={"image_path": IMAGE_PATH},
        )
        return RetrievalHit(
            id="hit_chart",
            source_type="chart",
            source_id=self.chart.id,
            document_id=self.chart.document_id,
            content=self.chart.summary,
            vector_score=0.91,
            merged_score=0.91,
            evidence=EvidenceBundle(evidences=[evidence]),
            metadata={"uri": IMAGE_PATH, "page_id": self.chart.page_id, "page_number": self.chart.page_number},
        )

    def _page_hit(self) -> RetrievalHit:
        evidence = Evidence(
            id="ev_page_image",
            document_id=self.document.id,
            page_id=PAGE_ID,
            page_number=1,
            source_type="page_image",
            source_id=PAGE_ID,
            snippet="Page image contains the quarterly revenue growth chart.",
            score=0.75,
            metadata={"image_path": IMAGE_PATH},
        )
        return RetrievalHit(
            id="hit_page_image",
            source_type="page",
            source_id=PAGE_ID,
            document_id=self.document.id,
            content="Page image contains the quarterly revenue growth chart.",
            vector_score=0.74,
            merged_score=0.74,
            evidence=EvidenceBundle(evidences=[evidence]),
            metadata={"uri": IMAGE_PATH, "page_id": PAGE_ID, "page_number": 1},
        )

    def _graph_edge_hit(self) -> RetrievalHit:
        evidence = Evidence(
            id="ev_graph_margin",
            document_id=self.document.id,
            page_id=PAGE_ID,
            page_number=1,
            source_type="graph_edge",
            source_id="edge-margin",
            snippet="Operating margin improved because of cloud subscriptions and stronger enterprise retention.",
            score=0.81,
        )
        source_node = GraphNode(
            id="node_revenue",
            label="Metric",
            properties={"name": "revenue growth"},
            source_reference=evidence,
        )
        target_node = GraphNode(
            id="node_margin",
            label="Metric",
            properties={"name": "operating margin"},
            source_reference=evidence,
        )
        edge = GraphEdge(
            id="edge-margin",
            type="EXPLAINED_BY",
            source_node_id=source_node.id,
            target_node_id=target_node.id,
            properties={"reason": "cloud subscriptions and retention"},
            source_reference=evidence,
        )
        return RetrievalHit(
            id="hit_graph_margin",
            source_type="graph_edge",
            source_id=edge.id,
            document_id=self.document.id,
            content="Revenue growth and margin improvement are explained by cloud subscriptions and stronger enterprise retention.",
            graph_score=0.82,
            merged_score=0.82,
            graph_nodes=[source_node, target_node],
            graph_edges=[edge],
            evidence=EvidenceBundle(evidences=[evidence]),
            metadata={"edge_type": edge.type},
        )

    def _graph_summary_hit(self) -> RetrievalHit:
        evidence = Evidence(
            id="ev_graph_summary",
            document_id=self.document.id,
            page_id=PAGE_ID,
            page_number=1,
            source_type="graph_edge",
            source_id="summary-1",
            snippet="Topic Performance: revenue increased 18% in 2025 Q1 and operating margin improved to 24%.",
            score=0.79,
        )
        return RetrievalHit(
            id="hit_graph_summary",
            source_type="graph_summary",
            source_id="summary-1",
            document_id=self.document.id,
            content="Topic Performance: revenue increased 18% in 2025 Q1 and operating margin improved to 24%.",
            graph_score=0.79,
            merged_score=0.79,
            evidence=EvidenceBundle(evidences=[evidence]),
            metadata={"topic": "Performance"},
        )


class SampleAnswerAgent:
    def __init__(self, *, chart: ChartSchema) -> None:
        self.chart = chart

    async def answer(
        self,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result=None,
        metadata=None,
        session_context=None,
        task_context=None,
        preference_context=None,
        retrieval_cache_summary=None,
        memory_hints=None,
        skill_context=None,
    ) -> QAResponse:
        await asyncio.sleep(0.002)
        lowered = question.lower()
        if not evidence_bundle.evidences:
            return QAResponse(
                answer="证据不足",
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=0.0,
                metadata={"answered_by": "SampleAnswerAgent"},
            )
        if "why" in lowered or "margin" in lowered:
            answer = "Operating margin improved because of cloud subscriptions and stronger enterprise retention."
        elif "chart" in lowered or "graph" in lowered or "figure" in lowered:
            chart_answer = (task_context or {}).get("chart_answer") or self.chart.summary
            answer = (
                f"{chart_answer} "
                "The document also states that revenue increased by 18% in 2025 Q1."
            )
        else:
            answer = "Revenue increased by 18% in 2025 Q1 while operating margin improved to 24%."
        return QAResponse(
            answer=answer,
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.84 if "证据不足" not in answer else 0.0,
            metadata={"answered_by": "SampleAnswerAgent"},
        )


class SampleGraphIndexService:
    async def index_graph_result(self, graph_result: GraphExtractionResult):
        from rag_runtime.services.graph_index_service import GraphIndexStats

        return GraphIndexStats(document_id=graph_result.document_id, status="indexed")


class SampleEmbeddingIndexService:
    async def index_text_blocks(self, document_id: str, text_blocks: list[TextBlock]):
        from rag_runtime.services.embedding_index_service import EmbeddingIndexResult

        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(text_blocks))

    async def index_pages(self, document_id: str, pages: list[DocumentPage]):
        from rag_runtime.services.embedding_index_service import EmbeddingIndexResult

        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(pages))

    async def index_charts(self, document_id: str, charts: list[ChartSchema]):
        from rag_runtime.services.embedding_index_service import EmbeddingIndexResult

        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(charts))
