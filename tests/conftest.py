from typing import Any

import pytest

from adapters.embedding.base import BaseEmbeddingAdapter
from adapters.graph_store.base import BaseGraphStore
from adapters.llm.base import BaseLLMAdapter
from adapters.vector_store.base import BaseVectorStore
from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock
from domain.schemas.embedding import EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.graph import GraphEdge, GraphNode, GraphQueryRequest, GraphQueryResult, GraphTriple
from domain.schemas.retrieval import RetrievalHit


@pytest.fixture
def sample_text_block() -> TextBlock:
    return TextBlock(
        id="tb1",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        text="Revenue increased in 2025.",
    )


@pytest.fixture
def sample_page(sample_text_block: TextBlock) -> DocumentPage:
    return DocumentPage(
        id="p1",
        document_id="doc1",
        page_number=1,
        image_uri="/tmp/page.png",
        text_blocks=[sample_text_block],
    )


@pytest.fixture
def sample_document(sample_page: DocumentPage) -> ParsedDocument:
    return ParsedDocument(
        id="doc1",
        filename="sample.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[sample_page],
    )


@pytest.fixture
def sample_chart() -> ChartSchema:
    return ChartSchema(
        id="chart1",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        chart_type="bar",
        title="Revenue",
        summary="Revenue increased in 2025.",
        confidence=0.8,
        metadata={"image_path": "/tmp/chart.png"},
    )


@pytest.fixture
def sample_evidence() -> Evidence:
    return Evidence(
        id="ev1",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        source_type="text_block",
        source_id="tb1",
        snippet="Revenue increased in 2025.",
    )


class MockLLMAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def _generate_structured(self, prompt: str, input_data: dict[str, Any], response_model: type):
        self.calls.append("generate_structured")
        if response_model is QAResponse:
            return QAResponse(
                answer="Revenue increased in 2025.",
                question=input_data["question"],
                evidence_bundle=EvidenceBundle.model_validate(input_data["evidence_bundle"]),
                confidence=0.8,
            )
        return response_model.model_validate(input_data)

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        self.calls.append("analyze_image_structured")
        return response_model(
            id="chart1",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_type="bar",
            title="Revenue",
            summary="Revenue increased in 2025.",
            confidence=0.8,
        )

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict[str, Any], response_model: type):
        self.calls.append("extract_graph_triples")
        ev = Evidence(
            id="ev1",
            document_id="doc1",
            source_type="text_block",
            source_id="tb1",
            snippet="Revenue increased in 2025.",
        )
        n1 = GraphNode(id="n1", label="Metric", properties={"name": "Revenue"}, source_reference=ev)
        n2 = GraphNode(id="n2", label="TimePeriod", properties={"name": "2025"}, source_reference=ev)
        edge = GraphEdge(
            id="e1",
            type="OCCURS_IN",
            source_node_id="n1",
            target_node_id="n2",
            properties={"confidence": 0.8},
            source_reference=ev,
        )
        return response_model(
            document_id=input_data["document_id"],
            nodes=[n1, n2],
            edges=[edge],
            triples=[GraphTriple(subject=n1, predicate=edge, object=n2)],
        )


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    async def _embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(model="mock-text", dimensions=2, values=[1.0, 0.0])

    async def _embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(model="mock-text", dimensions=2, values=[1.0, 0.0]) for _ in texts]

    async def _embed_image(self, image_path: str) -> EmbeddingVector:
        return EmbeddingVector(model="mock-mm", dimensions=2, values=[0.5, 0.5])

    async def _embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        return EmbeddingVector(model="mock-mm", dimensions=2, values=[0.2, 0.8])

    async def _embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        return EmbeddingVector(model="mock-mm", dimensions=2, values=[0.8, 0.2])


class MockVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: list[MultimodalEmbeddingRecord] = []
        self.last_filters: dict[str, Any] | None = None

    async def upsert_embedding(self, record: MultimodalEmbeddingRecord) -> None:
        self.records.append(record)

    async def upsert_embeddings(self, records: list[MultimodalEmbeddingRecord]) -> None:
        self.records.extend(records)

    async def search_by_vector(self, vector: EmbeddingVector, top_k: int, filters: dict[str, Any] | None = None) -> list[RetrievalHit]:
        self.last_filters = filters
        return [
            RetrievalHit(
                id="vh1",
                source_type="text_block",
                source_id="tb1",
                document_id="doc1",
                content="Revenue increased in 2025.",
                vector_score=0.9,
            )
        ][:top_k]

    async def search_similar_text(self, text: str, top_k: int) -> list[RetrievalHit]:
        return await self.search_by_vector(EmbeddingVector(model="mock", dimensions=2, values=[1, 0]), top_k)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        self.records = [record for record in self.records if record.item.document_id != doc_id]


class MockGraphStore(BaseGraphStore):
    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.last_query_request: GraphQueryRequest | None = None
        self.entity_search_calls: list[tuple[str, list[str]]] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        self.nodes.extend(nodes)

    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        self.edges.extend(edges)

    async def upsert_triples(self, triples: list[GraphTriple]) -> None:
        return None

    async def query_subgraph(self, query_request: GraphQueryRequest) -> GraphQueryResult:
        self.last_query_request = query_request
        ev = Evidence(id="gev1", document_id="doc1", source_type="graph_node", source_id="n1")
        n1 = GraphNode(id="n1", label="Metric", properties={"name": "Revenue"}, source_reference=ev)
        n2 = GraphNode(id="n2", label="TimePeriod", properties={"name": "2025"}, source_reference=ev)
        edge = GraphEdge(id="e1", type="OCCURS_IN", source_node_id="n1", target_node_id="n2", properties={}, source_reference=ev)
        return GraphQueryResult(
            query=query_request.query,
            nodes=[n1, n2],
            edges=[edge],
            triples=[GraphTriple(subject=n1, predicate=edge, object=n2)],
            evidences=[ev],
        )

    async def get_neighbors(self, node_id: str, depth: int) -> GraphQueryResult:
        return GraphQueryResult(query=node_id)

    async def search_entities(
        self,
        keyword: str,
        document_ids: list[str] | None = None,
    ) -> GraphQueryResult:
        resolved_document_ids = list(document_ids or [])
        self.entity_search_calls.append((keyword, resolved_document_ids))
        return await self.query_subgraph(
            GraphQueryRequest(query=keyword, document_ids=resolved_document_ids)
        )


@pytest.fixture
def mock_llm() -> MockLLMAdapter:
    return MockLLMAdapter()


@pytest.fixture
def mock_embedding() -> MockEmbeddingAdapter:
    return MockEmbeddingAdapter()


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture
def mock_graph_store() -> MockGraphStore:
    return MockGraphStore()
