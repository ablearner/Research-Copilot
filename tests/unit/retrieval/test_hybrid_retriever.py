import pytest

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery
from retrieval.cross_encoder import BaseCrossEncoderReranker
from retrieval.graph_retriever import GraphRetriever
from retrieval.graph_summary_retriever import GraphSummaryRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from domain.schemas.evidence import Evidence
from domain.schemas.graph_rag import GraphCommunitySummary


class _MockReranker(BaseCrossEncoderReranker):
    async def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [float(len(document)) for document in documents]


class _PriorityReranker(BaseCrossEncoderReranker):
    async def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        scores: list[float] = []
        for document in documents:
            score = 0.0
            if "shared" in document:
                score += 5.0
            if "graph" in document:
                score += 3.0
            if "vector" in document:
                score += 2.0
            if "sparse" in document:
                score += 1.0
            scores.append(score)
        return scores


class _NegativeScoreReranker(BaseCrossEncoderReranker):
    async def score(self, query: str, documents: list[str]) -> list[float]:
        del query, documents
        return [-5.82]


@pytest.mark.asyncio
async def test_hybrid_retriever_merges_and_builds_evidence(
    mock_graph_store,
    mock_embedding,
    mock_vector_store,
) -> None:
    retriever = HybridRetriever(
        graph_retriever=GraphRetriever(mock_graph_store),
        vector_retriever=VectorRetriever(mock_embedding, mock_vector_store),
        reranker=_MockReranker(),
    )

    result = await retriever.retrieve("Revenue in 2025?")

    assert result.hits
    assert result.evidence_bundle.evidences
    assert result.hits[0].merged_score is not None


@pytest.mark.asyncio
async def test_hybrid_retriever_can_merge_graph_summary_hits(
    mock_graph_store,
    mock_embedding,
    mock_vector_store,
) -> None:
    summary = GraphCommunitySummary(
        id="s1",
        community_id="c1",
        document_id="doc1",
        topic="Metric",
        summary="Topic Metric: Revenue OCCURS_IN 2025",
        node_ids=["n1"],
        edge_ids=["e1"],
        source_references=[
            Evidence(id="ev-summary", document_id="doc1", source_type="graph_edge", source_id="e1")
        ],
    )
    retriever = HybridRetriever(
        graph_retriever=GraphRetriever(mock_graph_store),
        vector_retriever=VectorRetriever(mock_embedding, mock_vector_store),
        graph_summary_retriever=GraphSummaryRetriever([summary]),
        reranker=_MockReranker(),
    )

    result = await retriever.retrieve("Revenue 2025")

    assert any(hit.source_type == "graph_summary" for hit in result.hits)
    assert result.metadata["summary_hit_count"] == 1


class _FailingGraphRetriever:
    async def ainvoke(self, query):
        raise RuntimeError("graph unavailable")


class _PassingVectorRetriever:
    async def ainvoke(self, query):
        return [
            RetrievalHit(
                id="vh1",
                source_type="text_block",
                source_id="tb1",
                document_id="doc1",
                content="Revenue increased in 2025.",
                vector_score=0.9,
                evidence=EvidenceBundle(
                    evidences=[
                        Evidence(
                            id="ev1",
                            document_id="doc1",
                            source_type="text_block",
                            source_id="tb1",
                            snippet="Revenue increased in 2025.",
                        )
                    ]
                ),
            )
        ]


class _CallTrackingGraphRetriever:
    def __init__(self) -> None:
        self.called = False

    async def ainvoke(self, query):
        self.called = True
        return []


class _StaticSparseRetriever:
    def __init__(self, hits):
        self._hits = hits

    async def ainvoke(self, query):
        return list(self._hits)


class _StaticVectorRetriever:
    def __init__(self, hits):
        self._hits = hits

    async def ainvoke(self, query):
        return list(self._hits)


class _StaticGraphRetriever:
    def __init__(self, hits):
        self._hits = hits

    async def ainvoke(self, query):
        return list(self._hits)


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_vector_mode_without_querying_graph(
    mock_embedding,
    mock_vector_store,
) -> None:
    graph_retriever = _CallTrackingGraphRetriever()
    retriever = HybridRetriever(
        graph_retriever=graph_retriever,
        vector_retriever=VectorRetriever(mock_embedding, mock_vector_store),
        reranker=_MockReranker(),
    )

    result = await retriever.retrieve(RetrievalQuery(query="Revenue", filters={"retrieval_mode": "vector"}))

    assert result.metadata["requested_retrieval_mode"] == "vector"
    assert graph_retriever.called is False
    assert result.metadata["vector_hit_count"] == 1
    assert result.metadata["graph_hit_count"] == 0


@pytest.mark.asyncio
async def test_hybrid_retriever_degrades_when_graph_fails_but_vector_succeeds() -> None:
    retriever = HybridRetriever(
        graph_retriever=_FailingGraphRetriever(),
        vector_retriever=_PassingVectorRetriever(),
        reranker=_MockReranker(),
    )

    result = await retriever.retrieve("Revenue in 2025?")

    assert result.hits
    assert result.metadata["partial_failure"] is True
    assert result.metadata["failed_sources"] == ["graph"]
    assert result.metadata["vector_hit_count"] == 1


@pytest.mark.asyncio
async def test_hybrid_retriever_uses_cross_encoder_to_promote_high_relevance_hits() -> None:
    shared_hit_sparse = RetrievalHit(
        id="shared-sparse",
        source_type="text_block",
        source_id="block-shared",
        document_id="doc-shared",
        content="shared hit from sparse",
        sparse_score=0.9,
    )
    sparse_only = RetrievalHit(
        id="sparse-only",
        source_type="text_block",
        source_id="block-sparse-only",
        document_id="doc-sparse-only",
        content="sparse only hit",
        sparse_score=0.85,
    )
    vector_only = RetrievalHit(
        id="vector-only",
        source_type="text_block",
        source_id="block-vector-only",
        document_id="doc-vector-only",
        content="vector only hit",
        vector_score=0.95,
    )
    shared_hit_vector = RetrievalHit(
        id="shared-vector",
        source_type="text_block",
        source_id="block-shared",
        document_id="doc-shared",
        content="shared hit from vector",
        vector_score=0.88,
    )
    shared_hit_graph = RetrievalHit(
        id="shared-graph",
        source_type="text_block",
        source_id="block-shared",
        document_id="doc-shared",
        content="shared hit from graph",
        graph_score=1.0,
    )

    retriever = HybridRetriever(
        graph_retriever=_StaticGraphRetriever([shared_hit_graph]),
        vector_retriever=_StaticVectorRetriever([vector_only, shared_hit_vector]),
        sparse_retriever=_StaticSparseRetriever([shared_hit_sparse, sparse_only]),
        reranker=_PriorityReranker(),
    )

    result = await retriever.retrieve(RetrievalQuery(query="shared result", top_k=3))

    assert result.metadata["fusion_strategy"] == "rrf"
    assert result.metadata["rerank_strategy"] == "cross_encoder"
    assert result.hits[0].document_id == "doc-shared"
    assert result.hits[0].metadata["retrieval_sources"] == ["graph", "sparse", "vector"]
    assert result.hits[0].metadata["rrf_score"] > result.hits[1].metadata["rrf_score"]
    assert result.hits[0].merged_score > result.hits[1].merged_score


@pytest.mark.asyncio
async def test_hybrid_retriever_clamps_negative_rerank_scores_in_evidence() -> None:
    vector_hit = RetrievalHit(
        id="vector-negative-score",
        source_type="text_block",
        source_id="block-negative",
        document_id="doc-negative",
        content="low relevance hit",
        vector_score=0.4,
    )

    retriever = HybridRetriever(
        graph_retriever=_StaticGraphRetriever([]),
        vector_retriever=_StaticVectorRetriever([vector_hit]),
        sparse_retriever=_StaticSparseRetriever([]),
        reranker=_NegativeScoreReranker(),
    )

    result = await retriever.retrieve(RetrievalQuery(query="irrelevant query", top_k=1))

    assert result.hits[0].merged_score == -5.82
    assert result.evidence_bundle.evidences[0].score == 0.0
    assert result.evidence_bundle.evidences[0].metadata["raw_retrieval_score"] == -5.82
