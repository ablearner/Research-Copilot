import pytest

from domain.schemas.evidence import Evidence
from domain.schemas.graph_rag import GraphCommunitySummary
from retrieval.graph_summary_retriever import GraphSummaryRetriever


@pytest.mark.asyncio
async def test_graph_summary_retriever_returns_summary_hits() -> None:
    summary = GraphCommunitySummary(
        id="s1",
        community_id="c1",
        document_id="doc1",
        topic="Metric",
        summary="Topic Metric: Revenue OCCURS_IN 2025",
        node_ids=["n1", "n2"],
        edge_ids=["e1"],
        source_references=[
            Evidence(id="ev1", document_id="doc1", source_type="graph_node", source_id="n1")
        ],
    )

    hits = await GraphSummaryRetriever([summary]).retrieve("Revenue 2025")

    assert hits
    assert hits[0].source_type == "graph_summary"
    assert hits[0].evidence.evidences[0].id == "ev1"
