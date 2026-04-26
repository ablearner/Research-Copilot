from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode, GraphTriple
from rag_runtime.services.graph_community_service import GraphCommunityService
from rag_runtime.services.graph_summary_service import GraphSummaryService


def test_graph_community_and_summary_services_build_structured_outputs() -> None:
    evidence = Evidence(id="ev1", document_id="doc1", source_type="text_block", source_id="tb1")
    subject = GraphNode(id="n1", label="Metric", properties={"name": "Revenue"}, source_reference=evidence)
    obj = GraphNode(id="n2", label="TimePeriod", properties={"name": "2025"}, source_reference=evidence)
    edge = GraphEdge(
        id="e1",
        type="OCCURS_IN",
        source_node_id="n1",
        target_node_id="n2",
        properties={"confidence": 0.8},
        source_reference=evidence,
    )
    graph_result = GraphExtractionResult(
        document_id="doc1",
        nodes=[subject, obj],
        edges=[edge],
        triples=[GraphTriple(subject=subject, predicate=edge, object=obj)],
    )

    communities = GraphCommunityService().build_communities(graph_result)
    summaries = GraphSummaryService().summarize_communities(communities)

    assert communities.communities
    assert summaries.summaries
    assert summaries.summaries[0].source_references
