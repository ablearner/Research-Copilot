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
    assert communities.strategy == "leiden"
    assert summaries.summaries
    assert summaries.summaries[0].source_references


def test_leiden_separates_disconnected_clusters() -> None:
    """Two disconnected triangles should produce two distinct communities."""
    ev = Evidence(id="ev1", document_id="doc1", source_type="text_block", source_id="tb1")

    # Cluster A: n1 -- n2 -- n3 -- n1
    nodes_a = [
        GraphNode(id="a1", label="Concept", properties={"name": "A1"}, source_reference=ev),
        GraphNode(id="a2", label="Concept", properties={"name": "A2"}, source_reference=ev),
        GraphNode(id="a3", label="Concept", properties={"name": "A3"}, source_reference=ev),
    ]
    edges_a = [
        GraphEdge(id="ea1", type="RELATED", source_node_id="a1", target_node_id="a2", source_reference=ev),
        GraphEdge(id="ea2", type="RELATED", source_node_id="a2", target_node_id="a3", source_reference=ev),
        GraphEdge(id="ea3", type="RELATED", source_node_id="a3", target_node_id="a1", source_reference=ev),
    ]

    # Cluster B: n4 -- n5 -- n6 -- n4
    nodes_b = [
        GraphNode(id="b1", label="Entity", properties={"name": "B1"}, source_reference=ev),
        GraphNode(id="b2", label="Entity", properties={"name": "B2"}, source_reference=ev),
        GraphNode(id="b3", label="Entity", properties={"name": "B3"}, source_reference=ev),
    ]
    edges_b = [
        GraphEdge(id="eb1", type="RELATED", source_node_id="b1", target_node_id="b2", source_reference=ev),
        GraphEdge(id="eb2", type="RELATED", source_node_id="b2", target_node_id="b3", source_reference=ev),
        GraphEdge(id="eb3", type="RELATED", source_node_id="b3", target_node_id="b1", source_reference=ev),
    ]

    graph_result = GraphExtractionResult(
        document_id="doc1",
        nodes=nodes_a + nodes_b,
        edges=edges_a + edges_b,
    )

    result = GraphCommunityService().build_communities(graph_result)

    assert result.strategy == "leiden"
    assert len(result.communities) == 2

    # Verify each community contains exactly one cluster
    community_node_sets = [set(c.node_ids) for c in result.communities]
    assert {"a1", "a2", "a3"} in community_node_sets
    assert {"b1", "b2", "b3"} in community_node_sets


def test_leiden_empty_graph_produces_no_communities() -> None:
    graph_result = GraphExtractionResult(document_id="doc1")
    result = GraphCommunityService().build_communities(graph_result)
    assert result.communities == []
    assert result.strategy == "leiden"


def test_leiden_isolated_nodes_get_singleton_communities() -> None:
    ev = Evidence(id="ev1", document_id="doc1", source_type="text_block", source_id="tb1")
    nodes = [
        GraphNode(id="iso1", label="Concept", properties={"name": "Lonely1"}, source_reference=ev),
        GraphNode(id="iso2", label="Concept", properties={"name": "Lonely2"}, source_reference=ev),
    ]
    graph_result = GraphExtractionResult(document_id="doc1", nodes=nodes)

    result = GraphCommunityService().build_communities(graph_result)
    assert len(result.communities) == 2
    all_node_ids = {nid for c in result.communities for nid in c.node_ids}
    assert all_node_ids == {"iso1", "iso2"}
