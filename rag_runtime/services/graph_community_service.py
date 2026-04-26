import hashlib
import logging
from collections import defaultdict

from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode
from domain.schemas.graph_rag import GraphCommunity, GraphCommunityBuildResult

logger = logging.getLogger(__name__)


class GraphCommunityService:
    def build_communities(
        self,
        graph_result: GraphExtractionResult,
        strategy: str = "label_topic",
        min_nodes: int = 1,
    ) -> GraphCommunityBuildResult:
        nodes = {node.id: node for node in graph_result.nodes}
        for triple in graph_result.triples:
            nodes.setdefault(triple.subject.id, triple.subject)
            nodes.setdefault(triple.object.id, triple.object)

        edges = {edge.id: edge for edge in graph_result.edges}
        for triple in graph_result.triples:
            edges.setdefault(triple.predicate.id, triple.predicate)

        buckets: dict[str, set[str]] = defaultdict(set)
        for node in nodes.values():
            buckets[self._topic_for_node(node, strategy)].add(node.id)

        communities: list[GraphCommunity] = []
        for topic, node_ids in buckets.items():
            if len(node_ids) < min_nodes:
                continue
            community_edges = [
                edge
                for edge in edges.values()
                if edge.source_node_id in node_ids or edge.target_node_id in node_ids
            ]
            community_triples = [
                triple
                for triple in graph_result.triples
                if triple.subject.id in node_ids or triple.object.id in node_ids
            ]
            community_nodes = [nodes[node_id] for node_id in sorted(node_ids)]
            communities.append(
                GraphCommunity(
                    id=self._community_id(graph_result.document_id, topic),
                    document_id=graph_result.document_id,
                    topic=topic,
                    node_ids=[node.id for node in community_nodes],
                    edge_ids=[edge.id for edge in community_edges],
                    nodes=community_nodes,
                    edges=community_edges,
                    triples=community_triples,
                    source_references=self._dedupe_evidence(
                        [node.source_reference for node in community_nodes]
                        + [edge.source_reference for edge in community_edges]
                    ),
                    confidence=self._community_confidence(community_nodes, community_edges),
                    metadata={"strategy": strategy},
                )
            )

        logger.info(
            "Built graph communities",
            extra={"document_id": graph_result.document_id, "community_count": len(communities)},
        )
        return GraphCommunityBuildResult(
            document_id=graph_result.document_id,
            communities=communities,
            strategy="label_topic" if strategy not in {"source_topic", "simple"} else strategy,
            metadata={"node_count": len(nodes), "edge_count": len(edges), "min_nodes": min_nodes},
        )

    def _topic_for_node(self, node: GraphNode, strategy: str) -> str:
        if strategy == "simple":
            return "document_graph"
        if strategy == "source_topic":
            return node.source_reference.source_type
        return node.label or "Entity"

    def _community_id(self, document_id: str, topic: str) -> str:
        digest = hashlib.sha256(f"{document_id}:{topic}".encode("utf-8")).hexdigest()[:16]
        return f"gcomm_{digest}"

    def _community_confidence(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> float | None:
        values = []
        for edge in edges:
            confidence = edge.properties.get("confidence")
            if isinstance(confidence, int | float):
                values.append(float(confidence))
        if values:
            return sum(values) / len(values)
        return 0.6 if nodes else None

    def _dedupe_evidence(self, evidences: list[Evidence]) -> list[Evidence]:
        deduped = {}
        for evidence in evidences:
            deduped[evidence.id] = evidence
        return list(deduped.values())
