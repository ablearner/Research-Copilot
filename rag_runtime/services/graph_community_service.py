import hashlib
import logging
from collections import defaultdict

import networkx as nx
from graspologic.partition import hierarchical_leiden

from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode
from domain.schemas.graph_rag import GraphCommunity, GraphCommunityBuildResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CLUSTER_SIZE = 10


class GraphCommunityService:
    """Build graph communities using the Leiden algorithm (standard GraphRAG)."""

    def build_communities(
        self,
        graph_result: GraphExtractionResult,
        *,
        max_cluster_size: int = _DEFAULT_MAX_CLUSTER_SIZE,
        min_nodes: int = 1,
    ) -> GraphCommunityBuildResult:
        nodes = {node.id: node for node in graph_result.nodes}
        for triple in graph_result.triples:
            nodes.setdefault(triple.subject.id, triple.subject)
            nodes.setdefault(triple.object.id, triple.object)

        edges = {edge.id: edge for edge in graph_result.edges}
        for triple in graph_result.triples:
            edges.setdefault(triple.predicate.id, triple.predicate)

        # Build networkx graph for Leiden
        G = nx.Graph()
        G.add_nodes_from(nodes.keys())
        for edge in edges.values():
            if edge.source_node_id in nodes and edge.target_node_id in nodes:
                G.add_edge(edge.source_node_id, edge.target_node_id)

        # Run Leiden community detection
        cluster_map = self._run_leiden(G, max_cluster_size=max_cluster_size)

        # Group node ids by cluster
        buckets: dict[int, set[str]] = defaultdict(set)
        for node_id, cluster_id in cluster_map.items():
            buckets[cluster_id].add(node_id)

        communities: list[GraphCommunity] = []
        for cluster_id, member_ids in sorted(buckets.items()):
            if len(member_ids) < min_nodes:
                continue
            community_nodes = [nodes[nid] for nid in sorted(member_ids)]
            community_edges = [
                edge
                for edge in edges.values()
                if edge.source_node_id in member_ids or edge.target_node_id in member_ids
            ]
            community_triples = [
                triple
                for triple in graph_result.triples
                if triple.subject.id in member_ids or triple.object.id in member_ids
            ]
            topic = self._infer_topic(community_nodes)
            communities.append(
                GraphCommunity(
                    id=self._community_id(graph_result.document_id, str(cluster_id)),
                    document_id=graph_result.document_id,
                    topic=topic,
                    node_ids=[n.id for n in community_nodes],
                    edge_ids=[e.id for e in community_edges],
                    nodes=community_nodes,
                    edges=community_edges,
                    triples=community_triples,
                    source_references=self._dedupe_evidence(
                        [n.source_reference for n in community_nodes]
                        + [e.source_reference for e in community_edges]
                    ),
                    confidence=self._community_confidence(community_nodes, community_edges),
                    metadata={"strategy": "leiden", "cluster_id": cluster_id},
                )
            )

        logger.info(
            "Built graph communities (Leiden)",
            extra={"document_id": graph_result.document_id, "community_count": len(communities)},
        )
        return GraphCommunityBuildResult(
            document_id=graph_result.document_id,
            communities=communities,
            strategy="leiden",
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "min_nodes": min_nodes,
                "max_cluster_size": max_cluster_size,
            },
        )

    # -- Leiden wrapper -------------------------------------------------------

    @staticmethod
    def _run_leiden(G: nx.Graph, *, max_cluster_size: int) -> dict[str, int]:
        """Run hierarchical Leiden and return {node_id: cluster_id}."""
        if G.number_of_nodes() == 0:
            return {}
        # Isolated nodes get their own singleton clusters
        if G.number_of_edges() == 0:
            return {node_id: idx for idx, node_id in enumerate(G.nodes())}
        results = hierarchical_leiden(G, max_cluster_size=max_cluster_size)
        # Use the finest-level (is_final_cluster=True) assignments
        cluster_map: dict[str, int] = {}
        for hc in results:
            if hc.is_final_cluster:
                cluster_map[str(hc.node)] = hc.cluster
        # Assign any missing nodes (isolated) to unique clusters
        next_cluster = max(cluster_map.values(), default=-1) + 1
        for node_id in G.nodes():
            if str(node_id) not in cluster_map:
                cluster_map[str(node_id)] = next_cluster
                next_cluster += 1
        return cluster_map

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _infer_topic(community_nodes: list[GraphNode]) -> str:
        """Derive a topic label from the most common node label in the community."""
        label_counts: dict[str, int] = defaultdict(int)
        for node in community_nodes:
            label_counts[node.label or "Entity"] += 1
        if not label_counts:
            return "Entity"
        return max(label_counts, key=label_counts.get)  # type: ignore[arg-type]

    @staticmethod
    def _community_id(document_id: str, cluster_key: str) -> str:
        digest = hashlib.sha256(f"{document_id}:{cluster_key}".encode("utf-8")).hexdigest()[:16]
        return f"gcomm_{digest}"

    @staticmethod
    def _community_confidence(nodes: list[GraphNode], edges: list[GraphEdge]) -> float | None:
        values = []
        for edge in edges:
            confidence = edge.properties.get("confidence")
            if isinstance(confidence, int | float):
                values.append(float(confidence))
        if values:
            return sum(values) / len(values)
        return 0.6 if nodes else None

    @staticmethod
    def _dedupe_evidence(evidences: list[Evidence]) -> list[Evidence]:
        deduped = {}
        for evidence in evidences:
            deduped[evidence.id] = evidence
        return list(deduped.values())
