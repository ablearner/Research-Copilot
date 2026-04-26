import logging
from typing import Any

from pydantic import BaseModel, Field

from adapters.graph_store.base import BaseGraphStore, GraphStoreError
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode
from domain.schemas.graph_rag import GraphCommunityBuildResult, GraphSummaryBuildResult

logger = logging.getLogger(__name__)


class GraphIndexServiceError(RuntimeError):
    """Raised when graph indexing fails."""


class GraphIndexStats(BaseModel):
    document_id: str
    status: str
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    skipped_node_count: int = Field(default=0, ge=0)
    skipped_edge_count: int = Field(default=0, ge=0)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphIndexService:
    def __init__(self, graph_store: BaseGraphStore) -> None:
        self.graph_store = graph_store

    async def index_graph_result(self, graph_result: GraphExtractionResult) -> GraphIndexStats:
        if graph_result.status == "failed":
            logger.warning(
                "Skipping failed graph extraction result",
                extra={"document_id": graph_result.document_id},
            )
            return GraphIndexStats(
                document_id=graph_result.document_id,
                status="skipped",
                error_message=graph_result.error_message,
                metadata={"reason": "graph_extraction_failed"},
            )

        try:
            normalized_nodes = self.normalize_nodes(graph_result)
            normalized_edges = self.normalize_edges(graph_result, normalized_nodes)
            await self.graph_store.upsert_nodes(normalized_nodes)
            await self.graph_store.upsert_edges(normalized_edges)
            logger.info(
                "Indexed graph extraction result",
                extra={
                    "document_id": graph_result.document_id,
                    "node_count": len(normalized_nodes),
                    "edge_count": len(normalized_edges),
                },
            )
            return GraphIndexStats(
                document_id=graph_result.document_id,
                status="indexed",
                node_count=len(normalized_nodes),
                edge_count=len(normalized_edges),
                skipped_node_count=self._candidate_node_count(graph_result) - len(normalized_nodes),
                skipped_edge_count=self._candidate_edge_count(graph_result) - len(normalized_edges),
                metadata={"source_status": graph_result.status},
            )
        except GraphStoreError as exc:
            logger.exception(
                "Graph store failed while indexing graph result",
                extra={"document_id": graph_result.document_id},
            )
            raise GraphIndexServiceError("Graph store failed while indexing graph result") from exc
        except Exception as exc:
            logger.exception(
                "Unexpected graph indexing failure",
                extra={"document_id": graph_result.document_id},
            )
            raise GraphIndexServiceError("Unexpected graph indexing failure") from exc

    async def index_graph_communities(
        self,
        community_result: GraphCommunityBuildResult,
        summary_result: GraphSummaryBuildResult | None = None,
    ) -> GraphIndexStats:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for community in community_result.communities:
            nodes.extend(community.nodes)
            edges.extend(community.edges)

        try:
            if nodes:
                await self.graph_store.upsert_nodes(self._dedupe_nodes(nodes))
            if edges:
                await self.graph_store.upsert_edges(self._dedupe_edges(edges))
            return GraphIndexStats(
                document_id=community_result.document_id,
                status="indexed" if community_result.communities else "skipped",
                node_count=len(self._dedupe_nodes(nodes)),
                edge_count=len(self._dedupe_edges(edges)),
                metadata={
                    "community_count": len(community_result.communities),
                    "summary_count": len(summary_result.summaries) if summary_result else 0,
                    "index_kind": "graph_communities",
                },
            )
        except GraphStoreError as exc:
            logger.exception(
                "Graph store failed while indexing graph communities",
                extra={"document_id": community_result.document_id},
            )
            raise GraphIndexServiceError("Graph store failed while indexing graph communities") from exc

    def normalize_nodes(self, graph_result: GraphExtractionResult) -> list[GraphNode]:
        nodes: dict[str, GraphNode] = {}
        for node in graph_result.nodes:
            normalized = self._normalize_node(graph_result.document_id, node)
            nodes[normalized.id] = normalized
        for triple in graph_result.triples:
            subject = self._normalize_node(graph_result.document_id, triple.subject)
            obj = self._normalize_node(graph_result.document_id, triple.object)
            nodes[subject.id] = subject
            nodes[obj.id] = obj
        return list(nodes.values())

    def normalize_edges(
        self,
        graph_result: GraphExtractionResult,
        nodes: list[GraphNode] | None = None,
    ) -> list[GraphEdge]:
        node_ids = {node.id for node in nodes or self.normalize_nodes(graph_result)}
        edges: dict[str, GraphEdge] = {}
        for edge in graph_result.edges:
            normalized = self._normalize_edge(graph_result.document_id, edge)
            if normalized.source_node_id in node_ids and normalized.target_node_id in node_ids:
                edges[normalized.id] = normalized
        for triple in graph_result.triples:
            normalized = self._normalize_edge(graph_result.document_id, triple.predicate)
            if normalized.source_node_id in node_ids and normalized.target_node_id in node_ids:
                edges[normalized.id] = normalized
        return list(edges.values())

    def _normalize_node(self, document_id: str, node: GraphNode) -> GraphNode:
        properties = {
            **node.properties,
            "document_id": node.properties.get("document_id") or document_id,
        }
        source_reference = node.source_reference.model_copy(
            update={"document_id": node.source_reference.document_id or document_id}
        )
        return node.model_copy(
            update={
                "label": node.label.strip() or "Entity",
                "properties": properties,
                "source_reference": source_reference,
            }
        )

    def _normalize_edge(self, document_id: str, edge: GraphEdge) -> GraphEdge:
        properties = {
            **edge.properties,
            "document_id": edge.properties.get("document_id") or document_id,
        }
        source_reference = edge.source_reference.model_copy(
            update={"document_id": edge.source_reference.document_id or document_id}
        )
        return edge.model_copy(
            update={
                "type": self._normalize_edge_type(edge.type),
                "properties": properties,
                "source_reference": source_reference,
            }
        )

    def _normalize_edge_type(self, edge_type: str) -> str:
        normalized = "_".join(edge_type.strip().upper().split())
        return normalized or "RELATED_TO"

    def _candidate_node_count(self, graph_result: GraphExtractionResult) -> int:
        node_ids = {node.id for node in graph_result.nodes}
        for triple in graph_result.triples:
            node_ids.add(triple.subject.id)
            node_ids.add(triple.object.id)
        return len(node_ids)

    def _candidate_edge_count(self, graph_result: GraphExtractionResult) -> int:
        edge_ids = {edge.id for edge in graph_result.edges}
        for triple in graph_result.triples:
            edge_ids.add(triple.predicate.id)
        return len(edge_ids)

    def _dedupe_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        deduped = {node.id: node for node in nodes}
        return list(deduped.values())

    def _dedupe_edges(self, edges: list[GraphEdge]) -> list[GraphEdge]:
        deduped = {edge.id: edge for edge in edges}
        return list(deduped.values())
