import json
import logging
import re
from typing import Any

from adapters.graph_store.base import BaseGraphStore, GraphStoreError
from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphNode, GraphQueryRequest, GraphQueryResult, GraphTriple

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"[^A-Za-z0-9_]")


class Neo4jGraphStore(BaseGraphStore):
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str | None = None,
        driver: Any | None = None,
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = driver
        self._owns_driver = driver is None

    async def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j graph store")
        except Exception as exc:
            logger.exception("Failed to connect to Neo4j graph store")
            raise GraphStoreError("Failed to connect to Neo4j graph store") from exc

    async def close(self) -> None:
        if self._driver is None or not self._owns_driver:
            return
        try:
            await self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j graph store connection")
        except Exception as exc:
            logger.exception("Failed to close Neo4j graph store connection")
            raise GraphStoreError("Failed to close Neo4j graph store connection") from exc

    async def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        if not nodes:
            return
        try:
            await self._ensure_connected()
            async with self._session() as session:
                for node in nodes:
                    label = self._safe_identifier(node.label, default="Entity")
                    await session.run(
                        f"""
                        MERGE (n:GraphNode {{id: $id}})
                        SET n:{label},
                            n.label = $label,
                            n.document_id = $document_id,
                            n.source_reference_json = $source_reference_json,
                            n.properties_json = $properties_json
                        SET n += $properties
                        """,
                        id=node.id,
                        label=node.label,
                        document_id=node.source_reference.document_id
                        or node.properties.get("document_id"),
                        properties=self._neo4j_properties(node.properties),
                        properties_json=json.dumps(node.properties, ensure_ascii=False),
                        source_reference_json=json.dumps(
                            node.source_reference.model_dump(mode="json"),
                            ensure_ascii=False,
                        ),
                    )
        except Exception as exc:
            logger.exception("Failed to upsert graph nodes", extra={"node_count": len(nodes)})
            raise GraphStoreError("Failed to upsert graph nodes") from exc

    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        if not edges:
            return
        try:
            await self._ensure_connected()
            async with self._session() as session:
                for edge in edges:
                    edge_type = self._safe_identifier(edge.type, default="RELATED_TO")
                    await session.run(
                        f"""
                        MATCH (source:GraphNode {{id: $source_node_id}})
                        MATCH (target:GraphNode {{id: $target_node_id}})
                        MERGE (source)-[r:{edge_type} {{id: $id}}]->(target)
                        SET r.type = $type,
                            r.source_node_id = $source_node_id,
                            r.target_node_id = $target_node_id,
                            r.document_id = $document_id,
                            r.source_reference_json = $source_reference_json,
                            r.properties_json = $properties_json
                        SET r += $properties
                        """,
                        id=edge.id,
                        type=edge.type,
                        source_node_id=edge.source_node_id,
                        target_node_id=edge.target_node_id,
                        document_id=edge.source_reference.document_id
                        or edge.properties.get("document_id"),
                        properties=self._neo4j_properties(edge.properties),
                        properties_json=json.dumps(edge.properties, ensure_ascii=False),
                        source_reference_json=json.dumps(
                            edge.source_reference.model_dump(mode="json"),
                            ensure_ascii=False,
                        ),
                    )
        except Exception as exc:
            logger.exception("Failed to upsert graph edges", extra={"edge_count": len(edges)})
            raise GraphStoreError("Failed to upsert graph edges") from exc

    async def upsert_triples(self, triples: list[GraphTriple]) -> None:
        if not triples:
            return
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for triple in triples:
            nodes[triple.subject.id] = triple.subject
            nodes[triple.object.id] = triple.object
            edges[triple.predicate.id] = triple.predicate
        await self.upsert_nodes(list(nodes.values()))
        await self.upsert_edges(list(edges.values()))

    async def query_subgraph(self, query_request: GraphQueryRequest) -> GraphQueryResult:
        try:
            await self._ensure_connected()
            async with self._session() as session:
                result = await session.run(
                    """
                    MATCH (n:GraphNode)
                    WHERE ($node_labels = [] OR n.label IN $node_labels)
                      AND (
                        $document_ids = []
                        OR n.document_id IN $document_ids
                      )
                      AND (
                        $query_text = ""
                        OR toLower(coalesce(n.name, "")) CONTAINS toLower($query_text)
                        OR toLower(coalesce(n.title, "")) CONTAINS toLower($query_text)
                        OR toLower(coalesce(n.label, "")) CONTAINS toLower($query_text)
                      )
                    OPTIONAL MATCH (n)-[r]-(m:GraphNode)
                    WHERE $edge_types = [] OR r.type IN $edge_types
                    RETURN collect(DISTINCT n)[0..$limit] AS nodes,
                           collect(DISTINCT m)[0..$limit] AS neighbor_nodes,
                           collect(DISTINCT r)[0..$limit] AS edges
                    """,
                    query_text=query_request.query,
                    document_ids=query_request.document_ids,
                    node_labels=query_request.node_labels,
                    edge_types=query_request.edge_types,
                    limit=query_request.limit,
                )
                record = await result.single()
                return self._record_to_query_result(query_request.query, record)
        except Exception as exc:
            logger.exception("Failed to query Neo4j subgraph")
            raise GraphStoreError("Failed to query Neo4j subgraph") from exc

    async def get_neighbors(self, node_id: str, depth: int) -> GraphQueryResult:
        safe_depth = max(1, min(depth, 5))
        try:
            await self._ensure_connected()
            async with self._session() as session:
                result = await session.run(
                    f"""
                    MATCH path = (n:GraphNode {{id: $node_id}})-[*1..{safe_depth}]-(m:GraphNode)
                    RETURN collect(DISTINCT n) + collect(DISTINCT m) AS nodes,
                           [] AS neighbor_nodes,
                           collect(DISTINCT relationships(path)) AS edge_paths
                    """,
                    node_id=node_id,
                )
                record = await result.single()
                return self._path_record_to_query_result(f"neighbors:{node_id}", record)
        except Exception as exc:
            logger.exception("Failed to get Neo4j neighbors", extra={"node_id": node_id})
            raise GraphStoreError("Failed to get Neo4j neighbors") from exc

    async def search_entities(
        self,
        keyword: str,
        document_ids: list[str] | None = None,
    ) -> GraphQueryResult:
        request = GraphQueryRequest(
            query=keyword,
            document_ids=list(document_ids or []),
            limit=20,
        )
        return await self.query_subgraph(request)

    async def _ensure_connected(self) -> None:
        if self._driver is None:
            await self.connect()

    def _session(self) -> Any:
        if self._driver is None:
            raise GraphStoreError("Neo4j driver is not connected")
        if self.database:
            return self._driver.session(database=self.database)
        return self._driver.session()

    def _record_to_query_result(self, query: str, record: Any | None) -> GraphQueryResult:
        if record is None:
            return GraphQueryResult(query=query)
        raw_nodes = [*record.get("nodes", []), *record.get("neighbor_nodes", [])]
        raw_edges = record.get("edges", [])
        nodes = self._dedupe_nodes([self._node_from_neo4j(node) for node in raw_nodes if node])
        edges = self._dedupe_edges([self._edge_from_neo4j(edge) for edge in raw_edges if edge])
        return GraphQueryResult(
            query=query,
            nodes=nodes,
            edges=edges,
            evidences=[node.source_reference for node in nodes] + [edge.source_reference for edge in edges],
        )

    def _path_record_to_query_result(self, query: str, record: Any | None) -> GraphQueryResult:
        if record is None:
            return GraphQueryResult(query=query)
        edge_paths = record.get("edge_paths", [])
        raw_edges = [edge for path_edges in edge_paths for edge in path_edges]
        nodes = self._dedupe_nodes(
            [self._node_from_neo4j(node) for node in record.get("nodes", []) if node]
        )
        edges = self._dedupe_edges([self._edge_from_neo4j(edge) for edge in raw_edges if edge])
        return GraphQueryResult(
            query=query,
            nodes=nodes,
            edges=edges,
            evidences=[node.source_reference for node in nodes] + [edge.source_reference for edge in edges],
        )

    def _node_from_neo4j(self, node: Any) -> GraphNode:
        data = dict(node)
        return GraphNode(
            id=str(data["id"]),
            label=str(data.get("label", "Entity")),
            properties=self._properties_from_data(data),
            source_reference=self._evidence_from_data(data.get("source_reference_json")),
        )

    def _edge_from_neo4j(self, edge: Any) -> GraphEdge:
        data = dict(edge)
        source_node_id = getattr(getattr(edge, "start_node", None), "get", lambda _key: "")("id")
        target_node_id = getattr(getattr(edge, "end_node", None), "get", lambda _key: "")("id")
        return GraphEdge(
            id=str(data["id"]),
            type=str(data.get("type", getattr(edge, "type", "RELATED_TO"))),
            source_node_id=str(data.get("source_node_id") or source_node_id),
            target_node_id=str(data.get("target_node_id") or target_node_id),
            properties=self._properties_from_data(data),
            source_reference=self._evidence_from_data(data.get("source_reference_json")),
        )

    def _evidence_from_data(self, data: Any) -> Evidence:
        if isinstance(data, Evidence):
            return data
        if isinstance(data, str) and data:
            return Evidence.model_validate(json.loads(data))
        if isinstance(data, dict):
            return Evidence.model_validate(data)
        return Evidence(id="unknown", source_type="document")

    def _properties_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        properties_json = data.get("properties_json")
        if isinstance(properties_json, str) and properties_json:
            return json.loads(properties_json)
        ignored = {"id", "label", "source_reference_json", "properties_json"}
        return {key: value for key, value in data.items() if key not in ignored}

    def _neo4j_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in properties.items()
            if isinstance(value, str | int | float | bool) or value is None
        }

    def _safe_identifier(self, value: str, default: str) -> str:
        cleaned = _IDENTIFIER_PATTERN.sub("_", value).strip("_").upper()
        if not cleaned:
            return default
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned

    def _dedupe_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        deduped: dict[str, GraphNode] = {}
        for node in nodes:
            deduped[node.id] = node
        return list(deduped.values())

    def _dedupe_edges(self, edges: list[GraphEdge]) -> list[GraphEdge]:
        deduped: dict[str, GraphEdge] = {}
        for edge in edges:
            deduped[edge.id] = edge
        return list(deduped.values())
