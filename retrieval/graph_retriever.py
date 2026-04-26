import logging
import re
from collections import Counter

from langchain_core.runnables import RunnableLambda
from adapters.graph_store.base import BaseGraphStore, GraphStoreError
from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.graph import GraphEdge, GraphNode, GraphQueryRequest, GraphQueryResult, GraphTriple
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "与",
    "和",
    "的",
    "了",
    "在",
    "是",
    "什么",
    "哪些",
    "如何",
    "为什么",
}


class GraphRetrieverError(RuntimeError):
    """Raised when graph retrieval fails."""


class GraphRetriever:
    def __init__(
        self,
        graph_store: BaseGraphStore,
        max_keywords: int = 8,
        neighbor_depth: int = 1,
    ) -> None:
        self.graph_store = graph_store
        self.max_keywords = max_keywords
        self.neighbor_depth = neighbor_depth
        self.runnable = RunnableLambda(self.ainvoke)

    async def ainvoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        return await self.retrieve(query)

    def invoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        raise NotImplementedError("Use async retrieval for graph retrieval")

    def as_runnable(self) -> RunnableLambda:
        return self.runnable

    async def retrieve(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        retrieval_query = self._coerce_query(query)
        keywords = self.extract_entity_keywords(retrieval_query.query)
        try:
            graph_results: list[GraphQueryResult] = []
            graph_results.append(
                await self.graph_store.query_subgraph(
                    GraphQueryRequest(
                        query=retrieval_query.query,
                        document_ids=retrieval_query.document_ids,
                        limit=retrieval_query.top_k,
                        metadata_filter=retrieval_query.filters,
                    )
                )
            )
            if retrieval_query.graph_query_mode in {"entity", "auto"}:
                for keyword in keywords:
                    graph_results.append(
                        await self.graph_store.search_entities(
                            keyword,
                            retrieval_query.document_ids,
                        )
                    )

            hits = self._results_to_hits(retrieval_query, graph_results, keywords)
            if retrieval_query.graph_query_mode in {"subgraph", "auto"}:
                hits.extend(self._results_to_subgraph_hits(retrieval_query, graph_results, keywords))
            ranked_hits = self.rank_graph_hits(hits, retrieval_query.query, keywords)
            return ranked_hits[: retrieval_query.top_k]
        except GraphStoreError as exc:
            logger.exception("Graph store failed during graph retrieval", extra={"query": retrieval_query.query})
            raise GraphRetrieverError("Graph store failed during graph retrieval") from exc
        except Exception as exc:
            logger.exception("Unexpected graph retrieval failure", extra={"query": retrieval_query.query})
            raise GraphRetrieverError("Unexpected graph retrieval failure") from exc

    def extract_entity_keywords(self, question: str) -> list[str]:
        tokens = [token.strip() for token in _TOKEN_PATTERN.findall(question) if token.strip()]
        candidates = [
            token
            for token in tokens
            if len(token) > 1 and token.lower() not in _STOPWORDS
        ]
        counts = Counter(candidates)
        return [keyword for keyword, _count in counts.most_common(self.max_keywords)]

    def rank_graph_hits(
        self,
        hits: list[RetrievalHit],
        question: str,
        keywords: list[str] | None = None,
    ) -> list[RetrievalHit]:
        keyword_set = {keyword.lower() for keyword in (keywords or self.extract_entity_keywords(question))}
        ranked: list[RetrievalHit] = []
        for hit in hits:
            content = (hit.content or "").lower()
            keyword_score = sum(1 for keyword in keyword_set if keyword and keyword in content)
            structure_score = len(hit.graph_nodes) * 0.1 + len(hit.graph_edges) * 0.15 + len(hit.graph_triples) * 0.2
            base_score = hit.graph_score or 0.0
            score = base_score + keyword_score + structure_score
            ranked.append(hit.model_copy(update={"graph_score": score, "merged_score": score}))
        return sorted(ranked, key=lambda item: item.graph_score or 0.0, reverse=True)

    def _results_to_hits(
        self,
        query: RetrievalQuery,
        graph_results: list[GraphQueryResult],
        keywords: list[str],
    ) -> list[RetrievalHit]:
        hits: dict[str, RetrievalHit] = {}
        for result in graph_results:
            for node in result.nodes:
                hit = self._node_to_hit(node, query, keywords)
                hits.setdefault(hit.id, hit)
            for edge in result.edges:
                hit = self._edge_to_hit(edge, result.nodes, query, keywords)
                hits.setdefault(hit.id, hit)
            for triple in result.triples:
                hit = self._triple_to_hit(triple, query, keywords)
                hits.setdefault(hit.id, hit)
        return list(hits.values())

    def _results_to_subgraph_hits(
        self,
        query: RetrievalQuery,
        graph_results: list[GraphQueryResult],
        keywords: list[str],
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for index, result in enumerate(graph_results):
            if not (result.nodes or result.edges or result.triples):
                continue
            content = self._subgraph_content(result)
            evidences = self._dedupe_evidences(
                result.evidences
                or [node.source_reference for node in result.nodes]
                + [edge.source_reference for edge in result.edges]
            )
            document_id = evidences[0].document_id if evidences else None
            hits.append(
                RetrievalHit(
                    id=f"graph_subgraph:{index}:{abs(hash(content))}",
                    source_type="graph_subgraph",
                    source_id=f"subgraph:{index}",
                    document_id=document_id,
                    content=content,
                    graph_score=self._keyword_score(content, keywords) + 0.4,
                    graph_nodes=result.nodes,
                    graph_edges=result.edges,
                    graph_triples=result.triples,
                    evidence=EvidenceBundle(evidences=evidences),
                    metadata={"query": query.query, "query_mode": query.graph_query_mode},
                )
            )
        return hits

    def _node_to_hit(
        self,
        node: GraphNode,
        query: RetrievalQuery,
        keywords: list[str],
    ) -> RetrievalHit:
        content = self._node_content(node)
        return RetrievalHit(
            id=f"graph_node:{node.id}",
            source_type="graph_node",
            source_id=node.id,
            document_id=node.source_reference.document_id,
            content=content,
            graph_score=self._keyword_score(content, keywords),
            graph_nodes=[node],
            evidence=EvidenceBundle(evidences=[node.source_reference]),
            metadata={"query": query.query, "node_label": node.label},
        )

    def _edge_to_hit(
        self,
        edge: GraphEdge,
        nodes: list[GraphNode],
        query: RetrievalQuery,
        keywords: list[str],
    ) -> RetrievalHit:
        related_nodes = [
            node for node in nodes if node.id in {edge.source_node_id, edge.target_node_id}
        ]
        content = self._edge_content(edge, related_nodes)
        return RetrievalHit(
            id=f"graph_edge:{edge.id}",
            source_type="graph_edge",
            source_id=edge.id,
            document_id=edge.source_reference.document_id,
            content=content,
            graph_score=self._keyword_score(content, keywords) + 0.15,
            graph_nodes=related_nodes,
            graph_edges=[edge],
            evidence=EvidenceBundle(evidences=[edge.source_reference]),
            metadata={"query": query.query, "edge_type": edge.type},
        )

    def _triple_to_hit(
        self,
        triple: GraphTriple,
        query: RetrievalQuery,
        keywords: list[str],
    ) -> RetrievalHit:
        content = (
            f"{self._node_content(triple.subject)} "
            f"{triple.predicate.type} "
            f"{self._node_content(triple.object)}"
        )
        evidences = self._dedupe_evidences(
            [
                triple.subject.source_reference,
                triple.predicate.source_reference,
                triple.object.source_reference,
            ]
        )
        return RetrievalHit(
            id=f"graph_triple:{triple.predicate.id}",
            source_type="graph_triple",
            source_id=triple.predicate.id,
            document_id=triple.predicate.source_reference.document_id,
            content=content,
            graph_score=self._keyword_score(content, keywords) + 0.3,
            graph_nodes=[triple.subject, triple.object],
            graph_edges=[triple.predicate],
            graph_triples=[triple],
            evidence=EvidenceBundle(evidences=evidences),
            metadata={"query": query.query, "edge_type": triple.predicate.type},
        )

    def _node_content(self, node: GraphNode) -> str:
        property_values = [
            str(value)
            for key, value in node.properties.items()
            if key in {"name", "title", "value", "description", "document_id"} and value is not None
        ]
        return " ".join([node.label, *property_values]).strip()

    def _edge_content(self, edge: GraphEdge, nodes: list[GraphNode]) -> str:
        node_content = " ".join(self._node_content(node) for node in nodes)
        property_values = [
            str(value)
            for key, value in edge.properties.items()
            if key in {"name", "description", "evidence", "document_id"} and value is not None
        ]
        return " ".join([edge.type, node_content, *property_values]).strip()

    def _subgraph_content(self, result: GraphQueryResult) -> str:
        node_text = "; ".join(self._node_content(node) for node in result.nodes)
        edge_text = "; ".join(self._edge_content(edge, result.nodes) for edge in result.edges)
        triple_text = "; ".join(
            f"{self._node_content(triple.subject)} {triple.predicate.type} {self._node_content(triple.object)}"
            for triple in result.triples
        )
        return " ".join(part for part in [node_text, edge_text, triple_text] if part)

    def _keyword_score(self, content: str, keywords: list[str]) -> float:
        lowered = content.lower()
        return float(sum(1 for keyword in keywords if keyword.lower() in lowered))

    def _dedupe_evidences(self, evidences: list[Evidence]) -> list[Evidence]:
        deduped: dict[str, Evidence] = {}
        for evidence in evidences:
            deduped[evidence.id] = evidence
        return list(deduped.values())

    def _coerce_query(self, query: RetrievalQuery | str) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query.model_copy(update={"mode": "graph"})
        return RetrievalQuery(query=query, mode="graph")
