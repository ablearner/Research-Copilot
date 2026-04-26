import logging
import re
from collections import Counter

from langchain_core.runnables import RunnableLambda

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph_rag import GraphCommunitySummary
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class GraphSummaryRetriever:
    def __init__(self, summaries: list[GraphCommunitySummary] | None = None) -> None:
        self.summaries = summaries or []
        self.runnable = RunnableLambda(self.ainvoke)

    def set_summaries(self, summaries: list[GraphCommunitySummary]) -> None:
        self.summaries = summaries

    async def ainvoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        return await self.retrieve(query)

    def invoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        raise NotImplementedError("Use async retrieval for graph summary retrieval")

    def as_runnable(self) -> RunnableLambda:
        return self.runnable

    async def retrieve(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        retrieval_query = query if isinstance(query, RetrievalQuery) else RetrievalQuery(query=query)
        tokens = self._tokens(retrieval_query.query)
        hits = [
            self._summary_to_hit(summary, self._score_summary(summary, tokens))
            for summary in self.summaries
            if not retrieval_query.document_ids or summary.document_id in retrieval_query.document_ids
        ]
        hits = [hit for hit in hits if (hit.graph_score or 0.0) > 0]
        ranked = sorted(hits, key=lambda hit: hit.graph_score or 0.0, reverse=True)
        logger.info(
            "Graph summary retrieval completed",
            extra={"query": retrieval_query.query, "hit_count": len(ranked)},
        )
        return ranked[: retrieval_query.top_k]

    def _summary_to_hit(self, summary: GraphCommunitySummary, score: float) -> RetrievalHit:
        return RetrievalHit(
            id=f"graph_summary:{summary.id}",
            source_type="graph_summary",
            source_id=summary.id,
            document_id=summary.document_id,
            content=summary.summary,
            graph_score=score,
            evidence=EvidenceBundle(evidences=summary.source_references),
            metadata={
                "community_id": summary.community_id,
                "topic": summary.topic,
                "node_ids": summary.node_ids,
                "edge_ids": summary.edge_ids,
                "summary_confidence": summary.confidence,
                "retrieval_sources": ["graph_summary"],
            },
        )

    def _score_summary(self, summary: GraphCommunitySummary, query_tokens: list[str]) -> float:
        haystack = " ".join([summary.topic, summary.summary, *summary.node_ids, *summary.edge_ids]).lower()
        if not query_tokens:
            return 0.0
        counts = Counter(token for token in query_tokens if token in haystack)
        return float(sum(counts.values())) + 0.1 * len(summary.node_ids) + 0.05 * len(summary.edge_ids)

    def _tokens(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text) if len(token) > 1]
