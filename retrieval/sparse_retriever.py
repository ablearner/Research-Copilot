from __future__ import annotations

import logging

from langchain_core.runnables import RunnableLambda

from adapters.vector_store.base import BaseVectorStore, VectorStoreError
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery

logger = logging.getLogger(__name__)


class SparseRetrieverError(RuntimeError):
    """Raised when sparse retrieval fails."""


class SparseRetriever:
    def __init__(self, vector_store: BaseVectorStore) -> None:
        self.vector_store = vector_store
        self.runnable = RunnableLambda(self.ainvoke)

    async def ainvoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        return await self.retrieve(query)

    def invoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        raise NotImplementedError("Use async retrieval for sparse retrieval")

    def as_runnable(self) -> RunnableLambda:
        return self.runnable

    async def retrieve(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        retrieval_query = self._coerce_query(query)
        try:
            filters = self._build_filters(retrieval_query)
            hits = await self.vector_store.search_sparse_text(
                text=retrieval_query.query,
                top_k=retrieval_query.top_k,
                filters=filters,
            )
            normalized_hits = self.normalize_hits(hits, retrieval_query)
            logger.info(
                "Sparse retrieval completed",
                extra={"query": retrieval_query.query, "hit_count": len(normalized_hits)},
            )
            return normalized_hits
        except VectorStoreError as exc:
            logger.exception("Sparse retrieval failed in vector store")
            raise SparseRetrieverError("Sparse retrieval failed in vector store") from exc
        except Exception as exc:
            logger.exception("Unexpected sparse retrieval failure")
            raise SparseRetrieverError("Unexpected sparse retrieval failure") from exc

    def normalize_hits(self, hits: list[RetrievalHit], query: RetrievalQuery) -> list[RetrievalHit]:
        normalized: list[RetrievalHit] = []
        for index, hit in enumerate(hits):
            sparse_score = hit.sparse_score if hit.sparse_score is not None else 0.0
            normalized.append(
                hit.model_copy(
                    update={
                        "sparse_score": sparse_score,
                        "merged_score": hit.merged_score if hit.merged_score is not None else sparse_score,
                        "metadata": {
                            **hit.metadata,
                            "retriever": self.__class__.__name__,
                            "rank": index + 1,
                            "query": query.query,
                        },
                    }
                )
            )
        return sorted(normalized, key=lambda item: item.sparse_score or 0.0, reverse=True)

    def _build_filters(self, query: RetrievalQuery) -> dict[str, object]:
        filters = dict(query.filters)
        if query.document_ids:
            filters["document_ids"] = query.document_ids
        source_types = query.filters.get("source_types") or []
        if source_types:
            filters["source_types"] = source_types
        else:
            filters["source_types"] = ["text_block", "page", "graph_summary"]
        return filters

    def _coerce_query(self, query: RetrievalQuery | str) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query.model_copy(update={"mode": "sparse"})
        return RetrievalQuery(query=query, mode="sparse")
