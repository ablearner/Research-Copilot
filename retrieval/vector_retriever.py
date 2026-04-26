import logging
from typing import Any

from langchain_core.runnables import RunnableLambda

from adapters.embedding.base import BaseEmbeddingAdapter, EmbeddingAdapterError
from adapters.vector_store.base import BaseVectorStore, VectorStoreError
from domain.schemas.embedding import EmbeddingVector
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery

logger = logging.getLogger(__name__)


class VectorRetrieverError(RuntimeError):
    """Raised when vector retrieval fails."""


class VectorRetriever:
    def __init__(
        self,
        embedding_adapter: BaseEmbeddingAdapter,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.vector_store = vector_store
        self.runnable = RunnableLambda(self.ainvoke)

    async def ainvoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        return await self.retrieve(query)

    def invoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        raise NotImplementedError("Use async retrieval for vector retrieval")

    def as_runnable(self) -> RunnableLambda:
        return self.runnable

    async def retrieve(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        retrieval_query = self._coerce_query(query)
        try:
            query_embedding = await self.build_query_embedding(retrieval_query.query)
            filters = self._build_filters(retrieval_query)
            hits = await self.vector_store.search_by_vector(
                vector=query_embedding,
                top_k=retrieval_query.top_k,
                filters=filters,
            )
            normalized_hits = self.normalize_hits(hits, retrieval_query)
            logger.info(
                "Vector retrieval completed",
                extra={"query": retrieval_query.query, "hit_count": len(normalized_hits)},
            )
            return normalized_hits
        except EmbeddingAdapterError as exc:
            logger.exception("Failed to build vector retrieval query embedding")
            raise VectorRetrieverError("Failed to build vector retrieval query embedding") from exc
        except VectorStoreError as exc:
            logger.exception("Vector store failed during retrieval")
            raise VectorRetrieverError("Vector store failed during retrieval") from exc
        except Exception as exc:
            logger.exception("Unexpected vector retrieval failure")
            raise VectorRetrieverError("Unexpected vector retrieval failure") from exc

    async def build_query_embedding(self, text: str) -> EmbeddingVector:
        if not text.strip():
            raise VectorRetrieverError("Query text must not be empty")
        return await self.embedding_adapter.embed_text(text)

    def normalize_hits(
        self,
        hits: list[RetrievalHit],
        query: RetrievalQuery,
    ) -> list[RetrievalHit]:
        normalized: list[RetrievalHit] = []
        for index, hit in enumerate(hits):
            vector_score = hit.vector_score if hit.vector_score is not None else 0.0
            normalized.append(
                hit.model_copy(
                    update={
                        "vector_score": vector_score,
                        "merged_score": hit.merged_score if hit.merged_score is not None else vector_score,
                        "metadata": {
                            **hit.metadata,
                            "retriever": self.__class__.__name__,
                            "rank": index + 1,
                            "query": query.query,
                        },
                    }
                )
            )
        return sorted(normalized, key=lambda item: item.vector_score or 0.0, reverse=True)

    def _build_filters(self, query: RetrievalQuery) -> dict[str, Any]:
        filters = dict(query.filters)
        if query.document_ids:
            filters["document_ids"] = query.document_ids
        modalities = query.modalities or query.filters.get("modalities") or []
        if modalities:
            filters["modalities"] = modalities
        source_types = query.filters.get("source_types") or []
        if source_types:
            filters["source_types"] = source_types
        return filters

    def _coerce_query(self, query: RetrievalQuery | str) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query
        return RetrievalQuery(query=query, mode="vector")
