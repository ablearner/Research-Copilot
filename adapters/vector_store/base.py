from abc import ABC, abstractmethod
from typing import Any

from domain.schemas.embedding import EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.retrieval import RetrievalHit


class VectorStoreError(RuntimeError):
    """Raised when a vector store operation fails."""


class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert_embedding(self, record: MultimodalEmbeddingRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_embeddings(self, records: list[MultimodalEmbeddingRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search_by_vector(
        self,
        vector: EmbeddingVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    async def search_similar_text(self, text: str, top_k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    async def search_sparse_text(
        self,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        return await self.search_similar_text(text=text, top_k=top_k)

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError
