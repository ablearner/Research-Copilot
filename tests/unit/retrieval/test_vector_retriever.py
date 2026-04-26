import pytest

from domain.schemas.retrieval import RetrievalQuery
from retrieval.vector_retriever import VectorRetriever


@pytest.mark.asyncio
async def test_vector_retriever_passes_filters_and_normalizes_hits(mock_embedding, mock_vector_store) -> None:
    query = RetrievalQuery(query="Revenue", document_ids=["doc1"], modalities=["text"], top_k=5)
    hits = await VectorRetriever(mock_embedding, mock_vector_store).retrieve(query)

    assert hits[0].merged_score == hits[0].vector_score
    assert mock_vector_store.last_filters["document_ids"] == ["doc1"]
    assert mock_vector_store.last_filters["modalities"] == ["text"]
