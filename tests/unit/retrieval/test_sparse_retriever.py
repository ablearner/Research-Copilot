import pytest

from adapters.local_runtime import InMemoryVectorStore
from domain.schemas.embedding import EmbeddingItem, EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.retrieval import RetrievalQuery
from retrieval.sparse_retriever import SparseRetriever


@pytest.mark.asyncio
async def test_sparse_retriever_uses_vector_store_sparse_search_and_normalizes_hits() -> None:
    vector_store = InMemoryVectorStore()
    await vector_store.upsert_embeddings(
        [
            MultimodalEmbeddingRecord(
                id="chunk1",
                item=EmbeddingItem(
                    id="tb1",
                    document_id="doc1",
                    source_type="text_block",
                    source_id="tb1",
                    content="BM25 keyword retrieval with inverted indexes.",
                ),
                embedding=EmbeddingVector(model="test", dimensions=2, values=[1.0, 0.0]),
                modality="text",
            )
        ]
    )

    hits = await SparseRetriever(vector_store).retrieve(
        RetrievalQuery(query="BM25 inverted", document_ids=["doc1"], top_k=5)
    )

    assert len(hits) == 1
    assert hits[0].document_id == "doc1"
    assert hits[0].merged_score == hits[0].sparse_score
    assert hits[0].metadata["retriever"] == "SparseRetriever"
    assert hits[0].metadata["rank"] == 1
