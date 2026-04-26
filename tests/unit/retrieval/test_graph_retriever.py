import pytest

from domain.schemas.retrieval import RetrievalQuery
from retrieval.graph_retriever import GraphRetriever


@pytest.mark.asyncio
async def test_graph_retriever_returns_graph_hits(mock_graph_store) -> None:
    hits = await GraphRetriever(mock_graph_store).retrieve("Revenue in 2025?")

    assert hits
    assert hits[0].graph_score is not None
    assert any(hit.evidence for hit in hits)


@pytest.mark.asyncio
async def test_graph_retriever_passes_document_ids_to_entity_search(mock_graph_store) -> None:
    query = RetrievalQuery(
        query="What is the Revenue in 2025?",
        document_ids=["doc-current"],
        graph_query_mode="entity",
    )

    await GraphRetriever(mock_graph_store).retrieve(query)

    assert mock_graph_store.entity_search_calls
    assert all(document_ids == ["doc-current"] for _keyword, document_ids in mock_graph_store.entity_search_calls)
