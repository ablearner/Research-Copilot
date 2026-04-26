import pytest

from rag_runtime.services.embedding_index_service import EmbeddingIndexService


@pytest.mark.asyncio
async def test_embedding_index_service_indexes_text_pages_and_charts(
    mock_embedding,
    mock_vector_store,
    sample_text_block,
    sample_page,
    sample_chart,
) -> None:
    service = EmbeddingIndexService(mock_embedding, mock_vector_store)

    text_result = await service.index_text_blocks("doc1", [sample_text_block])
    page_result = await service.index_pages("doc1", [sample_page])
    chart_result = await service.index_charts("doc1", [sample_chart])

    assert text_result.record_count == 1
    assert page_result.record_count == 1
    assert chart_result.record_count == 1
    assert len(mock_vector_store.records) == 3
