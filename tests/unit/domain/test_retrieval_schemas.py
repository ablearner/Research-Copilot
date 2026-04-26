from domain.schemas.embedding import EmbeddingItem
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery


def test_retrieval_schema_json_schema() -> None:
    assert HybridRetrievalResult.model_json_schema()["title"] == "HybridRetrievalResult"


def test_retrieval_hit_supports_multimodal_sources() -> None:
    EmbeddingItem(id="i1", document_id="doc1", source_type="image_region", source_id="r1")
    hit = RetrievalHit(id="h1", source_type="page", source_id="p1", vector_score=0.5)
    result = HybridRetrievalResult(query=RetrievalQuery(query="q"), hits=[hit])
    assert result.hits[0].source_type == "page"
