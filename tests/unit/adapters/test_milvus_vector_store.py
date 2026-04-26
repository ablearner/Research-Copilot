import json
from types import SimpleNamespace

import pytest

from adapters.vector_store.milvus_adapter import MilvusVectorStore
from adapters.vector_store.base import VectorStoreError
from domain.schemas.embedding import EmbeddingItem, EmbeddingVector, MultimodalEmbeddingRecord


def test_record_to_entity_serializes_metadata_json() -> None:
    store = MilvusVectorStore(collection_name="test_collection")
    record = MultimodalEmbeddingRecord(
        id="rec1",
        item=EmbeddingItem(
            id="item1",
            document_id="doc1",
            source_type="text_block",
            source_id="tb1",
            content="hello",
            uri="/tmp/a.txt",
            metadata={"page": 1, "tags": ["a", "b"], "source_type": "pdf"},
        ),
        embedding=EmbeddingVector(model="m1", dimensions=2, values=[1.0, 0.0]),
        modality="text",
        namespace="default",
        metadata={"score": 0.5, "published": True},
    )

    entity = store._record_to_entity(record)

    assert entity["id"] == "rec1"
    assert entity["document_id"] == "doc1"
    assert entity["source_type"] == "text_block"
    assert entity["source_id"] == "tb1"
    assert entity["embedding_dimensions"] == 2
    metadata = json.loads(entity["metadata_json"])
    assert metadata["page"] == 1
    assert metadata["tags"] == ["a", "b"]
    assert metadata["score"] == 0.5
    assert metadata["published"] is True
    assert "source_type" not in metadata


def test_build_filter_expression_supports_document_modality_namespace() -> None:
    store = MilvusVectorStore(collection_name="test_collection")

    expression = store._build_filter_expression(
        {
            "document_ids": ["doc1", "doc2"],
            "modalities": ["text"],
            "namespace": 'science"2026',
        }
    )

    assert expression == (
        'document_id in ["doc1", "doc2"] and '
        'modality in ["text"] and '
        'namespace == "science\\"2026"'
    )


def test_search_results_to_hits_maps_scores_and_metadata() -> None:
    store = MilvusVectorStore(collection_name="test_collection")

    hits = store._search_results_to_hits(
        [
            [
                {
                    "id": "rec1",
                    "distance": 0.72,
                    "entity": {
                        "id": "rec1",
                        "document_id": "doc1",
                        "source_type": "text_block",
                        "source_id": "tb1",
                        "content": "hello world",
                        "modality": "text",
                        "namespace": "science",
                        "uri": "/tmp/a.txt",
                        "embedding_model": "m1",
                        "embedding_dimensions": 2,
                        "metadata_json": '{"extra": "value"}',
                    },
                }
            ]
        ]
    )

    assert len(hits) == 1
    assert hits[0].id == "rec1"
    assert hits[0].document_id == "doc1"
    assert hits[0].source_type == "text_block"
    assert hits[0].source_id == "tb1"
    assert hits[0].content == "hello world"
    assert hits[0].vector_score == 0.72
    assert hits[0].metadata["modality"] == "text"
    assert hits[0].metadata["namespace"] == "science"
    assert hits[0].metadata["uri"] == "/tmp/a.txt"
    assert hits[0].metadata["extra"] == "value"


def test_search_results_to_hits_normalizes_legacy_page_source_type() -> None:
    store = MilvusVectorStore(collection_name="test_collection")

    hits = store._search_results_to_hits(
        [
            [
                {
                    "id": "rec_page_1",
                    "distance": 0.91,
                    "entity": {
                        "document_id": "doc1",
                        "source_type": "pdf",
                        "source_id": "page1",
                        "content": "rendered page text",
                        "modality": "page",
                        "metadata_json": '{"page_number": 1}',
                    },
                }
            ]
        ]
    )

    assert len(hits) == 1
    assert hits[0].source_type == "page"
    assert hits[0].source_id == "page1"
    assert hits[0].metadata["modality"] == "page"
    assert hits[0].metadata["page_number"] == 1


def test_collection_name_normalization_matches_milvus_rules() -> None:
    store = MilvusVectorStore(collection_name="2026-rag-vectors")

    assert store.collection_name == "c_2026_rag_vectors"


@pytest.mark.asyncio
async def test_get_collection_dimension_reads_vector_field_dim() -> None:
    store = MilvusVectorStore(collection_name="test_collection")
    store.client = SimpleNamespace(
        describe_collection=lambda collection_name: {
            "fields": [
                {"name": "id", "params": {"max_length": 512}},
                {"name": "vector", "params": {"dim": 3072}},
            ]
        }
    )

    dimension = await store._get_collection_dimension()

    assert dimension == 3072


@pytest.mark.asyncio
async def test_ensure_collection_rejects_dimension_mismatch() -> None:
    store = MilvusVectorStore(collection_name="test_collection", dimension=3072)
    store.client = SimpleNamespace(
        has_collection=lambda collection_name: True,
        describe_collection=lambda collection_name: {
            "fields": [
                {"name": "vector", "params": {"dim": 1024}},
            ]
        },
    )

    with pytest.raises(VectorStoreError, match="existing_dim=1024 requested_dim=3072"):
        await store._ensure_collection(3072)
