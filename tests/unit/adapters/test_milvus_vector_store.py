import json
from types import SimpleNamespace

import pytest
from pymilvus import DataType, FunctionType

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
                {"name": "content", "params": {"enable_analyzer": True}},
                {"name": "sparse"},
            ],
            "functions": [{"name": "content_bm25"}],
        },
    )

    with pytest.raises(VectorStoreError, match="existing_dim=1024 requested_dim=3072"):
        await store._ensure_collection(3072)


@pytest.mark.asyncio
async def test_upsert_embeddings_flushes_after_write() -> None:
    calls: list[str] = []

    class FakeClient:
        def has_collection(self, collection_name: str) -> bool:
            calls.append("has_collection")
            return True

        def describe_collection(self, collection_name: str) -> dict:
            calls.append("describe_collection")
            return {
                "fields": [
                    {"name": "vector", "params": {"dim": 2}},
                    {"name": "content", "params": {"enable_analyzer": True}},
                    {"name": "sparse"},
                ],
                "functions": [{"name": "content_bm25"}],
            }

        def load_collection(self, collection_name: str) -> None:
            calls.append("load_collection")

        def upsert(self, collection_name: str, data: list[dict]) -> dict:
            calls.append("upsert")
            return {"upsert_count": len(data)}

        def flush(self, collection_name: str) -> None:
            calls.append("flush")

    store = MilvusVectorStore(collection_name="test_collection")
    store.client = FakeClient()
    record = MultimodalEmbeddingRecord(
        id="rec1",
        item=EmbeddingItem(
            id="item1",
            document_id="doc1",
            source_type="text_block",
            source_id="tb1",
            content="hello",
        ),
        embedding=EmbeddingVector(model="m1", dimensions=2, values=[1.0, 0.0]),
        modality="text",
        namespace="default",
    )

    await store.upsert_embeddings([record])

    assert calls[-2:] == ["upsert", "flush"]


def test_build_schema_enables_milvus_bm25_sparse_function() -> None:
    store = MilvusVectorStore(collection_name="test_collection")

    schema = store._build_schema(2).to_dict()

    content_field = next(field for field in schema["fields"] if field["name"] == "content")
    sparse_field = next(field for field in schema["fields"] if field["name"] == "sparse")
    assert content_field["params"]["enable_analyzer"] is True
    assert sparse_field["type"] is DataType.SPARSE_FLOAT_VECTOR
    assert schema["functions"] == [
        {
            "name": "content_bm25",
            "description": "",
            "type": FunctionType.BM25,
            "input_field_names": ["content"],
            "output_field_names": ["sparse"],
            "params": {},
        }
    ]


def test_build_index_params_adds_sparse_inverted_index() -> None:
    store = MilvusVectorStore(collection_name="test_collection")
    store.client = SimpleNamespace(prepare_index_params=__import__("pymilvus").MilvusClient.prepare_index_params)

    index_params = [index.to_dict() for index in store._build_index_params()]

    assert any(index["field_name"] == "vector" for index in index_params)
    sparse_index = next(index for index in index_params if index["field_name"] == "sparse")
    assert sparse_index["index_type"] == "SPARSE_INVERTED_INDEX"
    assert sparse_index["metric_type"] == "BM25"
    assert sparse_index["inverted_index_algo"] == "DAAT_MAXSCORE"


@pytest.mark.asyncio
async def test_ensure_collection_rejects_existing_collection_without_bm25_schema() -> None:
    store = MilvusVectorStore(collection_name="test_collection", dimension=2)
    store.client = SimpleNamespace(
        has_collection=lambda collection_name: True,
        describe_collection=lambda collection_name: {
            "fields": [
                {"name": "vector", "params": {"dim": 2}},
                {"name": "content", "params": {"max_length": 65535}},
            ]
        },
    )

    with pytest.raises(VectorStoreError, match="missing native BM25 sparse schema"):
        await store._ensure_collection(2)


@pytest.mark.asyncio
async def test_search_sparse_text_uses_milvus_bm25_search() -> None:
    calls: list[dict] = []

    class FakeClient:
        def has_collection(self, collection_name: str) -> bool:
            return True

        def describe_collection(self, collection_name: str) -> dict:
            return {
                "fields": [
                    {"name": "vector", "params": {"dim": 2}},
                    {"name": "content", "params": {"enable_analyzer": True}},
                    {"name": "sparse"},
                ],
                "functions": [{"name": "content_bm25"}],
            }

        def load_collection(self, collection_name: str) -> None:
            return None

        def search(self, **kwargs):
            calls.append(kwargs)
            return [
                [
                    {
                        "id": "rec1",
                        "distance": 3.2,
                        "entity": {
                            "id": "rec1",
                            "document_id": "doc1",
                            "source_type": "text_block",
                            "source_id": "tb1",
                            "content": "BM25 sparse retrieval",
                            "modality": "text",
                            "metadata_json": "{}",
                        },
                    }
                ]
            ]

    store = MilvusVectorStore(collection_name="test_collection", dimension=2)
    store.client = FakeClient()

    hits = await store.search_sparse_text("BM25 retrieval", top_k=5, filters={"document_ids": ["doc1"]})

    assert calls[0]["anns_field"] == "sparse"
    assert calls[0]["data"] == ["BM25 retrieval"]
    assert calls[0]["search_params"]["metric_type"] == "BM25"
    assert calls[0]["filter"] == 'document_id in ["doc1"]'
    assert hits[0].sparse_score == 3.2
    assert hits[0].vector_score is None
