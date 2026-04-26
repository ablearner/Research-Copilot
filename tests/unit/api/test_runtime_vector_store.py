import pytest

from adapters.local_runtime import LocalHashEmbeddingAdapter
from adapters.vector_store.milvus_adapter import MilvusVectorStore
from apps.api.runtime import _build_vector_store
from core.config import Settings


def test_build_vector_store_uses_milvus_defaults() -> None:
    settings = Settings(vector_store_provider="milvus")

    store = _build_vector_store(settings, LocalHashEmbeddingAdapter())

    assert isinstance(store, MilvusVectorStore)
    assert store.collection_name == "multimodal_embeddings"
    assert store.uri == "http://localhost:19530"
    assert store.metric_type == "COSINE"
    assert store.index_type == "HNSW"


def test_build_vector_store_uses_milvus_config() -> None:
    settings = Settings(
        vector_store_provider="milvus",
        milvus_uri="http://127.0.0.1:19530",
        milvus_token="token",
        milvus_db_name="rag",
        milvus_collection_name="rag-vectors",
        milvus_dimension=1024,
        milvus_metric_type="ip",
        milvus_index_type="hnsw",
    )

    store = _build_vector_store(settings, LocalHashEmbeddingAdapter())

    assert isinstance(store, MilvusVectorStore)
    assert store.uri == "http://127.0.0.1:19530"
    assert store.token == "token"
    assert store.db_name == "rag"
    assert store.collection_name == "rag_vectors"
    assert store.dimension == 1024
    assert store.metric_type == "IP"
    assert store.index_type == "HNSW"


def test_build_vector_store_rejects_unsupported_provider() -> None:
    settings = Settings(vector_store_provider="unsupported")

    with pytest.raises(RuntimeError, match="Unsupported VECTOR_STORE_PROVIDER"):
        _build_vector_store(settings, LocalHashEmbeddingAdapter())
