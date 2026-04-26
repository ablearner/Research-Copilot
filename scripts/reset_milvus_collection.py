from __future__ import annotations

import asyncio

from adapters.local_runtime import LocalHashEmbeddingAdapter
from apps.api.runtime import _build_embedding_adapter, _build_vector_store
from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    embedding_adapter = (
        _build_embedding_adapter(settings)
        if settings.runtime_backend.lower() == "business"
        else LocalHashEmbeddingAdapter(model=settings.embedding_model)
    )
    vector_store = _build_vector_store(settings, embedding_adapter)
    reset_collection = getattr(vector_store, "reset_collection", None)
    if not callable(reset_collection):
        raise RuntimeError(
            f"Vector store provider '{settings.vector_store_provider}' does not support reset_collection()"
        )

    await reset_collection()
    close = getattr(vector_store, "close", None)
    if callable(close):
        await close()

    print(
        f"Reset Milvus collection '{settings.milvus_collection_name}' "
        f"with dimension={settings.milvus_dimension} uri={settings.milvus_uri}"
    )


if __name__ == "__main__":
    asyncio.run(main())
