from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.vector_store.milvus_adapter import MilvusVectorStore
from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    if settings.vector_store_provider.lower() not in {"milvus", "zilliz"}:
        raise RuntimeError(
            f"Vector store provider '{settings.vector_store_provider}' does not use Milvus"
        )

    vector_store = MilvusVectorStore(
        collection_name=settings.milvus_collection_name,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        dimension=settings.milvus_dimension,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
    )

    await vector_store.reset_collection()
    await vector_store.close()

    print(
        f"Reset Milvus collection '{settings.milvus_collection_name}' "
        f"with dimension={settings.milvus_dimension} uri={settings.milvus_uri}"
    )


if __name__ == "__main__":
    asyncio.run(main())
