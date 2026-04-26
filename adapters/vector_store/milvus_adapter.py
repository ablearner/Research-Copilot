import json
import logging
from collections.abc import Sequence
from inspect import isawaitable
from typing import Any

from adapters.embedding.base import BaseEmbeddingAdapter
from adapters.vector_store.base import BaseVectorStore, VectorStoreError
from domain.schemas.embedding import EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.retrieval import RetrievalHit
from retrieval.lexical import bm25_score_texts

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    _RESERVED_METADATA_KEYS = {
        "document_id",
        "source_type",
        "source_id",
        "modality",
        "namespace",
        "uri",
    }
    _VALID_SOURCE_TYPES = {
        "text_block",
        "page",
        "page_image",
        "chart",
        "image_region",
        "graph_node",
        "graph_edge",
        "graph_triple",
        "graph_subgraph",
        "graph_summary",
    }
    _OUTPUT_FIELDS = [
        "id",
        "content",
        "document_id",
        "source_type",
        "source_id",
        "modality",
        "namespace",
        "embedding_model",
        "embedding_dimensions",
        "uri",
        "metadata_json",
    ]

    def __init__(
        self,
        *,
        collection_name: str = "multimodal_embeddings",
        uri: str = "http://localhost:19530",
        token: str | None = None,
        db_name: str | None = None,
        dimension: int | None = None,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        embedding_adapter: BaseEmbeddingAdapter | None = None,
    ) -> None:
        self.collection_name = self._safe_collection_name(collection_name)
        self.uri = uri
        self.token = token or None
        self.db_name = db_name or None
        self.dimension = dimension
        self.metric_type = metric_type.upper()
        self.index_type = index_type.upper()
        self.embedding_adapter = embedding_adapter
        self.client: Any | None = None
        self._collection_loaded = False

    async def connect(self) -> None:
        if self.client is not None:
            return
        try:
            self.client = await self._call_sync(self._build_client)
            logger.info(
                "Connected to Milvus vector store",
                extra={"collection_name": self.collection_name, "uri": self.uri},
            )
        except Exception as exc:
            logger.exception("Failed to connect to Milvus vector store")
            raise VectorStoreError("Failed to connect to Milvus vector store") from exc

    async def ensure_schema(self) -> None:
        await self.connect()
        if self.dimension is not None:
            await self._ensure_collection(self.dimension)

    async def close(self) -> None:
        if self.client is None:
            return
        close = getattr(self.client, "close", None)
        if close:
            await self._call_sync(close)
        self.client = None
        self._collection_loaded = False

    async def upsert_embedding(self, record: MultimodalEmbeddingRecord) -> None:
        await self.upsert_embeddings([record])

    async def upsert_embeddings(self, records: list[MultimodalEmbeddingRecord]) -> None:
        if not records:
            return
        try:
            dimension = records[0].embedding.dimensions
            await self._ensure_collection(dimension)
            entities = [self._record_to_entity(record) for record in records]
            await self._call_sync(
                self.client.upsert,
                collection_name=self.collection_name,
                data=entities,
            )
        except Exception as exc:
            logger.exception("Failed to upsert Milvus records", extra={"record_count": len(records)})
            raise VectorStoreError("Failed to upsert Milvus records") from exc

    async def search_by_vector(
        self,
        vector: EmbeddingVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        try:
            collection_exists = await self._ensure_collection()
            if not collection_exists:
                return []
            safe_top_k = max(1, min(top_k, 100))
            filter_expr = self._build_filter_expression(filters or {})
            result = await self._call_sync(
                self.client.search,
                collection_name=self.collection_name,
                data=[vector.values],
                anns_field="vector",
                limit=safe_top_k,
                filter=filter_expr,
                output_fields=self._OUTPUT_FIELDS,
                search_params=self._search_params(safe_top_k),
            )
            return self._search_results_to_hits(result)
        except Exception as exc:
            logger.exception("Failed to search Milvus records")
            raise VectorStoreError("Failed to search Milvus records") from exc

    async def search_similar_text(self, text: str, top_k: int) -> list[RetrievalHit]:
        if self.embedding_adapter is None:
            raise VectorStoreError("Embedding adapter is required for text similarity search")
        vector = await self.embedding_adapter.embed_text(text)
        return await self.search_by_vector(vector=vector, top_k=top_k, filters=None)

    async def search_sparse_text(
        self,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        try:
            collection_exists = await self._ensure_collection()
            if not collection_exists:
                return []
            filter_expr = self._build_filter_expression(filters or {})
            candidate_limit = max(100, min(max(top_k, 1) * 20, 1000))
            rows = await self._call_sync(
                self.client.query,
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=self._OUTPUT_FIELDS,
                limit=candidate_limit,
            )
            hits = self._query_results_to_hits(rows)
            contents = [hit.content or "" for hit in hits]
            scores = bm25_score_texts(query=text, texts=contents)
            rescored: list[RetrievalHit] = []
            for hit, score in zip(hits, scores, strict=True):
                if score <= 0:
                    continue
                rescored.append(hit.model_copy(update={"sparse_score": score}))
            return sorted(rescored, key=lambda item: item.sparse_score or 0.0, reverse=True)[:top_k]
        except Exception as exc:
            logger.exception("Failed to search Milvus records with sparse retrieval")
            raise VectorStoreError("Failed to search Milvus records with sparse retrieval") from exc

    async def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            collection_exists = await self._ensure_collection()
            if not collection_exists:
                return
            await self._call_sync(
                self.client.delete,
                collection_name=self.collection_name,
                filter=f'document_id == "{self._escape_string(doc_id)}"',
            )
        except Exception as exc:
            logger.exception("Failed to delete Milvus records", extra={"document_id": doc_id})
            raise VectorStoreError("Failed to delete Milvus records") from exc

    async def reset_collection(self) -> None:
        await self.connect()
        exists = await self._call_sync(
            self.client.has_collection,
            collection_name=self.collection_name,
        )
        if exists:
            await self._call_sync(
                self.client.drop_collection,
                collection_name=self.collection_name,
            )
        self._collection_loaded = False
        if self.dimension is not None:
            await self._ensure_collection(self.dimension)

    async def _ensure_collection(self, dimension: int | None = None) -> bool:
        await self.connect()
        exists = await self._call_sync(
            self.client.has_collection,
            collection_name=self.collection_name,
        )
        if not exists:
            target_dimension = dimension or self.dimension
            if target_dimension is None:
                return False
            await self._create_collection(target_dimension)
            return True
        target_dimension = dimension or self.dimension
        if target_dimension is not None:
            existing_dimension = await self._get_collection_dimension()
            if existing_dimension is not None and existing_dimension != target_dimension:
                raise VectorStoreError(
                    "Milvus collection dimension mismatch: "
                    f"collection='{self.collection_name}' existing_dim={existing_dimension} "
                    f"requested_dim={target_dimension}. "
                    "Update MILVUS_DIMENSION to match the embedding model and reset the collection "
                    "with `python scripts/reset_milvus_collection.py`."
                )
        if not self._collection_loaded:
            await self._load_collection_quietly()
        return True

    async def _create_collection(self, dimension: int) -> None:
        schema = self._build_schema(dimension)
        index_params = self._build_index_params()
        await self._call_sync(
            self.client.create_collection,
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        await self._load_collection_quietly()
        self.dimension = dimension
        logger.info(
            "Created Milvus collection",
            extra={
                "collection_name": self.collection_name,
                "dimension": dimension,
                "metric_type": self.metric_type,
                "index_type": self.index_type,
            },
        )

    async def _load_collection_quietly(self) -> None:
        try:
            await self._call_sync(
                self.client.load_collection,
                collection_name=self.collection_name,
            )
            self._collection_loaded = True
        except Exception:
            logger.info("Milvus collection load skipped", exc_info=True)

    def _build_client(self) -> Any:
        from pymilvus import MilvusClient

        kwargs: dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        if self.db_name:
            kwargs["db_name"] = self.db_name
        return MilvusClient(**kwargs)

    async def _get_collection_dimension(self) -> int | None:
        try:
            description = await self._call_sync(
                self.client.describe_collection,
                collection_name=self.collection_name,
            )
        except Exception:
            logger.info("Milvus collection description skipped", exc_info=True)
            return None

        fields = description.get("fields", []) if isinstance(description, dict) else []
        for field in fields:
            if field.get("name") != "vector":
                continue
            params = field.get("params") or {}
            raw_dim = params.get("dim")
            if raw_dim is None:
                return None
            try:
                return int(raw_dim)
            except (TypeError, ValueError):
                return None
        return None

    def _build_schema(self, dimension: int) -> Any:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("document_id", DataType.VARCHAR, max_length=512)
        schema.add_field("source_type", DataType.VARCHAR, max_length=64)
        schema.add_field("source_id", DataType.VARCHAR, max_length=512)
        schema.add_field("modality", DataType.VARCHAR, max_length=64)
        schema.add_field("namespace", DataType.VARCHAR, max_length=128)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=256)
        schema.add_field("embedding_dimensions", DataType.INT64)
        schema.add_field("uri", DataType.VARCHAR, max_length=2048)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=65535)
        return schema

    def _build_index_params(self) -> Any:
        index_params = self.client.prepare_index_params()
        params: dict[str, Any] = {}
        if self.index_type == "HNSW":
            params = {"M": 16, "efConstruction": 200}
        index_params.add_index(
            field_name="vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=params,
        )
        return index_params

    def _search_params(self, top_k: int) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.index_type == "HNSW":
            params["ef"] = max(64, top_k)
        return {"metric_type": self.metric_type, "params": params}

    def _record_to_entity(self, record: MultimodalEmbeddingRecord) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in {**record.item.metadata, **record.metadata}.items()
            if key not in self._RESERVED_METADATA_KEYS
        }
        return {
            "id": self._truncate(record.id, 512),
            "vector": record.embedding.values,
            "content": self._truncate(record.item.content or "", 65535),
            "document_id": self._truncate(record.item.document_id, 512),
            "source_type": self._truncate(record.item.source_type, 64),
            "source_id": self._truncate(record.item.source_id, 512),
            "modality": self._truncate(record.modality, 64),
            "namespace": self._truncate(record.namespace, 128),
            "embedding_model": self._truncate(record.embedding.model, 256),
            "embedding_dimensions": record.embedding.dimensions,
            "uri": self._truncate(record.item.uri or "", 2048),
            "metadata_json": self._truncate(
                json.dumps(metadata, ensure_ascii=False, default=str),
                65535,
            ),
        }

    def _search_results_to_hits(self, result: Any) -> list[RetrievalHit]:
        rows = (result or [[]])[0]
        return self._rows_to_hits(rows)

    def _query_results_to_hits(self, rows: Any) -> list[RetrievalHit]:
        return self._rows_to_hits(rows or [])

    def _rows_to_hits(self, rows: Any) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for row in rows:
            entity = dict(row.get("entity") or row)
            record_id = row.get("id") or entity.get("id")
            metadata = self._metadata_json_to_dict(entity.pop("metadata_json", None))
            metadata = {**metadata}
            self._add_metadata_if_present(metadata, "uri", entity.get("uri"))
            self._add_metadata_if_present(metadata, "modality", entity.get("modality"))
            self._add_metadata_if_present(metadata, "namespace", entity.get("namespace"))
            self._add_metadata_if_present(metadata, "embedding_model", entity.get("embedding_model"))
            self._add_metadata_if_present(
                metadata,
                "embedding_dimensions",
                entity.get("embedding_dimensions"),
            )
            source_type = self._normalize_source_type(
                raw_source_type=entity.get("source_type"),
                modality=entity.get("modality"),
            )
            hits.append(
                RetrievalHit(
                    id=str(record_id),
                    source_type=source_type,
                    source_id=str(entity.get("source_id") or record_id),
                    document_id=str(entity["document_id"]) if entity.get("document_id") else None,
                    content=entity.get("content"),
                    sparse_score=row.get("sparse_score"),
                    vector_score=self._distance_to_score(row.get("distance", row.get("score"))),
                    metadata=metadata,
                )
            )
        return hits

    def _distance_to_score(self, distance: Any) -> float | None:
        if distance is None:
            return None
        value = float(distance)
        if self.metric_type == "L2":
            return max(1.0 / (1.0 + value), 0.0)
        return max(value, 0.0)

    def _metadata_json_to_dict(self, raw_value: Any) -> dict[str, Any]:
        if not raw_value:
            return {}
        if isinstance(raw_value, dict):
            return dict(raw_value)
        try:
            value = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _normalize_source_type(self, *, raw_source_type: Any, modality: Any) -> str:
        if isinstance(raw_source_type, str) and raw_source_type in self._VALID_SOURCE_TYPES:
            return raw_source_type
        if modality == "text":
            return "text_block"
        if modality == "page":
            return "page"
        if modality == "chart":
            return "chart"
        if modality == "image":
            return "image_region"
        return "text_block"

    def _build_filter_expression(self, filters: dict[str, Any]) -> str:
        clauses: list[str] = []
        self._add_sequence_filter(clauses, "document_id", filters.get("document_ids"))
        self._add_sequence_filter(clauses, "source_type", filters.get("source_types"))
        self._add_sequence_filter(clauses, "modality", filters.get("modalities"))
        if namespace := filters.get("namespace"):
            clauses.append(f'namespace == "{self._escape_string(namespace)}"')
        return " and ".join(clauses)

    def _add_sequence_filter(
        self,
        clauses: list[str],
        field_name: str,
        values: Sequence[str] | None,
    ) -> None:
        if not values:
            return
        escaped_values = ", ".join(f'"{self._escape_string(value)}"' for value in values)
        clauses.append(f"{field_name} in [{escaped_values}]")

    def _add_metadata_if_present(self, metadata: dict[str, Any], key: str, value: Any) -> None:
        if value in (None, ""):
            return
        metadata[key] = value

    def _escape_string(self, value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def _truncate(self, value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        return text[:limit]

    def _safe_collection_name(self, collection_name: str) -> str:
        normalized = collection_name.replace("-", "_")
        if not normalized:
            raise ValueError("collection_name must not be empty")
        if not normalized.replace("_", "").isalnum():
            raise ValueError("collection_name must contain only letters, numbers, hyphens, and underscores")
        if normalized[0].isdigit():
            normalized = f"c_{normalized}"
        return normalized[:255]

    async def _call_sync(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if isawaitable(result):
            return await result
        return result
