import logging
from collections.abc import Sequence
from typing import Any

from adapters.embedding.base import BaseEmbeddingAdapter
from adapters.vector_store.base import BaseVectorStore, VectorStoreError
from domain.schemas.embedding import EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.retrieval import RetrievalHit

logger = logging.getLogger(__name__)


class PgVectorStore(BaseVectorStore):
    _RESERVED_METADATA_KEYS = {
        "document_id",
        "source_type",
        "source_id",
        "modality",
        "namespace",
        "uri",
    }

    def __init__(
        self,
        dsn: str | None = None,
        table_name: str = "multimodal_embeddings",
        connection: Any | None = None,
        embedding_adapter: BaseEmbeddingAdapter | None = None,
    ) -> None:
        self.dsn = dsn
        self.table_name = self._safe_table_name(table_name)
        self.connection = connection
        self.embedding_adapter = embedding_adapter
        self._owns_connection = connection is None

    async def connect(self) -> None:
        if self.connection is not None:
            return
        if not self.dsn:
            raise VectorStoreError("Postgres DSN is required when no connection is provided")
        try:
            import psycopg

            self.connection = await psycopg.AsyncConnection.connect(self.dsn)
            logger.info("Connected to pgvector store")
        except Exception as exc:
            logger.exception("Failed to connect to pgvector store")
            raise VectorStoreError("Failed to connect to pgvector store") from exc

    async def ensure_schema(self) -> None:
        try:
            await self._ensure_connected()
            async with self.connection.cursor() as cursor:
                await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        modality TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        content TEXT,
                        uri TEXT,
                        embedding_model TEXT NOT NULL,
                        embedding_dimensions INTEGER NOT NULL,
                        embedding vector NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                await cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_document_id_idx "
                    f"ON {self.table_name} (document_id)"
                )
                await cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_source_type_idx "
                    f"ON {self.table_name} (source_type)"
                )
            await self.connection.commit()
        except Exception as exc:
            await self._rollback_quietly()
            logger.exception("Failed to ensure pgvector schema")
            raise VectorStoreError("Failed to ensure pgvector schema") from exc

    async def close(self) -> None:
        if self.connection is None or not self._owns_connection:
            return
        try:
            await self.connection.close()
            self.connection = None
            logger.info("Closed pgvector store connection")
        except Exception as exc:
            logger.exception("Failed to close pgvector store connection")
            raise VectorStoreError("Failed to close pgvector store connection") from exc

    async def upsert_embedding(self, record: MultimodalEmbeddingRecord) -> None:
        await self.upsert_embeddings([record])

    async def upsert_embeddings(self, records: list[MultimodalEmbeddingRecord]) -> None:
        if not records:
            return
        try:
            await self._ensure_connected()
            query = f"""
                INSERT INTO {self.table_name} (
                    id,
                    document_id,
                    source_type,
                    source_id,
                    modality,
                    namespace,
                    content,
                    uri,
                    embedding_model,
                    embedding_dimensions,
                    embedding,
                    metadata
                )
                VALUES (
                    %(id)s,
                    %(document_id)s,
                    %(source_type)s,
                    %(source_id)s,
                    %(modality)s,
                    %(namespace)s,
                    %(content)s,
                    %(uri)s,
                    %(embedding_model)s,
                    %(embedding_dimensions)s,
                    %(embedding)s,
                    %(metadata)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    source_type = EXCLUDED.source_type,
                    source_id = EXCLUDED.source_id,
                    modality = EXCLUDED.modality,
                    namespace = EXCLUDED.namespace,
                    content = EXCLUDED.content,
                    uri = EXCLUDED.uri,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
            """
            async with self.connection.cursor() as cursor:
                await cursor.executemany(query, [self._record_to_row(record) for record in records])
            await self.connection.commit()
        except Exception as exc:
            await self._rollback_quietly()
            logger.exception("Failed to upsert pgvector records", extra={"record_count": len(records)})
            raise VectorStoreError("Failed to upsert pgvector records") from exc

    async def search_by_vector(
        self,
        vector: EmbeddingVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        try:
            await self._ensure_connected()
            safe_top_k = max(1, min(top_k, 100))
            filter_clause, params = self._build_filter_clause(filters or {})
            params.update({"embedding": vector.values, "top_k": safe_top_k})
            query = f"""
                SELECT
                    id,
                    document_id,
                    source_type,
                    source_id,
                    content,
                    uri,
                    modality,
                    namespace,
                    metadata,
                    1 - (embedding <=> %(embedding)s::vector) AS vector_score
                FROM {self.table_name}
                {filter_clause}
                ORDER BY embedding <=> %(embedding)s::vector
                LIMIT %(top_k)s
            """
            async with self.connection.cursor() as cursor:
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                columns = [column.name for column in cursor.description]
            return [self._row_to_hit(dict(zip(columns, row, strict=True))) for row in rows]
        except Exception as exc:
            logger.exception("Failed to search pgvector records")
            raise VectorStoreError("Failed to search pgvector records") from exc

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
            await self._ensure_connected()
            safe_top_k = max(1, min(top_k, 100))
            filter_clause, params = self._build_filter_clause(filters or {})
            params.update({"query_text": text, "top_k": safe_top_k})
            ts_query = "plainto_tsquery('simple', %(query_text)s)"
            search_clause = f"to_tsvector('simple', coalesce(content, '')) @@ {ts_query}"
            where_clause = f"WHERE {search_clause}" if not filter_clause else f"{filter_clause} AND {search_clause}"
            query = f"""
                SELECT
                    id,
                    document_id,
                    source_type,
                    source_id,
                    content,
                    uri,
                    modality,
                    namespace,
                    metadata,
                    ts_rank_cd(
                        to_tsvector('simple', coalesce(content, '')),
                        {ts_query}
                    ) AS sparse_score
                FROM {self.table_name}
                {where_clause}
                ORDER BY sparse_score DESC, updated_at DESC
                LIMIT %(top_k)s
            """
            async with self.connection.cursor() as cursor:
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                columns = [column.name for column in cursor.description]
            return [self._row_to_hit(dict(zip(columns, row, strict=True))) for row in rows]
        except Exception as exc:
            logger.exception("Failed to search pgvector records with sparse retrieval")
            raise VectorStoreError("Failed to search pgvector records with sparse retrieval") from exc

    async def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            await self._ensure_connected()
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE document_id = %(document_id)s",
                    {"document_id": doc_id},
                )
            await self.connection.commit()
        except Exception as exc:
            await self._rollback_quietly()
            logger.exception("Failed to delete pgvector records", extra={"document_id": doc_id})
            raise VectorStoreError("Failed to delete pgvector records") from exc

    async def _ensure_connected(self) -> None:
        if self.connection is None:
            await self.connect()

    async def _rollback_quietly(self) -> None:
        if self.connection is None:
            return
        try:
            await self.connection.rollback()
        except Exception:
            logger.warning("Failed to rollback pgvector transaction", exc_info=True)

    def _record_to_row(self, record: MultimodalEmbeddingRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "document_id": record.item.document_id,
            "source_type": record.item.source_type,
            "source_id": record.item.source_id,
            "modality": record.modality,
            "namespace": record.namespace,
            "content": record.item.content,
            "uri": record.item.uri,
            "embedding_model": record.embedding.model,
            "embedding_dimensions": record.embedding.dimensions,
            "embedding": record.embedding.values,
            "metadata": self._jsonb(
                {
                    key: value
                    for key, value in {**record.item.metadata, **record.metadata}.items()
                    if key not in self._RESERVED_METADATA_KEYS
                }
            ),
        }

    def _row_to_hit(self, row: dict[str, Any]) -> RetrievalHit:
        metadata = row.get("metadata") or {}
        if row.get("uri"):
            metadata = {**metadata, "uri": row["uri"]}
        if row.get("modality"):
            metadata = {**metadata, "modality": row["modality"]}
        if row.get("namespace"):
            metadata = {**metadata, "namespace": row["namespace"]}
        return RetrievalHit(
            id=str(row["id"]),
            source_type=row["source_type"],
            source_id=str(row["source_id"]),
            document_id=str(row["document_id"]) if row.get("document_id") else None,
            content=row.get("content"),
            sparse_score=row.get("sparse_score"),
            vector_score=row.get("vector_score"),
            metadata=metadata,
        )

    def _build_filter_clause(self, filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        self._add_sequence_filter(clauses, params, "document_id", filters.get("document_ids"))
        self._add_sequence_filter(clauses, params, "source_type", filters.get("source_types"))
        self._add_sequence_filter(clauses, params, "modality", filters.get("modalities"))
        if namespace := filters.get("namespace"):
            clauses.append("namespace = %(namespace)s")
            params["namespace"] = namespace
        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params

    def _add_sequence_filter(
        self,
        clauses: list[str],
        params: dict[str, Any],
        column: str,
        values: Sequence[str] | None,
    ) -> None:
        if not values:
            return
        param_name = f"{column}_values"
        clauses.append(f"{column} = ANY(%({param_name})s)")
        params[param_name] = list(values)

    def _safe_table_name(self, table_name: str) -> str:
        if not table_name.replace("_", "").isalnum():
            raise ValueError("table_name must contain only letters, numbers, and underscores")
        return table_name

    def _jsonb(self, value: dict[str, Any]) -> Any:
        try:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        except Exception:
            return value
