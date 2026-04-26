import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from adapters.embedding.base import BaseEmbeddingAdapter, EmbeddingAdapterError
from adapters.vector_store.base import BaseVectorStore, VectorStoreError
from domain.schemas.chart import ChartSchema
from domain.schemas.document import DocumentPage, TextBlock
from domain.schemas.embedding import EmbeddingItem, EmbeddingVector, MultimodalEmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingIndexServiceError(RuntimeError):
    """Raised when embedding indexing fails."""


class EmbeddingIndexResult(BaseModel):
    document_id: str
    status: Literal["indexed", "partial", "skipped", "failed"]
    record_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    record_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingIndexService:
    def __init__(
        self,
        embedding_adapter: BaseEmbeddingAdapter,
        vector_store: BaseVectorStore,
        namespace: str = "default",
    ) -> None:
        self.embedding_adapter = embedding_adapter
        self.vector_store = vector_store
        self.namespace = namespace

    async def index_text_blocks(self, document_id: str, text_blocks: list[TextBlock]) -> EmbeddingIndexResult:
        blocks = [block for block in text_blocks if block.text.strip()]
        skipped_count = len(text_blocks) - len(blocks)
        if not blocks:
            return EmbeddingIndexResult(
                document_id=document_id,
                status="skipped",
                skipped_count=skipped_count,
                metadata={"source_type": "text_block", "reason": "empty_input"},
            )
        try:
            vectors = await self.embedding_adapter.embed_texts([block.text for block in blocks])
            records = [
                self._text_block_record(block=block, vector=vector)
                for block, vector in zip(blocks, vectors, strict=True)
            ]
            await self.vector_store.upsert_embeddings(records)
            logger.info(
                "Indexed text block embeddings",
                extra={"document_id": document_id, "record_count": len(records)},
            )
            return EmbeddingIndexResult(
                document_id=document_id,
                status="indexed" if skipped_count == 0 else "partial",
                record_count=len(records),
                skipped_count=skipped_count,
                record_ids=[record.id for record in records],
                metadata={"source_type": "text_block"},
            )
        except (EmbeddingAdapterError, VectorStoreError, ValueError) as exc:
            logger.exception("Failed to index text block embeddings", extra={"document_id": document_id})
            raise EmbeddingIndexServiceError("Failed to index text block embeddings") from exc

    async def index_pages(self, document_id: str, pages: list[DocumentPage]) -> EmbeddingIndexResult:
        indexable_pages = [page for page in pages if page.image_uri]
        skipped_count = len(pages) - len(indexable_pages)
        if not indexable_pages:
            return EmbeddingIndexResult(
                document_id=document_id,
                status="skipped",
                skipped_count=skipped_count,
                metadata={"source_type": "page", "reason": "missing_page_images"},
            )

        records: list[MultimodalEmbeddingRecord] = []
        failed_count = 0
        for page in indexable_pages:
            try:
                page_text = self._page_text(page)
                vector = await self.embedding_adapter.embed_page(
                    page_image_path=self._local_path(page.image_uri),
                    page_text=page_text,
                )
                records.append(self._page_record(page=page, page_text=page_text, vector=vector))
            except (EmbeddingAdapterError, OSError, ValueError) as exc:
                failed_count += 1
                logger.warning(
                    "Failed to embed document page",
                    extra={"document_id": document_id, "page_id": page.id},
                    exc_info=exc,
                )

        return await self._write_records(
            document_id=document_id,
            records=records,
            skipped_count=skipped_count,
            failed_count=failed_count,
            source_type="page",
        )

    async def index_charts(self, document_id: str, charts: list[ChartSchema]) -> EmbeddingIndexResult:
        indexable_charts = [
            chart for chart in charts if self._chart_image_uri(chart) and (chart.summary or chart.title)
        ]
        skipped_count = len(charts) - len(indexable_charts)
        if not indexable_charts:
            return EmbeddingIndexResult(
                document_id=document_id,
                status="skipped",
                skipped_count=skipped_count,
                metadata={"source_type": "chart", "reason": "missing_chart_image_or_summary"},
            )

        records: list[MultimodalEmbeddingRecord] = []
        failed_count = 0
        for chart in indexable_charts:
            try:
                chart_image_path = self._local_path(self._chart_image_uri(chart) or "")
                chart_summary = self._chart_summary(chart)
                vector = await self.embedding_adapter.embed_chart(
                    chart_image_path=chart_image_path,
                    chart_summary=chart_summary,
                )
                records.append(self._chart_record(chart=chart, chart_summary=chart_summary, vector=vector))
            except (EmbeddingAdapterError, OSError, ValueError) as exc:
                failed_count += 1
                logger.warning(
                    "Failed to embed chart",
                    extra={"document_id": document_id, "chart_id": chart.id},
                    exc_info=exc,
                )

        return await self._write_records(
            document_id=document_id,
            records=records,
            skipped_count=skipped_count,
            failed_count=failed_count,
            source_type="chart",
        )

    async def _write_records(
        self,
        document_id: str,
        records: list[MultimodalEmbeddingRecord],
        skipped_count: int,
        failed_count: int,
        source_type: str,
    ) -> EmbeddingIndexResult:
        if not records:
            return EmbeddingIndexResult(
                document_id=document_id,
                status="failed" if failed_count else "skipped",
                skipped_count=skipped_count,
                failed_count=failed_count,
                error_message="No embedding records were created" if failed_count else None,
                metadata={"source_type": source_type},
            )
        try:
            await self.vector_store.upsert_embeddings(records)
            status: Literal["indexed", "partial"] = (
                "indexed" if skipped_count == 0 and failed_count == 0 else "partial"
            )
            logger.info(
                "Indexed embedding records",
                extra={
                    "document_id": document_id,
                    "source_type": source_type,
                    "record_count": len(records),
                },
            )
            return EmbeddingIndexResult(
                document_id=document_id,
                status=status,
                record_count=len(records),
                skipped_count=skipped_count,
                failed_count=failed_count,
                record_ids=[record.id for record in records],
                metadata={"source_type": source_type},
            )
        except VectorStoreError as exc:
            logger.exception(
                "Failed to write embedding records",
                extra={"document_id": document_id, "source_type": source_type},
            )
            raise EmbeddingIndexServiceError("Failed to write embedding records") from exc

    def _text_block_record(
        self,
        block: TextBlock,
        vector: EmbeddingVector,
    ) -> MultimodalEmbeddingRecord:
        return MultimodalEmbeddingRecord(
            id=self._record_id(block.document_id, "text_block", block.id, vector.model),
            item=EmbeddingItem(
                id=block.id,
                document_id=block.document_id,
                source_type="text_block",
                source_id=block.id,
                content=block.text,
                metadata={
                    "page_id": block.page_id,
                    "page_number": block.page_number,
                    "block_type": block.block_type,
                    **block.metadata,
                },
            ),
            embedding=vector,
            modality="text",
            namespace=self.namespace,
            metadata={"indexed_by": self.__class__.__name__},
        )

    def _page_record(
        self,
        page: DocumentPage,
        page_text: str,
        vector: EmbeddingVector,
    ) -> MultimodalEmbeddingRecord:
        return MultimodalEmbeddingRecord(
            id=self._record_id(page.document_id, "page", page.id, vector.model),
            item=EmbeddingItem(
                id=page.id,
                document_id=page.document_id,
                source_type="page",
                source_id=page.id,
                content=page_text,
                uri=page.image_uri,
                metadata={"page_number": page.page_number, **page.metadata},
            ),
            embedding=vector,
            modality="page",
            namespace=self.namespace,
            metadata={"indexed_by": self.__class__.__name__},
        )

    def _chart_record(
        self,
        chart: ChartSchema,
        chart_summary: str,
        vector: EmbeddingVector,
    ) -> MultimodalEmbeddingRecord:
        return MultimodalEmbeddingRecord(
            id=self._record_id(chart.document_id, "chart", chart.id, vector.model),
            item=EmbeddingItem(
                id=chart.id,
                document_id=chart.document_id,
                source_type="chart",
                source_id=chart.id,
                content=chart_summary,
                uri=self._chart_image_uri(chart),
                metadata={
                    "page_id": chart.page_id,
                    "page_number": chart.page_number,
                    "chart_type": chart.chart_type,
                    "title": chart.title,
                    **chart.metadata,
                },
            ),
            embedding=vector,
            modality="chart",
            namespace=self.namespace,
            metadata={"indexed_by": self.__class__.__name__},
        )

    def _page_text(self, page: DocumentPage) -> str:
        return "\n".join(block.text for block in page.text_blocks if block.text.strip())

    def _chart_summary(self, chart: ChartSchema) -> str:
        parts = [part for part in [chart.title, chart.caption, chart.summary] if part]
        return "\n".join(parts)

    def _chart_image_uri(self, chart: ChartSchema) -> str | None:
        value = chart.metadata.get("image_uri") or chart.metadata.get("image_path")
        return str(value) if value else None

    def _local_path(self, uri: str) -> str:
        if uri.startswith("file://"):
            return uri.removeprefix("file://")
        return str(Path(uri))

    def _record_id(self, document_id: str, source_type: str, source_id: str, model: str) -> str:
        raw = f"{document_id}:{source_type}:{source_id}:{model}:{self.namespace}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"emb_{digest}"
