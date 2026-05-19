from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.local_runtime import LocalDocumentParser  # noqa: E402
from adapters.vector_store.milvus_adapter import MilvusVectorStore  # noqa: E402
from apps.api.runtime import _build_embedding_adapter  # noqa: E402
from core.config import get_settings  # noqa: E402
from domain.schemas.research import PaperCandidate  # noqa: E402
from rag_runtime.services.embedding_index_service import EmbeddingIndexService  # noqa: E402


@dataclass
class PaperSource:
    document_id: str
    storage_uri: str
    title: str = ""
    paper_id: str = ""
    source: str = ""
    origins: set[str] = field(default_factory=set)


@dataclass
class ReindexResult:
    document_id: str
    title: str
    status: str
    pages: int = 0
    text_records: int = 0
    page_records: int = 0
    error: str | None = None


def _resolve_path(value: str) -> Path:
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _merge_source(sources: dict[str, PaperSource], incoming: PaperSource) -> None:
    existing = sources.get(incoming.document_id)
    if existing is None:
        sources[incoming.document_id] = incoming
        return
    existing.origins.update(incoming.origins)
    if not existing.title and incoming.title:
        existing.title = incoming.title
    if not existing.paper_id and incoming.paper_id:
        existing.paper_id = incoming.paper_id
    if not existing.source and incoming.source:
        existing.source = incoming.source
    if not Path(existing.storage_uri).exists() and Path(incoming.storage_uri).exists():
        existing.storage_uri = incoming.storage_uri


def discover_research_papers(root: Path) -> dict[str, PaperSource]:
    sources: dict[str, PaperSource] = {}
    papers_root = root / "papers"
    for path in sorted(papers_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip unreadable paper metadata {path}: {exc}", flush=True)
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            try:
                paper = PaperCandidate.model_validate(item)
            except Exception:
                continue
            metadata = dict(paper.metadata or {})
            document_id = str(metadata.get("document_id") or "").strip()
            storage_uri = str(metadata.get("storage_uri") or "").strip()
            if not document_id or not storage_uri:
                continue
            resolved = _resolve_path(storage_uri)
            _merge_source(
                sources,
                PaperSource(
                    document_id=document_id,
                    storage_uri=str(resolved),
                    title=paper.title,
                    paper_id=paper.paper_id,
                    source=str(paper.source),
                    origins={f"research_metadata:{path.name}"},
                ),
            )
    return sources


def _find_upload_pdf(document_id: str) -> Path | None:
    upload_root = PROJECT_ROOT / ".data" / "uploads"
    matches = sorted(upload_root.glob(f"{document_id}*.pdf"))
    return matches[0].resolve() if matches else None


async def discover_current_milvus_papers(settings: Any) -> dict[str, PaperSource]:
    store = MilvusVectorStore(
        collection_name=settings.milvus_collection_name,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        dimension=settings.milvus_dimension,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
    )
    sources: dict[str, PaperSource] = {}
    try:
        await store.connect()
        exists = await store._call_sync(  # noqa: SLF001
            store.client.has_collection,
            collection_name=store.collection_name,
        )
        if not exists:
            return sources
        rows = await store._call_sync(  # noqa: SLF001
            store.client.query,
            collection_name=store.collection_name,
            filter='document_id != ""',
            output_fields=["document_id"],
            limit=10000,
        )
    except Exception as exc:
        print(f"skip current Milvus discovery: {exc}", flush=True)
        return sources
    finally:
        await store.close()

    document_ids = sorted({str(row.get("document_id") or "") for row in rows if row.get("document_id")})
    for document_id in document_ids:
        if not document_id.startswith("paper_"):
            continue
        path = _find_upload_pdf(document_id)
        if path is None:
            continue
        _merge_source(
            sources,
            PaperSource(
                document_id=document_id,
                storage_uri=str(path),
                title=path.stem,
                origins={"current_milvus"},
            ),
        )
    return sources


async def reset_collection(settings: Any, embedding_adapter: Any) -> MilvusVectorStore:
    vector_store = MilvusVectorStore(
        collection_name=settings.milvus_collection_name,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        dimension=settings.milvus_dimension,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
        embedding_adapter=embedding_adapter,
    )
    await vector_store.reset_collection()
    return vector_store


async def reindex_one(
    *,
    parser: LocalDocumentParser,
    index_service: EmbeddingIndexService,
    source: PaperSource,
    skip_pages: bool = False,
) -> ReindexResult:
    parsed = await parser.parse_document(source.storage_uri, document_id=source.document_id)
    text_blocks = [block for page in parsed.pages for block in page.text_blocks]
    text_result = await index_service.index_text_blocks(parsed.id, text_blocks)
    page_result = None if skip_pages else await index_service.index_pages(parsed.id, parsed.pages)
    return ReindexResult(
        document_id=source.document_id,
        title=source.title,
        status="indexed",
        pages=len(parsed.pages),
        text_records=text_result.record_count,
        page_records=page_result.record_count if page_result is not None else 0,
    )


async def verify_sparse_search(settings: Any, query: str) -> dict[str, Any]:
    store = MilvusVectorStore(
        collection_name=settings.milvus_collection_name,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        dimension=settings.milvus_dimension,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
    )
    try:
        await store.connect()
        description = await store._call_sync(  # noqa: SLF001
            store.client.describe_collection,
            collection_name=store.collection_name,
        )
        stats = await store._call_sync(  # noqa: SLF001
            store.client.get_collection_stats,
            store.collection_name,
        )
        sparse_hits = await store.search_sparse_text(query, top_k=3)
        return {
            "fields": [field.get("name") for field in description.get("fields", [])],
            "functions": [function.get("name") for function in description.get("functions", [])],
            "stats": stats,
            "sparse_top": [
                {
                    "document_id": hit.document_id,
                    "source_type": hit.source_type,
                    "score": hit.sparse_score,
                    "snippet": (hit.content or "")[:120],
                }
                for hit in sparse_hits
            ],
        }
    finally:
        await store.close()


async def amain() -> int:
    parser = argparse.ArgumentParser(description="Reset Milvus and reindex imported research papers.")
    parser.add_argument("--no-reset", action="store_true", help="Do not drop/recreate the Milvus collection.")
    parser.add_argument("--limit", type=int, default=0, help="Only index the first N papers, for debugging.")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Only index this document_id. May be passed multiple times.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed paper.")
    parser.add_argument("--skip-pages", action="store_true", help="Only index text blocks; skip page records.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.vector_store_provider.lower() not in {"milvus", "zilliz"}:
        raise RuntimeError(f"VECTOR_STORE_PROVIDER={settings.vector_store_provider!r} is not Milvus")

    research_sources = discover_research_papers(settings.resolve_path(settings.research_storage_root))
    current_sources = await discover_current_milvus_papers(settings)
    sources = dict(research_sources)
    for source in current_sources.values():
        _merge_source(sources, source)

    valid_sources: list[PaperSource] = []
    missing_sources: list[PaperSource] = []
    for source in sorted(sources.values(), key=lambda item: item.document_id):
        if Path(source.storage_uri).exists():
            valid_sources.append(source)
        else:
            missing_sources.append(source)
    if args.document_id:
        wanted = set(args.document_id)
        valid_sources = [source for source in valid_sources if source.document_id in wanted]
    if args.limit > 0:
        valid_sources = valid_sources[: args.limit]
    if not valid_sources:
        print("No imported research papers with existing PDF files were found.", flush=True)
        return 1

    print(
        json.dumps(
            {
                "research_metadata_docs": len(research_sources),
                "current_milvus_paper_docs": len(current_sources),
                "deduped_valid_docs": len(valid_sources),
                "missing_pdf_docs": len(missing_sources),
                "collection": settings.milvus_collection_name,
                "milvus_uri": settings.milvus_uri,
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.milvus_dimension,
                "reset": not args.no_reset,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    embedding_adapter = _build_embedding_adapter(settings)
    vector_store = await reset_collection(settings, embedding_adapter) if not args.no_reset else MilvusVectorStore(
        collection_name=settings.milvus_collection_name,
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        dimension=settings.milvus_dimension,
        metric_type=settings.milvus_metric_type,
        index_type=settings.milvus_index_type,
        embedding_adapter=embedding_adapter,
    )
    parser_service = LocalDocumentParser(storage_root=settings.resolve_path(settings.local_storage_root))
    index_service = EmbeddingIndexService(embedding_adapter, vector_store)

    started = time.perf_counter()
    results: list[ReindexResult] = []
    try:
        for index, source in enumerate(valid_sources, start=1):
            paper_started = time.perf_counter()
            try:
                result = await reindex_one(
                    parser=parser_service,
                    index_service=index_service,
                    source=source,
                    skip_pages=args.skip_pages,
                )
                elapsed = time.perf_counter() - paper_started
                print(
                    f"[{index}/{len(valid_sources)}] indexed {source.document_id} "
                    f"pages={result.pages} text_records={result.text_records} "
                    f"page_records={result.page_records} elapsed={elapsed:.1f}s "
                    f"title={source.title[:90]}",
                    flush=True,
                )
                results.append(result)
            except Exception as exc:
                elapsed = time.perf_counter() - paper_started
                message = f"{exc.__class__.__name__}: {exc}"
                print(
                    f"[{index}/{len(valid_sources)}] failed {source.document_id} "
                    f"elapsed={elapsed:.1f}s error={message}",
                    flush=True,
                )
                results.append(
                    ReindexResult(
                        document_id=source.document_id,
                        title=source.title,
                        status="failed",
                        error=message,
                    )
                )
                if args.fail_fast:
                    break
    finally:
        await vector_store.close()
        close = getattr(embedding_adapter, "close", None)
        if close:
            await close()

    indexed = [result for result in results if result.status == "indexed"]
    failed = [result for result in results if result.status != "indexed"]
    query_source = next((source for source in valid_sources if source.title), valid_sources[0])
    verification = await verify_sparse_search(settings, query_source.title or query_source.document_id)
    summary = {
        "indexed_docs": len(indexed),
        "failed_docs": len(failed),
        "total_pages": sum(result.pages for result in indexed),
        "total_text_records": sum(result.text_records for result in indexed),
        "total_page_records": sum(result.page_records for result in indexed),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "failed": [
            {"document_id": result.document_id, "title": result.title, "error": result.error}
            for result in failed
        ],
        "verification": verification,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failed else 2


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
