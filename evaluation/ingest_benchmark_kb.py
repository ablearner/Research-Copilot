from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.runtime import build_rag_runtime, close_rag_runtime, initialize_rag_runtime  # noqa: E402
from core.config import get_settings  # noqa: E402
from domain.schemas.document import TextBlock  # noqa: E402


VALID_BLOCK_TYPES = {
    "caption",
    "footer",
    "footnote",
    "header",
    "paragraph",
    "table_text",
    "title",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest an evaluation knowledge_base.json into the configured vector store."
    )
    parser.add_argument("--knowledge-base", type=Path, required=True)
    parser.add_argument("--limit-docs", type=int, default=0, help="Use 0 to ingest all documents.")
    parser.add_argument("--batch-size", type=int, default=64, help="Text blocks per embedding batch.")
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    payload = json.loads(args.knowledge_base.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if args.limit_docs > 0:
        documents = documents[: args.limit_docs]

    stats = {
        "knowledge_base": relpath(args.knowledge_base),
        "documents": 0,
        "text_blocks": 0,
        "indexed_records": 0,
        "skipped_blocks": 0,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        for document in documents:
            blocks = text_blocks_for_document(document)
            stats["documents"] += 1
            stats["text_blocks"] += len(blocks)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    runtime = build_rag_runtime(get_settings())
    await initialize_rag_runtime(runtime)
    try:
        pending_blocks: list[TextBlock] = []
        batch_size = max(args.batch_size, 1)
        for document in documents:
            document_id = clean_text(document.get("document_id"))
            if not document_id:
                continue
            blocks = text_blocks_for_document(document)
            stats["documents"] += 1
            stats["text_blocks"] += len(blocks)
            if args.delete_existing:
                await runtime.embedding_index_service.vector_store.delete_by_doc_id(document_id)
            pending_blocks.extend(blocks)
            while len(pending_blocks) >= batch_size:
                batch = pending_blocks[:batch_size]
                pending_blocks = pending_blocks[batch_size:]
                result = await runtime.embedding_index_service.index_text_blocks(
                    f"benchmark_batch_{stats['indexed_records']}",
                    batch,
                )
                stats["indexed_records"] += result.record_count
                stats["skipped_blocks"] += result.skipped_count
            if stats["documents"] % 100 == 0:
                print(
                    "ingested "
                    f"documents={stats['documents']} "
                    f"indexed_records={stats['indexed_records']}"
                )
        if pending_blocks:
            result = await runtime.embedding_index_service.index_text_blocks(
                f"benchmark_batch_{stats['indexed_records']}",
                pending_blocks,
            )
            stats["indexed_records"] += result.record_count
            stats["skipped_blocks"] += result.skipped_count
    finally:
        await close_rag_runtime(runtime)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def text_blocks_for_document(document: dict[str, Any]) -> list[TextBlock]:
    document_id = clean_text(document.get("document_id"))
    blocks = []
    for index, item in enumerate(document.get("text_blocks", [])):
        text = clean_text(item.get("text"))
        if not text:
            continue
        page_number = int(item.get("page_number") or 1)
        block_type = clean_text(item.get("block_type")) or "paragraph"
        if block_type not in VALID_BLOCK_TYPES:
            block_type = "paragraph"
        blocks.append(
            TextBlock(
                id=clean_text(item.get("id")) or f"{document_id}_block_{index}",
                document_id=clean_text(item.get("document_id")) or document_id,
                page_id=clean_text(item.get("page_id")) or f"{document_id}_page_1",
                page_number=max(page_number, 1),
                text=text,
                block_type=block_type,  # type: ignore[arg-type]
                metadata={
                    **dict(document.get("metadata") or {}),
                    **dict(item.get("metadata") or {}),
                    "benchmark_document_id": document_id,
                },
            )
        )
    return blocks


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    asyncio.run(main())
