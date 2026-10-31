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

from evaluation.runner import evaluate_cases  # noqa: E402
from evaluation.sample_runtime import build_sample_runtime  # noqa: E402
from evaluation.schemas import EvaluationCase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent/RAG system metrics for Research-Copilot.")
    parser.add_argument(
        "--runtime",
        choices=["sample", "live"],
        default="sample",
        help="Use the deterministic sample runtime or the live runtime from current settings.",
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).resolve().with_name("sample_cases.json")),
        help="Path to the evaluation cases JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to run from the selected case file.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the specified case id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=5,
        help="Top-k cutoff for Recall@k and Hit@k. Defaults to 5.",
    )
    parser.add_argument(
        "--knowledge-base",
        action="append",
        default=[],
        help=(
            "Optional benchmark knowledge_base.json to ingest into the runtime before evaluation. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--delete-existing-knowledge-base",
        action="store_true",
        help="Delete existing vector records for each benchmark document before ingesting.",
    )
    parser.add_argument(
        "--knowledge-base-limit-docs",
        type=int,
        default=0,
        help="Optional max documents to ingest from each knowledge base. Use 0 for all.",
    )
    parser.add_argument(
        "--knowledge-base-batch-size",
        type=int,
        default=64,
        help="Number of text blocks to embed per benchmark ingestion batch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print one progress line every N completed cases. Use 0 to disable.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases))
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case.id in selected_ids]
        missing_ids = sorted(selected_ids.difference({case.id for case in cases}))
        if missing_ids:
            raise ValueError(f"Requested case id(s) not found: {', '.join(missing_ids)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        cases = cases[: args.limit]
    if args.recall_k < 1:
        raise ValueError("--recall-k must be >= 1")
    runtime = None
    initialized = False
    try:
        if args.runtime == "live":
            from apps.api.runtime import build_rag_runtime, close_rag_runtime, initialize_rag_runtime
            from core.config import get_settings

            runtime = build_rag_runtime(get_settings())
            await initialize_rag_runtime(runtime)
            initialized = True
        else:
            runtime = build_sample_runtime()
        knowledge_base_stats = []
        for knowledge_base in args.knowledge_base:
            knowledge_base_stats.append(
                await ingest_knowledge_base(
                    graph_runtime=runtime,
                    path=Path(knowledge_base),
                    delete_existing=args.delete_existing_knowledge_base,
                    limit_docs=args.knowledge_base_limit_docs,
                    batch_size=args.knowledge_base_batch_size,
                )
            )
        report = await evaluate_cases(
            graph_runtime=runtime,
            cases=cases,
            runtime_mode=args.runtime,
            recall_k=args.recall_k,
            progress_callback=_progress_callback(args.progress_every),
        )
        if knowledge_base_stats:
            report.metadata["knowledge_bases"] = knowledge_base_stats
    finally:
        if initialized and runtime is not None:
            from apps.api.runtime import close_rag_runtime

            await close_rag_runtime(runtime)

    payload = report.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("cases", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported cases payload format: {path}")
    return [EvaluationCase.model_validate(item) for item in items]


def _progress_callback(progress_every: int):
    if progress_every <= 0:
        return None

    def callback(index: int, total: int, result: Any) -> None:
        if index % progress_every != 0 and index != total:
            return
        print(
            "[progress] "
            f"{index}/{total} "
            f"case_id={result.case_id} "
            f"success={result.task_success} "
            f"recall_at_k={result.recall_at_k} "
            f"groundedness={result.groundedness} "
            f"latency_ms={result.latency_ms}",
            file=sys.stderr,
            flush=True,
        )

    return callback


async def ingest_knowledge_base(
    *,
    graph_runtime: Any,
    path: Path,
    delete_existing: bool,
    limit_docs: int,
    batch_size: int,
) -> dict[str, Any]:
    from evaluation.ingest_benchmark_kb import text_blocks_for_document

    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if limit_docs > 0:
        documents = documents[:limit_docs]
    stats = {
        "knowledge_base": str(path),
        "documents": 0,
        "text_blocks": 0,
        "indexed_records": 0,
        "skipped_blocks": 0,
    }
    pending_blocks = []
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            continue
        blocks = text_blocks_for_document(document)
        stats["documents"] += 1
        stats["text_blocks"] += len(blocks)
        if delete_existing:
            await graph_runtime.embedding_index_service.vector_store.delete_by_doc_id(document_id)
        pending_blocks.extend(blocks)
        while len(pending_blocks) >= max(batch_size, 1):
            batch = pending_blocks[: max(batch_size, 1)]
            pending_blocks = pending_blocks[max(batch_size, 1) :]
            result = await graph_runtime.embedding_index_service.index_text_blocks(
                f"benchmark_batch_{stats['indexed_records']}",
                batch,
            )
            stats["indexed_records"] += result.record_count
            stats["skipped_blocks"] += result.skipped_count
    if pending_blocks:
        result = await graph_runtime.embedding_index_service.index_text_blocks(
            f"benchmark_batch_{stats['indexed_records']}",
            pending_blocks,
        )
        stats["indexed_records"] += result.record_count
        stats["skipped_blocks"] += result.skipped_count
    return stats


if __name__ == "__main__":
    asyncio.run(main())
