from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.runtime import build_rag_runtime, close_rag_runtime, initialize_rag_runtime  # noqa: E402
from core.config import get_settings  # noqa: E402
from domain.schemas.retrieval import RetrievalHit, RetrievalQuery  # noqa: E402
from evaluation.ingest_benchmark_kb import text_blocks_for_document  # noqa: E402
from evaluation.metrics import percentile, retrieval_recall_at_k  # noqa: E402
from evaluation.run_agent_metrics import load_cases  # noqa: E402
from evaluation.schemas import EvaluationCase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only Recall@k benchmark.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--delete-existing-knowledge-base", action="store_true")
    parser.add_argument("--knowledge-base-batch-size", type=int, default=64)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    runtime = build_rag_runtime(get_settings())
    await initialize_rag_runtime(runtime)
    try:
        ingest_stats = await ingest_knowledge_base(
            graph_runtime=runtime,
            path=args.knowledge_base,
            delete_existing=args.delete_existing_knowledge_base,
            batch_size=args.knowledge_base_batch_size,
        )
        report = await evaluate_retrieval(
            graph_runtime=runtime,
            cases=cases,
            recall_k=args.recall_k,
            retrieval_mode=args.retrieval_mode,
            ingest_stats=ingest_stats,
        )
    finally:
        await close_rag_runtime(runtime)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def ingest_knowledge_base(
    *,
    graph_runtime: Any,
    path: Path,
    delete_existing: bool,
    batch_size: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = {
        "knowledge_base": str(path),
        "documents": 0,
        "text_blocks": 0,
        "indexed_records": 0,
        "skipped_blocks": 0,
    }
    pending_blocks = []
    batch_size = max(batch_size, 1)
    for document in payload.get("documents", []):
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            continue
        blocks = text_blocks_for_document(document)
        stats["documents"] += 1
        stats["text_blocks"] += len(blocks)
        if delete_existing:
            await graph_runtime.embedding_index_service.vector_store.delete_by_doc_id(document_id)
        pending_blocks.extend(blocks)
        while len(pending_blocks) >= batch_size:
            batch = pending_blocks[:batch_size]
            pending_blocks = pending_blocks[batch_size:]
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


async def evaluate_retrieval(
    *,
    graph_runtime: Any,
    cases: list[EvaluationCase],
    recall_k: int,
    retrieval_mode: str,
    ingest_stats: dict[str, Any],
) -> dict[str, Any]:
    case_results = []
    for case in cases:
        started_at = time.perf_counter()
        query = RetrievalQuery(
            query=case.question or "",
            document_ids=case.resolved_document_ids,
            mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=max(case.top_k, recall_k),
            filters={**case.filters, "retrieval_mode": retrieval_mode},
        )
        if retrieval_mode == "vector":
            hits = await graph_runtime.retrieval_tools.hybrid_retriever.vector_retriever.retrieve(query)
        else:
            result = await graph_runtime.retrieval_tools.hybrid_retriever.retrieve(query)
            hits = result.hits
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        hit_payloads = [hit_payload(hit) for hit in hits]
        (
            hit_at_k,
            recall_at_k,
            matched_evidence_ids,
            matched_source_ids,
            matched_retrieval_keywords,
        ) = retrieval_recall_at_k(
            expected_evidence_ids=case.expected_evidence_ids,
            expected_source_ids=case.expected_source_ids,
            expected_retrieval_keywords=case.expected_retrieval_keywords,
            hit_payloads=hit_payloads,
            k=recall_k,
        )
        case_results.append(
            {
                "case_id": case.id,
                "hit_at_k": hit_at_k,
                "recall_at_k": recall_at_k,
                "matched_evidence_ids": matched_evidence_ids,
                "matched_source_ids": matched_source_ids,
                "matched_retrieval_keywords": matched_retrieval_keywords,
                "latency_ms": latency_ms,
                "top_hits": hit_payloads[:recall_k],
            }
        )

    hit_values = [1.0 if item["hit_at_k"] else 0.0 for item in case_results if item["hit_at_k"] is not None]
    recall_values = [item["recall_at_k"] for item in case_results if item["recall_at_k"] is not None]
    latency_values = [item["latency_ms"] for item in case_results]
    return {
        "runtime_mode": "retrieval_only",
        "retrieval_mode": retrieval_mode,
        "recall_k": recall_k,
        "metrics": {
            "total_cases": len(case_results),
            "hit_at_k": average(hit_values),
            "recall_at_k": average(recall_values),
            "latency_p50_ms": percentile(latency_values, 0.50),
            "latency_p95_ms": percentile(latency_values, 0.95),
        },
        "cases": case_results,
        "metadata": {
            "ingest": ingest_stats,
            "core_metric_scope": "retrieval_only",
        },
    }


def hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    evidence_ids = []
    if hit.evidence:
        evidence_ids.extend(evidence.id for evidence in hit.evidence.evidences)
    return {
        "id": hit.id,
        "content": hit.content or "",
        "source_type": hit.source_type,
        "source_id": hit.source_id,
        "document_id": hit.document_id,
        "evidence_ids": evidence_ids,
        "vector_score": hit.vector_score,
        "graph_score": hit.graph_score,
        "merged_score": hit.merged_score,
    }


def average(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


if __name__ == "__main__":
    asyncio.run(main())
