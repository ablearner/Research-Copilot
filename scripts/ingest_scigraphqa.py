from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.runtime import build_rag_runtime, close_rag_runtime, initialize_rag_runtime  # noqa: E402
from core.config import get_settings  # noqa: E402
from domain.schemas.chart import ChartSchema  # noqa: E402
from domain.schemas.document import TextBlock  # noqa: E402
from domain.schemas.evidence import Evidence  # noqa: E402
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode, GraphTriple  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/home/myc/ChartGPT/data/raw/scigraphqa")
DEFAULT_COPY_ROOT = Path(".data/scigraphqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest SciGraphQA parquet rows into Research-Copilot's configured vector store "
            "and graph store. This script only reads external data files; it does not "
            "import or modify the ChartGPT project."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--data-glob", default="data/*.parquet")
    parser.add_argument("--copy-root", type=Path, default=DEFAULT_COPY_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum rows to ingest. Use 0 to ingest all rows.",
    )
    parser.add_argument("--skip-images", action="store_true", help="Do not copy/index chart images.")
    parser.add_argument("--dry-run", action="store_true", help="Read and map rows without DB writes.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    parquet_files = sorted(args.dataset_root.glob(args.data_glob))
    if not parquet_files:
        raise SystemExit(f"No parquet files matched: {args.dataset_root / args.data_glob}")

    settings = get_settings()
    graph_runtime = build_rag_runtime(settings)
    if not args.dry_run:
        await initialize_rag_runtime(graph_runtime)

    stats = {"rows": 0, "text_blocks": 0, "charts": 0, "graphs": 0, "missing_images": 0}
    try:
        async for row in iter_scigraphqa_rows(parquet_files, args.batch_size):
            if args.limit and stats["rows"] >= args.limit:
                break
            mapped = map_row(
                row,
                args.dataset_root,
                args.copy_root,
                copy_image=not args.skip_images and not args.dry_run,
                inspect_image=not args.skip_images,
            )
            stats["rows"] += 1
            stats["text_blocks"] += len(mapped["text_blocks"])
            stats["charts"] += len(mapped["charts"])
            stats["missing_images"] += int(mapped["missing_image"])
            if args.dry_run:
                continue

            await graph_runtime.embedding_index_service.index_text_blocks(
                mapped["document_id"],
                mapped["text_blocks"],
            )
            if mapped["charts"]:
                await graph_runtime.embedding_index_service.index_charts(
                    mapped["document_id"],
                    mapped["charts"],
                )
            await graph_runtime.graph_index_service.index_graph_result(mapped["graph"])
            stats["graphs"] += 1

            if stats["rows"] % max(args.batch_size, 1) == 0:
                print(f"ingested rows={stats['rows']} charts={stats['charts']}")
    finally:
        if not args.dry_run:
            await close_rag_runtime(graph_runtime)

    print(
        "done "
        f"rows={stats['rows']} "
        f"text_blocks={stats['text_blocks']} "
        f"charts={stats['charts']} "
        f"graphs={stats['graphs']} "
        f"missing_images={stats['missing_images']}"
    )


async def iter_scigraphqa_rows(
    parquet_files: list[Path],
    batch_size: int,
):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=max(batch_size, 1)):
            for row in batch.to_pylist():
                yield row
            await asyncio.sleep(0)


def map_row(
    row: dict[str, Any],
    dataset_root: Path,
    copy_root: Path,
    *,
    copy_image: bool,
    inspect_image: bool,
) -> dict[str, Any]:
    source_id = str(row.get("id") or stable_id(row))
    document_id = f"scigraphqa_{safe_id(source_id)}"
    page_id = f"{document_id}_page_1"
    chart_id = f"{document_id}_chart_1"
    image_file = clean_text(row.get("image_file"))
    source_image = resolve_image_path(dataset_root, image_file) if inspect_image else None
    copied_image = copy_chart_image(dataset_root, copy_root, source_image) if copy_image else None
    image_path = copied_image or source_image

    text_blocks = build_text_blocks(row, document_id, page_id)
    chart_summary = build_chart_summary(row)
    charts: list[ChartSchema] = []
    if image_path:
        charts.append(
            ChartSchema(
                id=chart_id,
                document_id=document_id,
                page_id=page_id,
                page_number=1,
                chart_type="unknown",
                title=clean_text(row.get("title")) or None,
                caption=clean_text(row.get("caption")) or None,
                summary=chart_summary,
                confidence=1.0,
                metadata={
                    "image_path": str(image_path),
                    "image_uri": str(image_path),
                    "source_dataset": "scigraphqa",
                    "source_image_file": image_file,
                },
            )
        )

    return {
        "document_id": document_id,
        "text_blocks": text_blocks,
        "charts": charts,
        "graph": build_graph(row, document_id, chart_id, text_blocks, bool(image_path)),
        "missing_image": bool(image_file and image_path is None),
    }


def build_text_blocks(row: dict[str, Any], document_id: str, page_id: str) -> list[TextBlock]:
    fields = [
        ("title", "title"),
        ("abstract", "paragraph"),
        ("caption", "caption"),
        ("first_mention", "paragraph"),
        ("response", "paragraph"),
    ]
    blocks: list[TextBlock] = []
    for field_name, block_type in fields:
        text = clean_text(row.get(field_name))
        if not text:
            continue
        blocks.append(
            TextBlock(
                id=f"{document_id}_{field_name}",
                document_id=document_id,
                page_id=page_id,
                page_number=1,
                text=text,
                block_type=block_type,
                metadata={"field_name": field_name, "source_dataset": "scigraphqa"},
            )
        )

    qa_text = qa_pairs_text(row.get("q_a_pairs"), row.get("conversations"))
    if qa_text:
        blocks.append(
            TextBlock(
                id=f"{document_id}_qa",
                document_id=document_id,
                page_id=page_id,
                page_number=1,
                text=qa_text,
                block_type="paragraph",
                metadata={"field_name": "q_a_pairs", "source_dataset": "scigraphqa"},
            )
        )
    return blocks


def build_chart_summary(row: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            clean_text(row.get("title")),
            clean_text(row.get("caption")),
            clean_text(row.get("first_mention")),
        ]
        if part
    )


def build_graph(
    row: dict[str, Any],
    document_id: str,
    chart_id: str,
    text_blocks: list[TextBlock],
    has_chart_image: bool,
) -> GraphExtractionResult:
    evidence = Evidence(
        id=f"{document_id}_evidence",
        document_id=document_id,
        page_id=f"{document_id}_page_1",
        page_number=1,
        source_type="document",
        source_id=document_id,
        snippet=(clean_text(row.get("caption")) or clean_text(row.get("title")))[:300],
        metadata={"source_dataset": "scigraphqa"},
    )
    paper = GraphNode(
        id=f"{document_id}_paper",
        label="Paper",
        properties={
            "name": clean_text(row.get("title")) or document_id,
            "abstract": clean_text(row.get("abstract")),
            "document_id": document_id,
            "source_dataset": "scigraphqa",
        },
        source_reference=evidence,
    )
    figure = GraphNode(
        id=f"{document_id}_figure",
        label="Figure",
        properties={
            "name": clean_text(row.get("caption"))[:120] or chart_id,
            "chart_id": chart_id,
            "has_image": has_chart_image,
            "document_id": document_id,
            "image_file": clean_text(row.get("image_file")),
        },
        source_reference=evidence,
    )
    edge = GraphEdge(
        id=f"{document_id}_has_figure",
        type="HAS_FIGURE",
        source_node_id=paper.id,
        target_node_id=figure.id,
        properties={"document_id": document_id, "source_dataset": "scigraphqa"},
        source_reference=evidence,
    )
    nodes = [paper, figure]
    edges = [edge]
    triples = [GraphTriple(subject=paper, predicate=edge, object=figure)]

    for block in text_blocks:
        node = GraphNode(
            id=f"{block.id}_node",
            label="TextEvidence",
            properties={
                "name": block.text[:120],
                "field_name": block.metadata.get("field_name"),
                "document_id": document_id,
            },
            source_reference=Evidence(
                id=f"{block.id}_evidence",
                document_id=document_id,
                page_id=block.page_id,
                page_number=block.page_number,
                source_type="text_block",
                source_id=block.id,
                snippet=block.text[:300],
                metadata={"source_dataset": "scigraphqa"},
            ),
        )
        relation = GraphEdge(
            id=f"{block.id}_describes_figure",
            type="DESCRIBES_FIGURE",
            source_node_id=node.id,
            target_node_id=figure.id,
            properties={"document_id": document_id},
            source_reference=node.source_reference,
        )
        nodes.append(node)
        edges.append(relation)
        triples.append(GraphTriple(subject=node, predicate=relation, object=figure))

    return GraphExtractionResult(
        document_id=document_id,
        nodes=nodes,
        edges=edges,
        triples=triples,
        status="succeeded",
        metadata={"source_dataset": "scigraphqa", "source_row_id": str(row.get("id") or "")},
    )


def copy_chart_image(dataset_root: Path, copy_root: Path, source: Path | None) -> Path | None:
    if source is None:
        return None
    target = copy_root / "images" / source.relative_to(dataset_root / "images")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return target


def resolve_image_path(dataset_root: Path, image_file: str) -> Path | None:
    raw = Path(image_file)
    candidates = [
        dataset_root / "images" / raw,
        dataset_root / "images" / "imgs" / "train" / raw.name,
        dataset_root / "images" / "imgs" / "test" / raw.name,
        dataset_root / "images" / "imgs" / "validation" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list((dataset_root / "images").glob(f"**/{raw.name}"))
    return matches[0] if matches else None


def qa_pairs_text(q_a_pairs: Any, conversations: Any) -> str:
    lines: list[str] = []
    if isinstance(q_a_pairs, list):
        for item in q_a_pairs:
            if isinstance(item, list) and len(item) >= 2:
                lines.append(f"Q: {clean_text(item[0])}\nA: {clean_text(item[1])}")
            elif item:
                lines.append(clean_text(item))
    if not lines and isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict):
                speaker = clean_text(turn.get("from")) or "turn"
                value = clean_text(turn.get("value"))
                if value:
                    lines.append(f"{speaker}: {value}")
    return "\n\n".join(line for line in lines if line)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if clean_text(item))
    return " ".join(str(value).split())


def stable_id(row: dict[str, Any]) -> str:
    raw = "|".join(clean_text(row.get(key)) for key in ["image_file", "title", "caption"])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def safe_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() else "_" for char in value)[:64].strip("_")
    return f"{safe or 'row'}_{digest}"


if __name__ == "__main__":
    asyncio.run(main())
