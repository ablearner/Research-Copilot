from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.schemas import EvaluationCase  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "evaluation" / "datasets"
DEFAULT_JSON_OUTPUT = DEFAULT_DATASET_ROOT / "dataset_metrics.json"
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_DATASET_ROOT / "dataset_metrics.md"
PDFQA_INPUT_PREFIX = Path("real-pdfQA") / "01.2_Input_Files_PDF"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local evaluation datasets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    summary = build_summary(dataset_root)
    write_json(summary, args.json_output)
    write_markdown(summary, args.markdown_output)
    print(f"Wrote JSON metrics to {args.json_output}")
    print(f"Wrote Markdown metrics to {args.markdown_output}")


def build_summary(dataset_root: Path) -> dict[str, Any]:
    case_datasets = [summarize_case_dataset(path) for path in sorted(dataset_root.glob("*/cases.json"))]
    pdf_corpora = []
    pdfqa_manifest = dataset_root / "pdfqa_benchmark" / "pdf_manifest.json"
    if pdfqa_manifest.exists():
        pdf_corpora.append(summarize_pdfqa(pdfqa_manifest))

    all_files = [path for path in dataset_root.rglob("*") if path.is_file()]
    image_files = [path for path in all_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}]
    pdf_files = [path for path in all_files if path.suffix.lower() == ".pdf"]
    warnings = collect_warnings(case_datasets=case_datasets, pdf_corpora=pdf_corpora)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "overall": {
            "file_count": len(all_files),
            "total_size_bytes": sum(file_size(path) for path in all_files),
            "case_dataset_count": len(case_datasets),
            "case_count": sum(item["case_count"] for item in case_datasets),
            "image_file_count": len(image_files),
            "pdf_file_count": len(pdf_files),
            "pdf_corpus_count": len(pdf_corpora),
        },
        "case_datasets": case_datasets,
        "pdf_corpora": pdf_corpora,
        "warnings": warnings,
    }


def summarize_case_dataset(cases_path: Path) -> dict[str, Any]:
    dataset_dir = cases_path.parent
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    cases = [EvaluationCase.model_validate(item) for item in raw_cases]

    case_ids = [case.id for case in cases]
    duplicate_ids = sorted([case_id for case_id, count in Counter(case_ids).items() if count > 1])
    image_paths = [resolve_project_path(case.image_path) for case in cases if case.image_path]
    unique_image_paths = sorted(set(image_paths))
    missing_images = [str(path) for path in unique_image_paths if not path.exists()]
    image_dir = dataset_dir / "images"
    image_files = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    referenced_image_set = {Path(path).resolve() for path in unique_image_paths}
    orphan_images = [str(path.relative_to(ROOT)) for path in image_files if path.resolve() not in referenced_image_set]

    reference_answers = [
        str(case.metadata.get("reference_answer", ""))
        for case in cases
        if str(case.metadata.get("reference_answer", "")).strip()
    ]
    expected_keyword_lengths = [len(case.expected_keywords) for case in cases]
    question_lengths = [len(case.question or "") for case in cases]
    question_word_counts = [len((case.question or "").split()) for case in cases]
    top_ks = [case.top_k for case in cases]
    image_stats = summarize_images(image_files)

    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir.relative_to(ROOT)),
        "cases_path": str(cases_path.relative_to(ROOT)),
        "case_count": len(cases),
        "unique_case_id_count": len(set(case_ids)),
        "duplicate_case_ids": duplicate_ids,
        "kind_counts": dict(sorted(Counter(case.kind for case in cases).items())),
        "expected_route_counts": dict(sorted(Counter(case.expected_route or "" for case in cases).items())),
        "skill_counts": dict(sorted(Counter(case.skill_name or "" for case in cases).items())),
        "source_dataset_counts": dict(sorted(Counter(str(case.metadata.get("dataset", "")) for case in cases).items())),
        "dataset_split_counts": dict(sorted(Counter(str(case.metadata.get("dataset_split", "")) for case in cases).items())),
        "source_file_counts": dict(sorted(Counter(str(case.metadata.get("source_file", "")) for case in cases).items())),
        "unique_document_count": len({doc_id for case in cases for doc_id in case.resolved_document_ids}),
        "unique_image_count": len(unique_image_paths),
        "image_file_count": len(image_files),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "orphan_image_count": len(orphan_images),
        "orphan_images": orphan_images,
        "expected_tool_counts": dict(sorted(Counter(tool for case in cases for tool in case.expected_tool_names).items())),
        "require_evidence_counts": dict(sorted(Counter(str(case.require_evidence) for case in cases).items())),
        "top_k": numeric_summary(top_ks),
        "question_length_chars": numeric_summary(question_lengths),
        "question_word_count": numeric_summary(question_word_counts),
        "expected_keyword_count": numeric_summary(expected_keyword_lengths),
        "cases_with_expected_keywords": sum(1 for length in expected_keyword_lengths if length > 0),
        "cases_without_expected_keywords": sum(1 for length in expected_keyword_lengths if length == 0),
        "min_keyword_recall": numeric_summary([case.min_keyword_recall for case in cases]),
        "reference_answer": {
            "count": len(reference_answers),
            "type_counts": dict(sorted(Counter(classify_reference_answer(answer) for answer in reference_answers).items())),
            "char_length": numeric_summary([len(answer) for answer in reference_answers]),
            "word_count": numeric_summary([len(answer.split()) for answer in reference_answers]),
        },
        "images": image_stats,
        "size_bytes": directory_size(dataset_dir),
    }


def summarize_pdfqa(manifest_path: Path) -> dict[str, Any]:
    corpus_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_root = corpus_dir / PDFQA_INPUT_PREFIX
    per_dataset = {}
    all_pdfs = []
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        pdfs = sorted(dataset_dir.glob("*.pdf"))
        pdf_items = [summarize_pdf(path) for path in pdfs]
        all_pdfs.extend(pdf_items)
        manifest_item = manifest.get("datasets", {}).get(dataset_dir.name, {})
        page_counts = [item["page_count"] for item in pdf_items if item["page_count"] is not None]
        per_dataset[dataset_dir.name] = {
            "manifest_selected_count": manifest_item.get("selected_count"),
            "manifest_local_count": manifest_item.get("local_count"),
            "actual_pdf_count": len(pdf_items),
            "total_size_bytes": sum(item["size_bytes"] for item in pdf_items),
            "page_count": numeric_summary(page_counts),
            "total_pages": sum(page_counts),
            "pdfs": pdf_items,
        }
    page_counts = [item["page_count"] for item in all_pdfs if item["page_count"] is not None]
    return {
        "name": corpus_dir.name,
        "path": str(corpus_dir.relative_to(ROOT)),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "repo_id": manifest.get("repo_id"),
        "limit_per_dataset": manifest.get("limit_per_dataset"),
        "dataset_count": len(per_dataset),
        "pdf_count": len(all_pdfs),
        "total_size_bytes": sum(item["size_bytes"] for item in all_pdfs),
        "page_count": numeric_summary(page_counts),
        "total_pages": sum(page_counts),
        "datasets": per_dataset,
    }


def summarize_pdf(path: Path) -> dict[str, Any]:
    page_count = None
    error = None
    try:
        reader = PdfReader(str(path), strict=False)
        page_count = len(reader.pages)
    except Exception as exc:  # pragma: no cover - depends on external PDF contents.
        error = str(exc)
    return {
        "filename": path.name,
        "path": str(path.relative_to(ROOT)),
        "size_bytes": file_size(path),
        "size_mb": round(file_size(path) / 1024 / 1024, 2),
        "page_count": page_count,
        "read_error": error,
    }


def summarize_images(image_files: list[Path]) -> dict[str, Any]:
    dims = [read_png_size(path) for path in image_files if path.suffix.lower() == ".png"]
    dims = [item for item in dims if item is not None]
    widths = [width for width, _height in dims]
    heights = [height for _width, height in dims]
    sizes = [file_size(path) for path in image_files]
    return {
        "file_count": len(image_files),
        "total_size_bytes": sum(sizes),
        "size_bytes": numeric_summary(sizes),
        "width_px": numeric_summary(widths),
        "height_px": numeric_summary(heights),
    }


def read_png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", header[16:24])
    except OSError:
        return None


def classify_reference_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.fullmatch(r"[-+]?[$]?\d[\d,]*(\.\d+)?%?", normalized):
        return "numeric"
    if len(normalized.split()) <= 3 and len(normalized) <= 32:
        return "short_text"
    return "long_text"


def collect_warnings(*, case_datasets: list[dict[str, Any]], pdf_corpora: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for dataset in case_datasets:
        if dataset["missing_image_count"]:
            warnings.append(f"{dataset['name']} has {dataset['missing_image_count']} missing image reference(s).")
        if dataset["duplicate_case_ids"]:
            warnings.append(f"{dataset['name']} has duplicate case ids: {', '.join(dataset['duplicate_case_ids'])}.")
        if dataset["case_count"] == 0:
            warnings.append(f"{dataset['name']} has no cases.")
    for corpus in pdf_corpora:
        for dataset_name, dataset in corpus["datasets"].items():
            expected = dataset.get("manifest_local_count")
            actual = dataset.get("actual_pdf_count")
            if expected is not None and expected != actual:
                warnings.append(f"{corpus['name']}/{dataset_name} manifest local_count={expected}, actual_pdf_count={actual}.")
            failed_reads = [pdf["filename"] for pdf in dataset["pdfs"] if pdf["read_error"]]
            if failed_reads:
                warnings.append(f"{corpus['name']}/{dataset_name} has unreadable PDF(s): {', '.join(failed_reads)}.")
    if not warnings:
        warnings.append("No dataset integrity warnings.")
    return warnings


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    values = [value for value in values if value is not None and not math.isnan(float(value))]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
    }


def write_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Dataset Metrics",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total files: {summary['overall']['file_count']}",
        f"- Total size: {format_bytes(summary['overall']['total_size_bytes'])}",
        f"- Total evaluation cases: {summary['overall']['case_count']}",
        f"- Total images: {summary['overall']['image_file_count']}",
        f"- Total PDFs: {summary['overall']['pdf_file_count']}",
        "",
        "## Case Datasets",
        "",
        "| Dataset | Cases | Kind | Images | Missing Images | Avg Keywords | Reference Types | Size |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for dataset in summary["case_datasets"]:
        lines.append(
            "| {name} | {cases} | {kind} | {images} | {missing} | {avg_kw} | {ref_types} | {size} |".format(
                name=dataset["name"],
                cases=dataset["case_count"],
                kind=format_counts(dataset["kind_counts"]),
                images=dataset["image_file_count"],
                missing=dataset["missing_image_count"],
                avg_kw=dataset["expected_keyword_count"]["mean"],
                ref_types=format_counts(dataset["reference_answer"]["type_counts"]),
                size=format_bytes(dataset["size_bytes"]),
            )
        )
    lines.extend(["", "## PDF Corpora", ""])
    for corpus in summary["pdf_corpora"]:
        lines.extend(
            [
                f"### {corpus['name']}",
                "",
                f"- Repo: `{corpus.get('repo_id')}`",
                f"- PDFs: {corpus['pdf_count']}",
                f"- Total size: {format_bytes(corpus['total_size_bytes'])}",
                f"- Total pages: {corpus['total_pages']}",
                f"- Avg pages/PDF: {corpus['page_count']['mean']}",
                "",
                "| Dataset | PDFs | Pages | Avg Pages | Size |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset_name, dataset in corpus["datasets"].items():
            lines.append(
                "| {name} | {pdfs} | {pages} | {avg_pages} | {size} |".format(
                    name=dataset_name,
                    pdfs=dataset["actual_pdf_count"],
                    pages=dataset["total_pages"],
                    avg_pages=dataset["page_count"]["mean"],
                    size=format_bytes(dataset["total_size_bytes"]),
                )
            )
        lines.append("")
    lines.extend(["## Integrity Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key or '<empty>'}:{value}" for key, value in counts.items()) or "-"


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def resolve_project_path(path_value: str | None) -> Path:
    if not path_value:
        return Path()
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def directory_size(path: Path) -> int:
    return sum(file_size(item) for item in path.rglob("*") if item.is_file())


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


if __name__ == "__main__":
    main()
