from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import polarity_accuracy, reference_token_f1
from evaluation.run_agent_metrics import load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich and summarize benchmark evaluation reports.")
    parser.add_argument("--report", type=Path, required=True, help="Path to a run_agent_metrics JSON report.")
    parser.add_argument("--cases", type=Path, required=True, help="Path to the benchmark cases.json file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional enriched report output path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output path.")
    return parser.parse_args()


def average(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def enrich_report(report: dict[str, Any], cases_path: Path) -> dict[str, Any]:
    cases = {case.id: case for case in load_cases(cases_path)}
    enriched_cases = []
    for result in report.get("cases", []):
        case = cases.get(result.get("case_id"))
        if case is None:
            enriched_cases.append(result)
            continue
        reference_answer = case.metadata.get("reference_response")
        precision, recall, f1 = reference_token_f1(reference_answer, result.get("answer"))
        polarity_correct = polarity_accuracy(reference_answer, result.get("answer"))
        metadata = dict(result.get("metadata") or {})
        metadata.setdefault("benchmark", case.metadata.get("benchmark"))
        metadata.setdefault("benchmark_subset", case.metadata.get("subset"))
        metadata.setdefault("benchmark_split", case.metadata.get("split"))
        enriched_result = {
            **result,
            "reference_precision": result.get("reference_precision", precision),
            "reference_recall": result.get("reference_recall", recall),
            "reference_f1": result.get("reference_f1", f1),
            "polarity_correct": result.get("polarity_correct", polarity_correct),
            "metadata": metadata,
        }
        enriched_cases.append(enriched_result)

    metrics = dict(report.get("metrics") or {})
    reference_precision_values = [item.get("reference_precision") for item in enriched_cases if item.get("reference_precision") is not None]
    reference_recall_values = [item.get("reference_recall") for item in enriched_cases if item.get("reference_recall") is not None]
    reference_f1_values = [item.get("reference_f1") for item in enriched_cases if item.get("reference_f1") is not None]
    polarity_values = [1.0 if item.get("polarity_correct") else 0.0 for item in enriched_cases if item.get("polarity_correct") is not None]
    insufficient_values = [1.0 if "证据不足" in str(item.get("answer") or "").lower() or "insufficient evidence" in str(item.get("answer") or "").lower() else 0.0 for item in enriched_cases]
    warning_case_values = [1.0 if item.get("warnings") else 0.0 for item in enriched_cases]
    warning_count_values = [float(len(item.get("warnings") or [])) for item in enriched_cases]
    error_free_values = [1.0 if not item.get("errors") else 0.0 for item in enriched_cases]
    metrics.setdefault("reference_answer_precision", average(reference_precision_values))
    metrics.setdefault("reference_answer_recall", average(reference_recall_values))
    metrics.setdefault("reference_answer_f1", average(reference_f1_values))
    metrics.setdefault("answer_polarity_accuracy", average(polarity_values))
    metrics.setdefault("insufficient_answer_rate", average(insufficient_values))
    metrics.setdefault("warning_case_rate", average(warning_case_values))
    metrics.setdefault("avg_warning_count", average(warning_count_values))
    metrics.setdefault("error_free_rate", average(error_free_values))

    return {**report, "metrics": metrics, "cases": enriched_cases}


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases", [])
    benchmark_groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        benchmark = ((case.get("metadata") or {}).get("benchmark")) or "unknown"
        benchmark_groups.setdefault(str(benchmark), []).append(case)

    summary = {
        "runtime_mode": report.get("runtime_mode"),
        "top_level_metrics": report.get("metrics", {}),
        "benchmark_slices": {},
    }
    for benchmark, items in benchmark_groups.items():
        summary["benchmark_slices"][benchmark] = {
            "total_cases": len(items),
            "task_success_rate": average([1.0 if item.get("task_success") else 0.0 for item in items]),
            "recall_at_k": average([item.get("recall_at_k") for item in items]),
            "groundedness": average([item.get("groundedness") for item in items]),
            "keyword_recall": average([item.get("keyword_recall") for item in items]),
            "reference_f1": average([item.get("reference_f1") for item in items]),
            "polarity_accuracy": average(
                [1.0 if item.get("polarity_correct") else 0.0 for item in items if item.get("polarity_correct") is not None]
            ),
            "insufficient_answer_rate": average(
                [
                    1.0 if "证据不足" in str(item.get("answer") or "").lower() or "insufficient evidence" in str(item.get("answer") or "").lower() else 0.0
                    for item in items
                ]
            ),
            "warning_case_rate": average([1.0 if item.get("warnings") else 0.0 for item in items]),
            "latency_ms": average([item.get("latency_ms") for item in items]),
        }
    return summary


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    enriched = enrich_report(report, args.cases)
    summary = build_summary(enriched)

    enriched_payload = json.dumps(enriched, ensure_ascii=False, indent=2)
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_payload)
    if args.output:
        args.output.write_text(enriched_payload, encoding="utf-8")
    if args.summary_output:
        args.summary_output.write_text(summary_payload, encoding="utf-8")


if __name__ == "__main__":
    main()
