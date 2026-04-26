from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any

from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import RetrievalHit
from evaluation.metrics import (
    groundedness_score,
    keyword_recall,
    polarity_accuracy,
    percentile,
    reference_token_f1,
    retrieval_recall_at_k,
    route_accuracy,
    tool_call_success_rate,
)
from evaluation.schemas import (
    AggregateMetrics,
    CaseMetricResult,
    CoreMetricSummary,
    EvaluationCase,
    EvaluationReport,
)


async def evaluate_cases(
    *,
    graph_runtime: Any,
    cases: list[EvaluationCase],
    runtime_mode: str,
    recall_k: int = 5,
    progress_callback: Callable[[int, int, CaseMetricResult], Awaitable[None] | None] | None = None,
) -> EvaluationReport:
    results = []
    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        result = await evaluate_case(graph_runtime=graph_runtime, case=case, recall_k=recall_k)
        results.append(result)
        if progress_callback is not None:
            maybe_awaitable = progress_callback(index, total_cases, result)
            if isawaitable(maybe_awaitable):
                await maybe_awaitable
    metrics = aggregate_metrics(results, recall_k=recall_k)
    return EvaluationReport(
        runtime_mode=runtime_mode,  # type: ignore[arg-type]
        metrics=metrics,
        core_6_metrics=core_6_metrics(metrics=metrics, recall_k=recall_k),
        cases=results,
        metadata={
            "evaluated_case_ids": [case.id for case in cases],
            "recall_k": recall_k,
            "core_metric_names": [
                f"Recall@{recall_k}",
                "Groundedness",
                "Answer Keyword Recall",
                "Route Accuracy",
                "Tool Call Success Rate",
                "Latency P50/P95",
            ],
        },
    )


async def evaluate_case(
    *,
    graph_runtime: Any,
    case: EvaluationCase,
    recall_k: int = 5,
) -> CaseMetricResult:
    started_at = time.perf_counter()
    state, actual_route = await _invoke_case(graph_runtime=graph_runtime, case=case)
    fallback_latency = round((time.perf_counter() - started_at) * 1000, 2)

    tool_traces = [dict(trace) for trace in state.get("tool_traces", [])]
    hits = _collect_hits(state)
    hit_payloads = [_hit_payload(hit) for hit in hits]
    evidence_bundle = _state_evidence_bundle(state)
    answer_text = _extract_answer_text(case=case, state=state)
    metric_metadata = state.get("metadata", {})

    keyword_score, matched_keywords = keyword_recall(case.expected_keywords, answer_text)
    reference_precision, reference_recall, reference_f1 = reference_token_f1(
        case.metadata.get("reference_response"),
        answer_text,
    )
    polarity_correct = polarity_accuracy(
        case.metadata.get("reference_response"),
        answer_text,
    )
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
    groundedness = groundedness_score(
        answer=answer_text,
        evidence_texts=_evidence_texts(evidence_bundle=evidence_bundle, hits=hits),
        grounding_keywords=case.grounding_keywords,
    )
    route_correct = route_accuracy(case.expected_route, actual_route)
    tool_success_rate, tool_success_count, tool_total = tool_call_success_rate(
        tool_traces=tool_traces,
        expected_tool_names=case.expected_tool_names,
    )
    validation_retry = bool(state.get("retrieval_attempt", 0))
    latency_ms = metric_metadata.get("runtime_total_latency_ms", fallback_latency)
    step_count = len([trace for trace in tool_traces if trace.get("status") not in {"started"}])

    success_checks = [not bool(state.get("errors"))]
    if case.require_nonempty_answer and case.kind != "chart_understand":
        success_checks.append(bool((answer_text or "").strip()))
        success_checks.append(not _is_insufficient_answer(answer_text))
    if case.needs_evidence:
        success_checks.append(bool(evidence_bundle.evidences))
    if keyword_score is not None:
        success_checks.append(keyword_score >= case.min_keyword_recall)
    if route_correct is not None:
        success_checks.append(route_correct)
    task_success = all(success_checks)

    return CaseMetricResult(
        case_id=case.id,
        kind=case.kind,
        actual_route=actual_route,
        task_success=task_success,
        answer=answer_text,
        keyword_recall=keyword_score,
        reference_precision=reference_precision,
        reference_recall=reference_recall,
        reference_f1=reference_f1,
        polarity_correct=polarity_correct,
        matched_keywords=matched_keywords,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        hit_at_5=hit_at_k if recall_k == 5 else None,
        recall_at_5=recall_at_k if recall_k == 5 else None,
        matched_evidence_ids=matched_evidence_ids,
        matched_source_ids=matched_source_ids,
        matched_retrieval_keywords=matched_retrieval_keywords,
        groundedness=groundedness,
        route_correct=route_correct,
        tool_call_success_rate=tool_success_rate,
        tool_call_success_count=tool_success_count,
        tool_call_total=tool_total,
        validation_retry=validation_retry,
        latency_ms=latency_ms,
        step_count=step_count,
        warnings=list(state.get("warnings", [])),
        errors=list(state.get("errors", [])),
        metadata={
            "benchmark": case.metadata.get("benchmark"),
            "benchmark_subset": case.metadata.get("subset"),
            "benchmark_split": case.metadata.get("split"),
            "trace_id": metric_metadata.get("trace_id"),
            "thread_id": metric_metadata.get("thread_id"),
            "tool_trace_count": len(tool_traces),
            "warning_count": len(state.get("warnings", [])),
            "recall_k": recall_k,
        },
    )


def aggregate_metrics(results: list[CaseMetricResult], *, recall_k: int = 5) -> AggregateMetrics:
    total_cases = len(results)
    task_success_rate = sum(1 for result in results if result.task_success) / max(total_cases, 1)

    keyword_values = [result.keyword_recall for result in results if result.keyword_recall is not None]
    reference_precision_values = [result.reference_precision for result in results if result.reference_precision is not None]
    reference_recall_values = [result.reference_recall for result in results if result.reference_recall is not None]
    reference_f1_values = [result.reference_f1 for result in results if result.reference_f1 is not None]
    polarity_values = [1.0 if result.polarity_correct else 0.0 for result in results if result.polarity_correct is not None]
    hit_at_k_values = [1.0 if result.hit_at_k else 0.0 for result in results if result.hit_at_k is not None]
    recall_at_k_values = [result.recall_at_k for result in results if result.recall_at_k is not None]
    groundedness_values = [result.groundedness for result in results if result.groundedness is not None]
    route_values = [1.0 if result.route_correct else 0.0 for result in results if result.route_correct is not None]
    retry_values = [1.0 if result.validation_retry else 0.0 for result in results]
    insufficient_values = [1.0 if _is_insufficient_answer(result.answer) else 0.0 for result in results]
    warning_case_values = [1.0 if result.warnings else 0.0 for result in results]
    warning_count_values = [float(len(result.warnings)) for result in results]
    error_free_values = [1.0 if not result.errors else 0.0 for result in results]
    latency_values = [result.latency_ms for result in results if result.latency_ms is not None]
    step_values = [float(result.step_count) for result in results]

    tool_success_total = sum(result.tool_call_success_count for result in results)
    tool_total = sum(result.tool_call_total for result in results)

    return AggregateMetrics(
        total_cases=total_cases,
        task_success_rate=task_success_rate,
        answer_keyword_recall=_average(keyword_values),
        reference_answer_precision=_average(reference_precision_values),
        reference_answer_recall=_average(reference_recall_values),
        reference_answer_f1=_average(reference_f1_values),
        answer_polarity_accuracy=_average(polarity_values),
        hit_at_k=_average(hit_at_k_values),
        recall_at_k=_average(recall_at_k_values),
        hit_at_5=_average(hit_at_k_values) if recall_k == 5 else None,
        recall_at_5=_average(recall_at_k_values) if recall_k == 5 else None,
        groundedness=_average(groundedness_values),
        route_accuracy=_average(route_values),
        tool_call_success_rate=(tool_success_total / tool_total) if tool_total else None,
        validation_retry_rate=_average(retry_values),
        latency_p50_ms=percentile(latency_values, 0.50),
        latency_p95_ms=percentile(latency_values, 0.95),
        average_steps_per_task=_average(step_values),
        insufficient_answer_rate=_average(insufficient_values),
        warning_case_rate=_average(warning_case_values),
        avg_warning_count=_average(warning_count_values),
        error_free_rate=_average(error_free_values),
    )


def core_6_metrics(*, metrics: AggregateMetrics, recall_k: int = 5) -> CoreMetricSummary:
    return CoreMetricSummary(
        recall_k=recall_k,
        recall_at_k=metrics.recall_at_k,
        groundedness=metrics.groundedness,
        answer_keyword_recall=metrics.answer_keyword_recall,
        route_accuracy=metrics.route_accuracy,
        tool_call_success_rate=metrics.tool_call_success_rate,
        latency_p50_ms=metrics.latency_p50_ms,
        latency_p95_ms=metrics.latency_p95_ms,
    )


def _is_insufficient_answer(answer: str | None) -> bool:
    normalized = (answer or "").strip().lower()
    return "证据不足" in normalized or "insufficient evidence" in normalized


async def _invoke_case(*, graph_runtime: Any, case: EvaluationCase) -> tuple[dict[str, Any], str]:
    if case.kind == "ask_document":
        return await _invoke_ask_document_case(graph_runtime=graph_runtime, case=case)
    if case.kind == "ask_fused":
        return await _invoke_ask_fused_case(graph_runtime=graph_runtime, case=case), "ask_fused"
    if case.kind == "chart_understand":
        return await _invoke_chart_understand_case(graph_runtime=graph_runtime, case=case), "chart_understand"
    raise ValueError(f"Unsupported evaluation case kind: {case.kind}")


async def _invoke_ask_document_case(*, graph_runtime: Any, case: EvaluationCase) -> tuple[dict[str, Any], str]:
    skill_context = _skill_context(graph_runtime, task_type="ask_document", case=case)
    question = case.question or ""
    if graph_runtime._is_chart_like_question(question):  # noqa: SLF001 - internal evaluation helper
        visual_anchor = await graph_runtime._resolve_visual_anchor(  # noqa: SLF001 - mirrors runtime branch
            question=question,
            doc_id=case.document_id,
            document_ids=case.resolved_document_ids,
            top_k=case.top_k,
            filters=case.filters,
            session_id=case.session_id,
            skill_context=skill_context,
        )
        if visual_anchor is not None:
            state = await graph_runtime.invoke(
                _build_ask_state(
                    case=case,
                    task_intent="ask_fused",
                    skill_context=skill_context,
                    image_path=str(visual_anchor["image_path"]),
                    page_id=visual_anchor.get("page_id"),
                    page_number=visual_anchor.get("page_number", case.page_number),
                    chart_id=visual_anchor.get("chart_id"),
                    metadata={
                        **case.metadata,
                        "auto_fused": True,
                        "auto_fused_reason": "chart_like_question",
                        "visual_anchor": visual_anchor,
                    },
                )
            )
            return state, "ask_fused"
    state = await graph_runtime.invoke(
        _build_ask_state(
            case=case,
            task_intent="ask_document",
            skill_context=skill_context,
        )
    )
    return state, "ask_document"


async def _invoke_ask_fused_case(*, graph_runtime: Any, case: EvaluationCase) -> dict[str, Any]:
    return await graph_runtime.invoke(
        _build_ask_state(
            case=case,
            task_intent="ask_fused",
            skill_context=_skill_context(graph_runtime, task_type="ask_document", case=case),
            image_path=case.image_path,
            page_id=case.page_id,
            page_number=case.page_number,
            chart_id=case.chart_id,
        )
    )


async def _invoke_chart_understand_case(*, graph_runtime: Any, case: EvaluationCase) -> dict[str, Any]:
    return await graph_runtime.invoke(
        {
            "request_id": case.id,
            "thread_id": case.session_id or case.id,
            "task_type": "chart_understand",
            "user_input": case.image_path,
            "image_path": case.image_path,
            "document_id": case.document_id,
            "document_ids": case.resolved_document_ids,
            "page_id": case.page_id,
            "page_number": case.page_number,
            "chart_id": case.chart_id,
            "vector_hits": [],
            "graph_hits": [],
            "summary_hits": [],
            "graph_summary_hits": [],
            "warnings": [],
            "reasoning_summary": {},
            "react_trace": [],
            "messages": [],
            "tool_traces": [],
            "errors": [],
            "metadata": dict(case.metadata),
            "retrieval_mode": "hybrid",
            "top_k": case.top_k,
            "filters": dict(case.filters),
            "selected_skill": _skill_context(graph_runtime, task_type="understand_chart", case=case),
        }
    )


def _build_ask_state(
    *,
    case: EvaluationCase,
    task_intent: str,
    skill_context: dict[str, Any] | None,
    image_path: str | None = None,
    page_id: str | None = None,
    page_number: int | None = None,
    chart_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_metadata = dict(metadata or case.metadata)
    if case.reasoning_style:
        state_metadata["reasoning_style"] = case.reasoning_style
    return {
        "request_id": case.id,
        "thread_id": case.session_id or case.id,
        "task_type": "ask",
        "user_input": case.question,
        "document_id": case.document_id,
        "document_ids": case.resolved_document_ids,
        "session_id": case.session_id,
        "task_intent": task_intent,
        "image_path": image_path,
        "page_id": page_id,
        "page_number": page_number,
        "chart_id": chart_id,
        "top_k": case.top_k,
        "filters": dict(case.filters),
        "vector_hits": [],
        "graph_hits": [],
        "summary_hits": [],
        "graph_summary_hits": [],
        "warnings": [],
        "messages": [],
        "tool_traces": [],
        "errors": [],
        "metadata": state_metadata,
        "retrieval_mode": "hybrid",
        "reasoning_summary": {},
        "react_trace": [],
        "max_retrieval_attempts": 1,
        "selected_skill": skill_context,
    }


def _skill_context(graph_runtime: Any, *, task_type: str, case: EvaluationCase) -> dict[str, Any] | None:
    return graph_runtime.resolve_skill_context(
        task_type=task_type,
        preferred_skill_name=case.skill_name,
    ) if case.skill_name else None


def _extract_answer_text(*, case: EvaluationCase, state: dict[str, Any]) -> str | None:
    if case.kind == "chart_understand":
        chart_result = state.get("chart_result") or {}
        chart = chart_result.get("chart")
        summary_parts = [
            getattr(chart, "title", None),
            getattr(chart, "summary", None),
            chart_result.get("graph_text"),
        ]
        return "\n".join(part for part in summary_parts if part)
    final_answer = state.get("final_answer")
    if isinstance(final_answer, QAResponse):
        return final_answer.answer
    if isinstance(final_answer, dict):
        return str(final_answer.get("answer") or "")
    return None


def _state_evidence_bundle(state: dict[str, Any]) -> EvidenceBundle:
    bundle = state.get("evidence_bundle")
    if isinstance(bundle, EvidenceBundle):
        return bundle
    final_answer = state.get("final_answer")
    if isinstance(final_answer, QAResponse):
        return final_answer.evidence_bundle
    return EvidenceBundle()


def _collect_hits(state: dict[str, Any]) -> list[RetrievalHit]:
    return [
        *state.get("vector_hits", []),
        *state.get("graph_hits", []),
        *state.get("graph_summary_hits", state.get("summary_hits", [])),
    ]


def _hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    if isinstance(hit, dict):
        evidence_ids = []
        evidence = hit.get("evidence")
        if isinstance(evidence, dict):
            evidence_ids.extend(
                str(item.get("id"))
                for item in evidence.get("evidences", [])
                if isinstance(item, dict) and item.get("id") is not None
            )
        return {
            "id": hit.get("id"),
            "content": hit.get("content") or "",
            "source_type": hit.get("source_type"),
            "source_id": hit.get("source_id"),
            "document_id": hit.get("document_id"),
            "evidence_ids": evidence_ids,
        }
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
    }


def _evidence_texts(*, evidence_bundle: EvidenceBundle, hits: list[RetrievalHit]) -> list[str]:
    texts = [(evidence.snippet or "") for evidence in evidence_bundle.evidences if (evidence.snippet or "").strip()]
    for hit in hits:
        if isinstance(hit, dict):
            content = str(hit.get("content") or "")
        else:
            content = hit.content or ""
        if content.strip():
            texts.append(content)
    return texts


def _average(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
