from __future__ import annotations

from typing import Any

from domain.schemas.sub_manager import TaskEvaluation


class ResearchEvaluator:
    """Score research-task outputs against instruction, completeness, and schema expectations."""

    def evaluate_result(
        self,
        *,
        task_type: str,
        result_status: str,
        payload: dict[str, Any],
        task_instruction: str,
        expected_schema: dict[str, Any],
    ) -> TaskEvaluation:
        if result_status == "failed":
            reason = str(payload.get("reason") or "worker_failed")
            return TaskEvaluation(
                passed=False,
                score=1.0,
                issues=[reason],
                replan_suggestion="retry_with_more_specific_instruction",
                dimension_scores={
                    "content_completeness": 1.0,
                    "instruction_alignment": 1.0,
                    "format_compliance": 1.0,
                    "research_quality": 1.0,
                },
                metadata={"task_instruction": task_instruction, "skill": "ResearchEvaluator"},
            )

        issues: list[str] = []
        for field in list(expected_schema.get("required_fields") or []):
            value = payload.get(field)
            if value in (None, "", [], {}):
                issues.append(f"missing_field:{field}")

        thresholds: list[tuple[str, str]] = [
            ("min_paper_count", "insufficient_paper_count"),
            ("min_report_words", "report_too_short"),
            ("min_imported_or_skipped", "insufficient_import_progress"),
            ("min_answer_chars", "answer_too_short"),
        ]
        for threshold_key, issue_name in thresholds:
            if expected_schema.get(threshold_key) is None:
                continue
            actual_key = {
                "min_paper_count": "paper_count",
                "min_report_words": "report_word_count",
                "min_imported_or_skipped": None,
                "min_answer_chars": "answer_length",
            }[threshold_key]
            if threshold_key == "min_imported_or_skipped":
                progressed = int(payload.get("imported_count") or 0) + int(payload.get("skipped_count") or 0)
                if progressed < int(expected_schema[threshold_key]):
                    issues.append(issue_name)
                continue
            actual_value = int(payload.get(actual_key or "") or 0)
            if actual_value < int(expected_schema[threshold_key]):
                issues.append(issue_name)

        if expected_schema.get("require_citations") and not bool(payload.get("report_has_citations")):
            issues.append("missing_citations")
        if expected_schema.get("require_key_sections") and not bool(payload.get("report_has_key_sections")):
            issues.append("missing_key_sections")
        if expected_schema.get("min_evidence_count") is not None and payload.get("scope_mode") != "metadata_only":
            evidence_count = int(payload.get("evidence_count") or 0)
            if evidence_count < int(expected_schema["min_evidence_count"]):
                issues.append("insufficient_evidence")
        if task_type == "understand_document" and not payload.get("document_id"):
            issues.append("missing_document_id")
        if task_type == "understand_chart" and not payload.get("chart_id"):
            issues.append("missing_chart_id")

        score = max(0.0, 10.0 - (len(issues) * 2.0))
        passed = not issues and result_status in {"succeeded", "skipped"}
        replan_suggestion: str | None = None
        if not passed:
            if task_type == "write_review":
                replan_suggestion = "retry_review_quality"
            elif task_type == "answer_question":
                replan_suggestion = "retry_answer_quality"
            else:
                replan_suggestion = "retry_with_more_specific_instruction"
        return TaskEvaluation(
            passed=passed,
            score=score,
            issues=issues,
            replan_suggestion=replan_suggestion,
            dimension_scores={
                "content_completeness": 10.0 if not any("insufficient" in issue or "too_short" in issue for issue in issues) else 4.0,
                "instruction_alignment": 10.0 if not any("missing_field" in issue for issue in issues) else 5.0,
                "format_compliance": 10.0 if not any(
                    issue in {"missing_citations", "missing_key_sections", "missing_document_id", "missing_chart_id"}
                    for issue in issues
                ) else 5.0,
                "research_quality": 8.0 if not any(
                    issue in {"report_too_short", "answer_too_short", "insufficient_evidence"}
                    for issue in issues
                ) else 4.0,
            },
            metadata={"task_instruction": task_instruction, "skill": "ResearchEvaluator"},
        )
