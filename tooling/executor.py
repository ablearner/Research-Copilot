from __future__ import annotations

import inspect
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from tooling.registry import ToolRegistry
from tooling.schemas import ToolAttemptTrace, ToolCall, ToolCallTrace, ToolExecutionResult, ToolSpec
from tooling.serializers import to_jsonable

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._traces: list[ToolCallTrace] = []

    async def execute_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> ToolExecutionResult:
        call = ToolCall(
            call_id=call_id or f"call_{uuid4().hex}",
            tool_name=tool_name,
            arguments=tool_input or {},
        )
        started_at = perf_counter()
        tool_spec = self.registry.get_tool(call.tool_name, include_disabled=True)
        audit_context = self._extract_audit_context(call.arguments)

        if tool_spec is None:
            return self._build_result(
                tool_name=call.tool_name,
                call_id=call.call_id,
                tool_input=call.arguments,
                status="not_found",
                output=None,
                error_message=f"Tool not found: {call.tool_name}",
                started_at=started_at,
                audit_context=audit_context,
            )

        if not tool_spec.enabled:
            return self._build_result(
                tool_name=call.tool_name,
                call_id=call.call_id,
                tool_input=call.arguments,
                status="disabled",
                output=None,
                error_message=f"Tool is disabled: {call.tool_name}",
                started_at=started_at,
                audit_context=audit_context,
            )

        try:
            validated_input = self.validate_input(tool_spec, call.arguments)
        except ValidationError as exc:
            return self._build_result(
                tool_name=call.tool_name,
                call_id=call.call_id,
                tool_input=call.arguments,
                status="validation_error",
                output=None,
                error_message=str(exc),
                started_at=started_at,
                audit_context=audit_context,
            )

        attempts: list[ToolAttemptTrace] = []
        kwargs = self._validated_input_to_kwargs(validated_input)
        max_attempts = max(1, tool_spec.max_retries + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                output = tool_spec.handler(**kwargs)
                if inspect.isawaitable(output):
                    output = await output
                serialized_output = self.serialize_output(tool_spec, output)
                attempts.append(
                    ToolAttemptTrace(
                        attempt=attempt,
                        status="succeeded",
                        validation_passed=True,
                    )
                )
                return self._build_result(
                    tool_name=call.tool_name,
                    call_id=call.call_id,
                    tool_input=call.arguments,
                    status="succeeded",
                    output=serialized_output,
                    error_message=None,
                    started_at=started_at,
                    attempt_count=attempt,
                    attempts=attempts,
                    validation_passed=True,
                    tool_spec=tool_spec,
                    audit_context=audit_context,
                )
            except ValidationError as exc:
                attempts.append(
                    ToolAttemptTrace(
                        attempt=attempt,
                        status="validation_error",
                        error_message=str(exc),
                        validation_passed=False,
                    )
                )
                if attempt >= max_attempts:
                    return self._build_result(
                        tool_name=call.tool_name,
                        call_id=call.call_id,
                        tool_input=call.arguments,
                        status="validation_error",
                        output=None,
                        error_message=self._merge_attempt_errors(attempts),
                        started_at=started_at,
                        attempt_count=attempt,
                        attempts=attempts,
                        validation_passed=False,
                        tool_spec=tool_spec,
                        audit_context=audit_context,
                    )
            except Exception as exc:
                logger.exception(
                    "Tool execution failed",
                    extra={"tool_name": call.tool_name, "attempt": attempt},
                )
                attempts.append(
                    ToolAttemptTrace(
                        attempt=attempt,
                        status="failed",
                        error_message=str(exc),
                        validation_passed=True,
                    )
                )
                if attempt >= max_attempts:
                    return self._build_result(
                        tool_name=call.tool_name,
                        call_id=call.call_id,
                        tool_input=call.arguments,
                        status="failed",
                        output=None,
                        error_message=self._merge_attempt_errors(attempts),
                        started_at=started_at,
                        attempt_count=attempt,
                        attempts=attempts,
                        validation_passed=True,
                        tool_spec=tool_spec,
                        audit_context=audit_context,
                    )

    def validate_input(self, tool_spec: ToolSpec, tool_input: dict[str, Any]) -> BaseModel:
        return tool_spec.input_schema.model_validate(tool_input)

    def serialize_output(self, tool_spec: ToolSpec, output: Any) -> Any:
        if output is None:
            return None

        output_schema = tool_spec.output_schema
        if output_schema and isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            if not tool_spec.strict_output_validation:
                return to_jsonable(output)
            if isinstance(output, BaseModel):
                if isinstance(output, output_schema):
                    return output.model_dump(mode="json")
                return output_schema.model_validate(output.model_dump(mode="json")).model_dump(
                    mode="json"
                )
            return output_schema.model_validate(output).model_dump(mode="json")
        return to_jsonable(output)

    def record_trace(self, trace: ToolCallTrace) -> None:
        self._traces.append(trace)

    def get_traces(self, limit: int | None = None) -> list[ToolCallTrace]:
        if limit is None or limit <= 0:
            return list(self._traces)
        return self._traces[-limit:]

    def _validated_input_to_kwargs(self, validated_input: BaseModel) -> dict[str, Any]:
        return {
            field_name: getattr(validated_input, field_name)
            for field_name in validated_input.__class__.model_fields
        }

    def _build_result(
        self,
        tool_name: str,
        call_id: str,
        tool_input: dict[str, Any],
        status: str,
        output: Any,
        error_message: str | None,
        started_at: float,
        attempt_count: int = 1,
        attempts: list[ToolAttemptTrace] | None = None,
        validation_passed: bool = True,
        tool_spec: ToolSpec | None = None,
        audit_context: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        resolved_audit_context = audit_context or {}
        trace = ToolCallTrace(
            call_id=call_id,
            tool_name=tool_name,
            input=to_jsonable(tool_input),
            output=to_jsonable(output),
            status=status,
            tool_category=tool_spec.category if tool_spec is not None else "runtime",
            attempt_count=attempt_count,
            attempts=attempts or [],
            latency_ms=latency_ms,
            error_message=error_message,
            session_id=resolved_audit_context.get("session_id"),
            task_id=resolved_audit_context.get("task_id"),
            correlation_id=resolved_audit_context.get("correlation_id"),
            audit_metadata=to_jsonable(
                {
                    **(tool_spec.audit_metadata if tool_spec is not None else {}),
                    **resolved_audit_context.get("audit_metadata", {}),
                }
            ),
        )
        self.record_trace(trace)
        return ToolExecutionResult(
            call_id=call_id,
            tool_name=tool_name,
            status=status,
            output=output,
            error_message=error_message,
            attempt_count=attempt_count,
            validation_passed=validation_passed,
            trace=trace,
        )

    def _merge_attempt_errors(self, attempts: list[ToolAttemptTrace]) -> str:
        reasons = [
            f"attempt {attempt.attempt}: {attempt.status} - {attempt.error_message}"
            for attempt in attempts
            if attempt.error_message
        ]
        return "; ".join(reasons)

    def _extract_audit_context(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        audit_metadata = dict(tool_input.get("metadata") or {})
        return {
            "session_id": tool_input.get("session_id"),
            "task_id": tool_input.get("task_id"),
            "correlation_id": tool_input.get("correlation_id") or audit_metadata.get("correlation_id"),
            "audit_metadata": audit_metadata,
        }
