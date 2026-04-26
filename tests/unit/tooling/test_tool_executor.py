import pytest
from pydantic import BaseModel, Field

from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import ToolSpec


class RetryInput(BaseModel):
    value: int = Field(ge=0)


class RetryOutput(BaseModel):
    doubled: int


class WrongOutput(BaseModel):
    wrong_field: int


@pytest.mark.asyncio
async def test_tool_executor_retries_until_success() -> None:
    attempts = {"count": 0}

    async def flaky_handler(value: int):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary failure")
        return {"doubled": value * 2}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="retry_tool",
            description="retry",
            input_schema=RetryInput,
            output_schema=RetryOutput,
            handler=flaky_handler,
            max_retries=2,
        )
    )
    executor = ToolExecutor(registry)

    result = await executor.execute_tool_call("retry_tool", {"value": 3})

    assert result.status == "succeeded"
    assert result.output == {"doubled": 6}
    assert result.attempt_count == 3
    assert len(result.trace.attempts) == 3


@pytest.mark.asyncio
async def test_tool_executor_retries_on_output_validation_error() -> None:
    async def invalid_handler(value: int):
        return WrongOutput(wrong_field=value)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="invalid_tool",
            description="invalid",
            input_schema=RetryInput,
            output_schema=RetryOutput,
            handler=invalid_handler,
            max_retries=1,
        )
    )
    executor = ToolExecutor(registry)

    result = await executor.execute_tool_call("invalid_tool", {"value": 4})

    assert result.status == "validation_error"
    assert result.attempt_count == 2
    assert "attempt 1" in (result.error_message or "")


@pytest.mark.asyncio
async def test_tool_executor_records_audit_context() -> None:
    async def handler(value: int):
        return {"doubled": value * 2}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="audited_tool",
            description="audited",
            input_schema=RetryInput,
            output_schema=RetryOutput,
            handler=handler,
            category="research",
            audit_metadata={"owner": "research-runtime"},
        )
    )
    executor = ToolExecutor(registry)

    result = await executor.execute_tool_call(
        "audited_tool",
        {
            "value": 5,
            "task_id": "task-1",
            "session_id": "session-1",
            "correlation_id": "corr-1",
            "metadata": {"source": "unit-test"},
        },
    )

    assert result.status == "succeeded"
    assert result.trace.tool_category == "research"
    assert result.trace.task_id == "task-1"
    assert result.trace.session_id == "session-1"
    assert result.trace.correlation_id == "corr-1"
    assert result.trace.audit_metadata["owner"] == "research-runtime"
    assert result.trace.audit_metadata["source"] == "unit-test"
