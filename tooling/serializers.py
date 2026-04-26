from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from tooling.schemas import ToolSpec


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    return value


def tool_spec_to_openai_function(tool_spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool_spec.name,
        "description": tool_spec.description,
        "parameters": tool_spec.input_schema.model_json_schema(),
        "strict": True,
    }


def tool_specs_to_openai_functions(tool_specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [tool_spec_to_openai_function(tool_spec) for tool_spec in tool_specs]
