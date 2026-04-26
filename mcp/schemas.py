from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


MCPToolCallStatus = Literal[
    "succeeded",
    "failed",
    "not_found",
    "invalid_input",
    "disabled",
]


class MCPToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    source: Literal["local", "external"] = "local"
    server_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResult(BaseModel):
    call_id: str
    tool_name: str
    status: MCPToolCallStatus
    output: Any | None = None
    error_message: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    server_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MCPPromptSpec(BaseModel):
    name: str
    description: str
    prompt_key: str
    path: str
    skill_name: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPPromptContent(BaseModel):
    name: str
    prompt_key: str
    path: str
    content: str
    skill_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPResourceSpec(BaseModel):
    uri: str
    name: str
    description: str
    resource_type: Literal[
        "document_summary",
        "chart_summary",
        "graph_community_summary",
        "schema",
        "config",
        "custom",
    ] = "custom"
    mime_type: str = "application/json"
    read_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPResourceContent(BaseModel):
    uri: str
    mime_type: str = "application/json"
    content: Any
    read_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPServerDescriptor(BaseModel):
    name: str
    description: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPDiscoverySnapshot(BaseModel):
    server: MCPServerDescriptor
    tools: list[MCPToolSpec] = Field(default_factory=list)
    prompts: list[MCPPromptSpec] = Field(default_factory=list)
    resources: list[MCPResourceSpec] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
