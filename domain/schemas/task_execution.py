"""Trusted execution identity. Constructed by the worker, never by model tool arguments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionContext:
    workspace_id: str
    user_id: str
    project_id: str
    conversation_id: str | None
    task_id: str
    execution_epoch: int
    allowed_document_ids: tuple[str, ...]
    timeout_seconds: int
