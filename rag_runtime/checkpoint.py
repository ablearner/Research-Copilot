from __future__ import annotations

from typing import Any, Iterable, Protocol

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from rag_runtime.schemas import (
    ChartUnderstandingResult,
    DocumentIndexResult,
    FusedAskResult,
    GraphTaskRequest,
    GraphTaskResult,
    RuntimeHealthSummary,
)
from rag_runtime.state import ChartDocRAGState, GraphInput, GraphOutput

PROJECT_CHECKPOINT_SCHEMAS: tuple[type[Any], ...] = (
    ChartDocRAGState,
    GraphInput,
    GraphOutput,
    GraphTaskRequest,
    GraphTaskResult,
    DocumentIndexResult,
    ChartUnderstandingResult,
    FusedAskResult,
    RuntimeHealthSummary,
)


class CheckpointerFactory(Protocol):
    def create(self) -> Any:
        ...


class LangGraphCheckpointerFactory:
    """Factory wrapper so the storage backend can move from memory to DB/Redis later."""

    def __init__(self, backend: str = "memory") -> None:
        self.backend = backend

    def create(self) -> Any:
        if self.backend == "memory":
            return self._build_in_memory()
        raise RuntimeError(f"Unsupported LangGraph checkpoint backend: {self.backend}")

    def _build_in_memory(self) -> Any:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=()))
        except Exception:
            try:
                from langgraph.checkpoint.memory import MemorySaver

                return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=()))
            except Exception as exc:
                raise RuntimeError(
                    "LangGraph in-memory checkpointer is unavailable. Install a compatible langgraph version."
                ) from exc


def build_project_allowlist(*, extra_schemas: Iterable[type[Any]] | None = None) -> set[tuple[str, ...]]:
    schemas = [*PROJECT_CHECKPOINT_SCHEMAS, *(extra_schemas or ())]
    try:
        from langgraph._internal import _serde

        return _serde.build_serde_allowlist(schemas=schemas)
    except Exception:
        return {
            (schema.__module__, schema.__name__)
            for schema in schemas
            if schema is not None and getattr(schema, "__module__", None) and getattr(schema, "__name__", None)
        }


def apply_project_allowlist(
    checkpointer: Any,
    *,
    extra_schemas: Iterable[type[Any]] | None = None,
) -> Any:
    if checkpointer in (None, True, False):
        return checkpointer
    allowlist = build_project_allowlist(extra_schemas=extra_schemas)
    with_allowlist = getattr(checkpointer, "with_allowlist", None)
    if callable(with_allowlist):
        return with_allowlist(allowlist)
    return checkpointer


def build_checkpointer(backend: str = "memory") -> Any:
    return LangGraphCheckpointerFactory(backend=backend).create()
