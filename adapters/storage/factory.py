"""Factory for creating storage backend instances."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StorageBackend


def create_store(provider: str, **kwargs: object) -> StorageBackend:
    """Create a storage backend based on provider name.

    Args:
        provider: ``"sqlite"`` or ``"json"``.
        **kwargs: Provider-specific arguments.
            - ``db_path`` (Path): Required for sqlite.
            - ``storage_root`` (Path): Required for json.
    """
    if provider == "sqlite":
        from .sqlite_store import SQLiteStore

        db_path = kwargs.get("db_path")
        if db_path is None:
            raise ValueError("db_path is required for sqlite provider")
        return SQLiteStore(db_path=Path(db_path))  # type: ignore[arg-type]
    elif provider == "json":
        from services.research.research_report_service import ResearchReportService

        storage_root = kwargs.get("storage_root")
        if storage_root is None:
            raise ValueError("storage_root is required for json provider")
        return ResearchReportService(storage_root=Path(storage_root))  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown storage provider: {provider}")
