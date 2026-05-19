from __future__ import annotations

from pathlib import Path

from core.config import Settings
from memory.long_term_memory import (
    InMemoryLongTermMemoryStore,
    JsonLongTermMemoryStore,
    LongTermMemory,
    SQLiteLongTermMemoryStore,
)
from memory.memory_manager import MemoryManager
from memory.paper_knowledge_memory import (
    InMemoryPaperKnowledgeStore,
    JsonPaperKnowledgeStore,
    PaperKnowledgeMemory,
    SQLitePaperKnowledgeStore,
)
from memory.quality_gate import MemoryQualityGate
from memory.session_memory import (
    InMemorySessionMemoryStore,
    JsonSessionMemoryStore,
    SQLiteSessionMemoryStore,
    SessionMemory,
)
from memory.working_memory import WorkingMemory


_DEFAULT_RESEARCH_STORAGE_ROOT = ".data/research"
_DEFAULT_RESEARCH_SQLITE_DB_PATH = ".data/research/kepler.db"


def resolve_memory_db_path(settings: Settings) -> Path:
    """Resolve the SQLite memory DB path with test/local overrides respected."""

    storage_root = getattr(settings, "research_storage_root", _DEFAULT_RESEARCH_STORAGE_ROOT)
    db_path = getattr(settings, "research_sqlite_db_path", _DEFAULT_RESEARCH_SQLITE_DB_PATH)
    if (
        str(db_path) == _DEFAULT_RESEARCH_SQLITE_DB_PATH
        and str(storage_root) != _DEFAULT_RESEARCH_STORAGE_ROOT
    ):
        return settings.resolve_path(storage_root) / "kepler.db"
    return settings.resolve_path(db_path)


def build_long_term_memory(settings: Settings) -> LongTermMemory:
    provider = _provider(settings.long_term_memory_provider, default="sqlite")
    if provider in {"sqlite", "auto"}:
        return LongTermMemory(
            SQLiteLongTermMemoryStore(
                db_path=resolve_memory_db_path(settings),
                max_records=settings.long_term_memory_max_records,
            )
        )
    if provider in {"json", "file"}:
        return LongTermMemory(
            JsonLongTermMemoryStore(
                settings.resolve_path(settings.research_long_term_memory_dir)
            )
        )
    if provider in {"memory", "local", "inmemory"}:
        return LongTermMemory(InMemoryLongTermMemoryStore())
    raise RuntimeError(f"Unsupported LONG_TERM_MEMORY_PROVIDER: {settings.long_term_memory_provider}")


def build_session_memory(settings: Settings) -> SessionMemory:
    provider = _provider(settings.session_memory_provider, default="sqlite")
    if provider in {"sqlite", "auto"}:
        return SessionMemory(SQLiteSessionMemoryStore(db_path=resolve_memory_db_path(settings)))
    if provider in {"json", "file"}:
        return SessionMemory(
            JsonSessionMemoryStore(settings.resolve_path(settings.research_session_memory_dir))
        )
    if provider in {"memory", "local", "inmemory"}:
        return SessionMemory(InMemorySessionMemoryStore())
    raise RuntimeError(f"Unsupported SESSION_MEMORY_PROVIDER: {settings.session_memory_provider}")


def build_paper_knowledge_memory(settings: Settings) -> PaperKnowledgeMemory:
    configured = getattr(settings, "paper_knowledge_provider", None)
    provider = _provider(configured or "sqlite", default="sqlite")
    if provider in {"sqlite", "auto"}:
        return PaperKnowledgeMemory(SQLitePaperKnowledgeStore(db_path=resolve_memory_db_path(settings)))
    if provider in {"json", "file"}:
        return PaperKnowledgeMemory(
            JsonPaperKnowledgeStore(settings.resolve_path(settings.research_paper_knowledge_dir))
        )
    if provider in {"memory", "local", "inmemory"}:
        return PaperKnowledgeMemory(InMemoryPaperKnowledgeStore())
    raise RuntimeError(f"Unsupported PAPER_KNOWLEDGE_PROVIDER: {configured}")


def build_memory_manager(settings: Settings) -> MemoryManager:
    return MemoryManager(
        working_memory=WorkingMemory(max_turns=settings.research_working_memory_turns),
        session_memory=build_session_memory(settings),
        long_term_memory=build_long_term_memory(settings),
        paper_knowledge_memory=build_paper_knowledge_memory(settings),
        quality_gate=MemoryQualityGate(
            enabled=settings.memory_quality_gate_enabled,
            min_score=settings.memory_min_quality_score,
        ),
    )


def _provider(value: str | None, *, default: str) -> str:
    return str(value or default).strip().lower()
