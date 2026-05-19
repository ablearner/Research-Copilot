from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from domain.schemas.paper_knowledge import PaperKnowledgeRecord
from domain.schemas.research_memory import LongTermMemoryRecord, SessionMemoryRecord
from memory.factory import resolve_memory_db_path
from memory.long_term_memory import SQLiteLongTermMemoryStore
from memory.paper_knowledge_memory import SQLitePaperKnowledgeStore
from memory.session_memory import SQLiteSessionMemoryStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy JSON research memory files into kepler.db SQLite tables.",
    )
    parser.add_argument("--db-path", help="Target SQLite DB path. Defaults to configured research DB.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count records without writing.")
    return parser.parse_args()


def iter_json_files(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def migrate_long_term(paths: list[Path], store: SQLiteLongTermMemoryStore | None) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "skipped_duplicate": 0, "invalid": 0}
    markers: set[tuple[str, str, str]] = set()
    for path in iter_json_files(paths):
        stats["seen"] += 1
        try:
            record = LongTermMemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            stats["invalid"] += 1
            continue
        marker = (record.memory_type, record.topic.strip().lower(), record.content.strip().lower())
        if marker in markers:
            stats["skipped_duplicate"] += 1
            continue
        markers.add(marker)
        if store is not None:
            store.upsert(record)
        stats["written"] += 1
    return stats


def migrate_sessions(paths: list[Path], store: SQLiteSessionMemoryStore | None) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "invalid": 0}
    for path in iter_json_files(paths):
        stats["seen"] += 1
        try:
            record = SessionMemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            stats["invalid"] += 1
            continue
        if store is not None:
            store.put(record)
        stats["written"] += 1
    return stats


def migrate_paper_knowledge(paths: list[Path], store: SQLitePaperKnowledgeStore | None) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "invalid": 0}
    for path in iter_json_files(paths):
        stats["seen"] += 1
        try:
            record = PaperKnowledgeRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            stats["invalid"] += 1
            continue
        if store is not None:
            store.put(record)
        stats["written"] += 1
    return stats


def main() -> None:
    args = parse_args()
    settings = Settings()
    storage_root = settings.resolve_path(settings.research_storage_root)
    db_path = settings.resolve_path(args.db_path) if args.db_path else resolve_memory_db_path(settings)

    long_term_paths = [
        settings.resolve_path(settings.research_long_term_memory_dir),
        storage_root / "memory" / "long_term",
    ]
    session_paths = [
        settings.resolve_path(settings.research_session_memory_dir),
        storage_root / "memory" / "sessions",
    ]
    paper_knowledge_paths = [
        settings.resolve_path(settings.research_paper_knowledge_dir),
        storage_root / "memory" / "paper_knowledge",
    ]

    long_term_store = None if args.dry_run else SQLiteLongTermMemoryStore(
        db_path=db_path,
        max_records=settings.long_term_memory_max_records,
    )
    session_store = None if args.dry_run else SQLiteSessionMemoryStore(db_path=db_path)
    paper_knowledge_store = None if args.dry_run else SQLitePaperKnowledgeStore(db_path=db_path)

    result = {
        "db_path": str(db_path),
        "dry_run": args.dry_run,
        "sources": {
            "long_term": [str(path) for path in long_term_paths],
            "sessions": [str(path) for path in session_paths],
            "paper_knowledge": [str(path) for path in paper_knowledge_paths],
        },
        "long_term": migrate_long_term(long_term_paths, long_term_store),
        "sessions": migrate_sessions(session_paths, session_store),
        "paper_knowledge": migrate_paper_knowledge(paper_knowledge_paths, paper_knowledge_store),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
