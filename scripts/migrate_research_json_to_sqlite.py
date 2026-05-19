from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.storage.research_report_service import ResearchReportService
from adapters.storage.sqlite_store import SQLiteStore
from core.config import Settings
from domain.schemas.research import (
    PaperCandidate,
    ResearchConversation,
    ResearchJob,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
)

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy JSON research storage into the configured SQLite store.",
    )
    parser.add_argument(
        "--source-root",
        help="Legacy JSON storage root. Defaults to configured RESEARCH_STORAGE_ROOT.",
    )
    parser.add_argument(
        "--db-path",
        help="Target SQLite DB path. Defaults to configured RESEARCH_SQLITE_DB_PATH.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and count without writing.")
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def migrate_items(
    paths: list[Path],
    *,
    validate: Callable[[object], T],
    write: Callable[[T], None] | None,
) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "invalid": 0}
    for path in sorted(paths):
        stats["seen"] += 1
        try:
            item = validate(read_json(path))
        except Exception:
            stats["invalid"] += 1
            continue
        if write is not None:
            write(item)
        stats["written"] += 1
    return stats


def migrate_messages(
    service: ResearchReportService,
    store: SQLiteStore | None,
) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "invalid": 0}
    for path in sorted(service.messages_root.glob("*.json")):
        stats["seen"] += 1
        conversation_id = path.stem
        try:
            messages = [ResearchMessage.model_validate(item) for item in read_json(path)]
        except Exception:
            stats["invalid"] += 1
            continue
        if store is not None:
            store.save_messages(conversation_id, messages)
        stats["written"] += 1
    return stats


def migrate_reports(
    service: ResearchReportService,
    store: SQLiteStore | None,
) -> dict[str, int]:
    paths = sorted(service.reports_root.glob("*/*.json"))
    return migrate_items(
        paths,
        validate=ResearchReport.model_validate,
        write=store.save_report if store is not None else None,
    )


def migrate_papers(
    service: ResearchReportService,
    store: SQLiteStore | None,
) -> dict[str, int]:
    stats = {"seen": 0, "written": 0, "invalid": 0}
    for path in sorted(service.papers_root.glob("*.json")):
        stats["seen"] += 1
        task_id = path.stem
        try:
            papers = [PaperCandidate.model_validate(item) for item in read_json(path)]
        except Exception:
            stats["invalid"] += 1
            continue
        if store is not None:
            store.save_papers(task_id, papers)
        stats["written"] += 1
    return stats


def main() -> None:
    args = parse_args()
    settings = Settings()
    source_root = settings.resolve_path(args.source_root or settings.research_storage_root)
    db_path = settings.resolve_path(args.db_path or settings.research_sqlite_db_path)

    service = ResearchReportService(source_root)
    store = None if args.dry_run else SQLiteStore(db_path=db_path)

    result = {
        "source_root": str(source_root),
        "db_path": str(db_path),
        "dry_run": args.dry_run,
        "conversations": migrate_items(
            sorted(service.conversations_root.glob("*.json")),
            validate=ResearchConversation.model_validate,
            write=store.save_conversation if store is not None else None,
        ),
        "messages": migrate_messages(service, store),
        "tasks": migrate_items(
            sorted(service.tasks_root.glob("*.json")),
            validate=ResearchTask.model_validate,
            write=store.save_task if store is not None else None,
        ),
        "reports": migrate_reports(service, store),
        "papers": migrate_papers(service, store),
        "jobs": migrate_items(
            sorted(service.jobs_root.glob("*.json")),
            validate=ResearchJob.model_validate,
            write=store.save_job if store is not None else None,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
