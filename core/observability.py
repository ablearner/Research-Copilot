from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResearchObservabilityService:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.metrics_path = self.root_dir / "metrics.jsonl"
        self.failures_path = self.root_dir / "failures.jsonl"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def record_metric(self, *, metric_type: str, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.metrics_path,
            {"metric_type": metric_type, **payload},
        )

    def archive_failure(self, *, failure_type: str, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.failures_path,
            {"failure_type": failure_type, **payload},
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
