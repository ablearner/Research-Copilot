from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from skills.base import SkillSpec

logger = logging.getLogger(__name__)


class SkillLoaderError(RuntimeError):
    """Raised when skill loading fails."""


class SkillLoader:
    def __init__(self, specs_dir: str | Path = "skills/specs") -> None:
        self.specs_dir = Path(specs_dir)

    def load_from_file(self, file_path: str | Path) -> list[SkillSpec]:
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8")
            raw = self._parse_content(content)
            if raw is None:
                return []
            if isinstance(raw, list):
                return [SkillSpec.model_validate(item) for item in raw if item]
            if isinstance(raw, dict):
                return [SkillSpec.model_validate(raw)]
            raise SkillLoaderError(f"Unsupported skill spec format: {path}")
        except Exception as exc:
            logger.exception("Failed to load skill spec", extra={"path": str(path)})
            raise SkillLoaderError(f"Failed to load skill spec: {path}") from exc

    def load_from_directory(self, specs_dir: str | Path | None = None) -> list[SkillSpec]:
        base_dir = Path(specs_dir) if specs_dir else self.specs_dir
        if not base_dir.exists():
            logger.warning("Skill specs directory does not exist", extra={"path": str(base_dir)})
            return []

        loaded: list[SkillSpec] = []
        for path in sorted(base_dir.glob("*.yaml")):
            try:
                loaded.extend(self.load_from_file(path))
            except SkillLoaderError:
                logger.warning("Skip invalid skill spec", extra={"path": str(path)})
        return loaded

    def _parse_content(self, content: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        stripped = content.strip()
        if not stripped:
            return None

        try:
            import yaml  # type: ignore[import-not-found]

            return yaml.safe_load(stripped)
        except ModuleNotFoundError:
            # JSON is a valid subset of YAML. This fallback keeps the first version dependency-light.
            return json.loads(stripped)
