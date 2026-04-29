"""Loader for agent-editable knowledge entrys (Markdown + YAML frontmatter).

Knowledge entrys are stored as directories under a configurable skills_dir:

    .data/skills/
    ├── molecular-dynamics/
    │   └── SKILL.md
    └── meta-analysis/
        └── SKILL.md

Each SKILL.md has YAML frontmatter:

    ---
    name: molecular-dynamics
    description: Tips for molecular dynamics simulations
    tags: [md, simulation, force-field]
    ---
    # Molecular Dynamics Best Practices
    ...
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

DEFAULT_KNOWLEDGE_DIR = Path(".data/skills")


class KnowledgeEntryMeta(BaseModel):
    """Metadata for a knowledge entry."""
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class KnowledgeEntry(BaseModel):
    """A full knowledge entry with content."""
    meta: KnowledgeEntryMeta
    content: str
    path: str


def _parse_skill_file(skill_path: Path) -> KnowledgeEntry | None:
    """Parse a SKILL.md file into a KnowledgeEntry."""
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except OSError:
        return None

    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        fm_text = fm_match.group(1)
        content = raw[fm_match.end():]
        try:
            fm_data = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            fm_data = {}
    else:
        fm_data = {}
        content = raw

    meta = KnowledgeEntryMeta(
        name=fm_data.get("name", skill_path.parent.name),
        description=fm_data.get("description", ""),
        tags=fm_data.get("tags", []),
        created_at=fm_data.get("created_at", ""),
        updated_at=fm_data.get("updated_at", ""),
    )
    return KnowledgeEntry(meta=meta, content=content.strip(), path=str(skill_path))


class KnowledgeLoader:
    """Manages agent-editable knowledge entrys on disk."""

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir else DEFAULT_KNOWLEDGE_DIR

    def list_entries(self) -> list[KnowledgeEntryMeta]:
        """List metadata for all knowledge entrys."""
        if not self.skills_dir.exists():
            return []
        result: list[KnowledgeEntryMeta] = []
        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = _parse_skill_file(skill_file)
            if skill:
                result.append(skill.meta)
        return result

    def load_entry(self, name: str) -> KnowledgeEntry | None:
        """Load a full knowledge entry by name."""
        skill_file = self.skills_dir / name / "SKILL.md"
        if not skill_file.exists():
            return None
        return _parse_skill_file(skill_file)

    def save_entry(
        self,
        name: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> KnowledgeEntry:
        """Create or update a knowledge entry."""
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        existing = _parse_skill_file(skill_file) if skill_file.exists() else None
        created = existing.meta.created_at if existing else now

        frontmatter: dict[str, Any] = {
            "name": name,
            "description": description or (existing.meta.description if existing else ""),
            "tags": tags or (existing.meta.tags if existing else []),
            "created_at": created,
            "updated_at": now,
        }
        fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
        full_text = f"---\n{fm_text}\n---\n\n{content.strip()}\n"
        skill_file.write_text(full_text, encoding="utf-8")
        logger.info("Knowledge entry saved: %s", name)
        return _parse_skill_file(skill_file)  # type: ignore[return-value]

    def delete_entry(self, name: str) -> bool:
        """Delete a knowledge entry directory."""
        import shutil

        skill_dir = self.skills_dir / name
        if not skill_dir.exists():
            return False
        shutil.rmtree(skill_dir)
        logger.info("Knowledge entry deleted: %s", name)
        return True
