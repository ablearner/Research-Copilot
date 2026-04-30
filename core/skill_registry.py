"""Standardized Skill System for Research-Copilot.

A skill is a directory under ``skills/`` containing a ``SKILL.md`` file with
YAML frontmatter (metadata) and a Markdown body (instructions).  Users can
drop new skill folders into ``skills/builtin/`` or ``skills/community/`` and
they become available immediately after restart.

Architecture:

    skills/
    ├── builtin/                  # shipped with the project, git-tracked
    │   └── paper-comparison/
    │       └── SKILL.md
    ├── community/                # user-installed, gitignored
    │   └── arxiv-deep-search/
    │       └── SKILL.md
    └── .skill_config.json        # per-skill enable/disable overrides

Three-tier progressive disclosure:
    L1  frontmatter only  (loaded at startup for all skills)
    L2  body instructions (loaded on demand when a skill is matched)
    L3  references/templates/examples (loaded on demand from sub-dirs)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DEFAULT_SKILLS_DIR = Path("skills")
_SKILL_FILENAME = "SKILL.md"
_CONFIG_FILENAME = ".skill_config.json"
_MAX_SKILL_BODY_SIZE = 50_000  # chars – safety cap


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SkillMeta(BaseModel):
    """L1 metadata parsed from the YAML frontmatter of SKILL.md."""

    name: str
    description: str = ""
    version: str = ""
    author: str = ""
    category: str = "general"
    tags: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    requires_tools: list[str] = Field(default_factory=list)
    requires_skills: list[str] = Field(default_factory=list)
    trust_level: str = "community"   # builtin | trusted | community
    enabled: bool = True
    path: str = ""


class Skill(BaseModel):
    """Full skill object (L1 + L2)."""

    meta: SkillMeta
    body: str = ""                   # Markdown instructions (L2)
    references: dict[str, str] = Field(default_factory=dict)  # L3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _meta_from_frontmatter(fm_data: dict[str, Any], skill_path: Path) -> SkillMeta:
    """Build a SkillMeta from parsed YAML frontmatter data."""
    name = fm_data.get("name") or skill_path.parent.name
    requires = fm_data.get("requires") or {}
    if isinstance(requires, dict):
        requires_tools = requires.get("tools", [])
        requires_skills = requires.get("skills", [])
    else:
        requires_tools = []
        requires_skills = []
    return SkillMeta(
        name=name,
        description=fm_data.get("description", ""),
        version=fm_data.get("version", ""),
        author=fm_data.get("author", ""),
        category=fm_data.get("category", "general"),
        tags=fm_data.get("tags", []),
        triggers=fm_data.get("triggers", []),
        requires_tools=requires_tools if isinstance(requires_tools, list) else [],
        requires_skills=requires_skills if isinstance(requires_skills, list) else [],
        trust_level=fm_data.get("trust_level", "community"),
        enabled=fm_data.get("enabled", True),
        path=str(skill_path),
    )


def parse_skill_file(skill_path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill object (L1 + L2)."""
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Cannot read skill file: %s", skill_path)
        return None

    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        fm_text = fm_match.group(1)
        body = raw[fm_match.end():]
        try:
            fm_data: dict[str, Any] = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            fm_data = {}
    else:
        fm_data = {}
        body = raw

    meta = _meta_from_frontmatter(fm_data, skill_path)
    return Skill(meta=meta, body=body.strip()[:_MAX_SKILL_BODY_SIZE])


def parse_skill_meta(skill_path: Path) -> SkillMeta | None:
    """Parse only L1 metadata from a SKILL.md file (cheap)."""
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except OSError:
        return None

    fm_match = _FRONTMATTER_RE.match(raw)
    if not fm_match:
        return SkillMeta(name=skill_path.parent.name, path=str(skill_path))

    fm_text = fm_match.group(1)
    try:
        fm_data: dict[str, Any] = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm_data = {}

    return _meta_from_frontmatter(fm_data, skill_path)


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Scans skill directories, indexes metadata, and provides on-demand loading.

    Usage::

        registry = SkillRegistry()          # scans default skills/ dir
        registry.scan()                     # (re)scan
        metas = registry.list_skills()      # L1 only
        skill = registry.load_skill("paper-comparison")  # L1+L2
        ref = registry.load_reference("paper-comparison", "references/api.md")  # L3
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        scan_subdirs: list[str] | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
        self._scan_subdirs = scan_subdirs or ["builtin", "community"]
        self._index: dict[str, SkillMeta] = {}
        self._config_overrides: dict[str, dict[str, Any]] = {}
        self._load_config()

    # -- Config persistence --------------------------------------------------

    def _config_path(self) -> Path:
        return self.skills_dir / _CONFIG_FILENAME

    def _load_config(self) -> None:
        config_path = self._config_path()
        if not config_path.is_file():
            self._config_overrides = {}
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self._config_overrides = data.get("skills", {})
        except (json.JSONDecodeError, OSError):
            self._config_overrides = {}

    def _save_config(self) -> None:
        config_path = self._config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "global": {
                "auto_match": True,
                "max_active_skills": 3,
                "community_skills_enabled": True,
            },
            "skills": self._config_overrides,
        }
        config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # -- Scanning ------------------------------------------------------------

    def scan(self) -> list[SkillMeta]:
        """Scan all skill sub-directories and build the L1 index."""
        self._index.clear()
        self._load_config()
        for subdir_name in self._scan_subdirs:
            subdir = self.skills_dir / subdir_name
            if not subdir.is_dir():
                continue
            trust = "builtin" if subdir_name == "builtin" else "community"
            for skill_file in sorted(subdir.rglob(_SKILL_FILENAME)):
                meta = parse_skill_meta(skill_file)
                if meta is None:
                    continue
                meta.trust_level = trust
                # Apply config overrides
                override = self._config_overrides.get(meta.name, {})
                if "enabled" in override:
                    meta.enabled = override["enabled"]
                self._index[meta.name] = meta
        # Also scan root-level skills (flat layout)
        for skill_file in sorted(self.skills_dir.glob(f"*/{_SKILL_FILENAME}")):
            if any(
                skill_file.is_relative_to(self.skills_dir / sd)
                for sd in self._scan_subdirs
            ):
                continue  # already scanned
            meta = parse_skill_meta(skill_file)
            if meta is None:
                continue
            override = self._config_overrides.get(meta.name, {})
            if "enabled" in override:
                meta.enabled = override["enabled"]
            if meta.name not in self._index:
                self._index[meta.name] = meta

        logger.info("Skill registry scanned %d skill(s)", len(self._index))
        return list(self._index.values())

    # -- Queries -------------------------------------------------------------

    def list_skills(self, *, include_disabled: bool = False) -> list[SkillMeta]:
        """Return L1 metadata for all (or enabled-only) skills."""
        if not self._index:
            self.scan()
        if include_disabled:
            return list(self._index.values())
        return [m for m in self._index.values() if m.enabled]

    def get_meta(self, name: str) -> SkillMeta | None:
        """Get L1 metadata by skill name."""
        if not self._index:
            self.scan()
        return self._index.get(name)

    def load_skill(self, name: str) -> Skill | None:
        """Load a full skill (L1 + L2 body) by name."""
        meta = self.get_meta(name)
        if meta is None:
            return None
        skill_path = Path(meta.path)
        if not skill_path.is_file():
            return None
        return parse_skill_file(skill_path)

    def load_reference(self, skill_name: str, ref_path: str) -> str | None:
        """Load an L3 reference file from a skill directory."""
        meta = self.get_meta(skill_name)
        if meta is None:
            return None
        skill_dir = Path(meta.path).parent
        target = (skill_dir / ref_path).resolve()
        # Security: ensure the file is inside the skill directory
        if not str(target).startswith(str(skill_dir.resolve())):
            logger.warning("Skill reference path escape attempt: %s", ref_path)
            return None
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")[:_MAX_SKILL_BODY_SIZE]
        except OSError:
            return None

    # -- Enable / Disable ----------------------------------------------------

    def enable_skill(self, name: str) -> bool:
        meta = self._index.get(name)
        if meta is None:
            return False
        meta.enabled = True
        self._config_overrides.setdefault(name, {})["enabled"] = True
        self._save_config()
        return True

    def disable_skill(self, name: str) -> bool:
        meta = self._index.get(name)
        if meta is None:
            return False
        meta.enabled = False
        self._config_overrides.setdefault(name, {})["enabled"] = False
        self._save_config()
        return True
