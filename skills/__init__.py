"""Skill system for task-aware runtime behavior.

This package defines:

- declarative `SkillSpec` objects
- loading/registration for YAML skill specs
- research-specific skill helpers under `skills.research`

In the current architecture, `skills/` is still an active extension layer.
It shapes retrieval policy, prompt selection, memory policy, and output style.
"""

from skills.base import (
    SkillContext,
    SkillMemoryPolicy,
    SkillOutputStyle,
    SkillPromptSet,
    SkillRetrievalPolicy,
    SkillSpec,
    build_default_skill,
)
from skills.loader import SkillLoader, SkillLoaderError
from skills.registry import SkillRegistry, SkillRegistryError

__all__ = [
    "SkillContext",
    "SkillLoader",
    "SkillLoaderError",
    "SkillMemoryPolicy",
    "SkillOutputStyle",
    "SkillPromptSet",
    "SkillRegistry",
    "SkillRegistryError",
    "SkillRetrievalPolicy",
    "SkillSpec",
    "build_default_skill",
]
