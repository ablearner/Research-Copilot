from __future__ import annotations

import logging

from skills.base import SkillSpec, build_default_skill

logger = logging.getLogger(__name__)


class SkillRegistryError(RuntimeError):
    """Raised when skill registration or selection fails."""


class SkillRegistry:
    def __init__(self, default_skill: SkillSpec | None = None) -> None:
        self._skills: dict[str, SkillSpec] = {}
        seed_skill = default_skill or build_default_skill()
        self._default_skill_name = seed_skill.name
        self.register(seed_skill, replace=True)

    def register(self, skill: SkillSpec, replace: bool = False) -> SkillSpec:
        if skill.name in self._skills and not replace:
            raise SkillRegistryError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill
        logger.info(
            "Skill registered",
            extra={
                "skill_name": skill.name,
                "enabled": skill.enabled,
                "applicable_tasks": skill.applicable_tasks,
            },
        )
        return skill

    def register_many(self, skills: list[SkillSpec], replace: bool = False) -> None:
        for skill in skills:
            self.register(skill, replace=replace)

    def get_skill(self, name: str, include_disabled: bool = False) -> SkillSpec | None:
        skill = self._skills.get(name)
        if skill is None:
            return None
        if not include_disabled and not skill.enabled:
            return None
        return skill

    def list_skills(self, include_disabled: bool = True) -> list[SkillSpec]:
        skills = list(self._skills.values())
        if include_disabled:
            return skills
        return [skill for skill in skills if skill.enabled]

    def set_default_skill(self, name: str) -> None:
        skill = self.get_skill(name, include_disabled=True)
        if skill is None:
            raise SkillRegistryError(f"Default skill not found: {name}")
        self._default_skill_name = name

    def default_skill(self) -> SkillSpec:
        default_skill = self.get_skill(self._default_skill_name, include_disabled=True)
        if default_skill is None:
            raise SkillRegistryError(f"Default skill not found: {self._default_skill_name}")
        return default_skill

    def select_skill_for_task(
        self,
        task_type: str,
        preferred_skill_name: str | None = None,
    ) -> SkillSpec:
        if preferred_skill_name:
            preferred = self.get_skill(preferred_skill_name)
            if preferred and self._supports_task(preferred, task_type):
                return preferred

        default_skill = self.default_skill()
        if default_skill.enabled and self._supports_task(default_skill, task_type):
            return default_skill

        for skill in self.list_skills(include_disabled=False):
            if self._supports_task(skill, task_type):
                return skill

        return default_skill

    def allowed_external_mcp_tools(
        self,
        task_type: str,
        preferred_skill_name: str | None = None,
    ) -> set[str]:
        skill = self.select_skill_for_task(task_type=task_type, preferred_skill_name=preferred_skill_name)
        metadata = skill.metadata if isinstance(skill.metadata, dict) else {}
        values = metadata.get("allowed_external_mcp_tools")
        if isinstance(values, list):
            return {name for name in values if isinstance(name, str)}
        return set()

    def _supports_task(self, skill: SkillSpec, task_type: str) -> bool:
        return task_type in skill.applicable_tasks or "*" in skill.applicable_tasks
