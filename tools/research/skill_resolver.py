from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.skill_matcher import SkillMatcher, build_skill_context
from core.skill_registry import SkillRegistry
from core.skill_validator import SkillValidator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchSkillSelection:
    active_skill_names: list[str] = field(default_factory=list)
    skill_context: str = ""
    match_reasons: dict[str, str] = field(default_factory=dict)
    required_tools: list[str] = field(default_factory=list)
    missing_tools: dict[str, list[str]] = field(default_factory=dict)
    validation_warnings: list[dict[str, str]] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "active_skill_names": list(self.active_skill_names),
            "match_reasons": dict(self.match_reasons),
            "required_tools": list(self.required_tools),
            "missing_tools": {
                name: list(tools)
                for name, tools in self.missing_tools.items()
            },
            "validation_warnings": list(self.validation_warnings),
        }


class ResearchSkillResolver:
    """Resolve task skills without letting skills become tools or agents."""

    def __init__(
        self,
        *,
        registry: SkillRegistry | None = None,
        matcher: SkillMatcher | None = None,
        validator: SkillValidator | None = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        self.matcher = matcher or SkillMatcher(self.registry)
        self.validator = validator or SkillValidator()

    async def resolve(
        self,
        *,
        message: str,
        explicit_skill_name: str | None = None,
        available_tool_names: list[str] | None = None,
    ) -> ResearchSkillSelection:
        available_tools = {
            str(name).strip()
            for name in (available_tool_names or [])
            if str(name).strip()
        }
        tool_list = sorted(available_tools) if available_tools else None
        try:
            matches = await self.matcher.amatch(
                message,
                available_tool_names=tool_list,
            )
        except Exception:
            logger.warning("Async skill matching failed, falling back to sync", exc_info=True)
            matches = self.matcher.match(
                message,
                available_tool_names=tool_list,
            )

        skill_names: list[str] = []
        match_reasons: dict[str, str] = {}
        explicit_name = (explicit_skill_name or "").strip()
        if explicit_name:
            skill_names.append(explicit_name)
            match_reasons[explicit_name] = "explicit_request"

        for match in matches:
            if match.meta.name not in skill_names:
                skill_names.append(match.meta.name)
            match_reasons.setdefault(match.meta.name, match.match_reason)

        loaded_skills = []
        required_tools: list[str] = []
        missing_tools: dict[str, list[str]] = {}
        validation_warnings: list[dict[str, str]] = []

        for skill_name in skill_names:
            skill = self.registry.load_skill(skill_name)
            if skill is None:
                continue
            validation = self.validator.validate(skill)
            for issue in validation.issues:
                payload = {
                    "skill": skill.meta.name,
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                }
                if issue.severity == "warning":
                    validation_warnings.append(payload)
            if not validation.passed:
                logger.warning("Skipping invalid skill: %s", skill.meta.name)
                continue

            for tool_name in skill.meta.requires_tools:
                if tool_name not in required_tools:
                    required_tools.append(tool_name)
            if available_tools:
                missing = [
                    tool_name
                    for tool_name in skill.meta.requires_tools
                    if tool_name not in available_tools
                ]
                if missing:
                    missing_tools[skill.meta.name] = missing
            loaded_skills.append(skill)

        if not loaded_skills:
            return ResearchSkillSelection()

        return ResearchSkillSelection(
            active_skill_names=[skill.meta.name for skill in loaded_skills],
            skill_context=build_skill_context(loaded_skills),
            match_reasons=match_reasons,
            required_tools=required_tools,
            missing_tools=missing_tools,
            validation_warnings=validation_warnings,
        )
