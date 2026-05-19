from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.skill_matcher import SkillMatcher, build_skill_context
from core.skill_registry import SkillMeta
from core.skill_registry import SkillRegistry
from core.skill_validator import SkillValidator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchSkillCandidate:
    name: str
    description: str = ""
    planner_guidance: str = ""
    planning_policy: dict[str, object] = field(default_factory=dict)
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    score: float = 0.0
    match_reason: str = ""
    missing_tools: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "planner_guidance": self.planner_guidance,
            "planning_policy": dict(self.planning_policy),
            "category": self.category,
            "tags": list(self.tags),
            "required_tools": list(self.required_tools),
            "score": self.score,
            "match_reason": self.match_reason,
            "missing_tools": list(self.missing_tools),
        }


@dataclass(slots=True)
class ResearchSkillSelection:
    active_skill_names: list[str] = field(default_factory=list)
    skill_context: str = ""
    candidate_skills: list[ResearchSkillCandidate] = field(default_factory=list)
    match_reasons: dict[str, str] = field(default_factory=dict)
    required_tools: list[str] = field(default_factory=list)
    missing_tools: dict[str, list[str]] = field(default_factory=dict)
    validation_warnings: list[dict[str, str]] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "active_skill_names": list(self.active_skill_names),
            "candidate_skills": [candidate.metadata() for candidate in self.candidate_skills],
            "match_reasons": dict(self.match_reasons),
            "required_tools": list(self.required_tools),
            "missing_tools": {
                name: list(tools)
                for name, tools in self.missing_tools.items()
            },
            "validation_warnings": list(self.validation_warnings),
        }


class ResearchSkillResolver:
    """Resolve Markdown skills without letting them become tools or agents.

    The runtime uses this in two phases:

    1. Resolve lightweight candidate metadata for the Supervisor.
    2. Load full ``SKILL.md`` instructions only after the Supervisor selects
       skill names for a specialist task.
    """

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

    async def resolve_candidates(
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

        candidates: list[ResearchSkillCandidate] = []
        match_reasons: dict[str, str] = {}
        explicit_name = (explicit_skill_name or "").strip()
        if explicit_name:
            meta = self.registry.get_meta(explicit_name)
            if meta is not None:
                candidate = self._candidate_from_meta(
                    meta,
                    available_tools=available_tools,
                    score=1.0,
                    match_reason="explicit_request",
                )
                candidates.append(candidate)
                match_reasons[explicit_name] = "explicit_request"

        for match in matches:
            if match.meta.name not in {candidate.name for candidate in candidates}:
                candidates.append(
                    self._candidate_from_meta(
                        match.meta,
                        available_tools=available_tools,
                        score=match.score,
                        match_reason=match.match_reason,
                    )
                )
            match_reasons.setdefault(match.meta.name, match.match_reason)

        required_tools: list[str] = []
        missing_tools: dict[str, list[str]] = {}
        for candidate in candidates:
            for tool_name in candidate.required_tools:
                if tool_name not in required_tools:
                    required_tools.append(tool_name)
            if candidate.missing_tools:
                missing_tools[candidate.name] = list(candidate.missing_tools)

        return ResearchSkillSelection(
            candidate_skills=candidates,
            match_reasons=match_reasons,
            required_tools=required_tools,
            missing_tools=missing_tools,
        )

    def load_selected_skills(
        self,
        *,
        selected_skill_names: list[str],
        candidate_skill_names: list[str] | None = None,
        available_tool_names: list[str] | None = None,
    ) -> ResearchSkillSelection:
        has_candidate_filter = candidate_skill_names is not None
        allowed = {
            str(name).strip()
            for name in (candidate_skill_names or [])
            if str(name).strip()
        }
        available_tools = {
            str(name).strip()
            for name in (available_tool_names or [])
            if str(name).strip()
        }
        skill_names: list[str] = []
        for raw_name in selected_skill_names:
            name = str(raw_name).strip()
            if not name or name in skill_names:
                continue
            if has_candidate_filter and name not in allowed:
                logger.warning("Skipping supervisor-selected skill outside candidates: %s", name)
                continue
            skill_names.append(name)

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
            missing = [
                tool_name
                for tool_name in skill.meta.requires_tools
                if available_tools and tool_name not in available_tools
            ]
            if missing:
                missing_tools[skill.meta.name] = missing
            loaded_skills.append(skill)

        if not loaded_skills:
            return ResearchSkillSelection()

        return ResearchSkillSelection(
            active_skill_names=[skill.meta.name for skill in loaded_skills],
            skill_context=build_skill_context(loaded_skills),
            match_reasons={skill.meta.name: "supervisor_selected" for skill in loaded_skills},
            required_tools=required_tools,
            missing_tools=missing_tools,
            validation_warnings=validation_warnings,
        )

    async def resolve(
        self,
        *,
        message: str,
        explicit_skill_name: str | None = None,
        available_tool_names: list[str] | None = None,
    ) -> ResearchSkillSelection:
        """Backward-compatible helper that loads every matched candidate.

        New production paths should call ``resolve_candidates`` first and
        ``load_selected_skills`` only after the Supervisor has selected skills.
        """

        candidates = await self.resolve_candidates(
            message=message,
            explicit_skill_name=explicit_skill_name,
            available_tool_names=available_tool_names,
        )
        loaded = self.load_selected_skills(
            selected_skill_names=[candidate.name for candidate in candidates.candidate_skills],
            candidate_skill_names=[candidate.name for candidate in candidates.candidate_skills],
            available_tool_names=available_tool_names,
        )
        loaded.candidate_skills = candidates.candidate_skills
        loaded.match_reasons = candidates.match_reasons
        return loaded

    @staticmethod
    def _candidate_from_meta(
        meta: SkillMeta,
        *,
        available_tools: set[str],
        score: float,
        match_reason: str,
    ) -> ResearchSkillCandidate:
        missing_tools = [
            tool_name
            for tool_name in meta.requires_tools
            if available_tools and tool_name not in available_tools
        ]
        return ResearchSkillCandidate(
            name=meta.name,
            description=meta.description,
            planner_guidance=meta.planner_guidance,
            planning_policy=dict(meta.planning_policy),
            category=meta.category,
            tags=list(meta.tags),
            required_tools=list(meta.requires_tools),
            score=float(score),
            match_reason=match_reason,
            missing_tools=missing_tools,
        )
