from __future__ import annotations

import pytest

from core.skill_registry import SkillMeta
from tools.research.skill_resolver import ResearchSkillResolver


def test_research_skill_resolver_loads_only_supervisor_candidates() -> None:
    resolver = ResearchSkillResolver()

    selection = resolver.load_selected_skills(
        selected_skill_names=["paper-comparison", "research-qa"],
        candidate_skill_names=["paper-comparison"],
    )

    assert selection.active_skill_names == ["paper-comparison"]
    assert "Paper Comparison Skill" in selection.skill_context
    assert "research-qa" not in selection.active_skill_names


def test_research_skill_candidate_exposes_planner_guidance() -> None:
    candidate = ResearchSkillResolver._candidate_from_meta(
        SkillMeta(
            name="literature-survey",
            description="Write surveys.",
            planner_guidance="Search and write before optional import.",
            planning_policy={"action_policies": {"import_papers": {"default_enabled": False}}},
        ),
        available_tools=set(),
        score=1.0,
        match_reason="trigger:survey",
    )

    assert candidate.metadata()["planner_guidance"] == "Search and write before optional import."
    assert candidate.metadata()["planning_policy"] == {"action_policies": {"import_papers": {"default_enabled": False}}}


@pytest.mark.asyncio
async def test_literature_survey_candidate_carries_planner_guidance() -> None:
    selection = await ResearchSkillResolver().resolve_candidates(
        message="请做 On-Policy Distillation 文献调研报告",
    )
    candidate = next(
        candidate
        for candidate in selection.candidate_skills
        if candidate.name == "literature-survey"
    )

    assert "metadata/abstract-based review" in candidate.planner_guidance
    assert candidate.planning_policy["action_policies"]["import_papers"]["default_enabled"] is False
    assert candidate.metadata()["planner_guidance"] == candidate.planner_guidance
