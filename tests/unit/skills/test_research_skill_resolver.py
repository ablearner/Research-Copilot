from __future__ import annotations

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
