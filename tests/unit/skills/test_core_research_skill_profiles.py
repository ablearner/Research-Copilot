from skills.research import build_core_research_skill_profiles


def test_build_core_research_skill_profiles_returns_expected_profiles() -> None:
    profiles = build_core_research_skill_profiles()
    names = {profile.name for profile in profiles}

    assert {
        "literature_review",
        "paper_compare",
        "chart_interpretation",
        "research_gap_discovery",
        "paper_deep_analysis",
    }.issubset(names)
    assert all(profile.metadata.get("research_skill") for profile in profiles)
