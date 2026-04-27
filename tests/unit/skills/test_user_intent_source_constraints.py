"""Tests for source constraint extraction in user intent resolution."""
import pytest

from skills.research.user_intent import (
    ResearchUserIntentResolverSkill,
    _extract_source_constraints,
)


class TestExtractSourceConstraints:
    @pytest.mark.parametrize(
        "message, expected_constraints, topic_must_contain, topic_must_not_contain",
        [
            (
                "帮我在arxiv上找LLM agent相关的论文",
                ["arxiv"],
                "LLM agent",
                "arxiv",
            ),
            (
                "find papers on arxiv about transformer architecture",
                ["arxiv"],
                "transformer architecture",
                "arxiv",
            ),
            (
                "从semantic scholar搜索reinforcement learning",
                ["semantic_scholar"],
                "reinforcement learning",
                "semantic scholar",
            ),
            (
                "search on ieee for UAV path planning",
                ["ieee"],
                "UAV path planning",
                "ieee",
            ),
            (
                "search for LLM papers from arxiv and ieee",
                ["arxiv", "ieee"],
                "LLM papers",
                "arxiv",
            ),
            (
                "在google scholar上找deep learning的论文",
                ["google_scholar"],
                "deep learning",
                "google scholar",
            ),
            (
                "LLM agent papers",
                [],
                "LLM agent papers",
                None,
            ),
        ],
    )
    def test_extraction(
        self,
        message: str,
        expected_constraints: list[str],
        topic_must_contain: str,
        topic_must_not_contain: str | None,
    ) -> None:
        constraints, cleaned = _extract_source_constraints(message)
        assert sorted(constraints) == sorted(expected_constraints), (
            f"constraints mismatch for {message!r}: {constraints}"
        )
        assert topic_must_contain in cleaned, (
            f"cleaned topic {cleaned!r} should contain {topic_must_contain!r}"
        )
        if topic_must_not_contain:
            assert topic_must_not_contain.lower() not in cleaned.lower(), (
                f"cleaned topic {cleaned!r} should NOT contain {topic_must_not_contain!r}"
            )


class TestIntentResolverSourceConstraints:
    def test_literature_search_intent_includes_source_constraints(self) -> None:
        resolver = ResearchUserIntentResolverSkill(llm_adapter=None)
        result = resolver.resolve(
            message="帮我在arxiv上找LLM agent相关的论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.intent == "literature_search"
        assert result.source_constraints == ["arxiv"]
        assert "arxiv" not in result.extracted_topic.lower()
        assert "LLM agent" in result.extracted_topic

    def test_no_source_constraint_when_no_source_mentioned(self) -> None:
        resolver = ResearchUserIntentResolverSkill(llm_adapter=None)
        result = resolver.resolve(
            message="帮我找一下最新的VLN论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.intent == "literature_search"
        assert result.source_constraints == []
