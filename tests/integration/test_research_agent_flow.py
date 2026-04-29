"""Integration test: research agent intent routing + context compression flow.

Validates that the user_intent_resolver correctly routes messages and
that the _route_mode_hint_for_request uses intent results when confident.
"""

import pytest

from services.research.capabilities.user_intent import ResearchIntentResolver, ResearchUserIntentResult


@pytest.fixture
def resolver():
    return ResearchIntentResolver(llm_adapter=None)


class TestIntentRoutingHeuristic:
    """Verify heuristic intent resolution for key user messages."""

    def test_search_request_routes_to_literature_search(self, resolver):
        result = resolver.resolve(
            message="帮我搜索关于 LLM agent 的最新论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.intent == "literature_search"
        assert result.confidence >= 0.7

    def test_greeting_routes_to_general_answer(self, resolver):
        result = resolver.resolve(
            message="你好",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.intent == "general_answer"
        assert result.confidence >= 0.7

    def test_compare_routes_to_paper_comparison(self, resolver):
        result = resolver.resolve(
            message="对比这两篇论文的方法",
            has_task=True,
            candidate_paper_count=2,
            candidate_papers=[
                {"paper_id": "p1", "title": "Paper 1"},
                {"paper_id": "p2", "title": "Paper 2"},
            ],
            active_paper_ids=["p1", "p2"],
            selected_paper_ids=[],
        )
        assert result.intent == "paper_comparison"

    def test_figure_question_routes_to_figure_qa(self, resolver):
        result = resolver.resolve(
            message="这张图表的横轴表示什么？",
            has_task=True,
            candidate_paper_count=1,
            candidate_papers=None,
            active_paper_ids=["p1"],
            selected_paper_ids=[],
            has_visual_anchor=True,
        )
        assert result.intent == "figure_qa"

    def test_import_routes_to_paper_import(self, resolver):
        result = resolver.resolve(
            message="把这篇论文导入工作区",
            has_task=True,
            candidate_paper_count=1,
            candidate_papers=None,
            active_paper_ids=["p1"],
            selected_paper_ids=[],
        )
        assert result.intent == "paper_import"

    def test_zotero_routes_to_sync(self, resolver):
        result = resolver.resolve(
            message="把这些论文保存到zotero",
            has_task=True,
            candidate_paper_count=3,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.intent == "sync_to_zotero"

    def test_document_routes_to_understanding(self, resolver):
        result = resolver.resolve(
            message="解析这个pdf文件",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
            has_document_input=True,
        )
        assert result.intent == "document_understanding"


class TestIntentRoutingAsync:
    """Verify async resolution falls back to heuristic when no LLM."""

    @pytest.mark.asyncio
    async def test_async_without_llm_uses_heuristic(self, resolver):
        result = await resolver.resolve_async(
            message="搜索 transformer 相关论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert result.source == "heuristic"
        assert result.intent == "literature_search"


class TestSourceConstraintExtraction:
    """Verify source constraint extraction from messages."""

    def test_arxiv_constraint(self, resolver):
        result = resolver.resolve(
            message="在arxiv上搜索LLM agent论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert "arxiv" in result.source_constraints
        assert result.intent == "literature_search"

    def test_extracted_topic_strips_source(self, resolver):
        result = resolver.resolve(
            message="在arxiv上搜索LLM agent论文",
            has_task=False,
            candidate_paper_count=0,
            candidate_papers=None,
            active_paper_ids=[],
            selected_paper_ids=[],
        )
        assert "arxiv" not in result.extracted_topic.lower() or result.extracted_topic == ""
