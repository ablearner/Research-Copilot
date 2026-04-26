from __future__ import annotations

from apps.cli import (
    _figure_from_response,
    _format_compact_sources,
    _profile_summary_lines,
    _slash_command_entries,
    build_parser,
)
from domain.schemas.research import (
    ResearchAgentRunResponse,
    ResearchConversation,
    ResearchConversationResponse,
    ResearchConversationSnapshot,
    ResearchWorkspaceState,
)
from domain.schemas.research_memory import InterestTopic, UserResearchProfile


def test_cli_parser_supports_agent_and_management_commands() -> None:
    parser = build_parser()

    agent_args = parser.parse_args(["agent", "--topic", "graph rag"])
    assert agent_args.command == "agent"
    assert agent_args.topic == "graph rag"

    models_args = parser.parse_args(["models", "set", "--llm-provider", "openai", "--llm-model", "gpt-5.4-mini"])
    assert models_args.command == "models"
    assert models_args.models_command == "set"
    assert models_args.llm_provider == "openai"

    skills_args = parser.parse_args(["skills", "disable", "research_report"])
    assert skills_args.command == "skills"
    assert skills_args.skills_command == "disable"

    plugins_args = parser.parse_args(["plugins", "enable", "zotero_local_mcp"])
    assert plugins_args.command == "plugins"
    assert plugins_args.plugins_command == "enable"


def test_slash_command_entries_include_profile_aliases_and_clear() -> None:
    entries = dict(_slash_command_entries())

    assert "/clear" in entries
    assert "/profile" in entries
    assert "/profile clear" in entries
    assert "/profile remove <topic>" in entries
    assert "/preferences" in entries


def test_profile_summary_lines_show_interest_topics_and_sources() -> None:
    profile = UserResearchProfile(
        user_id="local-user",
        last_active_topic="GraphRAG",
        preferred_sources=["arxiv", "openalex"],
        preferred_keywords=["graph rag", "agent memory"],
        preferred_reasoning_style="concise",
        preferred_answer_language="zh-CN",
        interest_topics=[
            InterestTopic(
                topic_name="GraphRAG",
                normalized_topic="graphrag",
                weight=3.4,
                mention_count=5,
                recent_mention_count=3,
                preferred_sources=["arxiv"],
                preferred_keywords=["graph rag", "retrieval"],
                preferred_recency_days=30,
            )
        ],
    )

    lines = _profile_summary_lines(profile)
    rendered = "\n".join(lines)

    assert "last_active_topic: GraphRAG" in rendered
    assert "preferred_sources: arxiv, openalex" in rendered
    assert "1. GraphRAG | weight=3.40 | mentions=5 | recent=3" in rendered
    assert "sources=arxiv" in rendered
    assert "keywords=graph rag, retrieval" in rendered
    assert "reasoning_style: concise" in rendered
    assert "answer_language: zh-CN" in rendered
    assert "/profile remove <topic> | /profile clear" in rendered


def test_profile_summary_lines_show_empty_state_hint() -> None:
    lines = _profile_summary_lines(UserResearchProfile())
    rendered = "\n".join(lines)

    assert "还没有学到稳定偏好" in rendered
    assert "/profile 或 /preferences" in rendered


def test_format_compact_sources_limits_long_source_list() -> None:
    assert _format_compact_sources(["arxiv", "openalex"]) == "arxiv,openalex"
    assert _format_compact_sources(["arxiv", "openalex", "semantic_scholar", "ieee"]) == "arxiv,openalex,semantic_scholar+1"


def test_figure_from_response_ignores_cached_workspace_figure_for_non_chart_turn() -> None:
    figure = {"figure_id": "paper-1:fig-2", "image_path": "/tmp/paper-1-fig-2.png"}
    conversation = ResearchConversationResponse(
        conversation=ResearchConversation(
            conversation_id="conv-1",
            title="GraphRAG",
            created_at="2026-04-25T00:00:00+00:00",
            updated_at="2026-04-25T00:00:00+00:00",
            snapshot=ResearchConversationSnapshot(
                workspace=ResearchWorkspaceState(metadata={"last_visual_anchor_figure": figure})
            ),
        ),
        messages=[],
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        metadata={"route_mode": "research_follow_up"},
    )

    assert _figure_from_response(response, conversation) is None


def test_figure_from_response_uses_cached_workspace_figure_for_chart_turn() -> None:
    figure = {"figure_id": "paper-1:fig-2", "image_path": "/tmp/paper-1-fig-2.png"}
    conversation = ResearchConversationResponse(
        conversation=ResearchConversation(
            conversation_id="conv-1",
            title="GraphRAG",
            created_at="2026-04-25T00:00:00+00:00",
            updated_at="2026-04-25T00:00:00+00:00",
            snapshot=ResearchConversationSnapshot(
                workspace=ResearchWorkspaceState(metadata={"last_visual_anchor_figure": figure})
            ),
        ),
        messages=[],
    )
    response = ResearchAgentRunResponse(
        status="succeeded",
        metadata={"route_mode": "chart_drilldown"},
    )

    assert _figure_from_response(response, conversation) == figure
