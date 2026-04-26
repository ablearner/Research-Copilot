from tooling.registry import ToolRegistry
from tooling.research_function_registry import ResearchFunctionRegistry
from tooling.research_function_specs import RESEARCH_FUNCTION_DEFINITIONS, list_research_function_schemas


async def _dummy_handler(**kwargs):
    return {"research_topic": kwargs.get("topic", ""), "research_goals": [], "selected_papers": [], "imported_papers": [], "known_conclusions": [], "open_questions": [], "session_history": [], "user_preferences": {"review_style": "academic", "preferred_sources": [], "answer_language": "zh-CN", "citation_style": "inline_brackets", "max_selected_papers": 8, "max_history_turns": 10, "metadata": {}}, "paper_summaries": [], "metadata": {}, "updated_at": "2026-04-20T00:00:00+00:00"}


def test_research_function_definitions_cover_all_core_functions() -> None:
    expected = {
        "search_papers",
        "extract_paper_structure",
        "compare_papers",
        "generate_review",
        "ask_paper",
        "recommend_papers",
        "update_research_context",
        "decompose_task",
        "evaluate_result",
        "execute_research_plan",
    }

    assert expected == set(RESEARCH_FUNCTION_DEFINITIONS)

    schemas = list_research_function_schemas()
    assert len(schemas) == 10
    assert all(schema["input_schema"]["type"] == "object" for schema in schemas)


def test_research_function_registry_registers_tool_specs() -> None:
    registry = ResearchFunctionRegistry(ToolRegistry())

    registry.register("update_research_context", _dummy_handler)

    spec = registry.get_registry().get_tool("update_research_context")
    assert spec is not None
    assert spec.max_retries == 3
    assert "function_call" in spec.tags
