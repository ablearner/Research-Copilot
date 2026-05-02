from tooling.toolsets import AGENT_TOOLSET_MAP, resolve_agent_toolsets, resolve_toolset


def test_resolve_research_core():
    tools = resolve_toolset("research-core")
    assert "hybrid_retrieve" in tools
    assert "answer_with_evidence" in tools


def test_resolve_rag():
    tools = resolve_toolset("rag")
    assert "hybrid_retrieve" in tools
    assert "parse_document" in tools
    assert "ask_document" in tools


def test_resolve_supervisor_action():
    tools = resolve_toolset("supervisor_action")
    assert "search_literature" in tools
    assert "answer_question" in tools
    assert "supervisor_understand_chart" in tools


def test_resolve_kepler_cli_includes_all():
    tools = resolve_toolset("kepler-cli")
    assert "hybrid_retrieve" in tools
    assert "parse_document" in tools
    assert "arxiv_search" in tools
    assert "update_user_profile" in tools
    assert "search_papers" in tools
    assert "academic_search" in tools


def test_resolve_kepler_mcp_read_only():
    tools = resolve_toolset("kepler-mcp")
    assert "hybrid_retrieve" in tools
    assert "parse_document" not in tools
    assert "update_user_profile" not in tools


def test_resolve_unknown_returns_empty():
    assert resolve_toolset("nonexistent") == []


def test_no_duplicates():
    tools = resolve_toolset("kepler-cli")
    assert len(tools) == len(set(tools))


def test_agent_toolset_map_supervisor():
    toolsets = resolve_agent_toolsets("ResearchSupervisorAgent")
    assert toolsets == ["supervisor_action", "research-capability"]


def test_agent_toolset_map_unknown():
    assert resolve_agent_toolsets("UnknownAgent") == []


def test_agent_toolset_map_has_all_expected_agents():
    assert "ResearchQAAgent" in AGENT_TOOLSET_MAP
    assert "LiteratureScoutAgent" in AGENT_TOOLSET_MAP
