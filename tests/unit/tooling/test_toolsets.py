from tooling.toolsets import resolve_toolset


def test_resolve_research_core():
    tools = resolve_toolset("research-core")
    assert "hybrid_retrieve" in tools
    assert "answer_with_evidence" in tools


def test_resolve_kepler_cli_includes_all():
    tools = resolve_toolset("kepler-cli")
    assert "hybrid_retrieve" in tools
    assert "parse_document" in tools
    assert "arxiv_search" in tools
    assert "update_user_profile" in tools


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
