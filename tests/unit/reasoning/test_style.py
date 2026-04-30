"""Tests for reasoning style normalization (canonical: agents.research_qa_agent)."""

from agents.research_qa_agent import normalize_reasoning_style, DEFAULT_AGENT_REASONING_STYLE


class TestNormalizeReasoningStyle:
    def test_default_none(self):
        assert normalize_reasoning_style(None) == DEFAULT_AGENT_REASONING_STYLE

    def test_default_empty(self):
        assert normalize_reasoning_style("") == DEFAULT_AGENT_REASONING_STYLE

    def test_cot_alias_maps_to_react(self):
        assert normalize_reasoning_style("chain_of_thought") == "react"
        assert normalize_reasoning_style("chainofthought") == "react"
        assert normalize_reasoning_style("cot") == "react"

    def test_plan_and_execute(self):
        assert normalize_reasoning_style("planandsolve") == "plan_and_execute"
        assert normalize_reasoning_style("plan_and_solve") == "plan_and_execute"
        assert normalize_reasoning_style("plan_and_execute") == "plan_and_execute"

    def test_react(self):
        assert normalize_reasoning_style("react") == "react"

    def test_auto(self):
        assert normalize_reasoning_style("auto") == "auto"

    def test_unknown_passthrough(self):
        assert normalize_reasoning_style("custom_style") == "custom_style"

    def test_whitespace_and_case(self):
        assert normalize_reasoning_style("  CoT  ") == "react"
        assert normalize_reasoning_style("Chain-of-Thought") == "react"
