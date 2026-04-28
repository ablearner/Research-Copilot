"""Tests for reasoning.style normalization."""

from reasoning.style import normalize_reasoning_style, DEFAULT_AGENT_REASONING_STYLE


class TestNormalizeReasoningStyle:
    def test_default_none(self):
        assert normalize_reasoning_style(None) == DEFAULT_AGENT_REASONING_STYLE

    def test_default_empty(self):
        assert normalize_reasoning_style("") == DEFAULT_AGENT_REASONING_STYLE

    def test_cot_alias(self):
        assert normalize_reasoning_style("chain_of_thought") == "cot"
        assert normalize_reasoning_style("chainofthought") == "cot"
        assert normalize_reasoning_style("cot") == "cot"

    def test_plan_and_solve(self):
        assert normalize_reasoning_style("planandsolve") == "plan_and_solve"
        assert normalize_reasoning_style("plan_and_solve") == "plan_and_solve"

    def test_react(self):
        assert normalize_reasoning_style("react") == "react"

    def test_auto(self):
        assert normalize_reasoning_style("auto") == "auto"

    def test_unknown_passthrough(self):
        assert normalize_reasoning_style("custom_style") == "custom_style"

    def test_whitespace_and_case(self):
        assert normalize_reasoning_style("  CoT  ") == "cot"
        assert normalize_reasoning_style("Chain-of-Thought") == "cot"
