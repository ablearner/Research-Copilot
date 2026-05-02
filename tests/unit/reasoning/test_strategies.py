"""Tests for reasoning.strategies dataclass and protocols."""

from rag_runtime.strategies import ReasoningStrategySet


class TestReasoningStrategySet:
    def test_empty_set(self):
        s = ReasoningStrategySet()
        assert s.query_planning is None
        assert s.answer_synthesis is None
        assert s.tool_reasoning is None
        assert s.llm_adapter is None

    def test_llm_adapter_from_query_planning(self):
        class FakePlanner:
            llm_adapter = "adapter_a"
            async def plan_queries(self, **kwargs):
                return []

        s = ReasoningStrategySet(query_planning=FakePlanner())
        assert s.llm_adapter == "adapter_a"

    def test_llm_adapter_from_answer_synthesis(self):
        class FakeCoT:
            llm_adapter = "adapter_b"
            async def reason(self, **kwargs):
                return None

        s = ReasoningStrategySet(answer_synthesis=FakeCoT())
        assert s.llm_adapter == "adapter_b"

    def test_property_aliases(self):
        class FakePlanner:
            llm_adapter = None
            async def plan_queries(self, **kwargs):
                return []

        planner = FakePlanner()
        s = ReasoningStrategySet(query_planning=planner)
        assert s.plan_and_solve_reasoning_agent is planner
        assert s.cot_reasoning_agent is None
        assert s.react_reasoning_agent is None
