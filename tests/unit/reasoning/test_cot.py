"""Tests for reasoning.cot.CoTReasoningAgent."""

import pytest
from chains.cot import CoTReasoningAgent
from domain.schemas.retrieval import EvidenceBundle


class TestCoTReasoningAgentInit:
    def test_no_adapter(self):
        agent = CoTReasoningAgent(llm_adapter=None)
        assert agent.llm_adapter is None

    def test_with_adapter(self):
        agent = CoTReasoningAgent(llm_adapter="fake")
        assert agent.llm_adapter == "fake"


class TestCoTReasonEmptyEvidence:
    @pytest.mark.asyncio
    async def test_empty_evidence_returns_insufficient(self):
        agent = CoTReasoningAgent(llm_adapter=None)
        bundle = EvidenceBundle(evidences=[])
        result = await agent.reason(question="What is attention?", evidence_bundle=bundle)
        assert result.confidence == 0.0
        assert "证据不足" in result.answer
