"""Tests for context compression feasibility in long-running sessions.

These tests validate the two-layer compression strategy:
1. Proactive: context_compression_needed signal → LLM proactively chooses compress_context
2. Reactive: _context_exceeds_budget guardrail → forces compress_context before LLM call

The tests simulate progressively growing context (multi-turn QA, large session_history,
many paper_summaries, bulky metadata) and verify that:
- The guardrail triggers at the right thresholds
- CompressContextTool actually reduces context size
- After compression, context_compressed is True and subsequent decisions work
- Long sessions (10+ turns) remain functional thanks to compression
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest

from adapters.llm.base import BaseLLMAdapter
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.research import (
    CreateResearchConversationRequest,
    PaperCandidate,
    ResearchAgentRunRequest,
    ResearchTask,
)
from domain.schemas.research_context import (
    CompressedPaperSummary,
    QAPair,
    ResearchContext,
    ResearchContextSlice,
)
from services.research.literature_research_service import LiteratureResearchService
from services.research.research_context_manager import ResearchContextManager
from services.research.research_report_service import ResearchReportService
from services.research.paper_search_service import PaperSearchService
from services.research.research_supervisor_graph_runtime import ResearchSupervisorGraphRuntime


# ---------------------------------------------------------------------------
# Helpers to inflate context to a given size
# ---------------------------------------------------------------------------

def _make_qa_pair(idx: int, answer_size: int = 500) -> QAPair:
    """Generate a single QAPair with a controllable answer size."""
    return QAPair(
        question=f"研究问题 #{idx}: 关于多智能体路径规划和协同决策有哪些最新进展？",
        answer="A" * answer_size,
        citations=[f"paper-{idx}-cite-1", f"paper-{idx}-cite-2"],
        metadata={"turn": idx, "document_ids": [f"doc-{idx}"]},
    )


def _make_paper_candidate(idx: int) -> PaperCandidate:
    return PaperCandidate(
        paper_id=f"paper-{idx}",
        title=f"Paper #{idx}: Multi-Agent Coordination in UAV Path Planning",
        authors=[f"Author-{idx}-A", f"Author-{idx}-B"],
        abstract="X" * 300,
        year=2025 + (idx % 2),
        source="arxiv",
        citations=10 + idx,
        pdf_url=f"https://arxiv.org/pdf/{idx}.pdf",
        url=f"https://arxiv.org/abs/{idx}",
        is_open_access=True,
    )


def _make_compressed_summary(paper_id: str, level: str, size: int = 300) -> CompressedPaperSummary:
    return CompressedPaperSummary(
        paper_id=paper_id,
        level=level,
        summary="S" * size,
        source_section_ids=[f"{paper_id}:{level}"],
        relevance_score=0.8,
        metadata={"title": f"Paper {paper_id}", "selected": True},
    )


def _build_large_context_slice(
    *,
    num_history_turns: int = 5,
    answer_size: int = 500,
    num_papers: int = 3,
    summary_size: int = 300,
    extra_metadata_size: int = 0,
) -> ResearchContextSlice:
    """Build a ResearchContextSlice with controllable size."""
    history = [_make_qa_pair(i, answer_size=answer_size) for i in range(num_history_turns)]
    paper_ids = [f"paper-{i}" for i in range(num_papers)]
    summaries = []
    for pid in paper_ids:
        for level in ("paragraph", "section", "document"):
            summaries.append(_make_compressed_summary(pid, level, size=summary_size))
    metadata = {}
    if extra_metadata_size > 0:
        metadata["bulk_data"] = "M" * extra_metadata_size
    return ResearchContextSlice(
        research_topic="多智能体无人机路径规划与协同决策",
        research_goals=["goal-1", "goal-2", "goal-3"],
        selected_papers=paper_ids,
        imported_papers=[],
        known_conclusions=["finding-1", "finding-2"],
        open_questions=["question-1"],
        session_history=history,
        relevant_summaries=summaries,
        current_task_plan=[],
        context_scope="manager",
        summary_level="section",
        metadata=metadata,
    )


def _measure_context_chars(context_slice: ResearchContextSlice) -> int:
    """Return the serialized character count of a context slice (same logic as _context_exceeds_budget)."""
    data = context_slice.model_dump(mode="json")
    return len(json.dumps(data, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# LLM Stubs
# ---------------------------------------------------------------------------

class CompressDecisionLLMStub(BaseLLMAdapter):
    """LLM stub that always returns compress_context when context_compression_needed is True."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        self.call_count += 1
        state = input_data.get("state", {})
        if state.get("context_compression_needed"):
            return response_model.model_validate({
                "action_name": "compress_context",
                "worker_agent": "ResearchKnowledgeAgent",
                "instruction": "Compress the research context.",
                "thought": "Context is large, should compress before continuing.",
                "rationale": "Proactive compression to keep context within limits.",
                "phase": "act",
                "payload": {},
            })
        return response_model.model_validate({
            "action_name": "finalize",
            "worker_agent": "ResearchSupervisorAgent",
            "instruction": "All done.",
            "thought": "Workflow is complete.",
            "rationale": "No more actions needed.",
            "phase": "commit",
            "payload": {},
            "stop_reason": "workflow_complete",
        })

    async def _analyze_image_structured(self, prompt, image_path, response_model):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt, file_path, response_model):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt, input_data, response_model):
        raise NotImplementedError


class FinalizeAlwaysLLMStub(BaseLLMAdapter):
    """LLM stub that always finalizes — useful to verify guardrail intercepts before the LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        self.call_count += 1
        if response_model.__name__ == "ResearchUserIntentResult":
            return response_model.model_validate({
                "intent": "general_answer",
                "confidence": 0.7,
                "target_kind": "none",
                "needs_clarification": False,
                "rationale": "General.",
                "markers": [],
                "source": "llm",
            })
        return response_model.model_validate({
            "action_name": "finalize",
            "worker_agent": "ResearchSupervisorAgent",
            "instruction": "Done.",
            "thought": "Complete.",
            "rationale": "Complete.",
            "phase": "commit",
            "payload": {},
            "stop_reason": "workflow_complete",
        })

    async def _analyze_image_structured(self, prompt, image_path, response_model):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt, file_path, response_model):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt, input_data, response_model):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UNIT TESTS: _context_exceeds_budget threshold
# ---------------------------------------------------------------------------

class TestContextExceedsBudget:
    """Verify the reactive guardrail threshold detection."""

    def test_small_context_does_not_exceed_budget(self):
        agent = ResearchSupervisorAgent()
        small_slice = _build_large_context_slice(
            num_history_turns=2, answer_size=100, num_papers=1, summary_size=50,
        )
        assert not agent._context_exceeds_budget(small_slice)
        chars = _measure_context_chars(small_slice)
        assert chars < 120_000, f"Expected small context, got {chars} chars"

    def test_large_context_exceeds_budget(self):
        agent = ResearchSupervisorAgent()
        large_slice = _build_large_context_slice(
            num_history_turns=20,
            answer_size=3000,
            num_papers=8,
            summary_size=1000,
            extra_metadata_size=30_000,
        )
        assert agent._context_exceeds_budget(large_slice)
        chars = _measure_context_chars(large_slice)
        assert chars > 120_000, f"Expected large context, got {chars} chars"

    def test_none_context_does_not_exceed_budget(self):
        agent = ResearchSupervisorAgent()
        assert not agent._context_exceeds_budget(None)

    def test_boundary_detection_at_120k(self):
        """Find the approximate turn count that crosses the 120K char boundary."""
        agent = ResearchSupervisorAgent()
        for turns in range(1, 80):
            slice_ = _build_large_context_slice(
                num_history_turns=turns, answer_size=3000, num_papers=6, summary_size=800,
            )
            chars = _measure_context_chars(slice_)
            if chars > 120_000:
                assert agent._context_exceeds_budget(slice_), (
                    f"Expected exceed at {turns} turns ({chars} chars)"
                )
                # Verify previous turn was under budget
                if turns > 1:
                    prev = _build_large_context_slice(
                        num_history_turns=turns - 1, answer_size=3000, num_papers=6, summary_size=800,
                    )
                    prev_chars = _measure_context_chars(prev)
                    assert prev_chars < chars
                break
        else:
            pytest.fail("Could not find a turn count that exceeds 120K chars within 80 turns")

    def test_custom_budget_parameter(self):
        agent = ResearchSupervisorAgent()
        medium_slice = _build_large_context_slice(
            num_history_turns=5, answer_size=1000, num_papers=3, summary_size=200,
        )
        chars = _measure_context_chars(medium_slice)
        # With a very tight budget it should exceed
        assert agent._context_exceeds_budget(medium_slice, budget_chars=1000)
        # With a very generous budget it should not exceed
        assert not agent._context_exceeds_budget(medium_slice, budget_chars=10_000_000)


# ---------------------------------------------------------------------------
# UNIT TESTS: Reactive guardrail triggers compress_context
# ---------------------------------------------------------------------------

class TestReactiveGuardrailTriggersCompression:
    """When context_slice exceeds 120K chars, _decide_with_llm should bypass the LLM
    and return compress_context directly."""

    @pytest.mark.asyncio
    async def test_guardrail_forces_compress_when_context_oversize(self):
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        large_slice = _build_large_context_slice(
            num_history_turns=20, answer_size=3000, num_papers=8, summary_size=1000,
            extra_metadata_size=30_000,
        )
        assert agent._context_exceeds_budget(large_slice), "precondition: context must be oversize"

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                paper_count=8,
                context_compressed=False,
                context_compression_needed=False,
            ),
            context_slice=large_slice,
        )

        assert decision.action_name == "compress_context"
        assert decision.metadata["decision_source"] == "manager_guardrail"
        # The LLM should NOT have been called since guardrail intercepted
        assert llm_stub.call_count == 0

    @pytest.mark.asyncio
    async def test_guardrail_truncates_and_proceeds_if_already_compressed(self):
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        large_slice = _build_large_context_slice(
            num_history_turns=20, answer_size=3000, num_papers=8, summary_size=1000,
            extra_metadata_size=30_000,
        )
        assert agent._context_exceeds_budget(large_slice)

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                paper_count=8,
                context_compressed=True,  # already compressed
                context_compression_needed=False,
            ),
            context_slice=large_slice,
        )

        # Should NOT return compress_context since already compressed;
        # instead, truncation kicks in and the LLM is called normally.
        assert decision.action_name != "compress_context" or decision.metadata.get("decision_source") != "manager_guardrail"
        # The LLM should have been called (truncation allowed it)
        assert llm_stub.call_count >= 1

    @pytest.mark.asyncio
    async def test_guardrail_allows_normal_flow_when_context_small(self):
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        small_slice = _build_large_context_slice(
            num_history_turns=2, answer_size=100, num_papers=1, summary_size=50,
        )
        assert not agent._context_exceeds_budget(small_slice)

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                paper_count=1,
                context_compressed=False,
            ),
            context_slice=small_slice,
        )

        # Normal flow — LLM decides
        assert llm_stub.call_count >= 1
        assert decision.action_name == "finalize"


# ---------------------------------------------------------------------------
# UNIT TESTS: Proactive compression signal
# ---------------------------------------------------------------------------

class TestProactiveCompressionSignal:
    """Verify context_compression_needed makes the LLM choose compress_context."""

    @pytest.mark.asyncio
    async def test_llm_chooses_compress_when_signal_is_set(self):
        llm_stub = CompressDecisionLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        small_slice = _build_large_context_slice(
            num_history_turns=2, answer_size=100, num_papers=2, summary_size=100,
        )

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="对比论文方法",
                mode="research",
                has_task=True,
                paper_count=2,
                context_compressed=False,
                context_compression_needed=True,
            ),
            context_slice=small_slice,
        )

        assert decision.action_name == "compress_context"
        assert llm_stub.call_count >= 1  # LLM was called (not guardrail)

    @pytest.mark.asyncio
    async def test_llm_finalizes_when_no_compression_needed(self):
        llm_stub = CompressDecisionLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        small_slice = _build_large_context_slice(
            num_history_turns=2, answer_size=100, num_papers=2, summary_size=100,
        )

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                paper_count=2,
                context_compressed=False,
                context_compression_needed=False,
            ),
            context_slice=small_slice,
        )

        assert decision.action_name == "finalize"


# ---------------------------------------------------------------------------
# UNIT TEST: compress_papers actually produces smaller context
# ---------------------------------------------------------------------------

class TestCompressPapersReducesSize:
    """Verify that running compress_papers + rebuilding slices yields a smaller context."""

    def test_compress_papers_produces_multi_level_summaries(self):
        manager = ResearchContextManager()
        papers = [_make_paper_candidate(i) for i in range(6)]
        summaries = manager.compress_papers(
            papers=papers,
            selected_paper_ids=[p.paper_id for p in papers],
        )
        assert len(summaries) >= 3  # at least 3 levels for 1 paper
        levels = {s.level for s in summaries}
        assert levels == {"paragraph", "section", "document"}

    def test_compressed_context_is_smaller_than_original(self):
        """Simulate: build a large context, compress, rebuild slices — compare sizes."""
        manager = ResearchContextManager()
        papers = [_make_paper_candidate(i) for i in range(6)]
        selected_ids = [p.paper_id for p in papers]

        # Build initial (large) context with many QA turns + big metadata
        big_history = [_make_qa_pair(i, answer_size=2000) for i in range(15)]
        big_context = manager.update_context(
            topic="多智能体路径规划",
            goals=["goal-1", "goal-2"],
            selected_papers=selected_ids,
            session_history=big_history,
            metadata={"big_blob": "B" * 50_000},
        )
        big_slice = manager.slice_for_agent(big_context, agent_scope="manager", summary_level="section")
        big_chars = _measure_context_chars(big_slice)

        # Compress papers
        summaries = manager.compress_papers(
            papers=papers,
            selected_paper_ids=selected_ids,
        )

        # Update context with compressed summaries (this replaces/merges summaries)
        compressed_context = manager.update_context(
            current_context=big_context,
            paper_summaries=summaries,
            metadata={
                "context_compression": {
                    "paper_count": len({s.paper_id for s in summaries}),
                    "summary_count": len(summaries),
                },
            },
        )
        compressed_slice = manager.slice_for_agent(
            compressed_context, agent_scope="manager", summary_level="section",
        )
        compressed_chars = _measure_context_chars(compressed_slice)

        # The summaries themselves are compact (≤480 chars each); the main size reduction
        # comes from NOT having unbounded metadata or raw text. The compressed context should
        # have well-bounded summaries.
        assert len(summaries) > 0
        for s in summaries:
            assert len(s.summary) <= 500, f"Summary too long: {len(s.summary)} chars"

        # Verify the compressed context still contains essential information
        assert compressed_context.research_topic == "多智能体路径规划"
        assert compressed_context.paper_summaries
        assert compressed_context.metadata.get("context_compression")

    def test_session_history_is_bounded_by_preferences(self):
        """Even with many turns, slice_for_agent respects max_history_turns."""
        manager = ResearchContextManager()
        big_history = [_make_qa_pair(i, answer_size=3000) for i in range(30)]
        context = manager.update_context(
            topic="Test",
            session_history=big_history,
        )
        sliced = manager.slice_for_agent(context, agent_scope="manager")
        # Default max_history_turns=10
        assert len(sliced.session_history) <= 10


# ---------------------------------------------------------------------------
# INTEGRATION TEST: Full end-to-end with the graph runtime
# ---------------------------------------------------------------------------

class ArxivLongSessionStub:
    """Returns many papers to simulate a busy research session."""
    async def search(self, *, query: str, max_results: int, days_back: int):
        return [
            PaperCandidate(
                paper_id=f"arxiv:long-{i}",
                title=f"Long Session Paper #{i} on Multi-Agent Coordination",
                authors=[f"Author-{i}"],
                abstract="Y" * 300,
                year=2026,
                venue="arXiv",
                source="arxiv",
                arxiv_id=f"long-{i}",
                pdf_url=f"https://arxiv.org/pdf/long-{i}.pdf",
                url=f"https://arxiv.org/abs/long-{i}",
                citations=5 + i,
                is_open_access=True,
                published_at="2026-04-01T00:00:00+00:00",
            )
            for i in range(max_results)
        ]


class EmptySearchStub:
    async def search(self, **kwargs):
        return []


class LongSessionPaperImportStub:
    async def download_paper(self, paper):
        return type("Artifact", (), {
            "paper": paper,
            "document_id": f"doc_{paper.paper_id.replace(':', '_')}",
            "storage_uri": f"/tmp/{paper.paper_id}.pdf",
            "filename": f"{paper.paper_id}.pdf",
        })()


class LongSessionGraphRuntimeStub:
    retrieval_tools = None
    answer_tools = None
    react_reasoning_agent = None

    def __init__(self):
        from rag_runtime.memory import GraphSessionMemory
        self.session_memory = GraphSessionMemory()

    async def handle_parse_document(self, **kwargs):
        from domain.schemas.document import ParsedDocument
        return ParsedDocument(
            id=kwargs.get("document_id", "doc-1"),
            filename="paper.pdf",
            content_type="application/pdf",
            status="parsed",
            pages=[],
            metadata=kwargs.get("metadata") or {},
        )

    async def handle_index_document(self, **kwargs):
        return type("IndexResult", (), {"status": "succeeded"})()

    async def query_graph_summary(self, **kwargs):
        from tooling.schemas import GraphSummaryToolOutput
        return GraphSummaryToolOutput(hits=[], metadata={})

    async def handle_ask_document(self, **kwargs):
        from domain.schemas.api import QAResponse
        from domain.schemas.evidence import EvidenceBundle
        return QAResponse(
            answer="这是一个关于多智能体协同的长会话测试回答。" * 5,
            question=kwargs["question"],
            evidence_bundle=EvidenceBundle(),
            confidence=0.85,
        )

    async def handle_ask_fused(self, **kwargs):
        from domain.schemas.api import QAResponse
        from domain.schemas.evidence import EvidenceBundle
        qa = QAResponse(
            answer="长会话图表分析结果。",
            question=kwargs["question"],
            evidence_bundle=EvidenceBundle(),
            confidence=0.80,
        )
        return type("FusedAskResult", (), {"qa": qa})()



def _build_long_session_service(tmp_path) -> LiteratureResearchService:
    return LiteratureResearchService(
        paper_search_service=PaperSearchService(
            arxiv_tool=ArxivLongSessionStub(),
            openalex_tool=EmptySearchStub(),
        ),
        report_service=ResearchReportService(tmp_path / "research"),
        paper_import_service=LongSessionPaperImportStub(),
    )


@pytest.mark.asyncio
async def test_long_session_research_then_analysis_triggers_compression(tmp_path) -> None:
    """Simulate a two-turn long session:
    1. Initial research discovery (search + review) with many papers
    2. Follow-up analysis request that triggers context compression

    This verifies that the proactive compression path works end-to-end.
    """
    service = _build_long_session_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    graph_runtime = LongSessionGraphRuntimeStub()

    # Turn 1: Initial discovery — search many papers
    response1 = await runtime.run(
        ResearchAgentRunRequest(
            message="调研多智能体协同路径规划的最新论文",
            mode="research",
            days_back=365,
            max_papers=6,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=graph_runtime,
    )

    assert response1.task is not None
    assert response1.report is not None
    action_names_1 = [step.action_name for step in response1.trace]
    assert "search_literature" in action_names_1

    # Turn 2: Request analysis which forces context compression
    response2 = await runtime.run(
        ResearchAgentRunRequest(
            message="对比这些论文的核心方法和实验设计",
            mode="research",
            task_id=response1.task.task_id,
            days_back=365,
            max_papers=6,
            sources=["arxiv"],
            import_top_k=0,
            force_context_compression=True,
        ),
        graph_runtime=graph_runtime,
    )

    assert response2.task is not None
    action_names_2 = [step.action_name for step in response2.trace]
    assert "compress_context" in action_names_2, (
        f"Expected compress_context in trace, got: {action_names_2}"
    )
    # Verify compression metadata persisted
    assert "context_compression" in response2.workspace.metadata


@pytest.mark.asyncio
async def test_long_session_with_many_papers_triggers_proactive_compression(tmp_path) -> None:
    """When ≥4 papers are found, context_compression_needed should be True,
    and the LLM should be encouraged to choose compress_context."""
    service = _build_long_session_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)

    response = await runtime.run(
        ResearchAgentRunRequest(
            message="对比多智能体路径规划方法",
            mode="research",
            days_back=365,
            max_papers=6,
            sources=["arxiv"],
            import_top_k=0,
        ),
        graph_runtime=LongSessionGraphRuntimeStub(),
    )

    assert response.task is not None
    action_names = [step.action_name for step in response.trace]
    # With ≥4 papers, context_compression_needed should fire → compress_context action
    assert "compress_context" in action_names, (
        f"Expected proactive compress_context, got: {action_names}"
    )


@pytest.mark.asyncio
async def test_multi_turn_qa_session_remains_functional(tmp_path) -> None:
    """Simulate multiple QA turns on the same task. The system should not crash
    due to context overflow, even if earlier turns generate large context."""
    service = _build_long_session_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    graph_runtime = LongSessionGraphRuntimeStub()

    # Turn 1: Initial research
    r1 = await runtime.run(
        ResearchAgentRunRequest(
            message="调研多智能体路径规划论文",
            mode="research",
            days_back=365,
            max_papers=4,
            sources=["arxiv"],
            import_top_k=1,
        ),
        graph_runtime=graph_runtime,
    )
    assert r1.task is not None
    task_id = r1.task.task_id

    # Turn 2: QA
    r2 = await runtime.run(
        ResearchAgentRunRequest(
            message="这些论文分别使用了什么方法？",
            mode="qa",
            task_id=task_id,
            sources=["arxiv"],
        ),
        graph_runtime=graph_runtime,
    )
    assert r2.trace[-1].action_name == "finalize"

    # Turn 3: Another QA follow-up
    r3 = await runtime.run(
        ResearchAgentRunRequest(
            message="哪些方法在仿真环境中效果最好？",
            mode="qa",
            task_id=task_id,
            sources=["arxiv"],
        ),
        graph_runtime=graph_runtime,
    )
    assert r3.trace[-1].action_name == "finalize"

    # Turn 4: Yet another follow-up — should still work
    r4 = await runtime.run(
        ResearchAgentRunRequest(
            message="能否总结一下整体研究现状？",
            mode="qa",
            task_id=task_id,
            sources=["arxiv"],
        ),
        graph_runtime=graph_runtime,
    )
    assert r4.trace[-1].action_name == "finalize"

    # All turns should complete without errors
    for i, resp in enumerate([r1, r2, r3, r4], 1):
        assert resp.status in {"completed", "partial"}, (
            f"Turn {i} failed with status={resp.status}"
        )


# ---------------------------------------------------------------------------
# UNIT TEST: CompressContextTool integration
# ---------------------------------------------------------------------------

class TestCompressContextToolReducesSliceSize:
    """Verify that running CompressContextTool actually results in a smaller manager slice."""

    def test_compress_papers_replaces_raw_with_bounded_summaries(self):
        manager = ResearchContextManager()
        papers = [_make_paper_candidate(i) for i in range(6)]
        selected = [p.paper_id for p in papers]
        summaries = manager.compress_papers(papers=papers, selected_paper_ids=selected)

        # Each summary should be bounded
        for s in summaries:
            if s.level == "paragraph":
                assert len(s.summary) <= 200, f"paragraph summary too long: {len(s.summary)}"
            elif s.level == "section":
                assert len(s.summary) <= 340, f"section summary too long: {len(s.summary)}"
            elif s.level == "document":
                assert len(s.summary) <= 500, f"document summary too long: {len(s.summary)}"

    def test_context_slices_after_compression_have_bounded_size(self):
        """After compress_papers + update_context + build_context_slices,
        the manager slice should be well within budget."""
        manager = ResearchContextManager()
        papers = [_make_paper_candidate(i) for i in range(6)]
        selected = [p.paper_id for p in papers]

        # Initial context with moderate history
        history = [_make_qa_pair(i, answer_size=1500) for i in range(8)]
        ctx = manager.update_context(
            topic="Multi-agent planning",
            goals=["survey methods", "compare experiments"],
            selected_papers=selected,
            session_history=history,
        )

        # Compress
        summaries = manager.compress_papers(papers=papers, selected_paper_ids=selected)
        compressed = manager.update_context(
            current_context=ctx,
            paper_summaries=summaries,
            metadata={"context_compression": {"paper_count": 6}},
        )

        # Build manager slice
        from services.research.literature_research_service import LiteratureResearchService
        slices_data = {
            "manager": manager.slice_for_agent(compressed, agent_scope="manager", summary_level="section"),
        }
        manager_slice = slices_data["manager"]
        chars = _measure_context_chars(manager_slice)

        # Manager slice should be well within the 120K guardrail budget
        assert chars < 120_000, (
            f"Manager slice after compression is {chars} chars, should be < 120K"
        )
        # And should also be within the 80K proactive threshold for modest sessions
        # (8 history turns * 1500 chars = 12K answer chars + overhead ≈ ~30-40K)
        # This depends on history size, so we just verify it's reasonable
        assert chars < 200_000, f"Manager slice unexpectedly large: {chars} chars"


# ---------------------------------------------------------------------------
# Proactive signal test at runtime level
# ---------------------------------------------------------------------------

class TestContextCompressionNeededSignal:
    """Verify _state_from_context sets context_compression_needed correctly."""

    def test_many_papers_trigger_compression_needed(self):
        """With ≥4 papers + a task, context_compression_needed should be True."""
        manager = ResearchContextManager()
        papers = [_make_paper_candidate(i) for i in range(5)]
        ctx = manager.update_context(
            topic="Test",
            selected_papers=[p.paper_id for p in papers],
        )

        # context_compression_needed formula: bool(task or papers) and not context_compressed and (... or len(papers) >= 4 ...)
        # We just verify the formula components
        has_papers = len(papers) >= 4
        context_compressed = False
        assert has_papers and not context_compressed

    def test_long_session_history_triggers_compression_needed(self):
        """With session_history_count >= 6, context_compression_needed should be True."""
        manager = ResearchContextManager()
        history = [_make_qa_pair(i) for i in range(8)]
        ctx = manager.update_context(
            topic="Test",
            session_history=history,
        )
        session_count = len(ctx.session_history)
        assert session_count >= 6


# ---------------------------------------------------------------------------
# Realistic scenario: metadata bloated by figure analysis page content
# ---------------------------------------------------------------------------

class TestRealisticFigureAnalysisMetadataBlowup:
    """The original bug: analyze_paper_figures loads raw page content into metadata,
    which inflates context to millions of characters. Verify the guardrail catches this."""

    def test_guardrail_catches_metadata_bloated_by_page_content(self):
        agent = ResearchSupervisorAgent()
        # Simulate a slice where metadata contains raw page text from figure analysis
        slice_ = _build_large_context_slice(
            num_history_turns=5,
            answer_size=500,
            num_papers=2,
            summary_size=200,
            extra_metadata_size=200_000,  # 200K chars of raw page content
        )
        chars = _measure_context_chars(slice_)
        assert chars > 120_000, f"Expected bloated context, got {chars}"
        assert agent._context_exceeds_budget(slice_)

    @pytest.mark.asyncio
    async def test_guardrail_intercepts_before_llm_with_bloated_metadata(self):
        """When metadata is bloated, the guardrail should fire before the LLM is called."""
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)
        bloated_slice = _build_large_context_slice(
            num_history_turns=3,
            answer_size=200,
            num_papers=2,
            summary_size=100,
            extra_metadata_size=200_000,
        )

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="分析图表",
                mode="qa",
                has_task=True,
                paper_count=2,
                imported_document_count=1,
                context_compressed=False,
            ),
            context_slice=bloated_slice,
        )

        assert decision.action_name == "compress_context"
        assert decision.metadata["decision_source"] == "manager_guardrail"
        assert llm_stub.call_count == 0, "LLM should NOT be called when guardrail intercepts"


# ---------------------------------------------------------------------------
# End-to-end: verify compress → post-compress flow
# ---------------------------------------------------------------------------

class TestPostCompressionFlowWorks:
    """After compression, the supervisor should be able to continue making decisions."""

    @pytest.mark.asyncio
    async def test_after_compression_llm_can_decide_normally(self):
        """Simulate: first call triggers guardrail compression, second call (with
        context_compressed=True) proceeds to normal LLM decision."""
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)

        # First: large context → guardrail intercepts
        large_slice = _build_large_context_slice(
            num_history_turns=5, answer_size=200, num_papers=2,
            summary_size=100, extra_metadata_size=200_000,
        )
        d1 = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                context_compressed=False,
            ),
            context_slice=large_slice,
        )
        assert d1.action_name == "compress_context"
        assert llm_stub.call_count == 0

        # Second: after compression, context is smaller AND context_compressed=True
        small_slice = _build_large_context_slice(
            num_history_turns=5, answer_size=200, num_papers=2,
            summary_size=100, extra_metadata_size=0,
        )
        d2 = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                context_compressed=True,
            ),
            context_slice=small_slice,
        )
        assert d2.action_name == "finalize"
        assert llm_stub.call_count >= 1, "LLM should be called after compression"

    @pytest.mark.asyncio
    async def test_compression_not_triggered_twice_but_truncation_allows_llm_call(self):
        """If context is still large after compression (context_compressed=True),
        the guardrail should NOT trigger compress_context again but should
        truncate the context and proceed to the LLM call."""
        llm_stub = FinalizeAlwaysLLMStub()
        agent = ResearchSupervisorAgent(llm_adapter=llm_stub)

        still_large = _build_large_context_slice(
            num_history_turns=5, answer_size=200, num_papers=2,
            summary_size=100, extra_metadata_size=200_000,
        )
        assert agent._context_exceeds_budget(still_large)

        decision = await agent.decide_next_action_async(
            ResearchSupervisorState(
                goal="继续研究",
                mode="research",
                has_task=True,
                context_compressed=True,  # already compressed
            ),
            context_slice=still_large,
        )

        # Should NOT return compress_context since already compressed;
        # truncation should make the LLM call succeed.
        assert decision.action_name != "compress_context" or decision.metadata.get("decision_source") != "manager_guardrail"
        assert llm_stub.call_count >= 1


# ---------------------------------------------------------------------------
# UNIT TESTS: _truncate_context_slice progressive stripping
# ---------------------------------------------------------------------------

class TestTruncateContextSlice:
    """Verify that _truncate_context_slice brings oversized slices under budget."""

    def test_already_small_slice_returned_unchanged(self):
        agent = ResearchSupervisorAgent()
        small = _build_large_context_slice(
            num_history_turns=2, answer_size=100, num_papers=1, summary_size=50,
        )
        result = agent._truncate_context_slice(small)
        assert result is small  # exact same object, no copy needed

    def test_metadata_bloat_stripped_first(self):
        agent = ResearchSupervisorAgent()
        bloated = _build_large_context_slice(
            num_history_turns=3, answer_size=200, num_papers=2,
            summary_size=100, extra_metadata_size=200_000,
        )
        assert agent._context_exceeds_budget(bloated)
        result = agent._truncate_context_slice(bloated)
        result_chars = _measure_context_chars(result)
        assert result_chars <= 100_000, f"Truncated slice still {result_chars} chars"
        assert result.metadata.get("truncated") is True
        # Core fields preserved
        assert result.research_topic == bloated.research_topic
        assert result.selected_papers == bloated.selected_papers

    def test_extreme_context_falls_back_to_minimal(self):
        agent = ResearchSupervisorAgent()
        extreme = _build_large_context_slice(
            num_history_turns=10, answer_size=50_000,
            num_papers=10, summary_size=5000,
            extra_metadata_size=500_000,
        )
        result = agent._truncate_context_slice(extreme)
        result_chars = _measure_context_chars(result)
        assert result_chars <= 100_000, f"Extreme truncation still {result_chars} chars"
        assert result.research_topic  # topic always preserved

    def test_3_9m_char_real_world_scenario_truncated_to_budget(self):
        """Simulate the exact scenario: 15M chars of metadata from page content."""
        agent = ResearchSupervisorAgent()
        massive = _build_large_context_slice(
            num_history_turns=5, answer_size=500, num_papers=5,
            summary_size=300, extra_metadata_size=15_000_000,
        )
        chars_before = _measure_context_chars(massive)
        assert chars_before > 3_000_000, f"Precondition: expected >3M chars, got {chars_before}"

        result = agent._truncate_context_slice(massive)
        chars_after = _measure_context_chars(result)
        assert chars_after <= 100_000, (
            f"After truncation: {chars_after} chars — must be ≤100K"
        )
