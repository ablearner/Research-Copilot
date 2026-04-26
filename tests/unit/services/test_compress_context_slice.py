"""Tests for ResearchContextManager.compress_context_slice — Hermes-inspired 3-phase compression.

Covers:
  - Phase 1: metadata pruning (large values replaced with informative placeholders)
  - Phase 2: session history compaction (old QA → rolling summary + protected tail)
  - Phase 3: progressive field stripping (summaries, conclusions, papers, nuclear)
  - Idempotency: small slices pass through unchanged
  - Real-world scenarios: 15M metadata, 200K session history, mixed bloat
"""

import json

import pytest

from domain.schemas.research_context import (
    CompressedPaperSummary,
    QAPair,
    ResearchContextPaperMeta,
    ResearchContextSlice,
)
from services.research.research_context_manager import ResearchContextManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slice(
    *,
    num_history: int = 0,
    answer_size: int = 100,
    num_papers: int = 0,
    paper_meta_size: int = 0,
    num_summaries: int = 0,
    metadata_size: int = 0,
    memory_items: int = 0,
    memory_item_size: int = 0,
) -> ResearchContextSlice:
    history = [
        QAPair(
            question=f"Question {i}?",
            answer=f"Answer {i}. " + "x" * answer_size,
            citations=[f"paper_{i}"],
            metadata={"task_id": f"task_{i}"},
        )
        for i in range(num_history)
    ]
    papers = [
        ResearchContextPaperMeta(
            paper_id=f"paper_{i}",
            title=f"Paper Title {i}",
            authors=[f"Author {i}"],
            year=2024,
            summary=f"Paper summary {i}" + "s" * 100,
            metadata={"page_content": "P" * paper_meta_size} if paper_meta_size else {},
        )
        for i in range(num_papers)
    ]
    summaries = [
        CompressedPaperSummary(
            paper_id=f"paper_{i % max(num_papers, 1)}",
            level="section",
            summary=f"Summary {i}" + "s" * 100,
        )
        for i in range(num_summaries)
    ]
    metadata: dict = {}
    if metadata_size > 0:
        metadata["raw_page_content"] = "M" * metadata_size
    memory_context: dict = {}
    if memory_items > 0:
        memory_context["recalled_memories"] = [
            "m" * memory_item_size for _ in range(memory_items)
        ]
        memory_context["recalled_memory_ids"] = [f"mem_{i}" for i in range(memory_items)]
    return ResearchContextSlice(
        research_topic="Deep learning for NLP",
        research_goals=["Understand transformers", "Survey attention mechanisms"],
        selected_papers=[f"paper_{i}" for i in range(num_papers)],
        imported_papers=papers,
        session_history=history,
        relevant_summaries=summaries,
        known_conclusions=[f"Conclusion {i}" for i in range(5)],
        open_questions=[f"Question {i}" for i in range(5)],
        memory_context=memory_context,
        metadata=metadata,
    )


def _chars(s: ResearchContextSlice) -> int:
    return len(json.dumps(s.model_dump(mode="json"), ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Tests: Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_small_slice_returned_unchanged(self):
        mgr = ResearchContextManager()
        small = _make_slice(num_history=2, answer_size=50)
        result = mgr.compress_context_slice(small)
        assert result is small  # exact same object

    def test_empty_slice_returned_unchanged(self):
        mgr = ResearchContextManager()
        empty = ResearchContextSlice()
        result = mgr.compress_context_slice(empty)
        assert result is empty


# ---------------------------------------------------------------------------
# Tests: Phase 1 — Metadata Pruning
# ---------------------------------------------------------------------------

class TestPhase1MetadataPruning:
    def test_large_metadata_values_pruned(self):
        mgr = ResearchContextManager()
        s = _make_slice(metadata_size=200_000)
        assert _chars(s) > 200_000

        result = mgr.compress_context_slice(s)
        assert _chars(result) <= 100_000
        # Metadata was pruned — check the placeholder
        raw_content = result.metadata.get("raw_page_content", "")
        assert "pruned" in str(raw_content).lower()
        assert result.metadata.get("_compression") is not None
        assert "prune_metadata" in result.metadata["_compression"]["phases_applied"]

    def test_paper_metadata_pruned(self):
        mgr = ResearchContextManager()
        s = _make_slice(num_papers=5, paper_meta_size=30_000)
        assert _chars(s) > 100_000

        result = mgr.compress_context_slice(s)
        assert _chars(result) <= 100_000
        for paper in result.imported_papers:
            page_content = paper.metadata.get("page_content", "")
            assert len(str(page_content)) < 1000

    def test_memory_context_pruned(self):
        mgr = ResearchContextManager()
        s = _make_slice(memory_items=20, memory_item_size=5000)
        result = mgr.compress_context_slice(s)
        memories = result.memory_context.get("recalled_memories", [])
        assert len(memories) <= 10
        for m in memories:
            assert len(m) <= 200 + 5  # compact_text adds "…"

    def test_preserved_keys_survive_pruning(self):
        mgr = ResearchContextManager()
        s = _make_slice(metadata_size=200_000)
        s.metadata["context_scope"] = "manager"
        s.metadata["summary_level"] = "section"
        result = mgr.compress_context_slice(s)
        assert result.metadata.get("context_scope") == "manager"
        assert result.metadata.get("summary_level") == "section"


# ---------------------------------------------------------------------------
# Tests: Phase 2 — Session History Compaction
# ---------------------------------------------------------------------------

class TestPhase2HistoryCompaction:
    def test_old_history_folded_into_summary(self):
        mgr = ResearchContextManager()
        s = _make_slice(num_history=10, answer_size=12_000)
        assert _chars(s) > 100_000

        result = mgr.compress_context_slice(s)
        # Should have: 1 summary QA + 3 protected tail QAs = 4
        assert len(result.session_history) == 4
        summary_qa = result.session_history[0]
        assert "[CONTEXT COMPACTION]" in summary_qa.question
        assert "7 earlier QA turns" in summary_qa.question
        assert "## Compacted Conversation History" in summary_qa.answer
        assert "## Resolved Q&A" in summary_qa.answer

    def test_tail_qa_pairs_preserved_verbatim(self):
        mgr = ResearchContextManager()
        s = _make_slice(num_history=8, answer_size=15_000)
        original_tail = s.session_history[-3:]

        result = mgr.compress_context_slice(s)
        result_tail = result.session_history[-3:]
        for orig, comp in zip(original_tail, result_tail):
            assert orig.question == comp.question
            # Answer preserved fully for tail
            assert orig.answer == comp.answer

    def test_summary_contains_key_conclusions(self):
        mgr = ResearchContextManager()
        s = _make_slice(num_history=8, answer_size=15_000)
        # Give one answer a meaningful first sentence
        s.session_history[0].answer = "Transformers outperform RNNs on all NLP benchmarks。" + "x" * 2000

        result = mgr.compress_context_slice(s)
        summary_qa = result.session_history[0]
        assert "## Key Conclusions" in summary_qa.answer
        assert "Transformers outperform" in summary_qa.answer

    def test_custom_protect_tail(self):
        mgr = ResearchContextManager()
        s = _make_slice(num_history=10, answer_size=12_000)
        result = mgr.compress_context_slice(s, protect_tail=5)
        # 1 summary + 5 tail = 6
        assert len(result.session_history) == 6


# ---------------------------------------------------------------------------
# Tests: Phase 3 — Progressive Field Stripping
# ---------------------------------------------------------------------------

class TestPhase3FieldStripping:
    def test_summaries_capped(self):
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=8, answer_size=12_000,
            num_summaries=20, num_papers=5,
        )
        assert _chars(s) > 100_000
        result = mgr.compress_context_slice(s)
        # Phase 2 compresses history; if still over, Phase 3 caps summaries
        assert _chars(result) <= 100_000

    def test_extreme_data_compressed_to_budget(self):
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=10, answer_size=50_000,
            num_papers=10, paper_meta_size=50_000,
            num_summaries=20,
            metadata_size=500_000,
            memory_items=50, memory_item_size=5000,
        )
        assert _chars(s) > 1_000_000

        result = mgr.compress_context_slice(s)
        result_size = _chars(result)
        assert result_size <= 100_000, f"Still {result_size} chars after all phases"
        assert result.research_topic == "Deep learning for NLP"
        compression = result.metadata.get("_compression", {})
        assert "strip_fields" in compression.get("phases_applied", [])

    def test_nuclear_fallback_with_huge_tail_answers(self):
        """Force nuclear fallback: even 1 QA entry exceeds 100K after all strips."""
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=4, answer_size=150_000,
            num_papers=2,
            metadata_size=100_000,
        )
        # Even a single QA pair is ~150K > 100K budget → must reach nuclear
        result = mgr.compress_context_slice(s)
        result_size = _chars(result)
        assert result_size <= 100_000, f"Nuclear fallback still {result_size} chars"
        compression = result.metadata.get("_compression", {})
        assert "nuclear_fallback" in compression.get("phases_applied", [])


# ---------------------------------------------------------------------------
# Tests: Real-world scenarios
# ---------------------------------------------------------------------------

class TestRealWorldScenarios:
    def test_15m_metadata_from_figure_analysis(self):
        """Simulate the exact production bug: 15M chars of raw page content in metadata."""
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=6, answer_size=500,
            num_papers=5, paper_meta_size=100,
            metadata_size=15_000_000,
        )
        before = _chars(s)
        assert before > 10_000_000

        result = mgr.compress_context_slice(s)
        after = _chars(result)
        assert after <= 100_000, f"After compression: {after:,} chars"
        # Phase 1 alone should handle this
        compression = result.metadata.get("_compression", {})
        assert "prune_metadata" in compression.get("phases_applied", [])
        reduction_pct = (1 - after / before) * 100
        assert reduction_pct > 99, f"Expected >99% reduction, got {reduction_pct:.1f}%"

    def test_mixed_bloat_metadata_plus_history(self):
        """Both metadata and session history are large — needs Phase 1 + Phase 2."""
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=10, answer_size=12_000,
            num_papers=5, paper_meta_size=5000,
            metadata_size=50_000,
            memory_items=10, memory_item_size=3000,
        )
        before = _chars(s)
        assert before > 100_000

        result = mgr.compress_context_slice(s)
        assert _chars(result) <= 100_000
        compression = result.metadata.get("_compression", {})
        phases = compression.get("phases_applied", [])
        assert len(phases) >= 1  # at least one phase applied

    def test_compression_metadata_tracks_phases(self):
        """Compression records which phases were applied."""
        mgr = ResearchContextManager()
        s = _make_slice(
            num_history=10, answer_size=8000,
            metadata_size=50_000,
        )
        result = mgr.compress_context_slice(s)
        compression = result.metadata.get("_compression", {})
        assert isinstance(compression.get("phases_applied"), list)
        assert len(compression["phases_applied"]) >= 1
        assert "phase1_pruned_chars" in compression

    def test_custom_budget(self):
        """Compression respects a custom budget."""
        mgr = ResearchContextManager()
        s = _make_slice(num_history=5, answer_size=3000)
        result = mgr.compress_context_slice(s, budget_chars=5000)
        assert _chars(result) <= 5000
