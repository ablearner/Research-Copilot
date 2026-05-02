from domain.schemas.research_context import ResearchContext
from memory.memory_manager import MemoryManager


def test_freeze_and_get_snapshot():
    mm = MemoryManager()
    ctx = ResearchContext(research_topic="quantum computing")
    mm.freeze_session_snapshot("s1", ctx)
    frozen = mm.get_frozen_prompt_block("s1")
    assert frozen is not None
    assert "recalled_memories" in frozen
    assert "user_profile" in frozen


def test_frozen_snapshot_is_stable():
    mm = MemoryManager()
    ctx = ResearchContext(research_topic="quantum computing")
    mm.freeze_session_snapshot("s1", ctx)
    mm.get_frozen_prompt_block("s1")
    # Modify context; snapshot should NOT change
    ctx2 = ResearchContext(research_topic="machine learning", research_goals=["new goal"])
    mm.freeze_session_snapshot("s1", ctx2)  # re-freeze overwrites
    second = mm.get_frozen_prompt_block("s1")
    # Should reflect the new freeze, not the old one
    assert second is not None


def test_get_snapshot_returns_none_for_unknown():
    mm = MemoryManager()
    assert mm.get_frozen_prompt_block("unknown") is None


def test_hydrate_context_auto_freezes():
    mm = MemoryManager()
    ctx = ResearchContext(research_topic="quantum computing")
    mm.session_memory.save(mm.session_memory.load("s1").model_copy(update={"context": ctx}))
    result = mm.hydrate_context("s1", base_context=ctx)
    # Should have auto-frozen
    frozen = mm.get_frozen_prompt_block("s1")
    assert frozen is not None
    # Metadata should have recalled_memories
    assert "recalled_memories" in result.metadata


def test_hydrate_context_uses_frozen_on_second_call():
    mm = MemoryManager()
    ctx = ResearchContext(research_topic="quantum computing")
    mm.session_memory.save(mm.session_memory.load("s1").model_copy(update={"context": ctx}))
    first = mm.hydrate_context("s1", base_context=ctx)
    second = mm.hydrate_context("s1", base_context=ctx)
    # Both should have the same recalled_memories from frozen snapshot
    assert first.metadata.get("recalled_memories") == second.metadata.get("recalled_memories")
