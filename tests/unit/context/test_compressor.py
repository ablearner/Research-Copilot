import pytest

from context.compressor import ContextCompressor


def test_should_compress_above_threshold():
    comp = ContextCompressor(target_budget_ratio=0.75)
    assert comp.should_compress(8000, 10000)


def test_should_not_compress_below_threshold():
    comp = ContextCompressor(target_budget_ratio=0.75)
    assert not comp.should_compress(5000, 10000)


def test_anti_thrashing_blocks_after_ineffective():
    comp = ContextCompressor()
    comp._ineffective_count = 2
    assert not comp.should_compress(9000, 10000)


def test_prune_tool_outputs():
    comp = ContextCompressor(protect_last_n=1)
    messages = [
        {"role": "tool", "name": "hybrid_retrieve", "content": "x" * 1000},
        {"role": "tool", "name": "hybrid_retrieve", "content": "y" * 1000},
        {"role": "user", "content": "latest"},
    ]
    pruned = comp._prune_tool_outputs(messages, set())
    assert "pruned" in pruned[0]["content"]
    assert "pruned" in pruned[1]["content"]
    assert pruned[2]["content"] == "latest"


def test_prune_protects_tail():
    comp = ContextCompressor(protect_last_n=2)
    messages = [
        {"role": "tool", "name": "hybrid_retrieve", "content": "x" * 1000},
        {"role": "tool", "name": "hybrid_retrieve", "content": "y" * 1000},
    ]
    pruned = comp._prune_tool_outputs(messages, set())
    assert pruned[1]["content"] == "y" * 1000


def test_prune_protects_named_tools():
    comp = ContextCompressor(protect_last_n=0)
    messages = [
        {"role": "tool", "name": "hybrid_retrieve", "content": "x" * 1000},
    ]
    pruned = comp._prune_tool_outputs(messages, {"hybrid_retrieve"})
    assert pruned[0]["content"] == "x" * 1000


def test_sanitize_tool_pairs_injects_stubs():
    comp = ContextCompressor()
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}, {"id": "tc2"}]},
        {"role": "user", "content": "hi"},
    ]
    clean = comp._sanitize_tool_pairs(messages)
    tool_msgs = [m for m in clean if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["tool_call_id"] in ("tc1", "tc2") for m in tool_msgs)


def test_sanitize_tool_pairs_preserves_existing():
    comp = ContextCompressor()
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
    ]
    clean = comp._sanitize_tool_pairs(messages)
    assert len(clean) == 2
    assert clean[1]["content"] == "ok"


def test_fallback_summary():
    comp = ContextCompressor()
    turns = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    summary = comp._fallback_summary(turns)
    assert "2 conversation turns" in summary


@pytest.mark.asyncio
async def test_compress_messages_below_budget_returns_unchanged():
    comp = ContextCompressor()
    messages = [{"role": "user", "content": "short"}]
    result = await comp.compress_messages(messages, 100_000)
    assert result == messages
