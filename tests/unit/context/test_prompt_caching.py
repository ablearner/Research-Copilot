from context.prompt_caching import apply_anthropic_cache_control


def test_system_message_gets_cache_control():
    msgs = [{"role": "system", "content": "You are helpful"}]
    cached = apply_anthropic_cache_control(msgs)
    assert isinstance(cached[0]["content"], list)
    assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_does_not_mutate_original():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    cached = apply_anthropic_cache_control(msgs)
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["content"] == "hi"
    assert cached is not msgs


def test_up_to_4_breakpoints():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]
    cached = apply_anthropic_cache_control(msgs)
    count = sum(1 for m in cached if "cache_control" in str(m))
    assert count == 4


def test_last_non_system_messages_marked():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    cached = apply_anthropic_cache_control(msgs)
    assert "cache_control" in str(cached[-1])
    assert "cache_control" in str(cached[-2])
    assert "cache_control" in str(cached[-3])
