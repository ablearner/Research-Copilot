from context.token_counter import TokenBudget, estimate_tokens_rough, get_context_length


def test_known_model_context_length():
    assert get_context_length("gpt-4o") == 128_000
    assert get_context_length("qwen-plus") == 131_072
    assert get_context_length("claude-sonnet-4-20250514") == 200_000


def test_unknown_model_fallback():
    assert get_context_length("totally-unknown-model") == 32_768


def test_substring_match():
    assert get_context_length("gpt-4o-2024-08-06") == 128_000


def test_estimate_tokens_rough():
    msgs = [{"content": "hello world"}]
    tokens = estimate_tokens_rough(msgs)
    assert tokens > 0


def test_estimate_with_tool_calls():
    msgs = [
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"q": "test"}'}}
            ],
        }
    ]
    assert estimate_tokens_rough(msgs) > 0


def test_token_budget_basic():
    budget = TokenBudget("gpt-4o")
    assert budget.total == 128_000
    assert budget.remaining == budget.available
    budget.consume(100_000)
    assert budget.remaining == budget.available - 100_000
    assert not budget.should_compress(threshold=0.85)


def test_token_budget_compress_trigger():
    budget = TokenBudget("gpt-4", reserve_for_output=1000)
    # available = 8192 - 1000 = 7192
    budget.consume(6500)  # 6500/7192 ≈ 0.904
    assert budget.should_compress(threshold=0.85)


def test_token_budget_reset():
    budget = TokenBudget("gpt-4")
    budget.consume(5000)
    budget.reset_usage()
    assert budget.used == 0
    assert budget.remaining == budget.available


def test_context_overflow_probing():
    budget = TokenBudget("gpt-4")
    assert budget.total == 8192
    adjusted = budget.handle_context_overflow(
        "maximum context length is 4096 tokens"
    )
    assert adjusted
    assert budget.total == 4096


def test_context_overflow_probe_only_once():
    budget = TokenBudget("gpt-4")
    budget.handle_context_overflow("maximum context length is 4096 tokens")
    second = budget.handle_context_overflow("maximum context length is 2048 tokens")
    assert not second
    assert budget.total == 4096


def test_context_overflow_fallback_reduction():
    budget = TokenBudget("gpt-4")
    original = budget.total
    budget.handle_context_overflow("some generic error without token count")
    assert budget.total == int(original * 0.75)
