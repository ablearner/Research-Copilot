from adapters.llm.error_classifier import classify_llm_error, FailureReason


def _exc_with_status(msg, status_code):
    exc = Exception(msg)
    exc.status_code = status_code
    return exc


def test_rate_limit_by_status():
    c = classify_llm_error(_exc_with_status("too many requests", 429))
    assert c.reason == FailureReason.rate_limit
    assert c.retryable
    assert c.should_fallback


def test_rate_limit_by_message():
    c = classify_llm_error(Exception("rate limit exceeded"))
    assert c.reason == FailureReason.rate_limit


def test_context_overflow_413():
    c = classify_llm_error(_exc_with_status("payload too large", 413))
    assert c.reason == FailureReason.context_overflow
    assert c.should_compress
    assert not c.should_fallback


def test_context_overflow_400_with_keyword():
    c = classify_llm_error(_exc_with_status("context_length_exceeded", 400))
    assert c.reason == FailureReason.context_overflow


def test_context_overflow_keyword_only():
    c = classify_llm_error(Exception("context_length_exceeded"))
    assert c.reason == FailureReason.context_overflow
    assert c.should_compress


def test_billing():
    c = classify_llm_error(_exc_with_status("payment required", 402))
    assert c.reason == FailureReason.billing
    assert c.should_fallback


def test_billing_by_message():
    c = classify_llm_error(Exception("insufficient_quota"))
    assert c.reason == FailureReason.billing


def test_auth_401():
    c = classify_llm_error(_exc_with_status("unauthorized", 401))
    assert c.reason == FailureReason.auth


def test_auth_403():
    c = classify_llm_error(_exc_with_status("forbidden", 403))
    assert c.reason == FailureReason.auth


def test_content_filter():
    c = classify_llm_error(Exception("content_policy_violation"))
    assert c.reason == FailureReason.content_filter
    assert not c.retryable
    assert not c.should_fallback


def test_server_error():
    c = classify_llm_error(_exc_with_status("internal server error", 500))
    assert c.reason == FailureReason.server_error
    assert c.retryable


def test_timeout():
    c = classify_llm_error(TimeoutError("timed out"))
    assert c.reason == FailureReason.timeout
    assert c.retryable


def test_connection_error():
    c = classify_llm_error(OSError("connection refused"))
    assert c.reason == FailureReason.connection
    assert c.retryable


def test_unknown():
    c = classify_llm_error(Exception("something weird"))
    assert c.reason == FailureReason.unknown
