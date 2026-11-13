from services.task_execution import request_fingerprint


def test_fingerprint_is_canonical_and_sensitive():
    assert request_fingerprint({"a": 1, "b": 2}) == request_fingerprint({"b": 2, "a": 1})
    assert request_fingerprint({"a": 1}) != request_fingerprint({"a": 2})
