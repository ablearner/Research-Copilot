from security.redact import redact_sensitive_text


def test_openai_api_key():
    result = redact_sensitive_text("sk-abcdefghijklmnopqrstuvwxyz1234567890")
    assert "sk-" not in result
    assert "REDACTED" in result


def test_github_token():
    result = redact_sensitive_text("ghp_ABCDEFghijklmnopqrstuvwxyz1234567890")
    assert "ghp_" not in result


def test_aws_key():
    result = redact_sensitive_text("AKIA1234567890123456")
    assert "AKIA" not in result


def test_bearer_token():
    result = redact_sensitive_text("Bearer some-token-value-1234")
    assert "some-token-value" not in result


def test_connection_string():
    result = redact_sensitive_text("postgres://admin:secret@db.example.com:5432/mydb")
    assert "secret" not in result


def test_generic_key_value():
    result = redact_sensitive_text("api_key=super_secret_value_1234")
    assert "super_secret" not in result
    assert "REDACTED" in result


def test_normal_text_unchanged():
    text = "This is a normal research paper about quantum computing."
    assert redact_sensitive_text(text) == text
