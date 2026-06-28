from __future__ import annotations

from spice.llm.error_safety import sanitize_error_text


def test_sanitize_error_text_redacts_common_secret_shapes() -> None:
    text = "\n".join(
        [
            "url=https://user:ghp_abcdefghijklmnopqrstuvwxyz123456@api.example.test/path?api_key=abc123",
            "Proxy-Authorization: Basic abcdef",
            "X-Goog-Api-Key: AIzaSyabcdefghijklmnopqrstuvwxyz",
            "token=github_pat_abcdefghijklmnopqrstuvwxyz123456",
            "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n-----END OPENSSH PRIVATE KEY-----",
        ]
    )

    sanitized = sanitize_error_text(text)

    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in sanitized
    assert "abc123" not in sanitized
    assert "Basic abcdef" not in sanitized
    assert "AIzaSyabcdefghijklmnopqrstuvwxyz" not in sanitized
    assert "github_pat_abcdefghijklmnopqrstuvwxyz123456" not in sanitized
    assert "BEGIN OPENSSH PRIVATE KEY" not in sanitized
    assert sanitized.count("[redacted]") >= 5
