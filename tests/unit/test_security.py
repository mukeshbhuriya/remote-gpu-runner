"""Unit tests for gpu_core.security module."""

from pathlib import Path

import pytest

from gpu_core.errors import SecurityError
from gpu_core.security import (
    extract_bearer_token,
    generate_token,
    hash_token,
    sanitize_filename,
    validate_command_args,
    validate_safe_path,
    verify_token,
)


class TestTokens:
    def test_generate_token_length(self):
        token = generate_token()
        assert len(token) > 32

    def test_generate_unique(self):
        t1 = generate_token()
        t2 = generate_token()
        assert t1 != t2

    def test_hash_and_verify(self):
        token = generate_token()
        hashed = hash_token(token)
        assert verify_token(token, hashed)
        assert not verify_token("wrong-token", hashed)

    def test_hash_deterministic(self):
        token = "test-token-123"
        h1 = hash_token(token)
        h2 = hash_token(token)
        assert h1 == h2


class TestBearerExtraction:
    def test_valid_bearer(self):
        token = extract_bearer_token("Bearer abc123")
        assert token == "abc123"

    def test_case_insensitive(self):
        token = extract_bearer_token("bearer abc123")
        assert token == "abc123"

    def test_missing_header(self):
        from gpu_core.errors import AuthenticationError
        with pytest.raises(AuthenticationError):
            extract_bearer_token(None)

    def test_invalid_format(self):
        from gpu_core.errors import AuthenticationError
        with pytest.raises(AuthenticationError):
            extract_bearer_token("Basic abc123")

    def test_empty_token(self):
        from gpu_core.errors import AuthenticationError
        with pytest.raises(AuthenticationError):
            extract_bearer_token("Bearer ")


class TestFilenameSanitization:
    def test_normal_filename(self):
        assert sanitize_filename("train.py") == "train.py"

    def test_strips_path_components(self):
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("../../secret.txt") == "secret.txt"
        assert sanitize_filename("C:\\Users\\file.txt") == "file.txt"

    def test_removes_forbidden_chars(self):
        result = sanitize_filename("file<>name.txt")
        assert "<" not in result
        assert ">" not in result

    def test_rejects_dangerous_names(self):
        with pytest.raises(SecurityError):
            sanitize_filename("..")
        with pytest.raises(SecurityError):
            sanitize_filename(".")

    def test_rejects_empty(self):
        with pytest.raises(SecurityError):
            sanitize_filename("")


class TestPathValidation:
    def test_valid_path(self, tmp_path):
        safe = validate_safe_path("subdir/file.txt", tmp_path)
        assert str(safe).startswith(str(tmp_path.resolve()))

    def test_path_traversal_blocked(self, tmp_path):
        with pytest.raises(SecurityError):
            validate_safe_path("../../etc/passwd", tmp_path)

    def test_absolute_path_outside_root(self, tmp_path):
        with pytest.raises(SecurityError):
            validate_safe_path("/etc/passwd", tmp_path)


class TestCommandValidation:
    def test_safe_args(self):
        validate_command_args(["python", "train.py", "--epochs", "10"])

    def test_rejects_shell_injection(self):
        with pytest.raises(SecurityError):
            validate_command_args(["python", "; rm -rf /"])

    def test_rejects_pipe(self):
        with pytest.raises(SecurityError):
            validate_command_args(["cat", "file.txt", "|", "grep"])

    def test_rejects_backtick(self):
        with pytest.raises(SecurityError):
            validate_command_args(["echo", "`whoami`"])
