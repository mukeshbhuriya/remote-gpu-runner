"""
Authentication and security utilities.

Handles:
- Bearer token generation and validation
- Token hashing and comparison
- Path sanitization and traversal prevention
- Filename validation
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from gpu_core.errors import AuthenticationError, SecurityError, ErrorCode
from gpu_core.logging_config import get_logger

logger = get_logger("security")


# ─────────────────────────────────────
# Token Management
# ─────────────────────────────────────

def generate_token(nbytes: int = 48) -> str:
    """Generate a cryptographically secure URL-safe token."""
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """
    Hash a token for storage using SHA-256.

    For a single-user LAN platform, SHA-256 is sufficient.
    The token itself has high entropy (48 bytes = 384 bits).
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(plain_token: str, hashed_token: str) -> bool:
    """Verify a plain token against its hash using constant-time comparison."""
    return secrets.compare_digest(
        hash_token(plain_token),
        hashed_token,
    )


# ─────────────────────────────────────
# Bearer Token Extraction
# ─────────────────────────────────────

def extract_bearer_token(authorization: str | None) -> str:
    """
    Extract the token from an Authorization header.

    Expected format: "Bearer <token>"
    """
    if not authorization:
        raise AuthenticationError("Authorization header is missing.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError("Invalid Authorization header format. Expected: Bearer <token>")

    token = parts[1].strip()
    if not token:
        raise AuthenticationError("Empty token in Authorization header.")

    return token


# ─────────────────────────────────────
# Path Security
# ─────────────────────────────────────

# Characters forbidden in filenames
_FORBIDDEN_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

# Dangerous path components
_DANGEROUS_COMPONENTS = {".", "..", "~", "con", "prn", "aux", "nul"}
_DANGEROUS_PREFIXES = ("com", "lpt")  # Windows reserved names


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.

    - Remove path separators
    - Remove forbidden characters
    - Reject empty or dangerous names
    """
    # Strip any directory components — take only the basename
    filename = os.path.basename(filename)
    filename = PurePosixPath(filename).name  # Handle forward slashes too

    # Remove forbidden characters
    filename = _FORBIDDEN_CHARS.sub("_", filename)

    # Reject dangerous names — check both the full name and the stem
    full_lower = filename.lower().strip()
    if full_lower in _DANGEROUS_COMPONENTS or full_lower in ("..", "."):
        raise SecurityError(
            f"Forbidden filename: '{filename}'",
            code=ErrorCode.INVALID_FILENAME,
        )
    name_lower = full_lower.split(".")[0] if "." in full_lower else full_lower
    if name_lower and name_lower in _DANGEROUS_COMPONENTS:
        raise SecurityError(
            f"Forbidden filename: '{filename}'",
            code=ErrorCode.INVALID_FILENAME,
        )
    if name_lower and any(name_lower.startswith(p) and name_lower[len(p):].isdigit()
           for p in _DANGEROUS_PREFIXES):
        raise SecurityError(
            f"Reserved filename: '{filename}'",
            code=ErrorCode.INVALID_FILENAME,
        )

    # Reject empty
    if not filename or filename.isspace():
        raise SecurityError(
            "Empty filename after sanitization.",
            code=ErrorCode.INVALID_FILENAME,
        )

    return filename


def validate_safe_path(user_path: str, root: Path) -> Path:
    """
    Validate that a user-provided path stays within the allowed root.

    Returns the resolved absolute path if safe.
    Raises SecurityError if path traversal is detected.
    """
    root = root.resolve()

    # Normalize the user path
    try:
        candidate = (root / user_path).resolve()
    except (ValueError, OSError) as e:
        raise SecurityError(
            f"Invalid path: {e}",
            code=ErrorCode.PATH_TRAVERSAL,
        )

    # Check containment
    try:
        candidate.relative_to(root)
    except ValueError:
        logger.warning(
            f"Path traversal attempt blocked: '{user_path}' resolved to '{candidate}' "
            f"which is outside root '{root}'"
        )
        raise SecurityError(
            f"Path traversal detected. Access denied.",
            code=ErrorCode.PATH_TRAVERSAL,
        )

    return candidate


def validate_upload_size(size: int, max_size: int) -> None:
    """Raise if upload exceeds size limit."""
    if size > max_size:
        from gpu_core.errors import UploadTooLargeError
        raise UploadTooLargeError(size, max_size)


# ─────────────────────────────────────
# Subprocess Safety
# ─────────────────────────────────────

# Characters that could enable command injection
_SHELL_DANGEROUS = re.compile(r'[;&|`$(){}!\\<>\n\r]')


def validate_command_args(args: list[str]) -> None:
    """
    Validate subprocess arguments for safety.

    Reject arguments containing shell metacharacters that could enable injection.
    """
    for arg in args:
        if _SHELL_DANGEROUS.search(arg):
            raise SecurityError(
                f"Potentially dangerous character in command argument: '{arg}'",
                code=ErrorCode.PATH_TRAVERSAL,
            )
