"""
Structured logging for the Remote ML GPU Platform.

Provides JSON and text logging formatters, component-scoped loggers,
and log sanitization to prevent secret leakage.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any


# Patterns to redact from logs
_SENSITIVE_PATTERNS = [
    re.compile(r'(bearer\s+)\S+', re.IGNORECASE),
    re.compile(r'(token["\s:=]+)\S+', re.IGNORECASE),
    re.compile(r'(password["\s:=]+)\S+', re.IGNORECASE),
    re.compile(r'(secret[_\-]?key["\s:=]+)\S+', re.IGNORECASE),
    re.compile(r'(api[_\-]?key["\s:=]+)\S+', re.IGNORECASE),
    re.compile(r'(authorization["\s:=]+)\S+', re.IGNORECASE),
]


def _sanitize(message: str) -> str:
    """Redact sensitive values from log messages."""
    for pattern in _SENSITIVE_PATTERNS:
        message = pattern.sub(r'\1[REDACTED]', message)
    return message


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": _sanitize(record.getMessage()),
        }

        # Add extra fields
        for key in ("job_id", "request_id", "event", "status", "user", "ip"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["error"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": _sanitize(str(record.exc_info[1])),
            }

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        msg = _sanitize(record.getMessage())

        parts = [f"{ts} [{record.levelname:8s}] {record.name}: {msg}"]

        # Append extra context
        extras = []
        for key in ("job_id", "request_id", "event"):
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        if extras:
            parts.append(f"  ({', '.join(extras)})")

        if record.exc_info and record.exc_info[1] is not None:
            parts.append(f"  ERROR: {_sanitize(str(record.exc_info[1]))}")

        return "".join(parts)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: str = "",
) -> None:
    """
    Configure the root logging for the platform.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: 'json' or 'text'.
        log_file: Optional file path for log output.
    """
    root_logger = logging.getLogger("gpu_platform")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    formatter: logging.Formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    for noisy in ("uvicorn", "uvicorn.access", "sqlalchemy.engine", "httpx", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(component: str) -> logging.Logger:
    """
    Get a component-scoped logger.

    Usage:
        logger = get_logger("scheduler")
        logger.info("Job started", extra={"job_id": "abc123"})
    """
    return logging.getLogger(f"gpu_platform.{component}")
