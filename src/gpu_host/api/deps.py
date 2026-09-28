"""
FastAPI dependencies: auth, database session, lifespan.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, Header, FastAPI
from sqlalchemy.orm import Session

from gpu_core.config import get_config
from gpu_core.database import get_db, init_db
from gpu_core.errors import AuthenticationError
from gpu_core.logging_config import get_logger
from gpu_core.security import extract_bearer_token, verify_token, hash_token
from gpu_host.worker import start_worker, stop_worker

logger = get_logger("host.deps")

# Track server start time
_start_time: float = 0.0


def get_start_time() -> float:
    return _start_time


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    global _start_time
    _start_time = time.time()

    config = get_config()

    # Ensure storage directories exist
    config.ensure_storage()

    # Initialize database
    init_db()

    # Load or save secrets
    config.load_persisted_secrets()
    config.save_generated_secrets()

    logger.info(
        f"GPU Host started on {config.host.bind}:{config.host.port}",
        extra={"event": "startup"},
    )
    logger.info(f"API Token: {config.security.api_token[:8]}...  (first 8 chars)")

    # Start background worker
    start_worker()

    yield

    logger.info("GPU Host shutting down.", extra={"event": "shutdown"})
    stop_worker()


def get_session():  # type: ignore[no-untyped-def]
    """Get a database session (generator for Depends)."""
    yield from get_db()


async def verify_auth(
    authorization: str | None = Header(None, alias="Authorization"),
) -> str:
    """
    Verify the bearer token from the Authorization header.

    Returns the token if valid.
    Raises AuthenticationError if invalid.
    """
    config = get_config()
    if not config.security.authentication:
        return "no-auth"

    token = extract_bearer_token(authorization)

    # Compare against the configured API token
    if token == config.security.api_token:
        return token

    raise AuthenticationError("Invalid API token.")


# Dependency shortcuts
AuthDep = Depends(verify_auth)
DbDep = Depends(get_session)
