"""Health endpoint."""

from __future__ import annotations

import socket
import time

from fastapi import APIRouter

from gpu_core import __version__
from gpu_core.schemas import HealthResponse
from gpu_host.api.deps import get_start_time

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — no authentication required."""
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - get_start_time(), 1),
        hostname=socket.gethostname(),
    )
