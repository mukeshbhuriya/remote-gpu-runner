"""
FastAPI application for GPU Host.

This is the main API server that runs on GPU_HOST.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gpu_core.config import get_config
from gpu_core.database import init_db
from gpu_core.errors import PlatformError
from gpu_core.logging_config import get_logger, setup_logging
from gpu_host.api import health, system, gpu, ml_env, jobs, datasets, models_api, experiments, auth
from gpu_host.api.deps import lifespan

logger = get_logger("host.app")


def create_app(config_overrides: dict | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config(config_overrides)

    # Setup logging
    setup_logging(
        level=config.logging.log_level,
        log_format=config.logging.log_format,
        log_file=config.logging.log_file,
    )

    app = FastAPI(
        title="Remote ML GPU Platform",
        description="LAN AI/ML Remote GPU Compute Platform API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # LAN only; restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler for PlatformError
    @app.exception_handler(PlatformError)
    async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response().model_dump(),
        )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.debug(
            f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)"
        )
        return response

    # Register routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(system.router, tags=["System"])
    app.include_router(gpu.router, tags=["GPU"])
    app.include_router(ml_env.router, tags=["ML Environment"])
    app.include_router(jobs.router, tags=["Jobs"])
    app.include_router(datasets.router, tags=["Datasets"])
    app.include_router(models_api.router, tags=["Models"])
    app.include_router(experiments.router, tags=["Experiments"])

    return app
