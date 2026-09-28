"""
Pydantic schemas for API request/response validation.

These schemas define the public API contract and are separate from
the internal SQLAlchemy models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────
# Enums (mirrored from models for API use)
# ─────────────────────────────────────

class JobStatusSchema(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    RESUMING = "RESUMING"
    LOST = "LOST"


class FrameworkSchema(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    GENERIC = "generic"


class RuntimeModeSchema(str, Enum):
    SYSTEM = "system"
    VENV = "venv"
    DOCKER = "docker"


# ─────────────────────────────────────
# Health / System
# ─────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float
    hostname: str


class SystemInfoResponse(BaseModel):
    os: str
    os_version: str
    architecture: str
    hostname: str
    cpu: str
    cpu_cores: int
    ram_total_mb: int
    ram_available_mb: int
    storage_total_gb: float
    storage_free_gb: float
    python_version: str
    local_ip: str


# ─────────────────────────────────────
# GPU
# ─────────────────────────────────────

class GPUInfo(BaseModel):
    index: int
    name: str
    vendor: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    status: str = "IDLE"  # IDLE, TRAINING, INFERENCE, BUSY


class GPUStatusResponse(BaseModel):
    gpus: list[GPUInfo]
    gpu_count: int


# ─────────────────────────────────────
# ML Environment
# ─────────────────────────────────────

class MLComponentStatus(BaseModel):
    name: str
    installed: bool
    version: str | None = None
    gpu_support: bool | None = None
    status: str  # "PASS", "FAIL", "NOT_INSTALLED", "UNSUPPORTED"
    message: str = ""


class MLEnvironmentResponse(BaseModel):
    components: list[MLComponentStatus]
    overall_status: str  # "READY", "PARTIAL", "NOT_READY"
    overall_message: str = ""


# ─────────────────────────────────────
# Jobs
# ─────────────────────────────────────

class JobCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    framework: FrameworkSchema = FrameworkSchema.GENERIC
    runtime_mode: RuntimeModeSchema = RuntimeModeSchema.VENV
    entrypoint: str | None = None
    gpu: str = "auto"
    config: dict[str, Any] | None = None
    max_runtime_hours: float | None = None
    resume_checkpoint: str | None = None


class JobResponse(BaseModel):
    id: str
    name: str
    status: JobStatusSchema
    framework: FrameworkSchema
    runtime_mode: RuntimeModeSchema
    entrypoint: str | None = None
    gpu_device: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    exit_code: int | None = None
    error_message: str | None = None
    experiment_id: str | None = None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class JobLogEntry(BaseModel):
    timestamp: datetime
    stream: str
    message: str


class JobLogsResponse(BaseModel):
    job_id: str
    logs: list[JobLogEntry]
    total: int


class JobMetricEntry(BaseModel):
    timestamp: datetime
    name: str
    value: float
    step: int | None = None
    epoch: int | None = None


class JobMetricsResponse(BaseModel):
    job_id: str
    metrics: list[JobMetricEntry]


# ─────────────────────────────────────
# Datasets
# ─────────────────────────────────────

class DatasetResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    file_count: int | None = None
    format: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


# ─────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────

class CheckpointResponse(BaseModel):
    id: str
    job_id: str
    filename: str
    size_bytes: int | None = None
    epoch: int | None = None
    step: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CheckpointListResponse(BaseModel):
    job_id: str
    checkpoints: list[CheckpointResponse]
    total: int


# ─────────────────────────────────────
# Model Artifacts
# ─────────────────────────────────────

class ModelArtifactResponse(BaseModel):
    id: str
    job_id: str | None = None
    name: str
    framework: FrameworkSchema | None = None
    filename: str
    size_bytes: int | None = None
    format: str | None = None
    description: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    models: list[ModelArtifactResponse]
    total: int


# ─────────────────────────────────────
# Experiments
# ─────────────────────────────────────

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    framework: FrameworkSchema | None = None
    gpu_name: str | None = None
    gpu_backend: str | None = None
    python_version: str | None = None
    pytorch_version: str | None = None
    tensorflow_version: str | None = None
    dataset_name: str | None = None
    created_at: datetime
    job_count: int = 0

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    total: int


# ─────────────────────────────────────
# Benchmark
# ─────────────────────────────────────

class BenchmarkResult(BaseModel):
    framework: str
    test_name: str
    duration_ms: float
    gpu_name: str | None = None
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    message: str = ""


class BenchmarkResponse(BaseModel):
    gpu: GPUInfo | None = None
    results: list[BenchmarkResult]
    note: str = "Diagnostic benchmark only. Not a prediction of training performance."


# ─────────────────────────────────────
# Doctor / Diagnostics
# ─────────────────────────────────────

class DoctorCheck(BaseModel):
    component: str
    status: str  # "PASS", "FAIL", "WARNING", "UNSUPPORTED", "NOT_INSTALLED"
    message: str = ""
    detected: str | None = None
    expected: str | None = None
    fix: str | None = None


class DoctorResponse(BaseModel):
    checks: list[DoctorCheck]
    overall: str  # "READY FOR AI/ML TRAINING", "PARTIAL", "NOT READY"


# ─────────────────────────────────────
# Auth
# ─────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenRotateResponse(BaseModel):
    new_token: str
    message: str = "Token rotated. Use the new token for future requests."
