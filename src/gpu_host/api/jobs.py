"""Jobs API endpoints."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from gpu_core.config import get_config
from gpu_core.errors import JobNotFoundError, ValidationError
from gpu_core.logging_config import get_logger
from gpu_core.models import Job, JobLog, JobMetric, JobStatus, FrameworkType, RuntimeMode
from gpu_core.schemas import (
    JobCreateRequest,
    JobListResponse,
    JobLogEntry,
    JobLogsResponse,
    JobMetricEntry,
    JobMetricsResponse,
    JobResponse,
    CheckpointListResponse,
    CheckpointResponse,
)
from gpu_core.security import sanitize_filename, validate_safe_path
from gpu_host.api.deps import verify_auth, get_session

logger = get_logger("host.api.jobs")

router = APIRouter()


def _job_to_response(job: Job) -> JobResponse:
    """Convert ORM Job to response schema."""
    return JobResponse.model_validate(job)


def _get_job_or_404(db: Session, job_id: str) -> Job:
    """Get a job by ID or raise 404."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise JobNotFoundError(job_id)
    return job


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobListResponse:
    """List all jobs, optionally filtered by status."""
    query = db.query(Job)
    if status:
        try:
            status_enum = JobStatus(status.upper())
            query = query.filter(Job.status == status_enum)
        except ValueError:
            pass
    total = query.count()
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
    )


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    project_archive: UploadFile = File(...),
    name: str = Form(...),
    framework: str = Form("generic"),
    runtime_mode: str = Form("venv"),
    entrypoint: str = Form("train.py"),
    gpu: str = Form("auto"),
    max_runtime_hours: float | None = Form(None),
    config_json: str | None = Form(None),
    resume_checkpoint: str | None = Form(None),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobResponse:
    """
    Submit a new job.

    Upload a project archive (zip) and create a managed job.
    """
    config = get_config()

    # Validate framework
    try:
        fw = FrameworkType(framework.lower())
    except ValueError:
        raise ValidationError(f"Invalid framework: {framework}. Use: pytorch, tensorflow, generic")

    # Validate runtime mode
    try:
        rm = RuntimeMode(runtime_mode.lower())
    except ValueError:
        raise ValidationError(f"Invalid runtime mode: {runtime_mode}. Use: system, venv, docker")

    # Create job record
    job = Job(
        name=name,
        framework=fw,
        runtime_mode=rm,
        entrypoint=entrypoint,
        gpu_device=gpu,
        max_runtime_hours=max_runtime_hours or config.scheduler.max_runtime_hours,
        config_yaml=config_json,
        status=JobStatus.UPLOADING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Create job workspace
    job_dir = Path(config.storage.jobs_dir) / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded archive
        archive_path = job_dir / "project.zip"
        with open(archive_path, "wb") as f:
            while chunk := await project_archive.read(8 * 1024 * 1024):
                f.write(chunk)

        # Extract project
        project_dir = job_dir / "project"
        project_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as zf:
            # Validate no path traversal in zip entries
            for zip_info in zf.infolist():
                target = (project_dir / zip_info.filename).resolve()
                if not str(target).startswith(str(project_dir.resolve())):
                    raise ValidationError(f"Zip contains path traversal: {zip_info.filename}")
            zf.extractall(project_dir)

        # Update job
        job.working_dir = str(project_dir)
        job.status = JobStatus.QUEUED
        db.commit()
        db.refresh(job)

        logger.info(f"Job {job.id} created and queued.", extra={"job_id": job.id})

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        db.commit()
        logger.error(f"Job {job.id} upload failed: {e}", extra={"job_id": job.id})
        raise

    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobResponse:
    """Get job details by ID."""
    job = _get_job_or_404(db, job_id)
    return _job_to_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobResponse:
    """Request job cancellation."""
    job = _get_job_or_404(db, job_id)

    cancellable = {JobStatus.QUEUED, JobStatus.STARTING, JobStatus.RUNNING}
    if job.status not in cancellable:
        raise ValidationError(
            f"Cannot cancel job in state {job.status.value}. "
            f"Cancellable states: {[s.value for s in cancellable]}"
        )

    job.status = JobStatus.CANCEL_REQUESTED
    db.commit()
    db.refresh(job)

    logger.info(f"Job {job.id} cancel requested.", extra={"job_id": job.id})
    return _job_to_response(job)


@router.post("/jobs/{job_id}/resume", response_model=JobResponse)
async def resume_job(
    job_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobResponse:
    """Resume a failed or cancelled job."""
    job = _get_job_or_404(db, job_id)

    resumable = {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.COMPLETED}
    if job.status not in resumable:
        raise ValidationError(
            f"Cannot resume job in state {job.status.value}. "
            f"Resumable states: {[s.value for s in resumable]}"
        )

    job.status = JobStatus.RESUMING
    db.commit()
    db.refresh(job)

    logger.info(f"Job {job.id} resume requested.", extra={"job_id": job.id})
    return _job_to_response(job)


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
async def get_job_logs(
    job_id: str,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    stream: str | None = Query(None, description="Filter by stream: stdout, stderr"),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobLogsResponse:
    """Get job logs."""
    _get_job_or_404(db, job_id)  # Verify job exists

    query = db.query(JobLog).filter(JobLog.job_id == job_id)
    if stream:
        query = query.filter(JobLog.stream == stream)

    total = query.count()
    logs = query.order_by(JobLog.timestamp.asc()).offset(offset).limit(limit).all()

    return JobLogsResponse(
        job_id=job_id,
        logs=[
            JobLogEntry(timestamp=log.timestamp, stream=log.stream, message=log.message)
            for log in logs
        ],
        total=total,
    )


@router.get("/jobs/{job_id}/metrics", response_model=JobMetricsResponse)
async def get_job_metrics(
    job_id: str,
    name: str | None = Query(None, description="Filter by metric name"),
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> JobMetricsResponse:
    """Get job training metrics."""
    _get_job_or_404(db, job_id)

    query = db.query(JobMetric).filter(JobMetric.job_id == job_id)
    if name:
        query = query.filter(JobMetric.name == name)

    metrics = query.order_by(JobMetric.timestamp.asc()).all()

    return JobMetricsResponse(
        job_id=job_id,
        metrics=[
            JobMetricEntry(
                timestamp=m.timestamp,
                name=m.name,
                value=m.value,
                step=m.step,
                epoch=m.epoch,
            )
            for m in metrics
        ],
    )


@router.get("/jobs/{job_id}/checkpoints", response_model=CheckpointListResponse)
async def get_job_checkpoints(
    job_id: str,
    _token: str = Depends(verify_auth),
    db: Session = Depends(get_session),
) -> CheckpointListResponse:
    """Get checkpoints for a job."""
    job = _get_job_or_404(db, job_id)

    from gpu_core.models import Checkpoint as CheckpointModel
    checkpoints = db.query(CheckpointModel).filter(
        CheckpointModel.job_id == job_id
    ).order_by(CheckpointModel.created_at.desc()).all()

    return CheckpointListResponse(
        job_id=job_id,
        checkpoints=[
            CheckpointResponse.model_validate(c) for c in checkpoints
        ],
        total=len(checkpoints),
    )
