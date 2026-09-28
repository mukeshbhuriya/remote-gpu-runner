"""
SQLAlchemy ORM models for the Remote ML GPU Platform.

Minimum entities as specified:
- Credential (API tokens / users)
- Job
- JobLog
- JobMetric
- Dataset
- Checkpoint
- ModelArtifact
- Experiment
- EnvironmentSnapshot
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from gpu_core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ─────────────────────────────────────
# Enums
# ─────────────────────────────────────

class JobStatus(str, enum.Enum):
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


class FrameworkType(str, enum.Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    GENERIC = "generic"


class RuntimeMode(str, enum.Enum):
    SYSTEM = "system"
    VENV = "venv"
    DOCKER = "docker"


# ─────────────────────────────────────
# Credential
# ─────────────────────────────────────

class Credential(Base):
    __tablename__ = "credentials"

    id = Column(String(12), primary_key=True, default=_new_id)
    name = Column(String(255), nullable=False, unique=True)
    token_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Credential(id={self.id}, name={self.name})>"


# ─────────────────────────────────────
# Job
# ─────────────────────────────────────

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(12), primary_key=True, default=_new_id)
    name = Column(String(255), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.CREATED, nullable=False)
    framework = Column(Enum(FrameworkType), default=FrameworkType.GENERIC, nullable=False)
    runtime_mode = Column(Enum(RuntimeMode), default=RuntimeMode.VENV, nullable=False)

    # Execution details
    entrypoint = Column(String(512), nullable=True)
    working_dir = Column(String(1024), nullable=True)
    gpu_device = Column(String(64), nullable=True)
    pid = Column(Integer, nullable=True)

    # Configuration
    config_yaml = Column(Text, nullable=True)  # Stored job config
    environment_vars = Column(Text, nullable=True)  # JSON string
    requirements_file = Column(String(512), nullable=True)

    # Timing
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Results
    exit_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Resource limits
    max_runtime_hours = Column(Float, nullable=True)

    # Relationships
    logs = relationship("JobLog", back_populates="job", cascade="all, delete-orphan")
    metrics = relationship("JobMetric", back_populates="job", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="job", cascade="all, delete-orphan")
    artifacts = relationship("ModelArtifact", back_populates="job", cascade="all, delete-orphan")

    # Link to experiment
    experiment_id = Column(String(12), ForeignKey("experiments.id"), nullable=True)
    experiment = relationship("Experiment", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, name={self.name}, status={self.status})>"


# ─────────────────────────────────────
# JobLog
# ─────────────────────────────────────

class JobLog(Base):
    __tablename__ = "job_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(12), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    stream = Column(String(10), default="stdout", nullable=False)  # stdout, stderr
    message = Column(Text, nullable=False)

    job = relationship("Job", back_populates="logs")


# ─────────────────────────────────────
# JobMetric
# ─────────────────────────────────────

class JobMetric(Base):
    __tablename__ = "job_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(12), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    name = Column(String(255), nullable=False)   # e.g., "loss", "accuracy", "lr"
    value = Column(Float, nullable=False)
    step = Column(Integer, nullable=True)
    epoch = Column(Integer, nullable=True)

    job = relationship("Job", back_populates="metrics")


# ─────────────────────────────────────
# Dataset
# ─────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(12), primary_key=True, default=_new_id)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    path = Column(String(1024), nullable=False)
    size_bytes = Column(Integer, nullable=True)
    checksum = Column(String(128), nullable=True)  # SHA-256
    file_count = Column(Integer, nullable=True)
    format = Column(String(64), nullable=True)  # e.g., "directory", "tar.gz", "zip"
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", name="uq_dataset_name"),
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name={self.name})>"


# ─────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────

class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(String(12), primary_key=True, default=_new_id)
    job_id = Column(String(12), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
    size_bytes = Column(Integer, nullable=True)
    epoch = Column(Integer, nullable=True)
    step = Column(Integer, nullable=True)
    metrics_json = Column(Text, nullable=True)  # JSON of metrics at checkpoint time
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    job = relationship("Job", back_populates="checkpoints")

    def __repr__(self) -> str:
        return f"<Checkpoint(id={self.id}, job_id={self.job_id}, filename={self.filename})>"


# ─────────────────────────────────────
# ModelArtifact
# ─────────────────────────────────────

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id = Column(String(12), primary_key=True, default=_new_id)
    job_id = Column(String(12), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    framework = Column(Enum(FrameworkType), nullable=True)
    filename = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
    size_bytes = Column(Integer, nullable=True)
    format = Column(String(64), nullable=True)  # .pt, .pth, .keras, SavedModel
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    job = relationship("Job", back_populates="artifacts")

    def __repr__(self) -> str:
        return f"<ModelArtifact(id={self.id}, name={self.name})>"


# ─────────────────────────────────────
# Experiment
# ─────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(12), primary_key=True, default=_new_id)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    framework = Column(Enum(FrameworkType), nullable=True)

    # Environment snapshot at creation
    gpu_name = Column(String(255), nullable=True)
    gpu_vram_mb = Column(Integer, nullable=True)
    gpu_backend = Column(String(64), nullable=True)  # CUDA, ROCm
    python_version = Column(String(32), nullable=True)
    pytorch_version = Column(String(32), nullable=True)
    tensorflow_version = Column(String(32), nullable=True)
    cuda_version = Column(String(32), nullable=True)
    os_info = Column(String(255), nullable=True)
    git_commit = Column(String(64), nullable=True)

    # Dataset
    dataset_name = Column(String(255), nullable=True)
    dataset_id = Column(String(12), nullable=True)

    # Hyperparameters (JSON)
    hyperparameters_json = Column(Text, nullable=True)

    # Timing
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    jobs = relationship("Job", back_populates="experiment")

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name})>"


# ─────────────────────────────────────
# EnvironmentSnapshot
# ─────────────────────────────────────

class EnvironmentSnapshot(Base):
    __tablename__ = "environment_snapshots"

    id = Column(String(12), primary_key=True, default=_new_id)
    captured_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # System
    os_name = Column(String(64), nullable=True)
    os_version = Column(String(128), nullable=True)
    architecture = Column(String(32), nullable=True)
    hostname = Column(String(255), nullable=True)
    cpu_name = Column(String(255), nullable=True)
    cpu_cores = Column(Integer, nullable=True)
    ram_total_mb = Column(Integer, nullable=True)

    # GPU
    gpu_name = Column(String(255), nullable=True)
    gpu_vendor = Column(String(64), nullable=True)
    gpu_vram_mb = Column(Integer, nullable=True)
    gpu_driver = Column(String(64), nullable=True)
    cuda_version = Column(String(32), nullable=True)
    rocm_version = Column(String(32), nullable=True)

    # ML
    python_version = Column(String(32), nullable=True)
    pytorch_version = Column(String(32), nullable=True)
    pytorch_cuda = Column(Boolean, nullable=True)
    tensorflow_version = Column(String(32), nullable=True)
    tensorflow_gpu = Column(Boolean, nullable=True)

    # Tools
    docker_version = Column(String(32), nullable=True)
    jupyter_version = Column(String(32), nullable=True)
    tensorboard_version = Column(String(32), nullable=True)

    # Network
    local_ip = Column(String(45), nullable=True)

    # Storage
    storage_total_gb = Column(Float, nullable=True)
    storage_free_gb = Column(Float, nullable=True)

    # Full report (JSON)
    full_report_json = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<EnvironmentSnapshot(id={self.id}, captured_at={self.captured_at})>"
