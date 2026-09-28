"""Unit tests for gpu_core.database and gpu_core.models."""

import os

import pytest

from gpu_core.config import reset_config
from gpu_core.database import Base, get_engine, get_session_factory, init_db, reset_db_engine
from gpu_core.models import (
    Credential,
    Job,
    JobLog,
    JobMetric,
    JobStatus,
    FrameworkType,
    RuntimeMode,
    Dataset,
    Checkpoint,
    ModelArtifact,
    Experiment,
    EnvironmentSnapshot,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Create a fresh in-memory database for each test."""
    reset_config()
    reset_db_engine()
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ["GPU_PLATFORM_DATABASE_URL"] = db_url
    init_db(db_url)
    yield
    reset_db_engine()
    reset_config()
    os.environ.pop("GPU_PLATFORM_DATABASE_URL", None)


class TestDatabaseInit:
    def test_tables_created(self):
        engine = get_engine()
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        expected = [
            "credentials", "jobs", "job_logs", "job_metrics",
            "datasets", "checkpoints", "model_artifacts",
            "experiments", "environment_snapshots",
        ]
        for table in expected:
            assert table in tables, f"Missing table: {table}"


class TestJobModel:
    def test_create_job(self):
        factory = get_session_factory()
        session = factory()
        try:
            job = Job(
                name="test-training",
                framework=FrameworkType.PYTORCH,
                entrypoint="train.py",
            )
            session.add(job)
            session.commit()

            assert job.id is not None
            assert len(job.id) == 12
            assert job.status == JobStatus.CREATED
            assert job.framework == FrameworkType.PYTORCH
            assert job.created_at is not None
        finally:
            session.close()

    def test_job_status_transitions(self):
        factory = get_session_factory()
        session = factory()
        try:
            job = Job(name="test-job")
            session.add(job)
            session.commit()

            # Transition through states
            for status in [
                JobStatus.QUEUED,
                JobStatus.STARTING,
                JobStatus.RUNNING,
                JobStatus.COMPLETED,
            ]:
                job.status = status
                session.commit()
                session.refresh(job)
                assert job.status == status
        finally:
            session.close()

    def test_job_logs_relationship(self):
        factory = get_session_factory()
        session = factory()
        try:
            job = Job(name="test-job")
            session.add(job)
            session.commit()

            log1 = JobLog(job_id=job.id, stream="stdout", message="Starting training...")
            log2 = JobLog(job_id=job.id, stream="stderr", message="Warning: something")
            session.add_all([log1, log2])
            session.commit()

            session.refresh(job)
            assert len(job.logs) == 2
        finally:
            session.close()

    def test_job_metrics_relationship(self):
        factory = get_session_factory()
        session = factory()
        try:
            job = Job(name="test-job")
            session.add(job)
            session.commit()

            metric = JobMetric(
                job_id=job.id,
                name="loss",
                value=0.5,
                step=100,
                epoch=1,
            )
            session.add(metric)
            session.commit()

            session.refresh(job)
            assert len(job.metrics) == 1
            assert job.metrics[0].name == "loss"
            assert job.metrics[0].value == 0.5
        finally:
            session.close()


class TestDatasetModel:
    def test_create_dataset(self):
        factory = get_session_factory()
        session = factory()
        try:
            ds = Dataset(
                name="cifar10",
                path="/data/datasets/cifar10",
                size_bytes=170_000_000,
                format="directory",
            )
            session.add(ds)
            session.commit()

            assert ds.id is not None
            assert ds.name == "cifar10"
            assert ds.created_at is not None
        finally:
            session.close()

    def test_unique_name_constraint(self):
        factory = get_session_factory()
        session = factory()
        try:
            ds1 = Dataset(name="unique-ds", path="/data/ds1")
            ds2 = Dataset(name="unique-ds", path="/data/ds2")
            session.add(ds1)
            session.commit()
            session.add(ds2)
            with pytest.raises(Exception):  # IntegrityError
                session.commit()
            session.rollback()
        finally:
            session.close()


class TestCheckpointModel:
    def test_create_checkpoint(self):
        factory = get_session_factory()
        session = factory()
        try:
            job = Job(name="test-job")
            session.add(job)
            session.commit()

            ckpt = Checkpoint(
                job_id=job.id,
                filename="checkpoint_epoch_5.pt",
                path="/data/checkpoints/checkpoint_epoch_5.pt",
                epoch=5,
                step=5000,
            )
            session.add(ckpt)
            session.commit()

            session.refresh(job)
            assert len(job.checkpoints) == 1
            assert job.checkpoints[0].epoch == 5
        finally:
            session.close()


class TestExperimentModel:
    def test_experiment_with_jobs(self):
        factory = get_session_factory()
        session = factory()
        try:
            exp = Experiment(
                name="resnet-experiment",
                framework=FrameworkType.PYTORCH,
                gpu_name="RTX 2000 Ada",
                python_version="3.14.6",
            )
            session.add(exp)
            session.commit()

            job = Job(
                name="run-1",
                experiment_id=exp.id,
                framework=FrameworkType.PYTORCH,
            )
            session.add(job)
            session.commit()

            session.refresh(exp)
            assert len(exp.jobs) == 1
            assert exp.jobs[0].name == "run-1"
        finally:
            session.close()


class TestCredentialModel:
    def test_create_credential(self):
        factory = get_session_factory()
        session = factory()
        try:
            cred = Credential(
                name="admin",
                token_hash="abc123hash",
            )
            session.add(cred)
            session.commit()

            assert cred.id is not None
            assert cred.is_active is True
        finally:
            session.close()
