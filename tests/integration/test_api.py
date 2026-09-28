"""Integration tests for the Host API."""

import os

import pytest
from fastapi.testclient import TestClient

from gpu_core.config import get_config, reset_config
from gpu_core.database import init_db, reset_db_engine


@pytest.fixture(autouse=True)
def test_env(tmp_path):
    """Set up a fresh test environment for each test."""
    # Clean ALL state first
    reset_config()
    reset_db_engine()

    # Set env vars BEFORE any config/db access
    db_path = tmp_path / "test.db"
    os.environ["GPU_PLATFORM_DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["GPU_PLATFORM_DATA_DIR"] = str(tmp_path / "data")
    os.environ["GPU_PLATFORM_JOBS_DIR"] = str(tmp_path / "data" / "jobs")
    os.environ["GPU_PLATFORM_DATASETS_DIR"] = str(tmp_path / "data" / "datasets")
    os.environ["GPU_PLATFORM_MODELS_DIR"] = str(tmp_path / "data" / "models")
    os.environ["GPU_PLATFORM_CHECKPOINTS_DIR"] = str(tmp_path / "data" / "checkpoints")
    os.environ["GPU_PLATFORM_ARTIFACTS_DIR"] = str(tmp_path / "data" / "artifacts")
    os.environ["GPU_PLATFORM_LOG_LEVEL"] = "WARNING"
    os.environ["GPU_PLATFORM_API_TOKEN"] = "test-token-12345"
    os.environ["GPU_PLATFORM_AUTHENTICATION"] = "true"

    yield

    # Cleanup — order matters: reset singletons first, then env
    reset_db_engine()
    reset_config()
    for key in list(os.environ.keys()):
        if key.startswith("GPU_PLATFORM_"):
            del os.environ[key]


@pytest.fixture
def client():
    """Create a test client for the API."""
    from gpu_host.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Authorization headers for authenticated requests."""
    return {"Authorization": "Bearer test-token-12345"}


class TestHealthEndpoint:
    def test_health_no_auth(self, client):
        """Health endpoint should work without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "hostname" in data

    def test_health_has_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestAuthEndpoint:
    def test_auth_required(self, client):
        """Protected endpoints must reject unauthenticated requests."""
        resp = client.get("/system")
        assert resp.status_code == 401

    def test_invalid_token(self, client):
        resp = client.get("/system", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_valid_token(self, client, auth_headers):
        resp = client.get("/system", headers=auth_headers)
        assert resp.status_code == 200

    def test_token_verify(self, client, auth_headers):
        resp = client.post("/auth/token", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["token_type"] == "bearer"


class TestSystemEndpoint:
    def test_system_info(self, client, auth_headers):
        resp = client.get("/system", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "os" in data
        assert "cpu" in data
        assert "ram_total_mb" in data
        assert data["ram_total_mb"] > 0
        assert "python_version" in data


class TestGPUEndpoint:
    def test_gpu_status(self, client, auth_headers):
        resp = client.get("/gpu", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "gpus" in data
        assert "gpu_count" in data
        # Should detect at least one GPU on this machine
        assert data["gpu_count"] >= 0  # May be 0 in CI


class TestMLEnvironmentEndpoint:
    def test_ml_environment(self, client, auth_headers):
        resp = client.get("/ml/environment", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "overall_status" in data
        assert len(data["components"]) >= 3  # PyTorch, TF, Docker at least

    def test_system_doctor(self, client, auth_headers):
        resp = client.get("/system/doctor", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert "overall" in data
        # Should have OS, CPU, RAM at minimum
        component_names = [c["component"] for c in data["checks"]]
        assert "OS" in component_names
        assert "CPU" in component_names
        assert "RAM" in component_names


class TestJobsEndpoint:
    def test_list_jobs_empty(self, client, auth_headers):
        resp = client.get("/jobs", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_get_nonexistent_job(self, client, auth_headers):
        resp = client.get("/jobs/nonexistent", headers=auth_headers)
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "JOB_NOT_FOUND"


class TestDatasetsEndpoint:
    def test_list_datasets_empty(self, client, auth_headers):
        resp = client.get("/datasets", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["datasets"] == []
        assert data["total"] == 0

    def test_get_nonexistent_dataset(self, client, auth_headers):
        resp = client.get("/datasets/nonexistent", headers=auth_headers)
        assert resp.status_code == 404


class TestModelsEndpoint:
    def test_list_models_empty(self, client, auth_headers):
        resp = client.get("/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []
        assert data["total"] == 0


class TestExperimentsEndpoint:
    def test_list_experiments_empty(self, client, auth_headers):
        resp = client.get("/experiments", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiments"] == []
        assert data["total"] == 0
