"""
HTTP client for communicating with the GPU Host API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from gpu_core.errors import (
    AuthenticationError,
    PlatformError,
    ErrorCode,
)
from gpu_core.logging_config import get_logger

logger = get_logger("client.http")


class GPUHostClient:
    """
    HTTP client for the GPU Host API.

    Handles authentication, error mapping, and streaming.
    """

    def __init__(self, host: str, port: int, token: str, timeout: float = 30.0) -> None:
        self.base_url = f"http://{host}:{port}"
        self.token = token
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response, raising appropriate errors."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired token.")

        if response.status_code >= 400:
            try:
                data = response.json()
                error = data.get("error", {})
                raise PlatformError(
                    code=ErrorCode(error.get("code", "INTERNAL_ERROR")),
                    message=error.get("message", response.text),
                    status_code=response.status_code,
                    details=error.get("details", {}),
                )
            except (ValueError, KeyError):
                raise PlatformError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

        if response.status_code == 204:
            return {}

        return response.json()

    # ─── API Methods ───────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check host health (no auth needed)."""
        resp = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        return self._handle_response(resp)

    def system_info(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/system"))

    def gpu_status(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/gpu"))

    def ml_environment(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/ml/environment"))

    def doctor(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/system/doctor"))

    def benchmark(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/benchmark"))

    # Jobs
    def list_jobs(self, status: str | None = None) -> dict[str, Any]:
        params = {}
        if status:
            params["status"] = status
        return self._handle_response(self.client.get("/jobs", params=params))

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.get(f"/jobs/{job_id}"))

    def submit_job(
        self,
        archive_path: Path,
        name: str,
        framework: str = "generic",
        runtime_mode: str = "venv",
        entrypoint: str = "train.py",
        gpu: str = "auto",
        max_runtime_hours: float | None = None,
        config_json: str | None = None,
        resume_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Submit a job by uploading a project archive."""
        data = {
            "name": name,
            "framework": framework,
            "runtime_mode": runtime_mode,
            "entrypoint": entrypoint,
            "gpu": gpu,
        }
        if max_runtime_hours:
            data["max_runtime_hours"] = str(max_runtime_hours)
        if config_json:
            data["config_json"] = config_json
        if resume_checkpoint:
            data["resume_checkpoint"] = resume_checkpoint

        with open(archive_path, "rb") as f:
            files = {"project_archive": (archive_path.name, f, "application/zip")}
            resp = self.client.post("/jobs", data=data, files=files, timeout=300.0)

        return self._handle_response(resp)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.post(f"/jobs/{job_id}/cancel"))

    def resume_job(self, job_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.post(f"/jobs/{job_id}/resume"))

    def job_logs(self, job_id: str, limit: int = 500) -> dict[str, Any]:
        return self._handle_response(
            self.client.get(f"/jobs/{job_id}/logs", params={"limit": limit})
        )

    def job_metrics(self, job_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.get(f"/jobs/{job_id}/metrics"))

    def job_checkpoints(self, job_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.get(f"/jobs/{job_id}/checkpoints"))

    # Datasets
    def list_datasets(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/datasets"))

    def upload_dataset(self, file_path: Path, name: str, description: str = "") -> dict[str, Any]:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {"name": name, "description": description}
            resp = self.client.post("/datasets", data=data, files=files, timeout=600.0)
        return self._handle_response(resp)

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._handle_response(self.client.delete(f"/datasets/{dataset_id}"))

    # Models
    def list_models(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/models"))

    def download_model(self, model_id: str, output_path: Path) -> Path:
        resp = self.client.get(f"/models/{model_id}/download", timeout=600.0)
        if resp.status_code >= 400:
            self._handle_response(resp)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return output_path

    # Experiments
    def list_experiments(self) -> dict[str, Any]:
        return self._handle_response(self.client.get("/experiments"))

    # Auth
    def verify_token(self) -> dict[str, Any]:
        return self._handle_response(self.client.post("/auth/token"))

    def rotate_token(self) -> dict[str, Any]:
        return self._handle_response(self.client.post("/auth/rotate"))
