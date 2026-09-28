"""Unit tests for gpu_core.errors module."""

import pytest

from gpu_core.errors import (
    AuthenticationError,
    AuthorizationError,
    CheckpointNotFoundError,
    ConflictError,
    DatasetNotFoundError,
    ErrorCode,
    GPUError,
    GPUOutOfMemoryError,
    JobNotFoundError,
    ModelNotFoundError,
    NotFoundError,
    PlatformError,
    SecurityError,
    UploadTooLargeError,
    ValidationError,
)


class TestPlatformError:
    def test_basic_error(self):
        err = PlatformError(ErrorCode.INTERNAL_ERROR, "Something went wrong")
        assert err.code == ErrorCode.INTERNAL_ERROR
        assert err.message == "Something went wrong"
        assert err.status_code == 500
        assert err.details == {}

    def test_to_response(self):
        err = PlatformError(
            ErrorCode.JOB_NOT_FOUND,
            "Job not found",
            details={"job_id": "abc"},
            status_code=404,
        )
        resp = err.to_response()
        assert resp.error.code == "JOB_NOT_FOUND"
        assert resp.error.message == "Job not found"
        assert resp.error.details == {"job_id": "abc"}


class TestSpecificErrors:
    def test_auth_error(self):
        err = AuthenticationError()
        assert err.status_code == 401
        assert err.code == ErrorCode.AUTH_REQUIRED

    def test_authorization_error(self):
        err = AuthorizationError()
        assert err.status_code == 403

    def test_job_not_found(self):
        err = JobNotFoundError("abc123")
        assert err.status_code == 404
        assert "abc123" in err.message

    def test_dataset_not_found(self):
        err = DatasetNotFoundError("ds-001")
        assert err.status_code == 404
        assert "ds-001" in err.message

    def test_validation_error(self):
        err = ValidationError("Invalid batch size")
        assert err.status_code == 422

    def test_conflict_error(self):
        err = ConflictError("Dataset already exists")
        assert err.status_code == 409

    def test_gpu_oom(self):
        err = GPUOutOfMemoryError()
        assert err.status_code == 507
        assert err.code == ErrorCode.GPU_OOM
        assert "suggestions" in err.details
        assert len(err.details["suggestions"]) > 0

    def test_upload_too_large(self):
        err = UploadTooLargeError(size=200, limit=100)
        assert err.status_code == 413
        assert "200" in err.message
        assert "100" in err.message

    def test_security_error(self):
        err = SecurityError("Path traversal detected")
        assert err.status_code == 403
