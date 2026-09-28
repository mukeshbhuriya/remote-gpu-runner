"""
Structured error handling for the Remote ML GPU Platform.

Provides typed exception hierarchy and standardized API error responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standardized error codes used throughout the platform."""

    # General
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"

    # Auth
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_FORBIDDEN = "AUTH_FORBIDDEN"

    # Jobs
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_ALREADY_RUNNING = "JOB_ALREADY_RUNNING"
    JOB_CANNOT_CANCEL = "JOB_CANNOT_CANCEL"
    JOB_CANNOT_RESUME = "JOB_CANNOT_RESUME"
    JOB_UPLOAD_FAILED = "JOB_UPLOAD_FAILED"
    JOB_EXECUTION_FAILED = "JOB_EXECUTION_FAILED"
    JOB_LIMIT_REACHED = "JOB_LIMIT_REACHED"

    # Datasets
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    DATASET_ALREADY_EXISTS = "DATASET_ALREADY_EXISTS"
    DATASET_UPLOAD_FAILED = "DATASET_UPLOAD_FAILED"

    # Models / Checkpoints
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"

    # GPU
    GPU_NOT_AVAILABLE = "GPU_NOT_AVAILABLE"
    GPU_OOM = "GPU_OOM"
    GPU_DETECTION_FAILED = "GPU_DETECTION_FAILED"

    # ML Environment
    ML_ENV_ERROR = "ML_ENV_ERROR"
    ML_FRAMEWORK_NOT_FOUND = "ML_FRAMEWORK_NOT_FOUND"
    ML_INCOMPATIBLE = "ML_INCOMPATIBLE"

    # Security
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"
    INVALID_FILENAME = "INVALID_FILENAME"

    # Runtime
    DOCKER_NOT_AVAILABLE = "DOCKER_NOT_AVAILABLE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"


class ErrorDetail(BaseModel):
    """Standardized API error response body."""

    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Wrapper for API error responses."""

    error: ErrorDetail


class PlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int = 500,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_response(self) -> ErrorResponse:
        """Convert to a standardized API error response."""
        return ErrorResponse(
            error=ErrorDetail(
                code=self.code.value,
                message=self.message,
                details=self.details,
            )
        )


# --- Specific Exception Classes ---


class AuthenticationError(PlatformError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required.", **kwargs: Any) -> None:
        super().__init__(ErrorCode.AUTH_REQUIRED, message, status_code=401, **kwargs)


class AuthorizationError(PlatformError):
    """Raised when a user lacks permission."""

    def __init__(self, message: str = "Access forbidden.", **kwargs: Any) -> None:
        super().__init__(ErrorCode.AUTH_FORBIDDEN, message, status_code=403, **kwargs)


class NotFoundError(PlatformError):
    """Raised when a requested resource is not found."""

    def __init__(
        self, resource: str = "Resource", identifier: str = "", **kwargs: Any
    ) -> None:
        msg = f"{resource} not found."
        if identifier:
            msg = f"{resource} '{identifier}' not found."
        super().__init__(ErrorCode.NOT_FOUND, msg, status_code=404, **kwargs)


class JobNotFoundError(NotFoundError):
    """Raised when a job is not found."""

    def __init__(self, job_id: str, **kwargs: Any) -> None:
        super().__init__("Job", job_id, **kwargs)
        self.code = ErrorCode.JOB_NOT_FOUND


class DatasetNotFoundError(NotFoundError):
    """Raised when a dataset is not found."""

    def __init__(self, dataset_id: str, **kwargs: Any) -> None:
        super().__init__("Dataset", dataset_id, **kwargs)
        self.code = ErrorCode.DATASET_NOT_FOUND


class ModelNotFoundError(NotFoundError):
    """Raised when a model artifact is not found."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__("Model", model_id, **kwargs)
        self.code = ErrorCode.MODEL_NOT_FOUND


class CheckpointNotFoundError(NotFoundError):
    """Raised when a checkpoint is not found."""

    def __init__(self, checkpoint_id: str, **kwargs: Any) -> None:
        super().__init__("Checkpoint", checkpoint_id, **kwargs)
        self.code = ErrorCode.CHECKPOINT_NOT_FOUND


class ValidationError(PlatformError):
    """Raised when request validation fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(ErrorCode.VALIDATION_ERROR, message, status_code=422, **kwargs)


class ConflictError(PlatformError):
    """Raised when a resource conflict occurs."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(ErrorCode.CONFLICT, message, status_code=409, **kwargs)


class GPUError(PlatformError):
    """Raised when GPU operations fail."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.GPU_NOT_AVAILABLE, **kwargs: Any) -> None:
        super().__init__(code, message, status_code=503, **kwargs)


class GPUOutOfMemoryError(GPUError):
    """Raised when CUDA OOM is detected."""

    def __init__(self, message: str = "CUDA out of memory.", **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details.setdefault("suggestions", [
            "Reduce batch size",
            "Use mixed precision (AMP)",
            "Use gradient accumulation",
            "Reduce model size",
            "Inspect GPU memory usage with `gpu-client gpu`",
            "Check for other processes using GPU memory",
        ])
        super().__init__(message, code=ErrorCode.GPU_OOM, details=details, **kwargs)
        self.status_code = 507


class SecurityError(PlatformError):
    """Raised for security violations."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.PATH_TRAVERSAL, **kwargs: Any) -> None:
        super().__init__(code, message, status_code=403, **kwargs)


class UploadTooLargeError(PlatformError):
    """Raised when an upload exceeds the size limit."""

    def __init__(self, size: int, limit: int, **kwargs: Any) -> None:
        msg = f"Upload size ({size} bytes) exceeds limit ({limit} bytes)."
        super().__init__(ErrorCode.UPLOAD_TOO_LARGE, msg, status_code=413, **kwargs)


class RuntimeExecutionError(PlatformError):
    """Raised when job runtime execution fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(ErrorCode.RUNTIME_ERROR, message, status_code=500, **kwargs)
