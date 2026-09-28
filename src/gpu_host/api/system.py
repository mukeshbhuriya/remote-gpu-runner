"""System information endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from gpu_core.ml import get_ml_detector
from gpu_core.schemas import SystemInfoResponse
from gpu_host.api.deps import verify_auth

router = APIRouter()


@router.get("/system", response_model=SystemInfoResponse)
async def get_system_info(_token: str = Depends(verify_auth)) -> SystemInfoResponse:
    """Get host system information (CPU, RAM, storage, network)."""
    detector = get_ml_detector()
    info = detector.detect_system()

    return SystemInfoResponse(
        os=info.os_name,
        os_version=info.os_version,
        architecture=info.architecture,
        hostname=info.hostname,
        cpu=info.cpu_name,
        cpu_cores=info.cpu_cores,
        ram_total_mb=info.ram_total_mb,
        ram_available_mb=info.ram_available_mb,
        storage_total_gb=info.storage_total_gb,
        storage_free_gb=info.storage_free_gb,
        python_version=info.python_version,
        local_ip=info.local_ip,
    )
