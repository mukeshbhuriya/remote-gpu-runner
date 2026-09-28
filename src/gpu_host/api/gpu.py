"""GPU status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from gpu_core.gpu.detector import get_gpu_detector
from gpu_core.schemas import GPUInfo, GPUStatusResponse
from gpu_host.api.deps import verify_auth

router = APIRouter()


@router.get("/gpu", response_model=GPUStatusResponse)
async def get_gpu_status(_token: str = Depends(verify_auth)) -> GPUStatusResponse:
    """Get GPU hardware status: utilization, VRAM, temperature, power."""
    detector = get_gpu_detector()
    result = detector.get_live_status()

    gpus = []
    for device in result.devices:
        gpus.append(GPUInfo(
            index=device.index,
            name=device.name,
            vendor=device.vendor,
            vram_total_mb=device.vram_total_mb,
            vram_used_mb=device.vram_used_mb,
            vram_free_mb=device.vram_free_mb,
            utilization_percent=device.utilization_percent,
            temperature_c=device.temperature_c,
            power_watts=device.power_watts,
            driver_version=device.driver_version,
            cuda_version=device.cuda_version,
        ))

    return GPUStatusResponse(
        gpus=gpus,
        gpu_count=len(gpus),
    )
