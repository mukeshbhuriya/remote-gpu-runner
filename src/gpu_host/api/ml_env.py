"""ML environment / doctor endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from gpu_core.gpu.detector import get_gpu_detector
from gpu_core.ml import get_ml_detector
from gpu_core.schemas import (
    DoctorCheck,
    DoctorResponse,
    MLComponentStatus,
    MLEnvironmentResponse,
)
from gpu_host.api.deps import verify_auth

router = APIRouter()


@router.get("/ml/environment", response_model=MLEnvironmentResponse)
async def get_ml_environment(_token: str = Depends(verify_auth)) -> MLEnvironmentResponse:
    """Get ML framework status: PyTorch, TensorFlow, GPU support."""
    detector = get_ml_detector()
    checks = detector.run_full_check()
    status, message = detector.get_overall_status(checks)

    components = [
        MLComponentStatus(
            name=c.name,
            installed=c.installed,
            version=c.version,
            gpu_support=c.gpu_support,
            status=c.status,
            message=c.message,
        )
        for c in checks
    ]

    return MLEnvironmentResponse(
        components=components,
        overall_status=status,
        overall_message=message,
    )


@router.get("/system/doctor", response_model=DoctorResponse)
async def system_doctor(_token: str = Depends(verify_auth)) -> DoctorResponse:
    """
    Comprehensive system health check.

    Checks OS, CPU, RAM, GPU, CUDA/ROCm, PyTorch, TensorFlow,
    Jupyter, TensorBoard, Docker, Network, Storage.
    """
    ml_detector = get_ml_detector()
    gpu_detector = get_gpu_detector()

    results: list[DoctorCheck] = []

    # System info
    sys_info = ml_detector.detect_system()
    results.append(DoctorCheck(
        component="OS",
        status="PASS",
        detected=f"{sys_info.os_name} {sys_info.os_version}",
    ))
    results.append(DoctorCheck(
        component="CPU",
        status="PASS",
        detected=f"{sys_info.cpu_name} ({sys_info.cpu_cores} cores)",
    ))
    results.append(DoctorCheck(
        component="RAM",
        status="PASS" if sys_info.ram_total_mb > 4096 else "WARNING",
        detected=f"{sys_info.ram_total_mb} MB",
        message="" if sys_info.ram_total_mb > 4096 else "Low RAM may limit training.",
    ))

    # GPU
    gpu_result = gpu_detector.detect()
    if gpu_result.detected:
        for dev in gpu_result.devices:
            results.append(DoctorCheck(
                component="GPU",
                status="PASS",
                detected=f"{dev.name} ({dev.vram_total_mb} MB VRAM)",
            ))
        results.append(DoctorCheck(
            component="GPU Driver",
            status="PASS",
            detected=gpu_result.driver_version or "N/A",
        ))
        if gpu_result.cuda_version:
            results.append(DoctorCheck(
                component="CUDA/ROCm",
                status="PASS",
                detected=f"CUDA {gpu_result.cuda_version}",
            ))
        elif gpu_result.rocm_version:
            results.append(DoctorCheck(
                component="CUDA/ROCm",
                status="PASS",
                detected=f"ROCm {gpu_result.rocm_version}",
            ))
        else:
            results.append(DoctorCheck(
                component="CUDA/ROCm",
                status="FAIL",
                message="No CUDA or ROCm detected.",
                fix="Install CUDA toolkit or ROCm.",
            ))
    else:
        results.append(DoctorCheck(
            component="GPU",
            status="FAIL",
            message=gpu_result.error or "No GPU detected.",
            fix="Ensure GPU drivers are installed.",
        ))

    # ML frameworks
    ml_checks = ml_detector.run_full_check()
    for mc in ml_checks:
        results.append(DoctorCheck(
            component=mc.name,
            status=mc.status,
            detected=mc.version,
            message=mc.message,
        ))
        # Add separate GPU check for frameworks
        if mc.name in ("PyTorch", "TensorFlow") and mc.installed:
            gpu_label = f"{mc.name} GPU"
            results.append(DoctorCheck(
                component=gpu_label,
                status="PASS" if mc.gpu_support else "FAIL",
                message=mc.message if not mc.gpu_support else "",
                fix=f"Install {mc.name} with GPU support." if not mc.gpu_support else None,
            ))

    # Network
    results.append(DoctorCheck(
        component="Network",
        status="PASS",
        detected=sys_info.local_ip,
    ))

    # Storage
    storage_ok = sys_info.storage_free_gb > 10
    results.append(DoctorCheck(
        component="Storage",
        status="PASS" if storage_ok else "WARNING",
        detected=f"{sys_info.storage_free_gb} GB free / {sys_info.storage_total_gb} GB total",
        message="" if storage_ok else "Low disk space may cause issues.",
    ))

    # Determine overall
    statuses = [r.status for r in results]
    if all(s in ("PASS", "NOT_INSTALLED") for s in statuses):
        overall = "READY FOR AI/ML TRAINING"
    elif any(s == "FAIL" for s in statuses):
        overall = "NOT READY"
    else:
        overall = "PARTIAL"

    return DoctorResponse(checks=results, overall=overall)
