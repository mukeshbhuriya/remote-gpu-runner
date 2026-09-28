"""
GPU detection and monitoring.

Detects NVIDIA GPUs via PyNVML. Designed so AMD ROCm can be
added later without changing the public interface.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

from gpu_core.logging_config import get_logger

logger = get_logger("gpu.detector")


@dataclass
class GPUDevice:
    """Represents a detected GPU device."""

    index: int
    name: str
    vendor: str  # "NVIDIA", "AMD", "Unknown"
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None


@dataclass
class GPUDetectionResult:
    """Result of GPU detection."""

    detected: bool = False
    devices: list[GPUDevice] = field(default_factory=list)
    vendor: str = "Unknown"
    driver_version: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    error: str | None = None


class GPUDetector:
    """
    Detects and monitors GPU hardware.

    Currently supports NVIDIA via PyNVML.
    AMD ROCm support can be added by implementing a ROCmDetector.
    """

    def __init__(self) -> None:
        self._nvml_initialized = False

    def detect(self) -> GPUDetectionResult:
        """Detect all available GPUs."""
        result = GPUDetectionResult()

        # Try NVIDIA first
        nvidia_result = self._detect_nvidia()
        if nvidia_result.detected:
            return nvidia_result

        # Try nvidia-smi fallback
        smi_result = self._detect_nvidia_smi()
        if smi_result.detected:
            return smi_result

        # No GPU detected
        result.error = "No supported GPU detected."
        logger.warning("No GPU detected on this system.")
        return result

    def get_live_status(self) -> GPUDetectionResult:
        """Get current GPU status with live metrics (utilization, temp, power)."""
        return self.detect()

    def _detect_nvidia(self) -> GPUDetectionResult:
        """Detect NVIDIA GPUs using PyNVML."""
        result = GPUDetectionResult()

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            logger.debug(f"PyNVML init failed: {e}")
            result.error = str(e)
            return result

        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            result.driver_version = driver_version
            result.vendor = "NVIDIA"

            try:
                cuda_version_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                major = cuda_version_raw // 1000
                minor = (cuda_version_raw % 1000) // 10
                result.cuda_version = f"{major}.{minor}"
            except Exception:
                result.cuda_version = self._get_cuda_version_fallback()

            device_count = pynvml.nvmlDeviceGetCount()
            result.detected = device_count > 0

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                device = GPUDevice(
                    index=i,
                    name=name,
                    vendor="NVIDIA",
                    vram_total_mb=mem_info.total // (1024 * 1024),
                    vram_used_mb=mem_info.used // (1024 * 1024),
                    vram_free_mb=mem_info.free // (1024 * 1024),
                    driver_version=driver_version,
                    cuda_version=result.cuda_version,
                )

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    device.utilization_percent = float(util.gpu)
                except Exception:
                    pass

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    device.temperature_c = float(temp)
                except Exception:
                    pass

                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)
                    device.power_watts = power / 1000.0  # milliwatts to watts
                except Exception:
                    pass

                # Compute capability
                try:
                    major_cc, minor_cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    device.compute_capability = f"{major_cc}.{minor_cc}"
                except Exception:
                    pass

                result.devices.append(device)

            logger.info(
                f"Detected {device_count} NVIDIA GPU(s): "
                f"{', '.join(d.name for d in result.devices)}"
            )

        except Exception as e:
            result.error = f"NVML error: {e}"
            logger.error(f"NVML detection error: {e}")

        finally:
            try:
                if self._nvml_initialized:
                    pynvml.nvmlShutdown()
                    self._nvml_initialized = False
            except Exception:
                pass

        return result

    def _detect_nvidia_smi(self) -> GPUDetectionResult:
        """Fallback: detect via nvidia-smi command."""
        result = GPUDetectionResult()

        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if output.returncode != 0:
                return result

            lines = output.stdout.strip().split("\n")
            result.detected = len(lines) > 0
            result.vendor = "NVIDIA"

            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue

                try:
                    device = GPUDevice(
                        index=int(parts[0]),
                        name=parts[1],
                        vendor="NVIDIA",
                        vram_total_mb=int(float(parts[2])),
                        vram_used_mb=int(float(parts[3])),
                        vram_free_mb=int(float(parts[4])),
                        utilization_percent=float(parts[5]) if parts[5] not in ("[N/A]", "N/A", "") else None,
                        temperature_c=float(parts[6]) if parts[6] not in ("[N/A]", "N/A", "") else None,
                        power_watts=float(parts[7]) if parts[7] not in ("[N/A]", "N/A", "") else None,
                        driver_version=parts[8] if parts[8] not in ("[N/A]", "N/A", "") else None,
                    )
                    result.driver_version = device.driver_version
                    result.devices.append(device)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi line: {line} — {e}")

            # Get CUDA version
            result.cuda_version = self._get_cuda_version_fallback()

        except FileNotFoundError:
            logger.debug("nvidia-smi not found.")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out.")
        except Exception as e:
            logger.debug(f"nvidia-smi fallback failed: {e}")

        return result

    def _get_cuda_version_fallback(self) -> str | None:
        """Get CUDA version from nvidia-smi header."""
        try:
            output = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if output.returncode == 0:
                for line in output.stdout.split("\n"):
                    if "CUDA Version" in line:
                        # Parse "CUDA Version: XX.Y"
                        import re
                        match = re.search(r'CUDA Version:\s*([\d.]+)', line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return None


# Module-level singleton
_detector: GPUDetector | None = None


def get_gpu_detector() -> GPUDetector:
    """Get or create the GPU detector singleton."""
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector
