"""
ML environment detection — detects installed ML frameworks and tools.

Checks PyTorch, TensorFlow, Jupyter, TensorBoard, Docker, and their
GPU capabilities by running actual detection logic, not just imports.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

import psutil

from gpu_core.gpu.detector import get_gpu_detector
from gpu_core.logging_config import get_logger

logger = get_logger("ml.detector")


@dataclass
class ComponentCheck:
    """Result of checking a single ML component."""
    name: str
    installed: bool = False
    version: str | None = None
    gpu_support: bool | None = None
    status: str = "NOT_INSTALLED"  # PASS, FAIL, NOT_INSTALLED, UNSUPPORTED
    message: str = ""


@dataclass
class SystemInfo:
    """Full system environment information."""
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    hostname: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    storage_total_gb: float = 0.0
    storage_free_gb: float = 0.0
    python_version: str = ""
    local_ip: str = ""


class MLEnvironmentDetector:
    """Detects ML frameworks, tools, and system information."""

    def detect_system(self) -> SystemInfo:
        """Detect system-level information."""
        info = SystemInfo()
        info.os_name = platform.system()
        info.os_version = platform.version()
        info.architecture = platform.machine()
        info.hostname = socket.gethostname()
        info.python_version = platform.python_version()

        # CPU
        try:
            if platform.system() == "Windows":
                info.cpu_name = platform.processor()
                # Try more detailed name via subprocess
                try:
                    result = subprocess.run(
                        ["powershell", "-Command",
                         "(Get-CimInstance Win32_Processor).Name"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        info.cpu_name = result.stdout.strip()
                except Exception:
                    pass
            else:
                try:
                    with open("/proc/cpuinfo") as f:
                        for line in f:
                            if "model name" in line:
                                info.cpu_name = line.split(":")[1].strip()
                                break
                except Exception:
                    info.cpu_name = platform.processor()
        except Exception:
            info.cpu_name = "Unknown"

        info.cpu_cores = psutil.cpu_count(logical=False) or 0
        info.cpu_threads = psutil.cpu_count(logical=True) or 0

        # RAM
        mem = psutil.virtual_memory()
        info.ram_total_mb = int(mem.total / (1024 * 1024))
        info.ram_available_mb = int(mem.available / (1024 * 1024))

        # Storage (for the current drive)
        try:
            disk = psutil.disk_usage(".")
            info.storage_total_gb = round(disk.total / (1024**3), 1)
            info.storage_free_gb = round(disk.free / (1024**3), 1)
        except Exception:
            pass

        # Local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            info.local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            info.local_ip = "127.0.0.1"

        return info

    def check_pytorch(self) -> ComponentCheck:
        """Check PyTorch installation and GPU support."""
        check = ComponentCheck(name="PyTorch")
        try:
            torch = importlib.import_module("torch")
            check.installed = True
            check.version = torch.__version__

            if torch.cuda.is_available():
                check.gpu_support = True
                device_name = torch.cuda.get_device_name(0)
                check.status = "PASS"
                check.message = f"CUDA GPU: {device_name}"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                check.gpu_support = True
                check.status = "PASS"
                check.message = "MPS GPU available (Apple Silicon)"
            else:
                check.gpu_support = False
                check.status = "FAIL"
                check.message = "PyTorch installed but no GPU support detected."

        except ImportError:
            check.status = "NOT_INSTALLED"
            check.message = "PyTorch is not installed."
        except Exception as e:
            check.status = "FAIL"
            check.message = f"PyTorch detection error: {e}"

        return check

    def check_tensorflow(self) -> ComponentCheck:
        """Check TensorFlow installation and GPU support."""
        check = ComponentCheck(name="TensorFlow")
        try:
            tf = importlib.import_module("tensorflow")
            check.installed = True
            check.version = tf.__version__

            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                check.gpu_support = True
                check.status = "PASS"
                check.message = f"GPU devices: {[d.name for d in gpu_devices]}"
            else:
                check.gpu_support = False
                check.status = "FAIL"
                check.message = "TensorFlow installed but no GPU devices detected."

        except ImportError:
            check.status = "NOT_INSTALLED"
            check.message = "TensorFlow is not installed."
        except Exception as e:
            check.status = "FAIL"
            check.message = f"TensorFlow detection error: {e}"

        return check

    def check_jupyter(self) -> ComponentCheck:
        """Check Jupyter installation."""
        check = ComponentCheck(name="Jupyter")
        try:
            jupyter = importlib.import_module("jupyterlab")
            check.installed = True
            check.version = jupyter.__version__
            check.status = "PASS"
            check.message = f"JupyterLab {check.version}"
        except ImportError:
            try:
                jupyter = importlib.import_module("notebook")
                check.installed = True
                check.version = jupyter.__version__
                check.status = "PASS"
                check.message = f"Jupyter Notebook {check.version}"
            except ImportError:
                # Check CLI
                if shutil.which("jupyter"):
                    check.installed = True
                    check.status = "PASS"
                    check.message = "Jupyter available via CLI"
                else:
                    check.status = "NOT_INSTALLED"
                    check.message = "Jupyter is not installed."
        except Exception as e:
            check.status = "FAIL"
            check.message = f"Jupyter detection error: {e}"

        return check

    def check_tensorboard(self) -> ComponentCheck:
        """Check TensorBoard installation."""
        check = ComponentCheck(name="TensorBoard")
        try:
            tb = importlib.import_module("tensorboard")
            check.installed = True
            check.version = tb.__version__
            check.status = "PASS"
            check.message = f"TensorBoard {check.version}"
        except ImportError:
            if shutil.which("tensorboard"):
                check.installed = True
                check.status = "PASS"
                check.message = "TensorBoard available via CLI"
            else:
                check.status = "NOT_INSTALLED"
                check.message = "TensorBoard is not installed."
        except Exception as e:
            check.status = "FAIL"
            check.message = f"TensorBoard detection error: {e}"

        return check

    def check_docker(self) -> ComponentCheck:
        """Check Docker installation and GPU support."""
        check = ComponentCheck(name="Docker")
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                check.installed = True
                check.version = result.stdout.strip().replace("Docker version ", "").split(",")[0]
                check.status = "PASS"
                check.message = f"Docker {check.version}"

                # Check GPU support (nvidia-docker / NVIDIA Container Toolkit)
                try:
                    gpu_result = subprocess.run(
                        ["docker", "info", "--format", "{{.Runtimes}}"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if "nvidia" in gpu_result.stdout.lower():
                        check.gpu_support = True
                        check.message += " (NVIDIA GPU runtime available)"
                    else:
                        check.gpu_support = False
                except Exception:
                    check.gpu_support = None
            else:
                check.status = "FAIL"
                check.message = f"Docker command failed: {result.stderr.strip()}"
        except FileNotFoundError:
            check.status = "NOT_INSTALLED"
            check.message = "Docker is not installed."
        except Exception as e:
            check.status = "FAIL"
            check.message = f"Docker detection error: {e}"

        return check

    def run_full_check(self) -> list[ComponentCheck]:
        """Run all component checks."""
        checks = [
            self.check_pytorch(),
            self.check_tensorflow(),
            self.check_jupyter(),
            self.check_tensorboard(),
            self.check_docker(),
        ]
        return checks

    def get_overall_status(self, checks: list[ComponentCheck]) -> tuple[str, str]:
        """
        Determine overall ML readiness status.

        Returns (status, message) tuple.
        """
        pytorch_check = next((c for c in checks if c.name == "PyTorch"), None)
        tf_check = next((c for c in checks if c.name == "TensorFlow"), None)

        has_pytorch_gpu = pytorch_check and pytorch_check.gpu_support
        has_tf_gpu = tf_check and tf_check.gpu_support

        if has_pytorch_gpu and has_tf_gpu:
            return "READY", "READY FOR AI/ML TRAINING (PyTorch + TensorFlow GPU)"
        elif has_pytorch_gpu:
            return "READY", "READY FOR AI/ML TRAINING (PyTorch GPU)"
        elif has_tf_gpu:
            return "READY", "READY FOR AI/ML TRAINING (TensorFlow GPU)"
        elif (pytorch_check and pytorch_check.installed) or (tf_check and tf_check.installed):
            return "PARTIAL", "ML frameworks installed but no GPU support detected."
        else:
            return "NOT_READY", "No ML frameworks installed."


# Module-level singleton
_detector: MLEnvironmentDetector | None = None


def get_ml_detector() -> MLEnvironmentDetector:
    global _detector
    if _detector is None:
        _detector = MLEnvironmentDetector()
    return _detector
