# Local GPU Platform (LAN AI/ML Remote GPU Compute)

A robust, high-performance, and secure LAN-based AI/ML remote GPU platform. This system allows you to write ML scripts locally on any lightweight machine (like a laptop) and execute them seamlessly on a powerful centralized GPU server on the same network.

## Core Features
* **Client-Server Architecture:** Connect to the remote GPU system using a secure, token-authenticated CLI.
* **Intelligent Telemetry & Hardware Detection:** Automatically probes CUDA/ROCm capabilities, RAM, and storage, reporting the health and capacity of the GPU host.
* **Managed Job Queue & Execution:** 
  * Jobs are uploaded as zip archives containing project directories.
  * A background worker manages queuing, starting, and monitoring training scripts.
* **Environment Provisioning:** Supports virtual environments (`venv`) or the `system` environment.
* **Live Log Streaming:** Fetch the output of training scripts (`stdout` and `stderr`) remotely from the client machine.
* **PyTorch & TensorFlow Native Validation:** Includes a built-in `test-ml` diagnostic that performs physical matrix multiplications on the GPU remotely.
* **SQLite Backend:** Persistent tracking of all jobs, logs, metrics, and models.

## Project Structure
* `src/gpu_core/`: Shared data models (SQLAlchemy), constants, configuration, and hardware detectors.
* `src/gpu_host/`: The FastAPI-based daemon handling API requests and running the async worker queue.
* `src/gpu_client/`: The Typer-based CLI tool used by clients to submit and monitor jobs.
* `tests/`: End-to-end integration tests verifying FastAPI lifecycle, database interactions, and CLI capabilities.

## Setup & Installation
Requires Python 3.12+ (or 3.14).

```bash
# Clone the repository
git clone https://github.com/mukeshbhuriya/local-gpu-platform.git
cd local-gpu-platform

# Create a virtual environment
python -m venv .venv
# Activate virtual environment (Windows)
.venv\Scripts\activate
# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install requirements
pip install -e .
```

## Running the Host Server
Initialize the host database and generate a secure API token:
```bash
python -m gpu_host init
```

Start the host API (runs on `http://0.0.0.0:8765` by default):
```bash
python -m uvicorn gpu_host.main:create_app --factory --host 0.0.0.0 --port 8765
```
*(The host worker starts automatically upon server startup).*

## Connecting the Client
On your local laptop/machine, connect to the GPU host using the API token generated during `init`:
```bash
python -m gpu_client connect <HOST_IP> --port 8765 --token <YOUR_TOKEN>
```

Verify your connection and check remote GPU specs:
```bash
python -m gpu_client doctor
```

Test real ML execution (PyTorch / TensorFlow compute test):
```bash
python -m gpu_client test-ml
```

## Submitting a Job
Write your ML training script (e.g., `train.py`) and submit it to the queue:
```bash
python -m gpu_client submit train.py --runtime venv
```

Check the status of your jobs:
```bash
python -m gpu_client jobs
```

Fetch real-time logs from an active job:
```bash
python -m gpu_client logs <JOB_ID>
```

## Security Note
This platform uses token-based Bearer Authentication and is designed for secure, trusted LAN environments. Ensure that `data/.secrets.yaml` and `.venv` are never committed to version control.
