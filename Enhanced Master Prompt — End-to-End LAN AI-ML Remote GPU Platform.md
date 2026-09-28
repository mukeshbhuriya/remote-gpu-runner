# MASTER ENGINEERING PROMPT

# Build a Fully Functional LAN AI/ML Remote GPU Compute Platform

You are the lead software architect, senior Python/backend engineer, ML infrastructure engineer, DevOps engineer, security engineer, QA engineer, and technical writer responsible for building a **complete, working, production-quality personal LAN-based AI/ML remote GPU platform**.

This is an implementation task, not a design-only task.

**Build the actual system. Do not merely describe how it could be built.**

The final result must be runnable on real machines and must support real AI/ML workloads using the physical GPU installed in `GPU_HOST`.

---

# 1. PRIMARY OBJECTIVE

I have:

- one development machine (`GPU_CLIENT`)
- one machine containing a powerful physical GPU (`GPU_HOST`)
- a local network connecting them

I want to sit at `GPU_CLIENT` and use `GPU_HOST` as my personal AI/ML compute server.

The finished system should allow me to:

1. Discover `GPU_HOST` over the LAN.
2. Authenticate securely.
3. Inspect the remote machine.
4. Inspect GPU/CPU/RAM/storage/network status.
5. Diagnose the ML environment.
6. Upload datasets.
7. Submit ML projects.
8. Run PyTorch workloads.
9. Run TensorFlow workloads.
10. Run inference.
11. Run fine-tuning.
12. Run experiments.
13. Monitor jobs.
14. Stream logs.
15. Monitor GPU utilization.
16. Monitor VRAM.
17. Monitor temperature and power where supported.
18. Monitor training metrics.
19. Manage checkpoints.
20. Resume interrupted training.
21. Store model artifacts.
22. Download trained models.
23. Access TensorBoard.
24. Access Jupyter/JupyterLab.
25. Develop through VS Code Remote SSH.
26. Optionally execute jobs inside Docker.
27. Continue jobs when the client disconnects.
28. Reconnect later and see the same jobs.
29. Manage multiple experiments.
30. Provide a web dashboard.
31. Provide a CLI.
32. Provide diagnostics and troubleshooting.
33. Work on supported Windows and Linux configurations.
34. Never fake GPU functionality or report unverified capabilities.

The final product should effectively turn:

```text
GPU_HOST
```

into a:

```text
PERSONAL LAN AI/ML TRAINING SERVER
```

---

# 2. MOST IMPORTANT ARCHITECTURAL RULE

## DO NOT VIRTUALIZE THE REMOTE GPU

Do NOT attempt to make the remote GPU appear as a native PCIe GPU inside `GPU_CLIENT`.

Do NOT create a fake CUDA device.

Do NOT pretend this will work:

```python
torch.cuda.is_available()
```

on `GPU_CLIENT` merely because `GPU_HOST` has a GPU.

The physical GPU remains physically installed in:

```text
GPU_HOST
```

All GPU computation happens on:

```text
GPU_HOST
```

The client communicates with the host through the network.

Use a remote-execution architecture.

```text
                         LOCAL NETWORK
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      ┌───────────────┐                ┌────────────────────┐
      │  GPU_CLIENT   │                │     GPU_HOST       │
      │               │                │                    │
      │ VS Code       │                │ Physical GPU       │
      │ CLI           │                │ CUDA / ROCm        │
      │ Dashboard     │                │ PyTorch            │
      │ Jupyter UI    │                │ TensorFlow         │
      │               │                │ Docker             │
      │ Job control   │                │ Scheduler          │
      └───────┬───────┘                │ Worker             │
              │                        │ Datasets           │
              │ HTTPS/WebSocket        │ Checkpoints        │
              │ SSH                    │ Models             │
              └───────────────────────►│ Experiments        │
                                       └────────────────────┘
                                                │
                                                ▼
                                       PHYSICAL GPU COMPUTE
```

---

# 3. NON-NEGOTIABLE ENGINEERING PRINCIPLES

Follow these rules throughout the project.

## 3.1 Build, don't merely describe

Whenever implementation is possible, implement it.

Do not respond with:

> "You can implement this by..."

Instead:

- create the files
- write the code
- configure the system
- run tests
- inspect failures
- fix failures
- verify behavior
- document the result

---

## 3.2 Never fake functionality

Never claim:

- GPU support without testing it
- CUDA support without testing it
- ROCm support without testing it
- PyTorch GPU support without testing it
- TensorFlow GPU support without testing it
- Docker GPU support without testing it
- successful training without actually running it
- benchmark numbers that were not measured
- passing tests that were not executed

If something cannot be tested in the current environment, explicitly report:

```text
NOT VERIFIED
```

and explain why.

---

## 3.3 Inspect before modifying

Before substantial implementation:

1. Inspect the repository.
2. Inspect existing source files.
3. Inspect configuration.
4. Inspect installed dependencies.
5. Inspect operating system.
6. Inspect CPU.
7. Inspect RAM.
8. Inspect GPU.
9. Inspect GPU driver.
10. Inspect CUDA/ROCm.
11. Inspect Python.
12. Inspect PyTorch.
13. Inspect TensorFlow.
14. Inspect Docker.
15. Inspect Jupyter.
16. Inspect TensorBoard.
17. Inspect networking.
18. Inspect available storage.

Do not blindly overwrite an existing project.

Reuse good existing components where appropriate.

---

# 4. ENVIRONMENT DETECTION

Before implementing platform-specific behavior, detect the actual environment.

Report:

```text
Operating System:
OS Version:
Architecture:
CPU:
RAM:
GPU:
GPU Vendor:
GPU Model:
GPU VRAM:
GPU Driver:
CUDA:
ROCm:
Python:
pip:
PyTorch:
TensorFlow:
Docker:
Docker Compose:
NVIDIA Container Toolkit:
Jupyter:
TensorBoard:
Git:
Hostname:
Network Interfaces:
Local IP:
Available Storage:
```

For NVIDIA:

```bash
nvidia-smi
```

For Python:

```bash
python --version
```

For PyTorch:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

For TensorFlow:

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

Do not assume:

- NVIDIA
- AMD
- CUDA
- ROCm
- Linux
- Windows
- Docker
- TensorFlow GPU support

Detect reality.

---

# 5. COMPATIBILITY ENGINE

Implement a compatibility/diagnostic layer.

The platform must understand relationships between:

```text
OS
GPU
GPU Driver
CUDA
ROCm
Python
PyTorch
TensorFlow
cuDNN
Docker
GPU container runtime
```

Never blindly install incompatible versions.

The system should be able to explain:

```text
COMPONENT:
PyTorch

INSTALLED:
X.Y.Z

GPU:
...

GPU BACKEND:
CUDA

STATUS:
INCOMPATIBLE

REASON:
...

EXPECTED:
...

RECOMMENDED ACTION:
...
```

Do not automatically replace working ML environments unless explicitly configured to do so.

---

# 6. SYSTEM ARCHITECTURE

Use clean modular architecture.

At minimum, separate:

```text
Host API
Authentication
Configuration
GPU Detection
GPU Monitoring
Environment Diagnostics
Scheduler
Job Manager
Worker
Execution Runtime
Dataset Manager
Checkpoint Manager
Artifact Manager
Experiment Manager
Database
WebSocket Manager
Jupyter Manager
TensorBoard Manager
Docker Runtime
LAN Discovery
CLI
Dashboard
```

Use clean interfaces so future support can be added for:

- multiple GPUs
- multiple hosts
- distributed training
- PostgreSQL
- object storage
- Kubernetes
- Slurm
- Ray
- cloud GPUs
- model registries

Do not implement those future features unless required by the current core system.

---

# 7. RECOMMENDED TECHNOLOGY STACK

Prefer:

## Backend

```text
Python
FastAPI
Uvicorn
Pydantic
asyncio
SQLAlchemy
```

## Database

Initially:

```text
SQLite
```

Design the database layer so PostgreSQL can be added later.

## CLI

Use a maintainable Python CLI framework such as:

```text
Typer
```

or an equivalent well-structured CLI solution.

## Frontend

Use a simple maintainable frontend.

Avoid unnecessary frontend complexity.

Choose the smallest practical frontend stack that provides:

- dashboard
- jobs
- logs
- GPU monitoring
- experiments
- datasets
- models
- checkpoints
- system status

## Communication

Use:

```text
HTTPS/HTTP
WebSockets
SSH
```

where appropriate.

## Storage

Start with the local filesystem.

## Containers

Support:

```text
Docker
NVIDIA Container Toolkit
ROCm containers where applicable
```

---

# 8. GPU_HOST

Create the host component:

```text
gpu-host
```

Responsibilities:

- API server
- authentication
- GPU detection
- GPU monitoring
- ML environment detection
- job scheduling
- job execution
- worker management
- dataset management
- checkpoint management
- artifact management
- experiment tracking
- TensorBoard
- Jupyter
- Docker execution
- job persistence
- logging
- WebSocket streaming

The host must continue jobs even when `GPU_CLIENT` disconnects.

---

# 9. GPU_CLIENT

Create:

```text
gpu-client
```

Responsibilities:

- LAN discovery
- authentication
- host connection
- environment diagnostics
- job submission
- job monitoring
- logs
- cancellation
- dataset management
- checkpoint management
- model management
- TensorBoard access
- Jupyter access
- benchmarking
- configuration
- diagnostics

---

# 10. HOST API

Implement a documented API.

Minimum endpoints:

```text
GET    /health
GET    /system
GET    /gpu
GET    /ml/environment

GET    /jobs
POST   /jobs
GET    /jobs/{job_id}
POST   /jobs/{job_id}/cancel
POST   /jobs/{job_id}/resume
GET    /jobs/{job_id}/logs
GET    /jobs/{job_id}/metrics
GET    /jobs/{job_id}/checkpoints
GET    /jobs/{job_id}/artifacts

POST   /datasets
GET    /datasets
GET    /datasets/{dataset_id}
DELETE /datasets/{dataset_id}

GET    /models
GET    /experiments

GET    /system/doctor
GET    /benchmark
```

WebSockets:

```text
/ws/jobs/{job_id}/logs
/ws/jobs/{job_id}/metrics
/ws/gpu
```

Generate API documentation automatically through FastAPI/OpenAPI.

---

# 11. AUTHENTICATION

The host must never accept anonymous requests.

During setup:

```text
Generate authentication credentials
```

Use secure token handling.

Support:

```text
Bearer tokens
```

Example:

```text
Authorization: Bearer <TOKEN>
```

Provide token rotation.

Never store tokens in source code.

Never commit secrets to Git.

Do not log secrets.

---

# 12. NETWORK SECURITY

Default configuration:

```text
LAN only
Authentication required
Restricted filesystem
Validated paths
Upload limits
Job isolation
No unrestricted shell API
```

Never expose unrestricted shell execution through HTTP.

Never allow arbitrary filesystem paths.

Reject path traversal such as:

```text
../../etc/passwd
```

Normalize and validate paths.

All file operations must remain inside configured storage roots.

Implement:

- request validation
- upload size limits
- filename sanitization
- job ownership/access checks
- safe subprocess execution
- command argument validation
- timeout handling
- resource limits where practical
- audit logging

---

# 13. JOB SYSTEM

Every job receives a unique:

```text
job_id
```

Job states:

```text
CREATED
UPLOADING
QUEUED
STARTING
RUNNING
COMPLETED
FAILED
CANCEL_REQUESTED
CANCELLED
RESUMING
LOST
```

Persist state in the database.

The scheduler must recover correctly after:

- client disconnect
- API restart
- host reboot where practical
- worker failure
- process crash

Do not treat client connectivity as job lifetime.

---

# 14. JOB EXECUTION

Support:

```bash
gpu-client submit train.py
```

and:

```bash
gpu-client submit ./project
```

Example:

```text
project/
├── train.py
├── model.py
├── dataset.py
├── requirements.txt
├── config.yaml
└── src/
```

The project is transferred to `GPU_HOST`.

Execute it in a controlled workspace:

```text
jobs/<job_id>/
```

Never execute user projects directly from arbitrary filesystem locations.

---

# 15. JOB CONFIGURATION

Support YAML configuration:

```yaml
name: resnet-experiment

framework: pytorch

entrypoint: train.py

gpu: auto

runtime:
  mode: venv

environment:
  requirements: requirements.txt

dataset:
  name: cifar10

training:
  epochs: 100
  batch_size: 64

resources:
  max_runtime_hours: 24

outputs:
  checkpoints: checkpoints/
  models: models/
  results: results/
```

Support:

```text
existing environment
venv
Docker
```

The environment mode must be explicit and reproducible.

---

# 16. PYTORCH

Provide first-class PyTorch support.

Support and verify:

- CUDA
- GPU selection
- AMP
- mixed precision
- DataLoader
- checkpoints
- TensorBoard
- model saving
- inference

Verify actual GPU execution.

Example diagnostic:

```python
import torch

assert torch.cuda.is_available()

x = torch.randn(4096, 4096, device="cuda")
y = torch.randn(4096, 4096, device="cuda")

z = x @ y

torch.cuda.synchronize()

print(torch.cuda.get_device_name(0))
```

This must execute on the physical GPU in `GPU_HOST`.

---

# 17. TENSORFLOW

Provide first-class TensorFlow/Keras support.

Support and verify:

- TensorFlow
- Keras
- GPU
- tf.data
- mixed precision
- checkpoints
- TensorBoard
- SavedModel
- GPU selection

Verify:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print(gpus)
```

Also execute a real GPU workload.

---

# 18. ML DOCTOR

Implement:

```bash
gpu-client ml-doctor
```

Example:

```text
Remote ML Environment
────────────────────────────

OS                    ✓
CPU                   ✓
RAM                   ✓
GPU                   ✓
GPU Driver            ✓
CUDA/ROCm             ✓

PyTorch               ✓
PyTorch GPU           ✓

TensorFlow            ✓
TensorFlow GPU        ✓

Jupyter               ✓
TensorBoard           ✓

Docker                ✓

Network               ✓
Storage               ✓

Overall:
READY FOR AI/ML TRAINING
```

Every failure must explain:

1. What failed.
2. Why it failed.
3. What was detected.
4. What is expected.
5. How to fix it.

---

# 19. TEST-Ml

Implement:

```bash
gpu-client test-ml
```

This must perform real remote computation.

It must test:

```text
PyTorch
PyTorch GPU
TensorFlow
TensorFlow GPU
```

Do not merely import libraries.

The tests must perform actual computation.

---

# 20. DATASET MANAGEMENT

Implement a dataset registry.

Commands:

```bash
gpu-client dataset upload ./datasets/cifar10
gpu-client dataset list
gpu-client dataset delete DATASET_ID
```

Store datasets under:

```text
datasets/
```

Maintain metadata:

```text
dataset_id
name
size
checksum
created_at
path
format
```

Use checksums.

Avoid re-uploading datasets that already exist.

---

# 21. LARGE FILE TRANSFERS

Do not load entire datasets into RAM.

Implement:

- chunked uploads
- streaming
- progress reporting
- checksums
- resumable transfers where practical
- configurable upload limits

For large datasets:

```text
GPU_CLIENT
     │
     │ chunks
     ▼
GPU_HOST
     │
     ▼
persistent dataset storage
```

---

# 22. CHECKPOINTS

Checkpoints must survive client disconnection.

Support:

```text
PyTorch .pt
PyTorch .pth
TensorFlow/Keras checkpoints
```

Commands:

```bash
gpu-client checkpoints JOB_ID
gpu-client download-checkpoint JOB_ID
```

Never automatically delete checkpoints unless explicitly configured.

---

# 23. RESUME TRAINING

Support:

```bash
gpu-client resume JOB_ID
```

and:

```bash
gpu-client submit train.py --resume checkpoint.pt
```

Preserve:

- optimizer state
- scheduler state
- epoch
- global step
- model state
- relevant experiment metadata

where supported by the training framework/project.

---

# 24. MODEL ARTIFACTS

Store models separately from temporary job files.

Example:

```text
models/
├── pytorch/
└── tensorflow/
```

Support:

```text
.pt
.pth
.keras
SavedModel
```

Commands:

```bash
gpu-client models list
gpu-client models download MODEL_ID
```

Do not silently overwrite existing model artifacts.

---

# 25. EXPERIMENT TRACKING

Every job should record:

```text
experiment name
job ID
framework
GPU
GPU VRAM
Python version
PyTorch version
TensorFlow version
CUDA/ROCm
Git commit
dataset
hyperparameters
start time
end time
duration
status
metrics
checkpoints
artifacts
```

Use SQLite initially.

Design clean database models.

---

# 26. GPU MONITORING

Monitor where supported:

```text
GPU utilization
VRAM usage
temperature
power
GPU processes
GPU availability
```

For NVIDIA, use NVML/PyNVML where appropriate.

Expose:

```bash
gpu-client gpu
```

Example:

```text
GPU 0
────────────────────────

Model:
RTX XXXX

VRAM:
6.8 / 8.0 GB

Utilization:
92%

Temperature:
71°C

Power:
83 W

Status:
TRAINING
```

Do not display fabricated values.

If a metric is unavailable:

```text
N/A
```

---

# 27. GPU RESOURCE MANAGEMENT

Support:

```bash
--gpu auto
```

and:

```bash
--gpu 0
```

For NVIDIA, use:

```text
CUDA_VISIBLE_DEVICES
```

appropriately.

Design the scheduler so multi-GPU support can be added later.

Initially support:

```text
one GPU job at a time
```

unless the host has multiple GPUs and concurrent execution is explicitly configured.

---

# 28. CUDA OOM DIAGNOSTICS

Detect common CUDA OOM failures.

Display useful diagnostics:

```text
CUDA OUT OF MEMORY

Possible causes:
- batch size too large
- model too large
- another process is using GPU memory
- tensors retained unnecessarily

Suggested actions:
- reduce batch size
- use mixed precision
- use gradient accumulation
- reduce model size
- inspect GPU memory
```

Do not silently modify user training code.

---

# 29. DISCONNECT RESILIENCE

This is mandatory.

Scenario:

```text
GPU_CLIENT
    │
    │ submit
    ▼
GPU_HOST
    │
    ▼
TRAINING
```

Then:

```text
GPU_CLIENT DISCONNECTS
```

The training process must continue.

Later:

```text
GPU_CLIENT RECONNECTS
```

Running jobs must still be visible.

Logs, metrics, checkpoints, and artifacts must remain available.

---

# 30. JUPYTER/JUPYTERLAB

Provide:

```bash
gpu-client jupyter
```

Launch Jupyter/JupyterLab on `GPU_HOST`.

Access it securely through:

- authenticated proxy
- SSH tunnel
- another secure mechanism

Never expose an unauthenticated Jupyter server to the LAN.

A notebook running on the host:

```python
import torch

print(torch.cuda.get_device_name(0))
```

must show the physical host GPU when the environment supports it.

---

# 31. TENSORBOARD

Provide:

```bash
gpu-client tensorboard JOB_ID
```

Expose metrics such as:

```text
loss
accuracy
validation loss
validation accuracy
learning rate
epoch
step
```

Do not assume every training project emits all metrics.

---

# 32. VS CODE REMOTE SSH

Support the standard:

```text
VS Code
   │
   │ SSH
   ▼
GPU_HOST
   │
   ├── Python
   ├── PyTorch
   ├── TensorFlow
   ├── CUDA/ROCm
   └── GPU
```

Clearly distinguish:

### Remote development

VS Code Remote SSH allows code to execute directly on `GPU_HOST`.

### Platform job execution

The custom platform submits managed jobs through the API/scheduler.

Both workflows must be supported.

---

# 33. DOCKER RUNTIME

Support:

```bash
gpu-client submit ./project --runtime docker
```

For NVIDIA:

```text
Docker
  ↓
NVIDIA Container Toolkit
  ↓
CUDA
  ↓
PyTorch/TensorFlow
  ↓
GPU
```

Detect whether Docker and GPU container support are actually available.

Never claim Docker GPU support without testing it.

---

# 34. LAN DISCOVERY

Implement automatic LAN discovery.

Preferred:

```text
mDNS / Bonjour
```

Alternative:

```text
UDP discovery
```

Also support static configuration.

Command:

```bash
gpu-client discover
```

Example:

```text
Searching local network...

Found GPU_HOST

Hostname:
ai-gpu-host

IP:
192.168.1.25

GPU:
NVIDIA RTX XXXX

VRAM:
8 GB

CUDA:
AVAILABLE

PyTorch:
AVAILABLE

TensorFlow:
AVAILABLE
```

Never hardcode the user's IP.

---

# 35. WEB DASHBOARD

Create an ML-focused dashboard.

Minimum pages/views:

```text
Overview
GPU
Jobs
Job Details
Logs
Metrics
Datasets
Checkpoints
Models
Experiments
System
Settings
```

Overview should display:

```text
GPU status
GPU utilization
VRAM
temperature
running jobs
queued jobs
recent experiments
storage
host health
```

Job details should display:

```text
Job ID
Name
Framework
Status
GPU
Runtime
Started
Elapsed time
Epoch
Step
Metrics
Logs
Checkpoints
Artifacts
```

Allow:

```text
cancel
resume
view logs
view metrics
download artifacts
```

---

# 36. CLI

Implement at minimum:

```bash
gpu-client discover
gpu-client connect
gpu-client status
gpu-client gpu
gpu-client doctor
gpu-client ml-doctor
gpu-client test-ml

gpu-client submit
gpu-client jobs
gpu-client logs
gpu-client cancel
gpu-client resume

gpu-client dataset upload
gpu-client dataset list
gpu-client dataset delete

gpu-client checkpoints
gpu-client download-checkpoint

gpu-client models
gpu-client models download

gpu-client tensorboard
gpu-client jupyter

gpu-client benchmark

gpu-client config
```

Commands must provide useful errors and exit codes.

---

# 37. BENCHMARKING

Implement:

```bash
gpu-client benchmark
```

Run controlled GPU workloads through:

```text
PyTorch
TensorFlow
```

Report measured values only.

Example:

```text
GPU:
RTX XXXX

VRAM:
8 GB

PyTorch diagnostic:
X ms

TensorFlow diagnostic:
Y ms
```

Clearly state that this is a diagnostic benchmark, not a general prediction of training performance.

Never fabricate results.

---

# 38. WINDOWS SUPPORT

Support Windows 11 where the GPU/ML stack permits it.

Provide:

```text
install-host.ps1
install-client.ps1
```

Handle:

- Windows paths
- firewall
- services/processes
- environment variables
- Python
- Docker where supported
- GPU drivers

Document requirements.

---

# 39. LINUX SUPPORT

Support Ubuntu/Linux.

Provide:

```text
install-host.sh
install-client.sh
```

Document:

- Python
- GPU drivers
- CUDA
- ROCm where applicable
- Docker
- NVIDIA Container Toolkit
- permissions
- systemd

---

# 40. HOST SERVICE

Allow the host to run persistently.

Linux:

```text
systemd
```

Windows:

```text
Windows service/task mechanism
```

Commands:

```bash
gpu-host start
gpu-host stop
gpu-host restart
gpu-host status
```

The API should be able to start automatically after system reboot when configured.

---

# 41. CONFIGURATION

Use a configuration file such as:

```yaml
host:
  bind: "0.0.0.0"
  port: 8765

security:
  authentication: true

storage:
  root: "./data"
  jobs_dir: "./data/jobs"
  datasets_dir: "./data/datasets"
  models_dir: "./data/models"
  checkpoints_dir: "./data/checkpoints"
  artifacts_dir: "./data/artifacts"

scheduler:
  max_concurrent_jobs: 1

gpu:
  allowed_devices:
    - 0
```

Never hardcode IP addresses.

Allow environment variables to override sensitive/configurable values where appropriate.

---

# 42. DATABASE

Use a proper persistence layer.

Minimum entities:

```text
Host
User/Credential
Job
JobLog
JobMetric
Dataset
DatasetVersion
Checkpoint
ModelArtifact
Experiment
EnvironmentSnapshot
```

Use migrations.

Do not scatter raw SQL throughout the application.

---

# 43. PROJECT STRUCTURE

Use a maintainable structure similar to:

```text
remote-ml-gpu/
│
├── apps/
│   ├── host/
│   ├── client/
│   └── dashboard/
│
├── packages/
│   ├── api/
│   ├── core/
│   ├── database/
│   ├── scheduler/
│   ├── worker/
│   ├── gpu/
│   ├── ml/
│   │   ├── pytorch/
│   │   └── tensorflow/
│   ├── datasets/
│   ├── artifacts/
│   ├── security/
│   ├── discovery/
│   └── shared/
│
├── examples/
│   ├── pytorch/
│   └── tensorflow/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── e2e/
│
├── scripts/
├── docker/
├── docs/
│
├── migrations/
│
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

You may modify this structure if there is a technically superior architecture.

If you change it, explain why.

---

# 44. TESTING REQUIREMENTS

Create comprehensive automated tests.

## Unit tests

Test:

- configuration
- validation
- authentication
- authorization
- database
- job creation
- state transitions
- scheduler
- GPU detection
- environment detection
- path security
- dataset management
- checkpoint management
- artifact management
- CLI parsing

## Integration tests

Test:

```text
Client
 ↓
API
 ↓
Scheduler
 ↓
Worker
 ↓
Execution Runtime
 ↓
ML Framework
 ↓
GPU
 ↓
Results
```

## Security tests

Test:

- invalid token
- expired/invalid credentials where applicable
- path traversal
- command injection
- oversized upload
- unauthorized job access
- invalid job ID
- filesystem escape
- malicious filenames
- malformed requests

## End-to-end tests

Test the actual workflow from client through host.

---

# 45. EXAMPLE PROJECTS

Create working examples:

```text
examples/
├── pytorch/
│   ├── train.py
│   ├── model.py
│   ├── requirements.txt
│   └── config.yaml
│
└── tensorflow/
    ├── train.py
    ├── model.py
    ├── requirements.txt
    └── config.yaml
```

Use small workloads suitable for testing.

The examples must actually execute.

---

# 46. DOCUMENTATION

Create:

```text
README.md
ARCHITECTURE.md
INSTALL_WINDOWS.md
INSTALL_LINUX.md
ML_SETUP.md
PYTORCH.md
TENSORFLOW.md
JUPYTER.md
TENSORBOARD.md
DOCKER.md
DATASETS.md
CHECKPOINTS.md
SECURITY.md
TROUBLESHOOTING.md
API.md
CLI.md
DEVELOPMENT.md
```

Documentation must contain actual commands.

Do not write placeholder documentation.

---

# 47. DEVELOPMENT WORKFLOW

Do NOT attempt to generate the entire application blindly in one pass.

Build incrementally.

However, do not stop after producing a skeleton.

Use this execution loop:

```text
INSPECT
   ↓
DESIGN
   ↓
IMPLEMENT
   ↓
RUN
   ↓
TEST
   ↓
IDENTIFY FAILURES
   ↓
FIX
   ↓
RETEST
   ↓
VERIFY
   ↓
DOCUMENT
   ↓
NEXT PHASE
```

Never advance past a phase with unresolved critical failures.

---

# 48. IMPLEMENTATION PHASES

## Phase 0 — Repository and Environment Audit

Inspect:

- repository
- files
- dependencies
- OS
- hardware
- GPU
- ML stack
- Docker
- network

Produce an environment report.

---

## Phase 1 — Core Foundation

Implement:

- configuration
- logging
- database
- models
- migrations
- shared schemas
- error handling

Run tests.

---

## Phase 2 — GPU/ML Detection

Implement:

- GPU detection
- CPU/RAM detection
- CUDA/ROCm detection
- PyTorch detection
- TensorFlow detection
- Docker detection
- Jupyter detection
- TensorBoard detection

Run actual diagnostics.

---

## Phase 3 — Host API

Implement:

- FastAPI
- authentication
- health
- system
- GPU
- ML environment

Test from another machine.

---

## Phase 4 — Client

Implement:

- CLI
- configuration
- authentication
- discovery
- status
- GPU
- doctor

---

## Phase 5 — Job System

Implement:

- project upload
- job creation
- queue
- scheduler
- worker
- subprocess execution
- logs
- cancellation
- persistence
- disconnect resilience

Initially test using a simple CPU workload.

---

## Phase 6 — PyTorch

Run a real PyTorch GPU workload.

Verify:

```text
physical GPU
GPU utilization
GPU memory
successful completion
artifacts
logs
```

---

## Phase 7 — TensorFlow

Run a real TensorFlow GPU workload.

Verify the same lifecycle.

---

## Phase 8 — Dataset System

Implement:

- registry
- checksums
- uploads
- chunking
- caching
- resumable transfer where practical

---

## Phase 9 — ML Lifecycle

Implement:

- checkpoints
- resume
- model artifacts
- experiment metadata
- metrics
- TensorBoard

---

## Phase 10 — Developer Experience

Implement:

- Jupyter
- Jupyter secure access
- dashboard
- VS Code Remote SSH documentation
- improved CLI UX

---

## Phase 11 — Docker

Implement:

- Docker runtime
- GPU passthrough
- environment isolation
- image validation
- resource controls where practical

---

## Phase 12 — Packaging

Implement:

- Windows installation
- Linux installation
- services
- configuration
- documentation
- production startup

---

## Phase 13 — Full E2E Verification

Run the complete acceptance workflow.

Do not declare the project complete until it passes.

---

# 49. ACCEPTANCE TEST

The following workflow is the definition of "working."

## On GPU_HOST

Run:

```bash
gpu-host start
```

The host must start successfully.

---

## On GPU_CLIENT

Run:

```bash
gpu-client discover
```

The host must be discovered.

Then:

```bash
gpu-client connect <HOST>
```

Then:

```bash
gpu-client ml-doctor
```

Expected:

```text
GPU                  PASS
CUDA/ROCm            PASS
PyTorch              PASS
PyTorch GPU          PASS
TensorFlow           PASS
TensorFlow GPU       PASS
Jupyter              PASS
TensorBoard          PASS
```

If a capability is genuinely unsupported by the hardware/software environment, report:

```text
UNSUPPORTED
```

rather than falsely reporting PASS.

---

Run:

```bash
gpu-client test-ml
```

Expected:

```text
PyTorch GPU test: SUCCESS
TensorFlow GPU test: SUCCESS
```

where the installed environment supports both.

---

Run:

```bash
gpu-client submit examples/pytorch/
```

The job must actually train using the physical GPU on `GPU_HOST`.

Then:

```bash
gpu-client jobs
```

The job must appear.

Then:

```bash
gpu-client logs JOB_ID
```

Logs must stream/display.

Then:

```bash
gpu-client checkpoints JOB_ID
```

Checkpoints must appear when produced.

Then:

```bash
gpu-client tensorboard JOB_ID
```

Training metrics must be accessible when the training example emits them.

---

## DISCONNECT TEST

Disconnect `GPU_CLIENT`.

The training job must continue.

Reconnect.

Run:

```bash
gpu-client jobs
```

The running/completed job must still exist.

Run:

```bash
gpu-client logs JOB_ID
```

Historical logs must remain available.

Run:

```bash
gpu-client checkpoints JOB_ID
```

Checkpoints must remain available.

---

## TensorFlow test

Run:

```bash
gpu-client submit examples/tensorflow/
```

The TensorFlow workload must actually use the host GPU where TensorFlow GPU support is available.

---

## Benchmark

Run:

```bash
gpu-client benchmark
```

The benchmark must perform real computation.

---

# 50. FAILURE HANDLING

When something fails:

Do not simply stop.

Follow:

```text
1. Capture error
2. Identify root cause
3. Inspect relevant environment
4. Apply minimal safe fix
5. Re-run failed test
6. Run regression tests
7. Document the fix
8. Continue
```

Do not hide errors.

Do not silently ignore failures.

Do not mark a phase complete because the code "looks correct."

---

# 51. OBSERVABILITY

The platform should provide structured logs.

Use levels:

```text
DEBUG
INFO
WARNING
ERROR
CRITICAL
```

Include useful context:

```text
timestamp
component
job_id
request_id where applicable
event
status
error
```

Do not log:

- authentication tokens
- passwords
- secrets

---

# 52. API ERROR FORMAT

Use consistent errors.

Example:

```json
{
  "error": {
    "code": "JOB_NOT_FOUND",
    "message": "The requested job does not exist.",
    "details": {}
  }
}
```

Do not expose internal stack traces to normal clients.

Log detailed diagnostics server-side.

---

# 53. DATABASE AND STATE CONSISTENCY

Job state must be persistent.

Avoid in-memory-only job tracking.

If the API restarts:

```text
jobs must still exist
```

If a worker restarts:

```text
job state must be recoverable
```

If a client disconnects:

```text
job execution must continue
```

---

# 54. RESOURCE SAFETY

Do not allow jobs to accidentally consume the entire host indefinitely.

Support configurable:

```text
max runtime
allowed GPUs
workspace limits where practical
upload limits
concurrent job limits
```

Make resource management extensible.

---

# 55. GIT AND PROJECT HYGIENE

Create:

```text
.gitignore
.env.example
```

Never commit:

- secrets
- tokens
- credentials
- private datasets
- generated model files
- checkpoints
- logs
- virtual environments
- local configuration

---

# 56. CODE QUALITY

Write maintainable production-quality code.

Requirements:

- type hints
- clear naming
- modular functions
- modular classes
- docstrings where useful
- structured error handling
- no unnecessary duplication
- no giant monolithic files
- no hardcoded machine-specific paths
- no hardcoded IP addresses
- no hardcoded secrets

Prefer simple reliable solutions over unnecessary abstraction.

---

# 57. BACKWARD/FORWARD COMPATIBILITY

Design interfaces so that:

```text
SQLite → PostgreSQL
single GPU → multi GPU
single host → multi host
local filesystem → object storage
token auth → stronger authentication/TLS
```

can be introduced later without rewriting the entire system.

Do not prematurely implement these features.

---

# 58. SECURITY BOUNDARY

Treat submitted projects as potentially untrusted.

Clearly separate:

```text
API process
scheduler
worker
job workspace
host filesystem
datasets
models
system files
```

Never allow a submitted project to arbitrarily access:

```text
/etc
Windows system directories
SSH credentials
host secrets
application secrets
other users' data
```

Container execution should be preferred when stronger isolation is required.

---

# 59. PERFORMANCE

Optimize only after measuring.

Pay particular attention to:

- large file transfers
- dataset uploads
- WebSocket logs
- GPU polling
- database writes
- job scheduling
- dashboard refresh rates

Do not introduce unnecessary complexity without evidence.

---

# 60. FINAL DELIVERABLE

At completion, provide:

## Working source code

The complete runnable project.

## Installation

Working instructions for:

```text
GPU_HOST
GPU_CLIENT
```

## Configuration

Example configuration files.

## CLI

All documented commands.

## API

OpenAPI documentation.

## Dashboard

Working dashboard.

## Tests

Automated test suite.

## Examples

Working PyTorch and TensorFlow projects.

## Documentation

Complete operational documentation.

## Verification report

Include:

```text
Environment detected
Tests executed
Tests passed
Tests failed
Features verified
Features unsupported
Known limitations
Security notes
Installation status
End-to-end acceptance status
```

Do not claim success for anything that was not actually verified.

---

# 61. FINAL RESPONSE FORMAT

When reporting progress, use:

```text
PHASE:
Phase X — <name>

STATUS:
PASS / PARTIAL / BLOCKED

IMPLEMENTED:
- ...

TESTED:
- ...

VERIFIED:
- ...

FAILURES:
- ...

FIXES:
- ...

NEXT:
- ...
```

At the end provide:

```text
PROJECT STATUS
────────────────────────

Repository:
...

GPU_HOST:
...

GPU_CLIENT:
...

API:
...

CLI:
...

Dashboard:
...

PyTorch:
PASS / FAIL / UNSUPPORTED

TensorFlow:
PASS / FAIL / UNSUPPORTED

GPU execution:
PASS / FAIL

Dataset management:
PASS / FAIL

Checkpoints:
PASS / FAIL

Resume:
PASS / FAIL

TensorBoard:
PASS / FAIL

Jupyter:
PASS / FAIL

Docker:
PASS / FAIL / UNSUPPORTED

LAN discovery:
PASS / FAIL

Authentication:
PASS / FAIL

Disconnect resilience:
PASS / FAIL

End-to-end acceptance:
PASS / FAIL

Known limitations:
...

Next recommended engineering step:
...
```

---

# 62. CRITICAL RULES — NEVER VIOLATE

1. Never fake functionality.
2. Never fabricate test results.
3. Never fabricate benchmark results.
4. Never claim GPU support without verification.
5. Never assume NVIDIA.
6. Never assume AMD.
7. Never assume CUDA.
8. Never assume ROCm.
9. Never assume Linux.
10. Never assume Windows.
11. Never silently install incompatible ML dependencies.
12. Never expose the entire host filesystem.
13. Never expose unrestricted shell execution.
14. Never store authentication secrets in Git.
15. Never expose unauthenticated Jupyter.
16. Never expose unauthenticated APIs.
17. Never terminate training merely because the client disconnects.
18. Never lose checkpoints.
19. Never silently overwrite datasets or models.
20. Never hardcode IP addresses.
21. Never hardcode secrets.
22. Never mark an untested feature as complete.
23. Never skip critical security tests.
24. Never skip end-to-end verification.
25. Never stop at a mock, placeholder, skeleton, or pseudo-implementation when real implementation is possible.

---

# 63. START NOW

Do not begin by writing a huge amount of code blindly.

First:

```text
1. Inspect the current repository.
2. Inspect the existing architecture.
3. Inspect the current environment.
4. Detect GPU/CPU/RAM/OS/network.
5. Detect Python/PyTorch/TensorFlow/CUDA/ROCm.
6. Detect Docker/Jupyter/TensorBoard.
7. Identify what already exists.
8. Identify missing components.
9. Establish the smallest safe implementation path.
10. Begin Phase 0 / Phase 1.
```

Then implement incrementally.

After every meaningful implementation:

```text
RUN TESTS
→ INSPECT RESULTS
→ FIX FAILURES
→ RE-RUN
→ VERIFY
```

Do not proceed past unresolved critical failures.

The objective is not to produce impressive-looking source code.

The objective is to produce a **real, secure, tested, end-to-end working LAN AI/ML GPU platform** that can execute actual PyTorch and TensorFlow workloads on the physical GPU inside `GPU_HOST`.

**Build the system rather than merely describing it.**