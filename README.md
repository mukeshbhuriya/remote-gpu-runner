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

---

## Setup & Installation

Requires Python 3.12+ (or 3.14). 

### 1. Clone & Install
Run these commands on **both** your Host (GPU machine) and Client (Laptop):
```bash
# Clone the repository
git clone https://github.com/mukeshbhuriya-ctrl/local-gpu-platform.git
cd local-gpu-platform

# Create a virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate
# OR Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install requirements
pip install -e .
```

---

## Step-by-Step Workflow Guide

Using this platform involves two main parts: running the **Host Server** (the powerful machine with the GPU) and running the **Client** (your lightweight laptop or daily workstation). 

### Step 1: Start the GPU Host Server
On the powerful machine with the GPU, you need to initialize the system and start the host daemon. It will listen for incoming ML jobs from your laptop.

1. **Initialize the Database & Secrets (First time only):**
   ```bash
   python -m gpu_host init
   ```
   *(This will create your SQLite database and generate a secure API token inside `data/.secrets.yaml`)*

2. **Start the API server:**
   ```bash
   python -m uvicorn gpu_host.main:create_app --factory --host 0.0.0.0 --port 8765
   ```
   *(Keep this terminal open. The host worker starts automatically and is now listening for connections!)*

---

### Step 2: Connect your Laptop (Client)
On your laptop (where you write code), you need to connect to the host. You only need to do this once.

1. Find the IP address of the Host machine (e.g., `192.168.1.50`). 
2. Get the **API Token**. It's printed in the terminal of the Host when you ran `init`, or can be found in the Host's `data/.secrets.yaml` file.
3. Run the connect command on your laptop:
   ```bash
   python -m gpu_client connect 192.168.1.50 --port 8765 --token <YOUR_TOKEN>
   ```
   *(If you are running both the client and host on the exact same machine for testing, use `127.0.0.1` as the IP).*

You can verify the connection and see the remote GPU hardware by running:
```bash
python -m gpu_client doctor
```

*(Optional) Test real ML execution (PyTorch / TensorFlow compute test):*
```bash
python -m gpu_client test-ml
```

---

### Step 3: Write your ML Code (Locally)
On your laptop, you can now write your standard PyTorch or TensorFlow code just like you normally would. For example, create a file named `train.py`:

```python
# train.py
import torch
import time

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("Starting heavy training loop...")
for epoch in range(5):
    # Simulate heavy GPU compute
    x = torch.randn(10000, 10000, device="cuda")
    y = torch.randn(10000, 10000, device="cuda")
    z = x @ y
    
    print(f"Epoch {epoch} completed successfully!")
    time.sleep(2)

print("Training finished!")
```

---

### Step 4: Submit to the Remote GPU
Instead of running `python train.py` and freezing your laptop, you submit it to the GPU host!

Run this from the folder containing your script:
```bash
python -m gpu_client submit train.py --runtime system
```
*(Using `--runtime system` tells the remote host to use its globally installed PyTorch/TensorFlow, which is usually much faster than installing a fresh environment every time).*

The CLI will package your project, send it over the network, and return a **Job ID** (e.g., `61e83d364276`).

---

### Step 5: Watch the Results
You can view the real-time `print()` statements and errors from the remote GPU directly on your laptop using the Job ID:

```bash
python -m gpu_client logs <JOB_ID>
```

You can also list all your active, queued, or completed jobs by running:
```bash
python -m gpu_client jobs
```

---

## Security Note
This platform uses token-based Bearer Authentication and is designed for secure, trusted LAN environments. Ensure that `data/.secrets.yaml` and `.venv` are never committed to version control.
