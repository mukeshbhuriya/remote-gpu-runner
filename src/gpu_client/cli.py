"""
GPU Client CLI — commands to interact with GPU Host.

Commands cover discovery, connection, diagnostics, job management,
dataset management, model management, and developer tools.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpu_client.client import GPUHostClient
from gpu_client.config import get_connection, set_connection, load_client_config
from gpu_core.errors import PlatformError

app = typer.Typer(
    name="gpu-client",
    help="GPU Client — Remote ML GPU Platform CLI",
    no_args_is_help=True,
)
console = Console()


def _get_client() -> GPUHostClient:
    """Get an authenticated client from saved config."""
    try:
        conn = get_connection()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    return GPUHostClient(
        host=conn["host"],
        port=int(conn["port"]),
        token=conn["token"],
    )


# ─── Discovery & Connection ───────────────────────

@app.command()
def discover(
    timeout: float = typer.Option(5.0, "--timeout", "-t", help="Discovery timeout (seconds)"),
) -> None:
    """Discover GPU Hosts on the local network."""
    console.print("\n[cyan]Searching local network...[/cyan]\n")

    from gpu_client.discovery import discover_hosts, probe_host

    hosts = discover_hosts(timeout=timeout)
    if not hosts:
        console.print("[yellow]No GPU Hosts found via mDNS. Try 'gpu-client connect <ip>' directly.[/yellow]")
        return

    for host in hosts:
        console.print(Panel(
            f"[bold]{host['name']}[/bold]\n"
            f"IP:   {host['host']}\n"
            f"Port: {host['port']}",
            title="Found GPU Host",
            border_style="green",
        ))


@app.command()
def connect(
    host: str = typer.Argument(..., help="Host IP or hostname"),
    port: int = typer.Option(8765, "--port", "-p", help="Port"),
    token: str = typer.Option(..., "--token", "-t", help="API token", prompt=True),
) -> None:
    """Connect to a GPU Host."""
    from gpu_client.discovery import probe_host

    console.print(f"\n[cyan]Connecting to {host}:{port}...[/cyan]")

    # Verify connection
    health = probe_host(host, port)
    if not health:
        console.print(f"[red]Cannot reach {host}:{port}. Ensure the host is running.[/red]")
        raise typer.Exit(1)

    # Verify token
    client = GPUHostClient(host=host, port=port, token=token)
    try:
        client.verify_token()
    except PlatformError as e:
        console.print(f"[red]Authentication failed: {e.message}[/red]")
        raise typer.Exit(1)

    # Save connection
    set_connection(host, port, token)
    console.print(f"[green]✓ Connected to {host}:{port}[/green]")
    console.print(f"  Hostname: {health.get('hostname', 'N/A')}")
    console.print(f"  Version:  {health.get('version', 'N/A')}")


# ─── Status & Diagnostics ─────────────────────────

@app.command()
def status() -> None:
    """Show GPU Host status and system info."""
    client = _get_client()
    try:
        sys_info = client.system_info()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    table = Table(title="GPU Host System")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("OS", f"{sys_info['os']} {sys_info.get('os_version', '')}")
    table.add_row("CPU", sys_info.get("cpu", "N/A"))
    table.add_row("Cores", str(sys_info.get("cpu_cores", "N/A")))
    table.add_row("RAM", f"{sys_info.get('ram_total_mb', 0)} MB")
    table.add_row("RAM Available", f"{sys_info.get('ram_available_mb', 0)} MB")
    table.add_row("Storage", f"{sys_info.get('storage_free_gb', 0)} / {sys_info.get('storage_total_gb', 0)} GB")
    table.add_row("Python", sys_info.get("python_version", "N/A"))
    table.add_row("IP", sys_info.get("local_ip", "N/A"))

    console.print(table)


@app.command(name="gpu")
def gpu_status() -> None:
    """Show GPU hardware status."""
    client = _get_client()
    try:
        data = client.gpu_status()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if data["gpu_count"] == 0:
        console.print("[yellow]No GPUs detected.[/yellow]")
        return

    for gpu in data["gpus"]:
        vram_pct = (gpu["vram_used_mb"] / gpu["vram_total_mb"] * 100) if gpu["vram_total_mb"] > 0 else 0
        console.print(Panel(
            f"[bold]GPU {gpu['index']}[/bold]\n"
            f"{'─' * 30}\n"
            f"Model:        {gpu['name']}\n"
            f"VRAM:         {gpu['vram_used_mb']} / {gpu['vram_total_mb']} MB ({vram_pct:.0f}%)\n"
            f"Utilization:  {gpu.get('utilization_percent', 'N/A')}%\n"
            f"Temperature:  {gpu.get('temperature_c', 'N/A')}°C\n"
            f"Power:        {gpu.get('power_watts', 'N/A')} W\n"
            f"Driver:       {gpu.get('driver_version', 'N/A')}\n"
            f"CUDA:         {gpu.get('cuda_version', 'N/A')}",
            border_style="green",
        ))


@app.command()
def doctor() -> None:
    """Run system health check (doctor)."""
    client = _get_client()
    try:
        data = client.doctor()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]Remote System Health Check[/bold]")
    console.print("─" * 40)

    for check in data["checks"]:
        status_icon = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠", "NOT_INSTALLED": "○"}.get(
            check["status"], "?"
        )
        color = {"PASS": "green", "FAIL": "red", "WARNING": "yellow", "NOT_INSTALLED": "dim"}.get(
            check["status"], "white"
        )
        line = f"  [{color}]{status_icon}[/{color}] {check['component']:<20s}"
        if check.get("detected"):
            line += f"  {check['detected']}"
        if check.get("message"):
            line += f"  [dim]({check['message']})[/dim]"
        console.print(line)

    console.print(f"\n[bold]Overall: {data['overall']}[/bold]\n")


@app.command(name="ml-doctor")
def ml_doctor() -> None:
    """Check ML environment: PyTorch, TensorFlow, GPU support."""
    client = _get_client()
    try:
        data = client.ml_environment()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]Remote ML Environment[/bold]")
    console.print("─" * 40)

    for comp in data["components"]:
        icon = {"PASS": "✓", "FAIL": "✗", "NOT_INSTALLED": "○"}.get(comp["status"], "?")
        color = {"PASS": "green", "FAIL": "red", "NOT_INSTALLED": "dim"}.get(comp["status"], "white")
        version = f" v{comp['version']}" if comp.get("version") else ""
        gpu_label = ""
        if comp.get("gpu_support") is True:
            gpu_label = " [green](GPU ✓)[/green]"
        elif comp.get("gpu_support") is False:
            gpu_label = " [red](GPU ✗)[/red]"

        console.print(f"  [{color}]{icon}[/{color}] {comp['name']:<15s}{version}{gpu_label}")

    console.print(f"\n[bold]{data['overall_status']}: {data.get('overall_message', '')}[/bold]\n")


# ─── Jobs ──────────────────────────────────────────

@app.command()
def submit(
    path: str = typer.Argument(..., help="Path to training script or project directory"),
    name: str = typer.Option(None, "--name", "-n", help="Job name"),
    framework: str = typer.Option("generic", "--framework", "-f", help="Framework: pytorch, tensorflow, generic"),
    runtime: str = typer.Option("venv", "--runtime", "-r", help="Runtime: system, venv, docker"),
    entrypoint: str = typer.Option("train.py", "--entrypoint", "-e", help="Entrypoint script"),
    gpu_device: str = typer.Option("auto", "--gpu", help="GPU selection: auto, 0, 1, ..."),
    max_hours: float = typer.Option(None, "--max-hours", help="Max runtime hours"),
    resume_checkpoint: str = typer.Option(None, "--resume", help="Resume from checkpoint"),
) -> None:
    """Submit a training job to GPU Host."""
    source = Path(path)
    if not source.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    # Auto-detect name
    if not name:
        name = source.stem if source.is_file() else source.name

    console.print(f"\n[cyan]Packaging project: {source}[/cyan]")

    # Create zip archive
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if source.is_file():
                zf.write(source, source.name)
            else:
                for file in source.rglob("*"):
                    if file.is_file() and "__pycache__" not in str(file) and ".git" not in str(file):
                        zf.write(file, file.relative_to(source))

        console.print(f"[cyan]Uploading to GPU Host...[/cyan]")

        client = _get_client()
        result = client.submit_job(
            archive_path=archive_path,
            name=name,
            framework=framework,
            runtime_mode=runtime,
            entrypoint=entrypoint,
            gpu=gpu_device,
            max_runtime_hours=max_hours,
            resume_checkpoint=resume_checkpoint,
        )

        console.print(f"\n[green]✓ Job submitted: {result['id']}[/green]")
        console.print(f"  Name:      {result['name']}")
        console.print(f"  Framework: {result['framework']}")
        console.print(f"  Status:    {result['status']}")
        console.print(f"\n  Track with: [cyan]gpu-client logs {result['id']}[/cyan]")

    except PlatformError as e:
        console.print(f"[red]Submit failed: {e.message}[/red]")
        raise typer.Exit(1)
    finally:
        archive_path.unlink(missing_ok=True)


@app.command()
def jobs(
    status_filter: str = typer.Option(None, "--status", "-s", help="Filter by status"),
) -> None:
    """List all jobs."""
    client = _get_client()
    try:
        data = client.list_jobs(status=status_filter)
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if not data["jobs"]:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title=f"Jobs ({data['total']})")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Name", max_width=30)
    table.add_column("Status", max_width=15)
    table.add_column("Framework", max_width=12)
    table.add_column("Created", max_width=20)

    for job in data["jobs"]:
        status_color = {
            "RUNNING": "green", "COMPLETED": "blue", "FAILED": "red",
            "QUEUED": "yellow", "CANCELLED": "dim",
        }.get(job["status"], "white")
        table.add_row(
            job["id"],
            job["name"],
            f"[{status_color}]{job['status']}[/{status_color}]",
            job.get("framework", ""),
            job.get("created_at", "")[:19],
        )

    console.print(table)


@app.command()
def logs(
    job_id: str = typer.Argument(..., help="Job ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    limit: int = typer.Option(100, "--limit", "-n", help="Number of log lines"),
) -> None:
    """View job logs."""
    client = _get_client()
    try:
        data = client.job_logs(job_id, limit=limit)
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if not data["logs"]:
        console.print("[dim]No logs available yet.[/dim]")
        return

    for log in data["logs"]:
        stream_color = "white" if log["stream"] == "stdout" else "red"
        ts = log["timestamp"][:19] if log.get("timestamp") else ""
        console.print(f"[dim]{ts}[/dim] [{stream_color}]{log['message']}[/{stream_color}]")


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="Job ID to cancel"),
) -> None:
    """Cancel a running or queued job."""
    client = _get_client()
    try:
        result = client.cancel_job(job_id)
        console.print(f"[yellow]Job {job_id}: {result['status']}[/yellow]")
    except PlatformError as e:
        console.print(f"[red]Cancel failed: {e.message}[/red]")
        raise typer.Exit(1)


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
) -> None:
    """Resume a failed or cancelled job."""
    client = _get_client()
    try:
        result = client.resume_job(job_id)
        console.print(f"[green]Job {job_id}: {result['status']}[/green]")
    except PlatformError as e:
        console.print(f"[red]Resume failed: {e.message}[/red]")
        raise typer.Exit(1)


# ─── Datasets ──────────────────────────────────────

dataset_app = typer.Typer(help="Dataset management")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("upload")
def dataset_upload(
    path: str = typer.Argument(..., help="Path to dataset file or directory"),
    name: str = typer.Option(None, "--name", "-n", help="Dataset name"),
) -> None:
    """Upload a dataset to GPU Host."""
    source = Path(path)
    if not source.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    if not name:
        name = source.stem

    console.print(f"[cyan]Uploading dataset '{name}'...[/cyan]")

    client = _get_client()
    try:
        if source.is_dir():
            # Zip directory
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                zip_path = Path(tmp.name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in source.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(source))
            result = client.upload_dataset(zip_path, name)
            zip_path.unlink(missing_ok=True)
        else:
            result = client.upload_dataset(source, name)

        console.print(f"[green]✓ Dataset '{name}' uploaded ({result.get('size_bytes', 0)} bytes)[/green]")
    except PlatformError as e:
        console.print(f"[red]Upload failed: {e.message}[/red]")
        raise typer.Exit(1)


@dataset_app.command("list")
def dataset_list() -> None:
    """List all datasets."""
    client = _get_client()
    try:
        data = client.list_datasets()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if not data["datasets"]:
        console.print("[dim]No datasets registered.[/dim]")
        return

    table = Table(title=f"Datasets ({data['total']})")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Format")
    table.add_column("Created")

    for ds in data["datasets"]:
        size = f"{ds.get('size_bytes', 0) / (1024*1024):.1f} MB" if ds.get("size_bytes") else "N/A"
        table.add_row(ds["id"], ds["name"], size, ds.get("format", ""), ds.get("created_at", "")[:19])

    console.print(table)


@dataset_app.command("delete")
def dataset_delete(
    dataset_id: str = typer.Argument(..., help="Dataset ID to delete"),
) -> None:
    """Delete a dataset."""
    client = _get_client()
    try:
        client.delete_dataset(dataset_id)
        console.print(f"[green]✓ Dataset {dataset_id} deleted.[/green]")
    except PlatformError as e:
        console.print(f"[red]Delete failed: {e.message}[/red]")
        raise typer.Exit(1)


# ─── Checkpoints ───────────────────────────────────

@app.command()
def checkpoints(
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """List checkpoints for a job."""
    client = _get_client()
    try:
        data = client.job_checkpoints(job_id)
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if not data["checkpoints"]:
        console.print("[dim]No checkpoints found.[/dim]")
        return

    table = Table(title=f"Checkpoints for Job {job_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Filename")
    table.add_column("Epoch")
    table.add_column("Step")
    table.add_column("Created")

    for ckpt in data["checkpoints"]:
        table.add_row(
            ckpt["id"], ckpt["filename"],
            str(ckpt.get("epoch", "")), str(ckpt.get("step", "")),
            ckpt.get("created_at", "")[:19],
        )

    console.print(table)


# ─── Models ────────────────────────────────────────

models_app = typer.Typer(help="Model artifact management")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """List all model artifacts."""
    client = _get_client()
    try:
        data = client.list_models()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if not data["models"]:
        console.print("[dim]No models found.[/dim]")
        return

    table = Table(title=f"Models ({data['total']})")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Framework")
    table.add_column("Format")
    table.add_column("Created")

    for m in data["models"]:
        table.add_row(
            m["id"], m["name"], m.get("framework", ""),
            m.get("format", ""), m.get("created_at", "")[:19],
        )

    console.print(table)


@models_app.command("download")
def models_download(
    model_id: str = typer.Argument(..., help="Model ID"),
    output: str = typer.Option(".", "--output", "-o", help="Output directory"),
) -> None:
    """Download a model artifact."""
    client = _get_client()
    try:
        model_info = client.list_models()  # Get filename
        output_path = Path(output) / f"model_{model_id}"
        result = client.download_model(model_id, output_path)
        console.print(f"[green]✓ Model downloaded to {result}[/green]")
    except PlatformError as e:
        console.print(f"[red]Download failed: {e.message}[/red]")
        raise typer.Exit(1)


# ─── Developer Tools ──────────────────────────────

@app.command()
def benchmark() -> None:
    """Run GPU benchmark on the remote host."""
    client = _get_client()
    try:
        data = client.benchmark()
    except PlatformError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)

    if data.get("gpu"):
        console.print(f"\n[bold]GPU: {data['gpu']['name']}[/bold]")
        console.print(f"VRAM: {data['gpu']['vram_total_mb']} MB\n")

    for result in data.get("results", []):
        color = "green" if result["status"] == "SUCCESS" else "red"
        console.print(
            f"  [{color}]{result['framework']}[/{color}]: "
            f"{result.get('test_name', '')} — {result.get('duration_ms', 'N/A')} ms "
            f"({result['status']})"
        )

    if data.get("note"):
        console.print(f"\n[dim]{data['note']}[/dim]")


@app.command()
def config() -> None:
    """Show current client configuration."""
    cfg = load_client_config()
    conn = cfg.get("connection", {})

    table = Table(title="Client Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Host", conn.get("host", "[not set]"))
    table.add_row("Port", str(conn.get("port", "[not set]")))
    table.add_row("Token", f"{conn.get('token', '[not set]')[:12]}..." if conn.get("token") else "[not set]")
    table.add_row("Config file", str(get_connection.__module__))

    console.print(table)


@app.command()
def tensorboard(
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Open TensorBoard for a job (placeholder — opens URL)."""
    console.print(f"[cyan]TensorBoard for job {job_id} — launch via host.[/cyan]")
    console.print(f"[dim]Feature will proxy TensorBoard from GPU Host.[/dim]")


@app.command()
def jupyter() -> None:
    """Open Jupyter on GPU Host (placeholder — opens URL)."""
    console.print("[cyan]Jupyter — launch via host.[/cyan]")
    console.print("[dim]Feature will proxy Jupyter from GPU Host.[/dim]")


@app.command(name="test-ml")
def test_ml() -> None:
    """Run real ML computation tests on the remote GPU."""
    client = _get_client()
    console.print("\n[cyan]Running remote ML tests...[/cyan]\n")
    
    test_script = """import sys
import time

print("--- PyTorch Test ---")
try:
    import torch
    if torch.cuda.is_available():
        print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
        x = torch.randn(4096, 4096, device="cuda")
        y = torch.randn(4096, 4096, device="cuda")
        z = x @ y
        torch.cuda.synchronize()
        print("PyTorch Matrix Multiplication SUCCESS")
    else:
        print("PyTorch GPU: NOT AVAILABLE")
except ImportError:
    print("PyTorch NOT INSTALLED")
except Exception as e:
    print(f"PyTorch Error: {e}")

print("\\n--- TensorFlow Test ---")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"TensorFlow GPU: {gpus[0].name}")
        x = tf.random.normal((4096, 4096))
        y = tf.random.normal((4096, 4096))
        z = tf.matmul(x, y)
        print("TensorFlow Matrix Multiplication SUCCESS")
    else:
        print("TensorFlow GPU: NOT AVAILABLE")
except ImportError:
    print("TensorFlow NOT INSTALLED")
except Exception as e:
    print(f"TensorFlow Error: {e}")
"""
    
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_compute.py"
        script_path.write_text(test_script)
        
        # Zip it
        archive_path = Path(tmpdir) / "project.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(script_path, script_path.name)
            
        try:
            job = client.submit_job(
                archive_path=archive_path,
                name="test_ml_compute",
                framework="generic",
                runtime_mode="system",
                entrypoint="test_compute.py",
            )
            console.print(f"[yellow]Submitted test job: {job['id']}. Waiting for completion...[/yellow]")
            
            job_id = job["id"]
            # Poll for completion
            while True:
                status = client.get_job(job_id)["status"]
                if status in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                time.sleep(1)
                
            console.print(f"\\n[bold]Test Job Status: {status}[/bold]\\n")
            
            logs = client.job_logs(job_id, limit=50)
            for log in logs["logs"]:
                stream_color = "white" if log["stream"] == "stdout" else "red"
                console.print(f"[{stream_color}]{log['message']}[/{stream_color}]")
                
        except PlatformError as e:
            console.print(f"[red]Error: {e.message}[/red]")
            raise typer.Exit(1)

if __name__ == "__main__":
    app()

