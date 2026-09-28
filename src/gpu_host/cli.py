"""
GPU Host CLI — commands to manage the host server.

Commands:
    gpu-host start    Start the API server
    gpu-host stop     Stop the API server (signal-based)
    gpu-host status   Show server status
    gpu-host init     Initialize the platform (create dirs, DB, secrets)
"""

from __future__ import annotations

import os
import sys

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from gpu_core.config import get_config, reset_config
from gpu_core.database import init_db
from gpu_core.logging_config import setup_logging

app = typer.Typer(
    name="gpu-host",
    help="GPU Host — Remote ML GPU Platform Server",
    no_args_is_help=True,
)
console = Console()


@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8765, "--port", "-p", help="Port number"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev)"),
) -> None:
    """Start the GPU Host API server."""
    # Set env vars so config picks them up
    os.environ.setdefault("GPU_PLATFORM_HOST", host)
    os.environ.setdefault("GPU_PLATFORM_PORT", str(port))
    os.environ.setdefault("GPU_PLATFORM_LOG_LEVEL", log_level.upper())

    config = get_config()
    config.ensure_storage()

    console.print(f"\n[bold green]🚀 Starting GPU Host Server[/bold green]")
    console.print(f"   Address:  [cyan]{host}:{port}[/cyan]")
    console.print(f"   Workers:  {workers}")
    console.print(f"   Log:      {log_level.upper()}")
    console.print(f"   API Token (first 12 chars): [yellow]{config.security.api_token[:12]}...[/yellow]")
    console.print(f"   Docs:     [link]http://{host}:{port}/docs[/link]\n")

    uvicorn.run(
        "gpu_host.main:create_app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower(),
        reload=reload,
        factory=True,
    )


@app.command()
def init() -> None:
    """Initialize the platform: create directories, database, and secrets."""
    config = get_config()
    config.ensure_storage()
    init_db()
    config.load_persisted_secrets()
    config.save_generated_secrets()

    console.print("[bold green]✓ Platform initialized.[/bold green]")
    console.print(f"  Data directory: {config.storage.data_dir}")
    console.print(f"  Database: {config.database.database_url}")
    console.print(f"  API Token (first 12 chars): [yellow]{config.security.api_token[:12]}...[/yellow]")
    console.print(f"\n  Full token saved to: {config.storage.data_dir / '.secrets.yaml'}")


@app.command()
def status() -> None:
    """Show server configuration and status."""
    config = get_config()

    table = Table(title="GPU Host Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bind", config.host.bind)
    table.add_row("Port", str(config.host.port))
    table.add_row("Auth", str(config.security.authentication))
    table.add_row("Data Dir", str(config.storage.data_dir))
    table.add_row("Database", config.database.database_url)
    table.add_row("Max Concurrent Jobs", str(config.scheduler.max_concurrent_jobs))
    table.add_row("GPU Devices", config.gpu.allowed_devices)
    table.add_row("Log Level", config.logging.log_level)

    console.print(table)


@app.command()
def show_token() -> None:
    """Display the current API token."""
    config = get_config()
    config.load_persisted_secrets()
    console.print(f"[yellow]{config.security.api_token}[/yellow]")


if __name__ == "__main__":
    app()
