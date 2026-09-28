"""
Client-side configuration management.

Stores connection settings (host, port, token) in a local config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gpu_core.logging_config import get_logger

logger = get_logger("client.config")

# Default config file location
_CONFIG_DIR = Path.home() / ".gpu-client"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"


def get_config_path() -> Path:
    """Get the client config file path."""
    return _CONFIG_FILE


def load_client_config() -> dict[str, Any]:
    """Load client configuration from the config file."""
    path = get_config_path()
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_client_config(config: dict[str, Any]) -> None:
    """Save client configuration to the config file."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def get_connection() -> dict[str, str]:
    """
    Get the current connection settings.

    Returns dict with 'host', 'port', 'token'.
    Raises if not configured.
    """
    config = load_client_config()
    connection = config.get("connection", {})

    if not connection.get("host") or not connection.get("token"):
        raise RuntimeError(
            "Not connected to a GPU Host. Run 'gpu-client connect <host>' first."
        )

    return {
        "host": connection["host"],
        "port": str(connection.get("port", 8765)),
        "token": connection["token"],
    }


def set_connection(host: str, port: int, token: str) -> None:
    """Save connection settings."""
    config = load_client_config()
    config["connection"] = {
        "host": host,
        "port": port,
        "token": token,
    }
    save_client_config(config)
