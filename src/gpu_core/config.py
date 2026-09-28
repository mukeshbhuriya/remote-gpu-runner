"""
Platform configuration with YAML file support and environment variable overrides.

Configuration resolution order (highest priority first):
1. Environment variables (GPU_PLATFORM_*)
2. YAML config file (config.yaml)
3. Default values
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_config_file() -> Path | None:
    """Locate config.yaml in standard locations."""
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "config" / "config.yaml",
    ]
    env_path = os.environ.get("GPU_PLATFORM_CONFIG")
    if env_path:
        candidates.insert(0, Path(env_path))

    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML configuration file if it exists."""
    config_path = _find_config_file()
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class HostSettings(BaseSettings):
    """Host server settings."""

    bind: str = "0.0.0.0"
    port: int = 8765
    workers: int = 1

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""

    authentication: bool = True
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    api_token: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    token_algorithm: str = "HS256"
    token_expire_hours: int = 720  # 30 days

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")


class StorageSettings(BaseSettings):
    """Storage path settings. All paths are relative to data_dir unless absolute."""

    data_dir: Path = Path("./data")
    jobs_dir: Path = Path("./data/jobs")
    datasets_dir: Path = Path("./data/datasets")
    models_dir: Path = Path("./data/models")
    checkpoints_dir: Path = Path("./data/checkpoints")
    artifacts_dir: Path = Path("./data/artifacts")

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")

    def ensure_directories(self) -> None:
        """Create all storage directories if they don't exist."""
        for attr_name in [
            "data_dir",
            "jobs_dir",
            "datasets_dir",
            "models_dir",
            "checkpoints_dir",
            "artifacts_dir",
        ]:
            path = getattr(self, attr_name)
            path.mkdir(parents=True, exist_ok=True)


class SchedulerSettings(BaseSettings):
    """Job scheduler settings."""

    max_concurrent_jobs: int = 1
    max_runtime_hours: int = 72
    poll_interval_seconds: float = 2.0

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")


class GPUSettings(BaseSettings):
    """GPU resource settings."""

    allowed_devices: str = "0"  # Comma-separated GPU indices

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")

    @property
    def device_list(self) -> list[int]:
        """Parse allowed devices into a list of GPU indices."""
        return [int(d.strip()) for d in self.allowed_devices.split(",") if d.strip()]


class UploadSettings(BaseSettings):
    """File upload settings."""

    max_upload_size_mb: int = 10240  # 10 GB
    chunk_size_bytes: int = 8 * 1024 * 1024  # 8 MB
    allowed_extensions: str = ""  # Empty = allow all

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


class DatabaseSettings(BaseSettings):
    """Database settings."""

    database_url: str = "sqlite:///./data/platform.db"

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")


class DiscoverySettings(BaseSettings):
    """LAN discovery settings."""

    discovery_enabled: bool = True
    service_name: str = "_gpu-ml-platform._tcp.local."

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")


class LoggingSettings(BaseSettings):
    """Logging settings."""

    log_level: str = "INFO"
    log_file: str = ""  # Empty = stdout only
    log_format: str = "json"  # "json" or "text"

    model_config = SettingsConfigDict(env_prefix="GPU_PLATFORM_")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            msg = f"Invalid log level: {v}. Must be one of: {valid}"
            raise ValueError(msg)
        return upper


class PlatformConfig:
    """
    Root configuration container.

    Aggregates all sub-configurations and supports YAML + env var overrides.
    """

    def __init__(self, config_overrides: dict[str, Any] | None = None) -> None:
        yaml_data = _load_yaml_config()
        if config_overrides:
            yaml_data.update(config_overrides)

        # Map YAML sections to environment variables for sub-settings
        self._apply_yaml_to_env(yaml_data)

        self.host = HostSettings()
        self.security = SecuritySettings()
        self.storage = StorageSettings()
        self.scheduler = SchedulerSettings()
        self.gpu = GPUSettings()
        self.upload = UploadSettings()
        self.database = DatabaseSettings()
        self.discovery = DiscoverySettings()
        self.logging = LoggingSettings()

    def _apply_yaml_to_env(self, yaml_data: dict[str, Any]) -> None:
        """
        Map nested YAML keys to flat environment variables.
        Only sets env vars that aren't already set (env takes priority).
        """
        mapping = {
            "host": {
                "bind": "GPU_PLATFORM_BIND",
                "port": "GPU_PLATFORM_PORT",
                "workers": "GPU_PLATFORM_WORKERS",
            },
            "security": {
                "authentication": "GPU_PLATFORM_AUTHENTICATION",
                "secret_key": "GPU_PLATFORM_SECRET_KEY",
                "api_token": "GPU_PLATFORM_API_TOKEN",
            },
            "storage": {
                "root": "GPU_PLATFORM_DATA_DIR",
                "data_dir": "GPU_PLATFORM_DATA_DIR",
                "jobs_dir": "GPU_PLATFORM_JOBS_DIR",
                "datasets_dir": "GPU_PLATFORM_DATASETS_DIR",
                "models_dir": "GPU_PLATFORM_MODELS_DIR",
                "checkpoints_dir": "GPU_PLATFORM_CHECKPOINTS_DIR",
                "artifacts_dir": "GPU_PLATFORM_ARTIFACTS_DIR",
            },
            "scheduler": {
                "max_concurrent_jobs": "GPU_PLATFORM_MAX_CONCURRENT_JOBS",
                "max_runtime_hours": "GPU_PLATFORM_MAX_RUNTIME_HOURS",
            },
            "gpu": {
                "allowed_devices": "GPU_PLATFORM_ALLOWED_DEVICES",
            },
            "database": {
                "url": "GPU_PLATFORM_DATABASE_URL",
                "database_url": "GPU_PLATFORM_DATABASE_URL",
            },
            "logging": {
                "level": "GPU_PLATFORM_LOG_LEVEL",
                "log_level": "GPU_PLATFORM_LOG_LEVEL",
            },
        }

        for section, keys in mapping.items():
            section_data = yaml_data.get(section, {})
            if not isinstance(section_data, dict):
                continue
            for yaml_key, env_key in keys.items():
                if yaml_key in section_data and env_key not in os.environ:
                    os.environ[env_key] = str(section_data[yaml_key])

    def ensure_storage(self) -> None:
        """Create all required storage directories."""
        self.storage.ensure_directories()

    def save_generated_secrets(self, path: Path | None = None) -> None:
        """Save auto-generated secrets to a file (first-run only)."""
        target = path or (self.storage.data_dir / ".secrets.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return  # Don't overwrite existing secrets
        data = {
            "security": {
                "secret_key": self.security.secret_key,
                "api_token": self.security.api_token,
            }
        }
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def load_persisted_secrets(self, path: Path | None = None) -> None:
        """Load previously saved secrets."""
        target = path or (self.storage.data_dir / ".secrets.yaml")
        if not target.is_file():
            return
        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return
        sec = data.get("security", {})
        if "secret_key" in sec:
            self.security.secret_key = sec["secret_key"]
        if "api_token" in sec:
            self.security.api_token = sec["api_token"]


# Module-level singleton (lazy-init)
_config: PlatformConfig | None = None


def get_config(overrides: dict[str, Any] | None = None) -> PlatformConfig:
    """Get or create the global configuration singleton."""
    global _config
    if _config is None:
        _config = PlatformConfig(overrides)
    return _config


def reset_config() -> None:
    """Reset the configuration singleton (for testing)."""
    global _config
    _config = None
