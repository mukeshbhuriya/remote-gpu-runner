"""Unit tests for gpu_core.config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from gpu_core.config import (
    PlatformConfig,
    HostSettings,
    SecuritySettings,
    StorageSettings,
    LoggingSettings,
    get_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config singleton and env vars between tests."""
    reset_config()
    # Save and restore env vars
    saved = {}
    prefix = "GPU_PLATFORM_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            saved[key] = os.environ.pop(key)
    yield
    # Restore
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            del os.environ[key]
    os.environ.update(saved)
    reset_config()


class TestHostSettings:
    def test_defaults(self):
        settings = HostSettings()
        assert settings.bind == "0.0.0.0"
        assert settings.port == 8765
        assert settings.workers == 1

    def test_env_override(self):
        os.environ["GPU_PLATFORM_PORT"] = "9999"
        settings = HostSettings()
        assert settings.port == 9999


class TestSecuritySettings:
    def test_auto_generates_secrets(self):
        settings = SecuritySettings()
        assert len(settings.secret_key) > 0
        assert len(settings.api_token) > 0

    def test_different_tokens_each_time(self):
        s1 = SecuritySettings()
        reset_config()
        s2 = SecuritySettings()
        assert s1.secret_key != s2.secret_key


class TestStorageSettings:
    def test_defaults(self):
        settings = StorageSettings()
        assert settings.data_dir == Path("./data")

    def test_ensure_directories(self, tmp_path):
        settings = StorageSettings(
            data_dir=tmp_path / "data",
            jobs_dir=tmp_path / "data" / "jobs",
            datasets_dir=tmp_path / "data" / "datasets",
            models_dir=tmp_path / "data" / "models",
            checkpoints_dir=tmp_path / "data" / "checkpoints",
            artifacts_dir=tmp_path / "data" / "artifacts",
        )
        settings.ensure_directories()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "data" / "jobs").is_dir()
        assert (tmp_path / "data" / "datasets").is_dir()


class TestLoggingSettings:
    def test_valid_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            settings = LoggingSettings(log_level=level)
            assert settings.log_level == level

    def test_invalid_level(self):
        with pytest.raises(Exception):
            LoggingSettings(log_level="INVALID")

    def test_case_insensitive(self):
        settings = LoggingSettings(log_level="debug")
        assert settings.log_level == "DEBUG"


class TestPlatformConfig:
    def test_creates_with_defaults(self):
        config = PlatformConfig()
        assert config.host.port == 8765
        assert config.security.authentication is True
        assert config.scheduler.max_concurrent_jobs == 1

    def test_yaml_config(self, tmp_path):
        yaml_config = {
            "host": {"port": 1234, "bind": "127.0.0.1"},
            "scheduler": {"max_concurrent_jobs": 4},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(yaml_config, f)

        os.environ["GPU_PLATFORM_CONFIG"] = str(config_file)
        config = PlatformConfig()
        assert config.host.port == 1234
        assert config.host.bind == "127.0.0.1"
        assert config.scheduler.max_concurrent_jobs == 4

    def test_env_overrides_yaml(self, tmp_path):
        yaml_config = {"host": {"port": 1234}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(yaml_config, f)

        os.environ["GPU_PLATFORM_CONFIG"] = str(config_file)
        os.environ["GPU_PLATFORM_PORT"] = "5555"
        config = PlatformConfig()
        # Env var takes priority over YAML
        assert config.host.port == 5555

    def test_save_and_load_secrets(self, tmp_path):
        config = PlatformConfig()
        config.storage.data_dir = tmp_path
        config.save_generated_secrets()

        original_key = config.security.secret_key
        original_token = config.security.api_token

        # Modify in memory
        config.security.secret_key = "temporary"

        # Reload
        config.load_persisted_secrets()
        assert config.security.secret_key == original_key
        assert config.security.api_token == original_token


class TestGetConfig:
    def test_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2
