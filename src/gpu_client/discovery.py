"""
LAN discovery using mDNS/Zeroconf.

Discovers GPU Host services on the local network.
"""

from __future__ import annotations

import socket
import time
from typing import Any

from gpu_core.logging_config import get_logger

logger = get_logger("client.discovery")


def discover_hosts(timeout: float = 5.0, service_type: str = "_gpu-ml-platform._tcp.local.") -> list[dict[str, Any]]:
    """
    Discover GPU Host services on the LAN using mDNS.

    Returns a list of discovered hosts with their info.
    """
    from zeroconf import ServiceBrowser, Zeroconf, ServiceStateChange

    hosts: list[dict[str, Any]] = []
    zc = Zeroconf()

    class Handler:
        def __init__(self) -> None:
            self.found: list[str] = []

        def on_service_state_change(
            self, zeroconf: Zeroconf, service_type: str,
            name: str, state_change: ServiceStateChange,
        ) -> None:
            if state_change == ServiceStateChange.Added:
                self.found.append(name)
                info = zeroconf.get_service_info(service_type, name)
                if info:
                    addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
                    host_info = {
                        "name": name.replace(f".{service_type}", ""),
                        "host": addresses[0] if addresses else "unknown",
                        "port": info.port,
                        "properties": {
                            k.decode(): v.decode() if isinstance(v, bytes) else str(v)
                            for k, v in info.properties.items()
                        } if info.properties else {},
                    }
                    hosts.append(host_info)
                    logger.info(f"Discovered host: {host_info['name']} at {host_info['host']}:{host_info['port']}")

    handler = Handler()
    browser = ServiceBrowser(zc, service_type, handlers=[handler.on_service_state_change])

    # Wait for discovery
    time.sleep(timeout)

    zc.close()
    return hosts


def probe_host(host: str, port: int = 8765, timeout: float = 5.0) -> dict[str, Any] | None:
    """
    Probe a specific host to check if it's running the GPU platform.

    Returns health info if reachable, None otherwise.
    """
    import httpx

    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Probe failed for {host}:{port}: {e}")

    return None
