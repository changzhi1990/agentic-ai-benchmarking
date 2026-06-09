from __future__ import annotations

from typing import Callable

from .remote import RemoteConfig, run_remote_command


def summarize_preflight(
    hostname: str, docker_version: str, gpu_lines: list[str]
) -> dict[str, object]:
    gpu_names: list[str] = []
    driver_versions: list[str] = []
    for line in gpu_lines:
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 2:
            gpu_names.append(parts[0])
            driver_versions.append(parts[1])

    return {
        "status": "ok",
        "hostname": hostname.strip(),
        "docker_available": bool(docker_version.strip()),
        "docker_version": docker_version.strip(),
        "nvidia_smi_available": bool(gpu_lines),
        "gpu_count": len(gpu_lines),
        "gpu_names": sorted(set(gpu_names)),
        "driver_versions": sorted(set(driver_versions)),
    }


def run_remote_preflight(
    config: RemoteConfig,
    executor: Callable[[RemoteConfig, str, int], str] = run_remote_command,
) -> dict[str, object]:
    hostname = executor(config, "hostname", 30)
    docker_version = executor(config, "docker --version", 30)
    gpu_output = executor(
        config,
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        30,
    )
    gpu_lines = [line.strip() for line in gpu_output.splitlines() if line.strip()]
    return summarize_preflight(hostname, docker_version, gpu_lines)
