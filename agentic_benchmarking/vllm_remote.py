from __future__ import annotations

import json
from pathlib import PurePosixPath
import time
from typing import Callable

from .remote import RemoteConfig, quote_remote, run_remote_command


def summarize_vllm_health(
    container_name: str,
    image: str,
    model: str,
    port: int,
    models_payload: dict,
    startup_seconds: float,
) -> dict[str, object]:
    served_models = models_payload.get("data", [])
    return {
        "status": "ok",
        "ready": bool(served_models),
        "container_name": container_name,
        "image": image,
        "model": model,
        "port": port,
        "startup_seconds": round(startup_seconds, 2),
        "served_model_count": len(served_models),
        "served_model_ids": [item.get("id") for item in served_models],
    }


def build_vllm_docker_run_command(
    container_name: str,
    image: str,
    model: str,
    port: int,
    tensor_parallel_size: int = 1,
) -> str:
    mount_clause = ""
    if model.startswith("/"):
        model_parent = str(PurePosixPath(model).parent)
        mount_clause = f"-v {quote_remote(f'{model_parent}:{model_parent}:ro')} "

    return (
        "docker run -d --rm --gpus all "
        f"--name {container_name} "
        f"-p {port}:8000 "
        f"{mount_clause}"
        f"{quote_remote(image)} "
        f"--model {quote_remote(model)} "
        f"--tensor-parallel-size {tensor_parallel_size} "
        "--host 0.0.0.0 --port 8000"
    )


def run_remote_vllm_healthcheck(
    config: RemoteConfig,
    model: str,
    port: int = 8000,
    image: str = "vllm/vllm-openai:latest",
    container_name: str = "aab-vllm",
    executor: Callable[[RemoteConfig, str, int], str] = run_remote_command,
    max_attempts: int = 30,
    sleep_seconds: float = 2.0,
    tensor_parallel_size: int = 1,
    cleanup: bool = True,
) -> dict[str, object]:
    executor(config, f"docker rm -f {container_name} >/dev/null 2>&1 || true", 30)
    run_command = build_vllm_docker_run_command(
        container_name=container_name,
        image=image,
        model=model,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
    )
    executor(config, run_command, 120)

    start = time.monotonic()
    try:
        for _ in range(max_attempts):
            try:
                payload = executor(
                    config,
                    f"curl -sf http://127.0.0.1:{port}/v1/models",
                    30,
                )
                models_payload = json.loads(payload)
                return summarize_vllm_health(
                    container_name=container_name,
                    image=image,
                    model=model,
                    port=port,
                    models_payload=models_payload,
                    startup_seconds=time.monotonic() - start,
                )
            except Exception:
                time.sleep(sleep_seconds)
        raise RuntimeError("vLLM health check did not become ready in time")
    finally:
        if cleanup:
            executor(config, f"docker rm -f {container_name} >/dev/null 2>&1 || true", 30)
