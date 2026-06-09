from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .io_utils import dump_json, load_json
from .preflight import run_remote_preflight
from .remote import RemoteConfig
from .validation import validate_final_report, validate_profile, validate_run_manifest
from .vllm_remote import run_remote_vllm_healthcheck


def run_remote_protocol_smoke(
    manifest_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    remote_config: RemoteConfig,
    model: str,
    port: int = 8000,
    image: str = "vllm/vllm-openai:latest",
    tensor_parallel_size: int = 1,
    preflight_runner: Callable[[RemoteConfig], dict] = run_remote_preflight,
    healthcheck_runner: Callable[[RemoteConfig, str, int, str], dict] = run_remote_vllm_healthcheck,
) -> dict[str, object]:
    manifest = load_json(manifest_path)
    profile = load_json(profile_path)

    manifest_errors = validate_run_manifest(manifest)
    profile_errors = validate_profile(profile)
    if manifest_errors or profile_errors:
        raise ValueError(
            "invalid benchmark inputs: " + "; ".join(manifest_errors + profile_errors)
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{manifest['profile_id']}-remote-smoke-run"
    manifest_snapshot_path = output_root / "run-manifest.snapshot.json"
    preflight_path = output_root / "remote-preflight.json"
    health_path = output_root / "remote-vllm-health.json"
    iteration_records_path = output_root / "iteration-records.jsonl"
    final_report_path = output_root / "final-report.json"

    dump_json(manifest_snapshot_path, manifest)

    preflight_summary = preflight_runner(remote_config)
    health_summary = healthcheck_runner(
        remote_config,
        model,
        port,
        image,
        tensor_parallel_size=tensor_parallel_size,
    )
    dump_json(preflight_path, preflight_summary)
    dump_json(health_path, health_summary)

    iteration_records = [
        {
            "run_id": run_id,
            "iteration": 1,
            "status": "completed",
            "timing": {"llm_generation_s": 0.0, "tool_execution_s": 1.0, "total_s": 1.0},
            "tokens": {"input": 0, "output": 0, "total": 0},
            "verdict": "continue",
            "target_satisfied": False,
        },
        {
            "run_id": run_id,
            "iteration": 2,
            "status": "completed",
            "timing": {
                "llm_generation_s": 0.0,
                "tool_execution_s": float(health_summary.get("startup_seconds", 0.0)),
                "total_s": float(health_summary.get("startup_seconds", 0.0)),
            },
            "tokens": {"input": 0, "output": 0, "total": 0},
            "verdict": "success" if health_summary.get("ready") else "failed",
            "target_satisfied": bool(health_summary.get("ready")),
        },
    ]

    with iteration_records_path.open("w", encoding="utf-8") as handle:
        for record in iteration_records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    total_wall_clock_s = round(sum(item["timing"]["total_s"] for item in iteration_records), 2)
    final_report = {
        "run_id": run_id,
        "profile_id": profile["profile_id"],
        "summary": {
            "iterations": len(iteration_records),
            "success": bool(health_summary.get("ready")),
            "termination_reason": "remote_vllm_ready"
            if health_summary.get("ready")
            else "remote_vllm_unavailable",
            "total_wall_clock_s": total_wall_clock_s,
        },
        "metrics": {
            "agent_construction_speed": round(len(iteration_records) / max(total_wall_clock_s, 1.0), 4),
            "success_rate": 1.0 if health_summary.get("ready") else 0.0,
            "median_iterations_to_success": len(iteration_records) if health_summary.get("ready") else 0,
            "convergence_probability": 1.0 if health_summary.get("ready") else 0.0,
        },
        "artifacts": {
            "iteration_records_path": str(iteration_records_path),
            "manifest_snapshot_path": str(manifest_snapshot_path),
            "remote_preflight_path": str(preflight_path),
            "remote_vllm_health_path": str(health_path),
        },
        "remote": {
            "host": remote_config.host,
            "user": remote_config.user,
            "port": remote_config.port,
            "model": model,
            "image": image,
            "tensor_parallel_size": tensor_parallel_size,
        },
    }

    report_errors = validate_final_report(final_report)
    if report_errors:
        raise ValueError("invalid remote smoke report: " + "; ".join(report_errors))
    dump_json(final_report_path, final_report)

    return {
        "status": "success" if health_summary.get("ready") else "failed",
        "run_id": run_id,
        "output_dir": str(output_root),
        "ready": bool(health_summary.get("ready")),
    }
