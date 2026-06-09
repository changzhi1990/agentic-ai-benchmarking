from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import dump_json, load_json
from .validation import validate_final_report, validate_profile, validate_run_manifest


def _build_iteration_records(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "iteration": 1,
            "status": "completed",
            "timing": {"llm_generation_s": 2.5, "tool_execution_s": 4.0, "total_s": 6.5},
            "tokens": {"input": 800, "output": 320, "total": 1120},
            "verdict": "continue",
            "target_satisfied": False,
        },
        {
            "run_id": run_id,
            "iteration": 2,
            "status": "completed",
            "timing": {"llm_generation_s": 2.0, "tool_execution_s": 3.2, "total_s": 5.2},
            "tokens": {"input": 820, "output": 280, "total": 1100},
            "verdict": "continue",
            "target_satisfied": False,
        },
        {
            "run_id": run_id,
            "iteration": 3,
            "status": "completed",
            "timing": {"llm_generation_s": 1.8, "tool_execution_s": 2.7, "total_s": 4.5},
            "tokens": {"input": 760, "output": 220, "total": 980},
            "verdict": "success",
            "target_satisfied": True,
        },
    ]


def run_protocol_smoke(
    manifest_path: str | Path, profile_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    profile = load_json(profile_path)

    manifest_errors = validate_run_manifest(manifest)
    profile_errors = validate_profile(profile)
    if manifest_errors or profile_errors:
        raise ValueError(
            "invalid benchmark inputs: "
            + "; ".join(manifest_errors + profile_errors)
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{manifest['profile_id']}-smoke-run"
    manifest_snapshot_path = output_root / "run-manifest.snapshot.json"
    iteration_records_path = output_root / "iteration-records.jsonl"
    final_report_path = output_root / "final-report.json"

    dump_json(manifest_snapshot_path, manifest)

    iteration_records = _build_iteration_records(run_id)
    with iteration_records_path.open("w", encoding="utf-8") as handle:
        for record in iteration_records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    total_wall_clock_s = round(sum(item["timing"]["total_s"] for item in iteration_records), 2)
    final_report = {
        "run_id": run_id,
        "profile_id": profile["profile_id"],
        "manifest_path": str(Path(manifest_path)),
        "summary": {
            "iterations": len(iteration_records),
            "success": True,
            "termination_reason": "target_reached",
            "total_wall_clock_s": total_wall_clock_s,
        },
        "metrics": {
            "agent_construction_speed": round(len(iteration_records) / total_wall_clock_s, 4),
            "success_rate": 1.0,
            "median_iterations_to_success": len(iteration_records),
            "convergence_probability": 1.0,
        },
        "artifacts": {
            "iteration_records_path": str(iteration_records_path),
            "manifest_snapshot_path": str(manifest_snapshot_path),
        },
    }

    report_errors = validate_final_report(final_report)
    if report_errors:
        raise ValueError("invalid smoke report: " + "; ".join(report_errors))

    dump_json(final_report_path, final_report)

    return {
        "status": "success",
        "run_id": run_id,
        "iterations": len(iteration_records),
        "output_dir": str(output_root),
    }
