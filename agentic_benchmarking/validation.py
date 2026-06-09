from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import load_json


def _require_keys(payload: dict[str, Any], required: list[str]) -> list[str]:
    errors: list[str] = []
    for key in required:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")
    return errors


def _require_nested_keys(
    payload: dict[str, Any], parent_key: str, required: list[str]
) -> list[str]:
    parent = payload.get(parent_key)
    if not isinstance(parent, dict):
        return [f"missing object key: {parent_key}"]

    errors: list[str] = []
    for key in required:
        if key not in parent:
            errors.append(f"missing nested key: {parent_key}.{key}")
    return errors


def validate_run_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = _require_keys(
        manifest,
        [
            "schema_version",
            "profile_id",
            "benchmark",
            "runtime",
            "system",
            "budget",
            "artifacts",
            "controlled_variables",
        ],
    )
    errors.extend(
        _require_nested_keys(
            manifest,
            "benchmark",
            [
                "task_id",
                "system_prompt_id",
                "user_task_id",
                "initial_repo",
                "test_script",
                "tool_permission_profile",
            ],
        )
    )
    errors.extend(
        _require_nested_keys(
            manifest,
            "runtime",
            ["model_id", "tokenizer_version", "seed", "vllm_launch_command"],
        )
    )
    errors.extend(_require_nested_keys(manifest, "system", ["cpu", "gpu"]))
    errors.extend(_require_nested_keys(manifest, "budget", ["max_iterations", "token_budget"]))
    errors.extend(_require_nested_keys(manifest, "artifacts", ["output_dir"]))

    controlled_variables = manifest.get("controlled_variables")
    if isinstance(controlled_variables, list):
        if not controlled_variables:
            errors.append("controlled_variables must not be empty")
    elif "controlled_variables" in manifest:
        errors.append("controlled_variables must be a list")

    return errors


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors = _require_keys(profile, ["profile_id", "goal", "execution", "measurement"])
    errors.extend(
        _require_nested_keys(
            profile,
            "execution",
            ["repeat_runs_min", "single_tenant", "pin_cpu_affinity", "pin_numa"],
        )
    )
    errors.extend(_require_nested_keys(profile, "measurement", ["warmup_runs", "statistic"]))
    return errors


def validate_iteration_record(record: dict[str, Any]) -> list[str]:
    errors = _require_keys(
        record,
        ["run_id", "iteration", "status", "timing", "tokens", "verdict", "target_satisfied"],
    )
    errors.extend(_require_nested_keys(record, "timing", ["llm_generation_s", "tool_execution_s", "total_s"]))
    errors.extend(_require_nested_keys(record, "tokens", ["input", "output", "total"]))

    tokens = record.get("tokens")
    if isinstance(tokens, dict):
        if tokens.get("input", 0) + tokens.get("output", 0) != tokens.get("total"):
            errors.append("tokens.total must equal tokens.input + tokens.output")

    return errors


def validate_final_report(report: dict[str, Any]) -> list[str]:
    errors = _require_keys(report, ["run_id", "profile_id", "summary", "metrics", "artifacts"])
    errors.extend(
        _require_nested_keys(
            report,
            "summary",
            ["iterations", "success", "termination_reason", "total_wall_clock_s"],
        )
    )
    errors.extend(
        _require_nested_keys(
            report,
            "metrics",
            [
                "agent_construction_speed",
                "success_rate",
                "median_iterations_to_success",
                "convergence_probability",
            ],
        )
    )
    errors.extend(
        _require_nested_keys(
            report,
            "artifacts",
            ["iteration_records_path", "manifest_snapshot_path"],
        )
    )
    return errors


def validate_run_artifacts(report_path: str | Path) -> list[str]:
    final_report_path = Path(report_path)
    report = load_json(final_report_path)
    errors = validate_final_report(report)
    if errors:
        return errors

    artifact_paths = report["artifacts"]
    manifest_path = Path(artifact_paths["manifest_snapshot_path"])
    iteration_path = Path(artifact_paths["iteration_records_path"])

    for artifact_path in [manifest_path, iteration_path]:
        if not artifact_path.exists():
            errors.append(f"missing artifact file: {artifact_path}")

    if errors:
        return errors

    manifest_errors = validate_run_manifest(load_json(manifest_path))
    errors.extend(manifest_errors)

    lines = iteration_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        errors.append("iteration record file must not be empty")
        return errors

    for line_number, line in enumerate(lines, start=1):
        record = json.loads(line)
        record_errors = validate_iteration_record(record)
        errors.extend([f"iteration line {line_number}: {error}" for error in record_errors])

    summary = report.get("summary", {})
    if summary.get("iterations") != len(lines):
        errors.append("summary.iterations must match iteration record count")

    return errors


def _build_parser(kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Validate a benchmark {kind} JSON file.")
    parser.add_argument("path", type=Path, help=f"Path to the {kind} JSON file")
    return parser


def _run_cli(kind: str, validator) -> int:
    parser = _build_parser(kind)
    args = parser.parse_args()
    payload = load_json(args.path)
    errors = validator(payload)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"{kind} is valid: {args.path}")
    return 0


def main_validate_manifest() -> int:
    return _run_cli("manifest", validate_run_manifest)


def main_validate_report() -> int:
    return _run_cli("report", validate_final_report)
