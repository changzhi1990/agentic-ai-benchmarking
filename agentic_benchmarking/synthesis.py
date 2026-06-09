from __future__ import annotations

from typing import Any, Callable

from .evaluator import evaluate_stable_targets
from .generator import summarize_feedback


GenerationRunner = Callable[[dict[str, Any], str | None], dict[str, str]]
ExecutionRunner = Callable[[str], dict[str, Any]]
DEFAULT_CONCURRENCY_SCHEDULE = [2, 4, 8, 16, 32, 64, 128]


def summarize_iteration_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "cpu_avg": 0.0,
            "gpu_avg": 0.0,
            "memory_avg": 0.0,
            "memory_bandwidth_avg_gbps": 0.0,
        }

    count = len(samples)
    return {
        "cpu_avg": round(sum(float(s.get("cpu_util_percent", 0.0)) for s in samples) / count, 2),
        "gpu_avg": round(sum(float(s.get("gpu_util_percent", 0.0)) for s in samples) / count, 2),
        "memory_avg": round(sum(float(s.get("memory_util_percent", 0.0)) for s in samples) / count, 2),
        "memory_bandwidth_avg_gbps": 0.0,
    }


def _format_gap(record: dict[str, Any]) -> str:
    evaluation = record.get("evaluation", {})
    if evaluation.get("success"):
        return "met"
    missing_targets = evaluation.get("missing_targets", {})
    parts = []
    for metric, values in missing_targets.items():
        parts.append(f"{metric}: {values['observed']}/{values['target']}")
    return "; ".join(parts) if parts else "not met"


def render_iteration_table(iteration_records: list[dict[str, Any]]) -> str:
    lines = [
        "| Iteration | Request Workers | CPU Avg | GPU Avg | Memory Util Avg | Memory BW Avg (GB/s) | Success | Gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in iteration_records:
        metrics = record.get("metrics", {})
        success = "yes" if record.get("evaluation", {}).get("success") else "no"
        lines.append(
            f"| {record['iteration']} | {record.get('requested_concurrency', 0)} | {metrics.get('cpu_avg', 0.0):.2f} | {metrics.get('gpu_avg', 0.0):.2f} | {metrics.get('memory_avg', 0.0):.2f} | {metrics.get('memory_bandwidth_avg_gbps', 0.0):.2f} | {success} | {_format_gap(record)} |"
        )
    return "\n".join(lines)


def run_synthesis_loop(
    target_spec: dict[str, Any],
    max_iterations: int,
    generation_runner: GenerationRunner,
    execution_runner: ExecutionRunner,
    window_size: int = 3,
    concurrency_schedule: list[int] | None = None,
) -> dict[str, Any]:
    iteration_records: list[dict[str, Any]] = []
    feedback_summary: str | None = None
    schedule = concurrency_schedule or DEFAULT_CONCURRENCY_SCHEDULE

    for iteration in range(1, max_iterations + 1):
        requested_concurrency = schedule[min(iteration - 1, len(schedule) - 1)]
        iteration_target_spec = {
            **target_spec,
            "iteration_number": iteration,
            "preferred_request_workers": requested_concurrency,
            "allowed_request_workers": schedule,
        }
        candidate = generation_runner(iteration_target_spec, feedback_summary)
        execution = execution_runner(candidate["code"])
        metrics = summarize_iteration_metrics(execution["samples"])
        metrics["memory_bandwidth_avg_gbps"] = round(
            float(execution.get("memory_bandwidth_gbps", 0.0)), 2
        )
        evaluation = evaluate_stable_targets(
            execution["samples"], target_spec, window_size=window_size
        )
        record = {
            "iteration": iteration,
            "requested_concurrency": requested_concurrency,
            "code": candidate["code"],
            "raw_response": candidate["raw_response"],
            "execution": execution,
            "metrics": metrics,
            "evaluation": evaluation,
        }
        iteration_records.append(record)
        if evaluation["success"]:
            return {
                "success": True,
                "iterations": iteration,
                "iteration_records": iteration_records,
                "iteration_table": render_iteration_table(iteration_records),
                "final_candidate_code": candidate["code"],
            }
        feedback_summary = summarize_feedback(evaluation)

    return {
        "success": False,
        "iterations": max_iterations,
        "iteration_records": iteration_records,
        "iteration_table": render_iteration_table(iteration_records),
        "final_feedback": feedback_summary,
    }
