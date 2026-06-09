from __future__ import annotations

from typing import Any


def _window_average(samples: list[dict[str, Any]], metric: str) -> float:
    return round(sum(float(sample.get(metric, 0.0)) for sample in samples) / len(samples), 2)


def evaluate_stable_targets(
    samples: list[dict[str, Any]],
    target_spec: dict[str, float],
    window_size: int,
) -> dict[str, Any]:
    if len(samples) < window_size:
        missing = {
            metric: {"target": target, "observed": _window_average(samples, metric) if samples else 0.0}
            for metric, target in target_spec.items()
        }
        return {
            "success": False,
            "reason": "insufficient_samples",
            "missing_targets": missing,
        }

    for start in range(0, len(samples) - window_size + 1):
        window = samples[start : start + window_size]
        missing: dict[str, dict[str, float]] = {}
        for metric, target in target_spec.items():
            observed = _window_average(window, metric)
            if observed < float(target):
                missing[metric] = {"target": float(target), "observed": observed}
        if not missing:
            return {
                "success": True,
                "window_start_index": start,
                "window_size": window_size,
                "averages": {metric: _window_average(window, metric) for metric in target_spec},
            }

    trailing_window = samples[-window_size:]
    missing = {
        metric: {
            "target": float(target),
            "observed": _window_average(trailing_window, metric),
        }
        for metric, target in target_spec.items()
        if _window_average(trailing_window, metric) < float(target)
    }
    return {
        "success": False,
        "reason": "targets_not_met",
        "window_start_index": len(samples) - window_size,
        "window_size": window_size,
        "missing_targets": missing,
    }
