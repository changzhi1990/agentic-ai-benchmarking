from __future__ import annotations

import json
import re
from typing import Any

from .remote import RemoteConfig, quote_remote, run_remote_command


def extract_python_code(response_text: str) -> str:
    fenced_match = re.search(r"```python\s+(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    generic_match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return response_text.strip()


def summarize_feedback(evaluation: dict[str, Any]) -> str:
    if evaluation.get("success"):
        return "all targets satisfied"

    parts = []
    for metric, values in evaluation.get("missing_targets", {}).items():
        parts.append(
            f"{metric}: observed={values['observed']}, target={values['target']}"
        )
    if not parts:
        return "candidate did not satisfy targets"
    return "; ".join(parts)


def build_generation_messages(
    target_spec: dict[str, Any], feedback_summary: str | None
) -> list[dict[str, str]]:
    system_prompt = (
        "You generate only Python 3 benchmark code. "
        "Return a single python code block and no explanation. "
        "The script must run on Linux, use the Python standard library when possible, "
        "and drive CPU/GPU utilization toward the target metrics. "
        "Sustain concurrent load across the full measurement window rather than producing short bursts. "
        "Model a staged agentic workload with continuous request generation, CPU-side post-processing, "
        "result persistence, and summary aggregation, similar in spirit to swe-agentic-ai-benchmarking."
    )
    content = {
        "task": "Generate a Python benchmark workload candidate.",
        "targets": target_spec,
        "workload_shape": [
            "stage 1: continuously send OpenAI-compatible coding requests to the local vLLM server",
            "stage 2: on CPU, parse responses, validate response structure, hash or transform outputs, and persist per-request records",
            "stage 3: aggregate rolling business metrics during the run instead of waiting for the end",
            "stage 4: keep the whole pipeline active concurrently for the full measurement window",
        ],
        "requirements": [
            "Use urllib.request to send OpenAI-compatible requests to http://127.0.0.1:8000/v1/chat/completions",
            "Use the exact model identifier provided in targets.serving_model_id for every generation request",
            "Use targets.preferred_request_workers as the request worker count for this iteration unless the code explains why a higher count is needed",
            "Treat targets.allowed_request_workers as the concurrency ladder for later iterations",
            "Use multiprocessing, threading, or asyncio to keep request generation, CPU processing, and aggregation active at the same time",
            "Generate CPU pressure from real parsing, hashing, serialization, validation, and aggregation work instead of pure busy loops where possible",
            "Generate memory-bandwidth pressure from in-process Python bytearray, memoryview, array, or list copy/transform loops, not from stress-ng, sysbench, fio, dd, or similar tools",
            "Measure and print memory bandwidth periodically as lines in the form AAB_MEMORY_BW_GBPS=<value>",
            "Keep the script self-contained and executable with python3",
            "Run long enough for stable monitoring windows",
            "Avoid idle gaps, sleeps, staged pauses, or bursty phases that let utilization fall between samples",
            "Prefer sustained concurrent request workers over sequential request loops",
            "Persist per-request or per-batch results to local files during the run",
        ],
    }
    if feedback_summary:
        content["feedback"] = feedback_summary

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(content, sort_keys=True, indent=2)},
    ]


def generate_candidate_via_remote_vllm(
    remote_config: RemoteConfig,
    target_spec: dict[str, Any],
    feedback_summary: str | None,
    model: str,
    port: int = 8000,
    timeout: int = 300,
) -> dict[str, str]:
    messages = build_generation_messages(target_spec, feedback_summary)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1800,
    }
    payload_json = json.dumps(payload)
    remote_command = (
        "python3 - <<'PY'\n"
        "import json, urllib.request\n"
        f"payload = json.loads({payload_json!r})\n"
        f"url = 'http://127.0.0.1:{port}/v1/chat/completions'\n"
        "req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), "
        "headers={'Content-Type': 'application/json'})\n"
        "with urllib.request.urlopen(req, timeout=300) as resp:\n"
        "    print(resp.read().decode('utf-8'))\n"
        "PY"
    )
    raw_response = run_remote_command(remote_config, remote_command, timeout=timeout)
    response_payload = json.loads(raw_response)
    content = response_payload["choices"][0]["message"]["content"]
    return {
        "raw_response": content,
        "code": extract_python_code(content),
    }
