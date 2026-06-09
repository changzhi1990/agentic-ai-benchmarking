from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any, Callable

from .remote import RemoteConfig, run_remote_command

DEFAULT_AMDUPROF_PCM_PATH = "/home/user/zhi/AMDuProf_Nda_Linux_x64_5.0.1479/bin/AMDuProfPcm"


def _build_amduprof_pcm_command(
    tool_path: str,
    sudo_password: str,
    duration_seconds: int,
    sample_interval_seconds: int,
) -> str:
    interval_ms = max(sample_interval_seconds * 1000, 1200)
    return (
        "bash -lc "
        + repr(
            "printf "
            + repr(sudo_password + "\n")
            + " | sudo -S -p '' "
            + tool_path
            + f" -m memory -a --msr -r -d {duration_seconds} -I {interval_ms} 2>&1"
        )
    )


def _extract_amduprof_pcm_bandwidth_samples_gbps(output_text: str) -> list[float]:
    if "Total Mem Bw (GB/s)" not in output_text or "Profiling started" not in output_text:
        return []
    samples = []
    after_start = output_text.split("Profiling started", 1)[1]
    for line in after_start.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"^([0-9]+(?:\.[0-9]+)?),([0-9]+(?:\.[0-9]+)?),([0-9]+(?:\.[0-9]+)?),?$",
            line,
        )
        if not match:
            continue
        total_bw = float(match.group(1))
        if total_bw <= 1.0:
            continue
        samples.append(total_bw)
    return samples


def _extract_amduprof_pcm_bandwidth_gbps(output_text: str) -> float:
    samples = _extract_amduprof_pcm_bandwidth_samples_gbps(output_text)
    if not samples:
        return 0.0
    return round(sum(samples) / len(samples), 2)


def _extract_summary_bandwidth_gbps(output_text: str) -> float:
    if '"memory_reports"' not in output_text:
        return 0.0

    start = output_text.rfind("{")
    while start != -1:
        candidate = output_text[start:].strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = output_text.rfind("{", 0, start)
            continue
        reports = payload.get("memory_reports", [])
        if reports:
            total = sum(float(item.get("bandwidth_gbps", 0.0)) for item in reports)
            return round(total, 2)
        start = output_text.rfind("{", 0, start)
    return 0.0


def _extract_memory_bandwidth_gbps(stderr_text: str) -> float:
    amduprof_bandwidth = _extract_amduprof_pcm_bandwidth_gbps(stderr_text)
    if amduprof_bandwidth > 0:
        return amduprof_bandwidth

    summary_bandwidth = _extract_summary_bandwidth_gbps(stderr_text)
    if summary_bandwidth > 0:
        return summary_bandwidth

    stdout_matches = re.findall(r"AAB_MEMORY_BW_GBPS=([0-9]+(?:\.[0-9]+)?)", stderr_text)
    if stdout_matches:
        values = [float(item) for item in stdout_matches]
        return round(sum(values) / len(values), 2)

    matches = re.findall(r"memory rate:\s+([0-9]+(?:\.[0-9]+)?)\s+MB/sec", stderr_text)
    if not matches:
        return 0.0
    values = [float(item) / 1024.0 for item in matches]
    return round(sum(values) / len(values), 2)


def execute_remote_candidate(
    remote_config: RemoteConfig,
    code: str,
    duration_seconds: int = 90,
    sample_interval_seconds: int = 5,
    grace_period_seconds: int = 10,
    amduprof_pcm_path: str = DEFAULT_AMDUPROF_PCM_PATH,
    runner: Callable[[RemoteConfig, str, int], str] = run_remote_command,
) -> dict[str, Any]:
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    remote_command = f"""python3 - <<'PY'
import base64
import json
import pathlib
import subprocess
import tempfile
import time

workdir = pathlib.Path(tempfile.mkdtemp(prefix="aab-workload-"))
candidate_path = workdir / "candidate.py"
candidate_path.write_text(base64.b64decode("{encoded}").decode("utf-8"), encoding="utf-8")
stdout_path = workdir / "stdout.log"
stderr_path = workdir / "stderr.log"
pcm_path = workdir / "amduprof-pcm.log"
samples_path = workdir / "samples.jsonl"

with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(["python3", str(candidate_path)], stdout=stdout_handle, stderr=stderr_handle)
    pcm_command = {_build_amduprof_pcm_command(amduprof_pcm_path, remote_config.password, duration_seconds, sample_interval_seconds)!r}
    with pcm_path.open("w", encoding="utf-8") as pcm_handle:
        pcm_process = subprocess.Popen(["bash", "-lc", pcm_command], stdout=pcm_handle, stderr=subprocess.STDOUT)
    start = time.time()
    timestamp_index = 0
    with samples_path.open("w", encoding="utf-8") as samples_handle:
        while process.poll() is None and time.time() - start < {duration_seconds}:
            cpu_cmd = "mpstat 1 1 | awk '/Average:/ && $2==\\"all\\" {{printf \\"%.2f\\",100-$NF}}'"
            mem_cmd = "free | awk '/Mem:/ {{printf \\"%.2f\\", $3*100/$2}}'"
            gpu_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{{sum+=$1}} END {{if (NR>0) printf \\"%.2f\\", sum/NR; else printf \\"0.00\\"}}'"
            cpu = subprocess.check_output(["bash", "-lc", cpu_cmd], text=True).strip() or "0.00"
            mem = subprocess.check_output(["bash", "-lc", mem_cmd], text=True).strip() or "0.00"
            gpu = subprocess.check_output(["bash", "-lc", gpu_cmd], text=True).strip() or "0.00"
            sample = {{
                "timestamp_s": timestamp_index * {sample_interval_seconds},
                "cpu_util_percent": float(cpu),
                "memory_util_percent": float(mem),
                "gpu_util_percent": float(gpu),
            }}
            samples_handle.write(json.dumps(sample) + "\\n")
            samples_handle.flush()
            timestamp_index += 1
            time.sleep({sample_interval_seconds})
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout={grace_period_seconds})
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout={grace_period_seconds})
    if pcm_process.poll() is None:
        pcm_process.terminate()
        try:
            pcm_process.wait(timeout={grace_period_seconds})
        except subprocess.TimeoutExpired:
            pcm_process.kill()
            pcm_process.wait(timeout={grace_period_seconds})
    payload = {{
        "returncode": process.returncode,
        "amduprof_pcm": pcm_path.read_text(encoding="utf-8"),
        "stdout": stdout_path.read_text(encoding="utf-8"),
        "stderr": stderr_path.read_text(encoding="utf-8"),
        "samples": [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()],
    }}
    result = json.dumps(payload)
print(result)
PY"""
    raw = runner(
        remote_config,
        remote_command,
        timeout=duration_seconds + 180 + grace_period_seconds,
    )
    payload = json.loads(raw)
    combined_output = "\n".join(
        [payload.get("amduprof_pcm", ""), payload.get("stdout", ""), payload.get("stderr", "")]
    )
    amduprof_samples = _extract_amduprof_pcm_bandwidth_samples_gbps(
        payload.get("amduprof_pcm", "")
    )
    payload["memory_bandwidth_samples_gbps"] = amduprof_samples
    payload["memory_bandwidth_peak_gbps"] = (
        round(max(amduprof_samples), 2) if amduprof_samples else 0.0
    )
    payload["memory_bandwidth_gbps"] = _extract_memory_bandwidth_gbps(combined_output)
    return payload
