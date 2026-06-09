from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Callable

from .remote import RemoteConfig, run_remote_command
from .remote_workload import execute_remote_candidate
from .synthesis import summarize_iteration_metrics
from .vllm_remote import run_remote_vllm_healthcheck

DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_MODEL = "/home/user/models/Qwen2.5-Coder-32B-Instruct"
DEFAULT_TENSOR_PARALLEL_SIZE = 8
DEFAULT_BUG_REPORT = {
    "title": "Bug report: retry state is lost after the first timeout",
    "observed_behavior": (
        "Requests that should succeed after a retry fail with a stale retry_state and "
        "leave follow-up tasks marked as permanently errored."
    ),
    "repro_steps": [
        "Create a task with retry_policy.max_attempts=3.",
        "Force the first upstream call to raise TimeoutError.",
        "Run the worker loop and inspect the persisted task state.",
    ],
    "failure_artifacts": [
        "AssertionError: expected retry_state.attempt=2, got 1",
        "worker.py:218 in apply_retry_update",
        "task_store.py:91 in persist_task_snapshot",
    ],
    "expected_behavior": (
        "The retry counter should advance, the refreshed state should be persisted, "
        "and downstream tasks should observe the new attempt metadata."
    ),
}


def _resolve_workers(workers: int | None, memory_workers: int | None) -> int:
    if workers is not None and memory_workers is not None and workers != memory_workers:
        raise ValueError("workers and memory_workers must match when both are provided")
    return workers if workers is not None else (memory_workers if memory_workers is not None else 32)


def probe_existing_remote_vllm(
    remote_config: RemoteConfig,
    model: str,
    port: int,
) -> dict[str, Any] | None:
    try:
        raw = run_remote_command(
            remote_config,
            f"curl -sf http://127.0.0.1:{port}/v1/models",
            timeout=30,
        )
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    model_ids = [item.get("id") for item in payload.get("data", [])]
    if model not in model_ids:
        return None
    return {
        "status": "ok",
        "ready": True,
        "reused_existing": True,
        "model": model,
        "port": port,
        "served_model_count": len(model_ids),
        "served_model_ids": model_ids,
    }


def build_bug_report_candidate(
    model: str,
    port: int = 8000,
    duration_seconds: int = 90,
    request_workers: int = 2,
    workers: int | None = None,
    memory_workers: int | None = None,
    compute_workers: int = 64,
    repo_files: int = 1536,
    chunk_repeats: int = 6,
    memory_block_mb: int = 128,
    memory_stream_rounds: int = 8,
    top_k: int = 10,
    bug_report: dict[str, Any] | None = None,
) -> str:
    report = bug_report or DEFAULT_BUG_REPORT
    request_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    worker_count = _resolve_workers(workers, memory_workers)

    return textwrap.dedent(
        f"""\
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import random
import tempfile
import time
import urllib.request

model = {model!r}
port = {port}
request_url = {request_url!r}
duration_seconds = {duration_seconds}
request_workers = {request_workers}
workers = {worker_count}
compute_workers = {compute_workers}
repo_files = {repo_files}
chunk_repeats = {chunk_repeats}
memory_block_bytes = {memory_block_mb} * 1024 * 1024
memory_stream_rounds = {memory_stream_rounds}
top_k = {top_k}
grace_seconds = 6
bug_report = {report!r}
output_dir = Path(tempfile.mkdtemp(prefix="aab-bug-report-"))
summary_path = output_dir / "summary.json"

report_terms = sorted({{
    token.strip(".,:[]()").lower()
    for field in (
        [bug_report["title"], bug_report["observed_behavior"], bug_report["expected_behavior"]]
        + bug_report["failure_artifacts"]
        + bug_report["repro_steps"]
    )
    for token in field.split()
    if len(token.strip(".,:[]()")) >= 4
}})


def build_repo_corpus() -> list[dict[str, str]]:
    corpus = []
    for index in range(repo_files):
        module_name = f"module_{{index:05d}}"
        dependency_name = f"module_{{(index * 7 + 11) % repo_files:05d}}"
        symbol_name = f"apply_retry_update_{{index % 37}}"
        noisy_symbol = f"persist_task_snapshot_{{(index * 3) % 41}}"
        repeated = []
        for _ in range(chunk_repeats):
            repeated.append(
                "def "
                + symbol_name
                + "(task, retry_state, timeout_error):\\n"
                + "    task['module'] = '"
                + module_name
                + "'\\n"
                + "    task['dependency'] = '"
                + dependency_name
                + "'\\n"
                + "    task['retry_state'] = retry_state\\n"
                + "    if timeout_error:\\n"
                + "        return "
                + noisy_symbol
                + "(task, retry_state)\\n"
                + "    return task\\n\\n"
            )
        repeated.append(
            "class RetryCoordinator:\\n"
            + "    def persist_task_snapshot(self, task_store, task):\\n"
            + "        task_store.write(task['module'], task)\\n\\n"
        )
        repeated.append(
            "# failure keywords: timeout retry stale state snapshot persisted metadata downstream\\n"
        )
        corpus.append({{"path": f"src/{{module_name}}.py", "text": "".join(repeated)}})
    return corpus


corpus = build_repo_corpus()


def build_posting_lists() -> dict[str, list[int]]:
    posting_lists = {{term: [] for term in report_terms}}
    for chunk_id, entry in enumerate(corpus):
        lowered = entry["text"].lower()
        for term in report_terms:
            if term in lowered:
                posting_lists[term].append(chunk_id)
    return posting_lists


def build_symbol_table() -> list[dict[str, object]]:
    symbols = []
    for chunk_id, entry in enumerate(corpus):
        path = entry["path"]
        symbols.append(
            {{
                "chunk_id": chunk_id,
                "path": path,
                "symbol": f"retry_symbol_{{chunk_id % 257}}",
                "score_bias": (chunk_id * 13) % 17,
            }}
        )
    return symbols


def build_dependency_graph() -> list[list[int]]:
    graph = []
    total = len(corpus)
    for chunk_id in range(total):
        graph.append(
            [
                (chunk_id + 1) % total,
                (chunk_id * 7 + 11) % total,
                (chunk_id * 13 + 17) % total,
            ]
        )
    return graph


def build_test_affinity_map() -> list[list[int]]:
    affinity = []
    total = len(corpus)
    for chunk_id in range(total):
        affinity.append(
            [
                (chunk_id * 5 + 3) % total,
                (chunk_id * 11 + 7) % total,
            ]
        )
    return affinity


posting_lists = build_posting_lists()
symbol_table = build_symbol_table()
dependency_graph = build_dependency_graph()
test_affinity_map = build_test_affinity_map()


def build_role_cores() -> dict[str, list[int]]:
    cpu_total = max(os.cpu_count() or 1, 1)
    cores = list(range(cpu_total))
    request_count = min(request_workers, max(1, cpu_total // 12)) if request_workers > 0 else 0
    memory_count = min(max(workers, 1), max(1, cpu_total - request_count))
    request_cores = cores[:request_count]
    memory_end = min(cpu_total, request_count + memory_count)
    memory_cores = cores[request_count:memory_end]
    compute_cores = cores[memory_end:]
    if not memory_cores:
        memory_cores = request_cores or cores
    if not compute_cores:
        compute_cores = cores
    return {{
        "request": request_cores or cores,
        "memory": memory_cores or cores,
        "compute": compute_cores or cores,
    }}


ROLE_CORES = build_role_cores()


def bind_worker(role: str, worker_id: int) -> None:
    cores = ROLE_CORES.get(role) or [0]
    try:
        os.sched_setaffinity(0, {{cores[worker_id % len(cores)]}})
    except (AttributeError, OSError):
        return None


def select_context(worker_id: int, iteration: int) -> tuple[str, int]:
    hits: list[tuple[int, str, str]] = []
    scanned_bytes = 0
    for entry in corpus:
        text = entry["text"]
        scanned_bytes += len(text)
        score = 0
        lowered = text.lower()
        for term in report_terms:
            score += lowered.count(term)
        score += lowered.count("retry_state")
        score += lowered.count("persist_task_snapshot")
        if score:
            hits.append((score, entry["path"], text[:768]))
    hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = hits[:top_k]
    lines = [
        "Bug report",
        json.dumps(bug_report, indent=2, sort_keys=True),
        "",
        f"worker_id={{worker_id}} iteration={{iteration}}",
    ]
    for score, path, snippet in selected:
        lines.append(f"### score={{score}} file={{path}}")
        lines.append(snippet)
    return "\\n".join(lines), scanned_bytes


def candidate_rerank(worker_id: int, iteration: int) -> tuple[list[int], int]:
    candidate_scores = {{}}
    scanned_bytes = 0
    for term in report_terms:
        for chunk_id in posting_lists.get(term, []):
            candidate_scores[chunk_id] = candidate_scores.get(chunk_id, 0) + 1
            scanned_bytes += 8
    frontier = list(candidate_scores)[: max(32, top_k * 8)]
    for _ in range(2):
        expanded = []
        for chunk_id in frontier:
            neighbors = dependency_graph[chunk_id]
            tests = test_affinity_map[chunk_id]
            scanned_bytes += (len(neighbors) + len(tests)) * 8
            for neighbor in neighbors + tests:
                candidate_scores[neighbor] = candidate_scores.get(neighbor, 0) + 1
                expanded.append(neighbor)
        frontier = expanded[: max(32, top_k * 8)]
    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (item[1] + symbol_table[item[0]]["score_bias"], item[0]),
        reverse=True,
    )
    return [chunk_id for chunk_id, _score in ranked[: max(top_k * 4, 32)]], scanned_bytes


def context_packing(candidate_ids: list[int]) -> tuple[list[dict[str, object]], int]:
    packed = []
    packed_ids = set()
    moved_bytes = 0
    for chunk_id in candidate_ids:
        if chunk_id in packed_ids:
            continue
        entry = corpus[chunk_id]
        snippet = entry["text"][:768]
        packed.append(
            {{
                "chunk_id": chunk_id,
                "path": entry["path"],
                "symbol": symbol_table[chunk_id]["symbol"],
                "snippet": snippet,
            }}
        )
        packed_ids.add(chunk_id)
        moved_bytes += len(snippet)
        if len(packed) >= top_k:
            break
    return packed, moved_bytes


def impact_scan(candidate_ids: list[int]) -> int:
    scanned = 0
    for chunk_id in candidate_ids:
        for neighbor in dependency_graph[chunk_id]:
            scanned += len(corpus[neighbor]["text"])
        for test_chunk in test_affinity_map[chunk_id]:
            scanned += len(corpus[test_chunk]["text"])
    return scanned


def invoke_vllm(context: str, timeout_seconds: float) -> dict[str, object]:
    payload = {{
        "model": model,
        "temperature": 0.1,
        "max_tokens": 192,
        "messages": [
            {{
                "role": "system",
                "content": (
                    "You are a bug-fixing coding agent. "
                    "Read the bug report and candidate files, explain the likely root cause, "
                    "and propose a concise patch plan."
                ),
            }},
            {{
                "role": "user",
                "content": context,
            }},
        ],
    }}
    request = urllib.request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={{"Content-Type": "application/json"}},
    )
    with urllib.request.urlopen(request, timeout=max(timeout_seconds, 5.0)) as response:
        return json.loads(response.read().decode("utf-8"))


def request_worker(worker_id: int, deadline: float) -> None:
    bind_worker("request", worker_id)
    request_records_path = output_dir / f"request-records-{{worker_id:02d}}.jsonl"
    iteration = 0
    completed_requests = 0
    failed_requests = 0
    scanned_bytes_total = 0
    context_bytes_total = 0
    response_chars_total = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0
    latencies_seconds = []
    successful_latencies_seconds = []
    failed_latencies_seconds = []
    while time.monotonic() < deadline:
        base_context, scanned_bytes = select_context(worker_id, iteration)
        candidate_ids, rerank_bytes = candidate_rerank(worker_id, iteration)
        packed_context, packed_bytes = context_packing(candidate_ids)
        impact_bytes = impact_scan(candidate_ids[:top_k])
        scanned_bytes_total += scanned_bytes + rerank_bytes + packed_bytes + impact_bytes
        context_lines = [base_context, "", "### context_packing"]
        for item in packed_context:
            context_lines.append(
                item["path"]
                + "::"
                + item["symbol"]
                + "::"
                + str(item["chunk_id"])
                + "\\n"
                + item["snippet"]
            )
        context = "\\n".join(context_lines)
        context_bytes = len(context.encode("utf-8"))
        context_bytes_total += context_bytes
        started = time.monotonic()
        record = {{
            "worker_id": worker_id,
            "iteration": iteration,
            "started_at": started,
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "context_bytes": context_bytes,
        }}
        try:
            remaining = max(deadline - time.monotonic(), 0.0)
            if remaining <= 0:
                break
            payload = invoke_vllm(context, timeout_seconds=min(remaining + grace_seconds, 60.0))
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {{}})
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)
            record["status"] = "ok"
            latency_seconds = round(time.monotonic() - started, 3)
            record["latency_seconds"] = latency_seconds
            record["response_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            record["response_chars"] = len(content)
            record["prompt_tokens"] = prompt_tokens
            record["completion_tokens"] = completion_tokens
            record["total_tokens"] = total_tokens
            completed_requests += 1
            response_chars_total += len(content)
            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens
            total_tokens_total += total_tokens
            latencies_seconds.append(latency_seconds)
            successful_latencies_seconds.append(latency_seconds)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            latency_seconds = round(time.monotonic() - started, 3)
            record["latency_seconds"] = latency_seconds
            failed_requests += 1
            latencies_seconds.append(latency_seconds)
            failed_latencies_seconds.append(latency_seconds)
        with request_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\\n")
        iteration += 1
    return {{
        "kind": "request",
        "worker_id": worker_id,
        "completed_requests": completed_requests,
        "failed_requests": failed_requests,
        "scanned_bytes": scanned_bytes_total,
        "context_bytes": context_bytes_total,
        "response_chars": response_chars_total,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total,
        "total_tokens": total_tokens_total,
        "latencies_seconds": latencies_seconds,
        "successful_latencies_seconds": successful_latencies_seconds,
        "failed_latencies_seconds": failed_latencies_seconds,
    }}


def request_process_main(worker_id: int, deadline: float, result_queue) -> None:
    result_queue.put(request_worker(worker_id, deadline))


def memory_worker(worker_id: int, deadline: float, result_queue) -> None:
    bind_worker("memory", worker_id)
    random.seed(worker_id)
    source = bytearray(((worker_id * 17) + index) % 251 for index in range(memory_block_bytes))
    target = bytearray(memory_block_bytes)
    scatter_buffer = bytearray(memory_block_bytes)
    shadow_buffer = bytearray(memory_block_bytes)
    copied_bytes = 0
    scanned_bytes = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        for _ in range(memory_stream_rounds):
            target[:] = source
            scatter_buffer[:] = target
            shadow_buffer[:] = scatter_buffer
            source[:] = shadow_buffer
            source[:65536] = target[-65536:]
            copied_bytes += (len(source) + len(target)) * 4
        gathered_scores = []
        for posting in posting_lists.values():
            scanned_bytes += len(posting) * 8
            for chunk_id in posting[: max(top_k * 4, 32)]:
                symbol = symbol_table[chunk_id]
                gathered_scores.append(
                    (chunk_id, symbol["score_bias"], dependency_graph[chunk_id], test_affinity_map[chunk_id])
                )
                scanned_bytes += 64
        gathered_scores.sort(key=lambda item: (item[1], item[0]), reverse=True)
        packed_chunk_ids = []
        for chunk_id, _score_bias, neighbors, test_neighbors in gathered_scores[: max(top_k * 8, 64)]:
            packed_chunk_ids.append(chunk_id)
            for neighbor in neighbors:
                packed_chunk_ids.append(neighbor)
            for test_chunk in test_neighbors:
                packed_chunk_ids.append(test_chunk)
        packed_chunk_ids = packed_chunk_ids[: max(top_k * 16, 128)]
        packed_cursor = 0
        for chunk_id in packed_chunk_ids:
            snippet = corpus[chunk_id]["text"].encode("utf-8")[:2048]
            snippet_len = len(snippet)
            scatter_buffer[packed_cursor:packed_cursor + snippet_len] = snippet
            packed_cursor = (packed_cursor + snippet_len) % max(memory_block_bytes - 4096, 1)
            copied_bytes += snippet_len
            scanned_bytes += snippet_len
        scatter_buffer[:262144] = target[-262144:]
        copied_bytes += 262144
    elapsed = max(time.monotonic() - started, 1e-6)
    worker_bandwidth = copied_bytes / elapsed / 1_000_000_000
    print(f"AAB_MEMORY_BW_GBPS={{worker_bandwidth:.2f}}", flush=True)
    result_queue.put({{
        "kind": "memory",
        "worker_id": worker_id,
        "copied_bytes": copied_bytes,
        "scanned_bytes": scanned_bytes,
        "bandwidth_gbps": round(worker_bandwidth, 2),
    }})


def compute_worker(worker_id: int, deadline: float, result_queue) -> None:
    bind_worker("compute", worker_id)
    random.seed(worker_id + 10_000)
    block = bytearray(((worker_id * 29) + index) % 251 for index in range(8 * 1024 * 1024))
    corpus_bytes = [entry["text"].encode("utf-8")[:4096] for entry in corpus[: min(128, len(corpus))]]
    digests = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        random.shuffle(corpus_bytes)
        for snippet in corpus_bytes:
            digest = hashlib.blake2b(snippet + block[:4096], digest_size=32).digest()
            block[:32] = digest
            digests += 1
        block.reverse()
    elapsed = max(time.monotonic() - started, 1e-6)
    result_queue.put({{
        "kind": "compute",
        "worker_id": worker_id,
        "digests": digests,
        "digests_per_second": round(digests / elapsed, 2),
    }})


def main() -> None:
    runtime_seconds = max(5, duration_seconds - 8)
    deadline = time.monotonic() + runtime_seconds
    result_queue = multiprocessing.Queue()
    processes = []
    expected_reports = 0
    for worker_id in range(request_workers):
        process = multiprocessing.Process(
            target=request_process_main,
            args=(worker_id, deadline, result_queue),
        )
        process.daemon = True
        process.start()
        processes.append(process)
        expected_reports += 1
    for worker_id in range(workers):
        process = multiprocessing.Process(
            target=memory_worker,
            args=(worker_id, deadline, result_queue),
        )
        process.daemon = True
        process.start()
        processes.append(process)
        expected_reports += 1
    for worker_id in range(compute_workers):
        process = multiprocessing.Process(
            target=compute_worker,
            args=(worker_id, deadline, result_queue),
        )
        process.daemon = True
        process.start()
        processes.append(process)
        expected_reports += 1
    memory_reports = []
    compute_reports = []
    request_reports = []
    reap_deadline = deadline + grace_seconds
    for process in processes:
        remaining_wait = max(reap_deadline - time.monotonic(), 0.0)
        process.join(timeout=remaining_wait)
        if process.is_alive():
            process.terminate()
            process.join(timeout=grace_seconds)
    for _ in range(expected_reports):
        try:
            report = result_queue.get(timeout=1.0)
        except queue.Empty:
            break
        if report.get("kind") == "memory":
            memory_reports.append(report)
        elif report.get("kind") == "compute":
            compute_reports.append(report)
        elif report.get("kind") == "request":
            request_reports.append(report)
    memory_reports.sort(key=lambda item: item["worker_id"])
    compute_reports.sort(key=lambda item: item["worker_id"])
    request_reports.sort(key=lambda item: item["worker_id"])
    final_progress = {{
        "completed_requests": sum(item.get("completed_requests", 0) for item in request_reports),
        "failed_requests": sum(item.get("failed_requests", 0) for item in request_reports),
        "scanned_bytes": sum(item.get("scanned_bytes", 0) for item in request_reports),
        "context_bytes": sum(item.get("context_bytes", 0) for item in request_reports),
        "response_chars": sum(item.get("response_chars", 0) for item in request_reports),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in request_reports),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in request_reports),
        "total_tokens": sum(item.get("total_tokens", 0) for item in request_reports),
    }}
    summary = {{
        "compute_reports": compute_reports,
        "compute_workers": compute_workers,
        "output_dir": str(output_dir),
        "memory_reports": memory_reports,
        "progress": final_progress,
        "request_reports": request_reports,
        "request_workers": request_workers,
        "workers": workers,
        "repo_files": repo_files,
    }}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""
    )


def run_remote_bug_report_workload(
    remote_config: RemoteConfig,
    model: str = DEFAULT_MODEL,
    image: str = DEFAULT_IMAGE,
    port: int = 8000,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    duration_seconds: int = 90,
    sample_interval_seconds: int = 5,
    request_workers: int = 2,
    workers: int | None = None,
    memory_workers: int | None = None,
    compute_workers: int = 64,
    repo_files: int = 1536,
    chunk_repeats: int = 6,
    memory_block_mb: int = 128,
    memory_stream_rounds: int = 8,
    top_k: int = 10,
    bug_report: dict[str, Any] | None = None,
    skip_vllm_healthcheck: bool = False,
    existing_vllm_probe: Callable[[RemoteConfig, str, int], dict[str, Any] | None] = probe_existing_remote_vllm,
    healthcheck_runner: Callable[..., dict[str, Any]] = run_remote_vllm_healthcheck,
    execution_runner: Callable[..., dict[str, Any]] = execute_remote_candidate,
) -> dict[str, Any]:
    worker_count = _resolve_workers(workers, memory_workers)
    if skip_vllm_healthcheck or request_workers <= 0:
        health = {
            "status": "skipped",
            "reason": "no vLLM requests configured",
            "model": model,
            "port": port,
        }
    else:
        health = existing_vllm_probe(remote_config, model, port)
        if health is not None:
            pass
        else:
            health = healthcheck_runner(
                remote_config,
                model=model,
                port=port,
                image=image,
                tensor_parallel_size=tensor_parallel_size,
                max_attempts=90,
                sleep_seconds=2.0,
                cleanup=False,
            )
    code = build_bug_report_candidate(
        model=model,
        port=port,
        duration_seconds=duration_seconds,
        request_workers=request_workers,
        workers=worker_count,
        compute_workers=compute_workers,
        repo_files=repo_files,
        chunk_repeats=chunk_repeats,
        memory_block_mb=memory_block_mb,
        memory_stream_rounds=memory_stream_rounds,
        top_k=top_k,
        bug_report=bug_report,
    )
    execution = execution_runner(
        remote_config,
        code,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
    )
    summary = summarize_iteration_metrics(execution.get("samples", []))
    summary["memory_bandwidth_avg_gbps"] = round(
        float(execution.get("memory_bandwidth_gbps", 0.0)),
        2,
    )

    return {
        "status": "success" if execution.get("returncode", 1) == 0 else "error",
        "healthcheck": health,
        "summary": summary,
        "execution": execution,
        "config": {
            "model": model,
            "image": image,
            "port": port,
            "tensor_parallel_size": tensor_parallel_size,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "request_workers": request_workers,
            "workers": worker_count,
            "compute_workers": compute_workers,
            "repo_files": repo_files,
            "chunk_repeats": chunk_repeats,
            "memory_block_mb": memory_block_mb,
            "memory_stream_rounds": memory_stream_rounds,
            "top_k": top_k,
        },
        "candidate_code": code,
    }


def write_bug_report_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bug-report-workload-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
