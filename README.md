# Bug Report Coding-Agent Benchmark

This repository contains a synthetic coding-agent benchmark built around a
realistic bug-fixing workflow. The workload starts from a bug report, searches a
large synthetic codebase, expands candidate files through metadata and dependency
graphs, builds a prompt context, and sends coding-agent requests to a local vLLM
server.

The benchmark is designed to study how a local coding agent behaves when code
search, context construction, model inference, and host-side tool execution run
at the same time.

## Scenario

The simulated task is a bug report for a retry-state failure:

```text
retry state is lost after the first timeout
```

The report describes a task system where the first upstream timeout should
advance and persist `retry_state`, but the persisted state remains stale. A real
coding agent would need to inspect retry update logic, task persistence code,
caller/callee relationships, and tests that exercise the failure path.

This benchmark models that workflow with a generated codebase containing many
similar modules and symbols, such as:

- `apply_retry_update_*`
- `persist_task_snapshot_*`
- `retry_state`
- `task_store`
- timeout and downstream task state paths

The goal is not to test whether the model can produce a correct patch. The goal
is to reproduce the system behavior of a coding agent while it performs bug
localization and context preparation at scale.

## Workflow

Each benchmark run follows this coding-agent flow:

1. Read a structured bug report with observed behavior, reproduction steps,
   failure artifacts, and expected behavior.
2. Generate a large synthetic repository with repeated but slightly different
   modules and symbols.
3. Build repository metadata: posting lists, symbol records, dependency graph,
   and test-affinity graph.
4. Scan repository text for bug-report terms and related symbols.
5. Expand candidates through dependency and test-affinity relationships.
6. Rerank candidate chunks and pack a compact context for the model.
7. Send OpenAI-compatible chat completion requests to local vLLM.
8. Record request latency, success/failure count, token usage, scanned bytes,
   context size, and response size.
9. Sample host resources while the agent loop runs.

This represents the bug-localization and context-construction stage of a coding
agent. It does not yet apply patches or run a test suite.

## Worker Types

`request_workers` represent concurrent coding-agent loops. They perform code
search, candidate expansion, context packing, and vLLM requests.

`workers` represent CPU-side tool workers. They model the parallel repository
search and metadata traversal that support coding-agent context construction.
This is the main scaling axis in the included benchmark results.

`compute_workers` represent additional CPU-heavy tool work, such as hashing,
shuffle, and buffer mutation. They are used to study contention between CPU
compute work, code-search work, and vLLM request serving.

## Measurements

The benchmark records both system metrics and agent/business metrics.

System metrics include:

- CPU utilization from `mpstat`
- GPU utilization from `nvidia-smi`
- host memory utilization from `free`
- CPU memory bandwidth from `AMDuProfPcm`
- per-run memory bandwidth average, peak, and raw samples

Agent/business metrics include:

- completed requests
- failed requests
- failure rate
- request throughput
- request latency average, p50, and p95
- prompt token throughput
- completion token throughput
- total token throughput
- scanned repository bytes
- packed context bytes
- retrieval amplification
- response character count

## Remote Setup Used

The remote server used during testing was:

- Host: `10.83.32.172`
- User: `user`
- Model: `/home/user/models/Qwen2.5-Coder-32B-Instruct`
- vLLM image: `vllm/vllm-openai:latest`
- Tensor parallel size: `8`
- AMDuProfPcm: `/home/user/zhi/AMDuProf_Nda_Linux_x64_5.0.1479/bin/AMDuProfPcm`

AMDuProfPcm is invoked with:

```bash
sudo AMDuProfPcm -m memory -a --msr -r
```

## Run Tests

```bash
python3 -m unittest tests.test_bug_report_workload tests.test_remote -v
```

## Run The Coding-Agent Workload

This run uses vLLM requests and CPU-side tool workers:

```bash
AAB_REMOTE_HOST=10.83.32.172 \
AAB_REMOTE_USER=user \
AAB_REMOTE_PASSWORD=000000 \
python3 scripts/run_remote_bug_report_workload.py \
  --duration-seconds 36 \
  --sample-interval 5 \
  --request-workers 2 \
  --workers 96 \
  --compute-workers 64 \
  --repo-files 2048 \
  --chunk-repeats 8 \
  --memory-block-mb 128 \
  --memory-stream-rounds 8 \
  --top-k 8 \
  --output-dir artifacts/balanced-workers-96
```

## Run A Workers Sweep

The latest sweep varied the main `workers` axis:

```text
workers=16,32,48,64,96,128,160,190,256
request_workers=2
compute_workers=64
duration_seconds=36
repo_files=2048
chunk_repeats=8
memory_block_mb=128
memory_stream_rounds=8
top_k=8
```

The latest sweep artifacts are:

- `artifacts/workers-sweep-business-metrics.csv`
- `artifacts/workers-sweep-business-metrics.json`

Best balanced point observed:

```text
workers=96
CPU avg: 98.38%
GPU avg: 91.30%
AMDuProfPcm memory bandwidth avg: 414.81 GB/s
AMDuProfPcm memory bandwidth peak: 583.19 GB/s
request throughput: 0.500 req/s
latency avg: 4.119 s
```

