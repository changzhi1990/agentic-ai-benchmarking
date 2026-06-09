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

## Resource Sensitivity By Stage

Bug report parsing is CPU sensitive. The benchmark reads a structured issue
description, extracts terms such as retry state, timeout, persistence, and
failure artifacts, and prepares the query terms used by later repository search.
This stage does not use the GPU.

Synthetic repository generation is CPU and host-memory sensitive. The workload
creates many similar modules and symbols to mimic a large codebase with repeated
patterns and distracting candidates. It mainly stresses Python object creation,
string construction, and host memory capacity.

Metadata construction is CPU and CPU-memory-bandwidth sensitive. The benchmark
builds posting lists, symbol records, dependency graphs, and test-affinity
graphs. These structures model the indexes a coding agent would use for code
search, symbol lookup, caller/callee expansion, and test-related localization.

Repository scanning is CPU-memory-bandwidth sensitive. Request workers scan
source text and metadata to find bug-report terms and related symbols. This is a
host-side search stage and does not use the GPU.

Candidate expansion is CPU and CPU-memory-bandwidth sensitive. The workload
expands initial matches through dependency and test-affinity graphs. This stage
models a coding agent following imports, call paths, and likely test coverage
relationships to find files that should be included in context.

Candidate reranking is CPU sensitive. Candidate chunks are scored and sorted so
that only the most relevant code regions are kept. This models the ranking step
between broad repository search and final prompt construction.

Context packing is CPU and CPU-memory-bandwidth sensitive. The selected chunks
are converted into a compact model prompt with file paths, symbols, and snippets.
This stage is sensitive to host memory movement because it repeatedly reads,
filters, and assembles code fragments.

CPU-side tool work is CPU and CPU-memory-bandwidth sensitive. The `workers`
processes perform repository-support work, metadata traversal, and large host
memory movement. The `compute_workers` processes add CPU-heavy hashing and
buffer mutation to model tool-side compute pressure.

vLLM request construction is CPU sensitive. Request workers serialize the prompt
as OpenAI-compatible JSON and send it to the local vLLM endpoint. This stage
uses host CPU, Python runtime, and loopback HTTP, but not GPU compute directly.

Model inference is GPU sensitive, with CPU participation. vLLM performs
transformer inference on the GPUs with tensor parallelism. The GPU handles the
attention and token generation work, while CPU still participates in request
scheduling, HTTP serving, tokenization-related overhead, and response handling.

Response processing is CPU sensitive. The benchmark parses the vLLM response,
records token usage, latency, response size, and request success or failure.

Metric collection is CPU-side system work. `mpstat` records CPU utilization,
`nvidia-smi` records GPU utilization, GPU memory-controller utilization, and
VRAM capacity usage, and AMDuProfPcm records host CPU memory bandwidth.

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

## Metric Definitions

`workers` is the number of CPU-side tool workers. It is the main sweep axis and
controls how many concurrent workers perform repository metadata traversal,
context-support work, and memory movement.

`cpu_avg_percent` is the average whole-machine CPU utilization during the run,
computed from `mpstat` as `100 - idle`.

`cpu_peak_percent` is the highest sampled whole-machine CPU utilization.

`gpu_avg_percent` is the average GPU compute utilization across all visible GPUs,
reported by `nvidia-smi`.

`gpu_peak_percent` is the highest sampled average GPU compute utilization.

`gpu_memctl_avg_percent` is the average GPU memory-controller utilization from
`nvidia-smi utilization.memory`. It is a percentage, not a GB/s bandwidth value.

`gpu_memctl_peak_percent` is the peak sampled GPU memory-controller utilization.

`gpu_vram_avg_percent` is average GPU memory capacity usage, computed from
`memory.used / memory.total`.

`gpu_vram_peak_percent` is the peak sampled GPU memory capacity usage.

`gpu_vram_used_avg_mb` is the average GPU memory capacity used per GPU in MB.

`mem_bw_avg_gbps` is average host CPU memory bandwidth from AMDuProfPcm
`Total Mem Bw (GB/s)`.

`mem_bw_peak_gbps` is the peak host CPU memory bandwidth sample from AMDuProfPcm.

`completed` is the count of successful vLLM chat completion requests.

`failed` is the count of failed vLLM requests, including timeouts, HTTP errors,
connection failures, or malformed responses.

`failure_rate_percent` is `failed / (completed + failed) * 100`.

`request_throughput_rps` is successful request throughput, computed as
`completed / active_window_seconds`.

`attempt_throughput_rps` includes both successful and failed request attempts.

`latency_avg_s` is average successful request latency in seconds. It includes
context construction, HTTP request, vLLM inference, and response parsing.

`latency_p50_s` is median successful request latency.

`latency_p95_s` is p95 successful request latency.

`prompt_tokens_per_s`, `completion_tokens_per_s`, and `total_tokens_per_s` are
derived from vLLM OpenAI-compatible `usage` fields divided by the active
workload window.

`scanned_gb` is the amount of repository and metadata data processed by request
workers while building model context.

`context_mb` is the final packed context size sent to vLLM.

`retrieval_amplification` is `scanned_bytes / context_bytes`. It shows how much
CPU-side search and filtering work was required to produce one byte of model
context.

`response_chars` is the total number of response characters returned by vLLM.

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
