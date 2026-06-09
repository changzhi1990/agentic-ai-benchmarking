# Agentic AI CPU Memory Workload

This directory contains a standalone copy of the bug-report driven coding-agent
workload and the latest benchmark artifacts.

## What It Simulates

The workload models a coding agent that receives a bug report, scans a synthetic
large codebase, builds metadata indexes, expands candidates through dependency
and test-affinity graphs, packs context, and sends requests to a local vLLM
server.

The main scaling axis is `workers`, which represents CPU-side tool workers that
stress code retrieval, metadata traversal, context packing, and memory movement.

## Key Files

- `agentic_benchmarking/bug_report_workload.py`: workload generator and remote
  orchestration.
- `agentic_benchmarking/remote_workload.py`: remote execution, CPU/GPU sampling,
  and AMDuProfPcm memory-bandwidth parsing.
- `scripts/run_remote_bug_report_workload.py`: CLI entrypoint.
- `tests/test_bug_report_workload.py`: workload behavior tests.
- `tests/test_remote.py`: remote runner and metric parser tests.
- `artifacts/workers-sweep-business-metrics.csv`: latest workers sweep table.
- `artifacts/workers-sweep-business-metrics.json`: latest workers sweep raw data.

## Remote Requirements

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

From this directory:

```bash
python3 -m unittest tests.test_bug_report_workload tests.test_remote -v
```

## Run CPU-Only Memory Bandwidth Test

```bash
AAB_REMOTE_HOST=10.83.32.172 \
AAB_REMOTE_USER=user \
AAB_REMOTE_PASSWORD=000000 \
python3 scripts/run_remote_bug_report_workload.py \
  --duration-seconds 36 \
  --sample-interval 5 \
  --request-workers 0 \
  --workers 40 \
  --compute-workers 0 \
  --repo-files 2048 \
  --chunk-repeats 8 \
  --memory-block-mb 128 \
  --memory-stream-rounds 8 \
  --top-k 8 \
  --skip-vllm-healthcheck \
  --output-dir artifacts/cpu-memory-bandwidth-amduprof-rerun
```

## Run Balanced CPU/GPU Test

This uses vLLM requests plus CPU-side workers:

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

## Latest Workers Sweep

The latest sweep used:

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

