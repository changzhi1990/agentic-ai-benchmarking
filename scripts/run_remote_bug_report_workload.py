#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(globals().get("__file__", "scripts/run_remote_bug_report_workload.py")).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))

from agentic_benchmarking.bug_report_workload import (  # noqa: E402
    DEFAULT_IMAGE,
    DEFAULT_MODEL,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    run_remote_bug_report_workload,
    write_bug_report_artifacts,
)
from agentic_benchmarking.remote import RemoteConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the remote bug report-driven coding-agent workload."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument("--duration-seconds", type=int, default=90)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--request-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--memory-workers", type=int, default=None, help="Deprecated alias for --workers")
    parser.add_argument("--compute-workers", type=int, default=64)
    parser.add_argument("--repo-files", type=int, default=1536)
    parser.add_argument("--chunk-repeats", type=int, default=6)
    parser.add_argument("--memory-block-mb", type=int, default=128)
    parser.add_argument("--memory-stream-rounds", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-vllm-healthcheck", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_remote_bug_report_workload(
        remote_config=RemoteConfig.from_env(),
        model=args.model,
        image=args.image,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        duration_seconds=args.duration_seconds,
        sample_interval_seconds=args.sample_interval,
        request_workers=args.request_workers,
        workers=args.workers,
        memory_workers=args.memory_workers,
        compute_workers=args.compute_workers,
        repo_files=args.repo_files,
        chunk_repeats=args.chunk_repeats,
        memory_block_mb=args.memory_block_mb,
        memory_stream_rounds=args.memory_stream_rounds,
        top_k=args.top_k,
        skip_vllm_healthcheck=args.skip_vllm_healthcheck,
    )
    if args.output_dir is not None:
        write_bug_report_artifacts(result, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
