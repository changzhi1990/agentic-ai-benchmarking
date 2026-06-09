import unittest
from pathlib import Path

from agentic_benchmarking.bug_report_workload import (
    DEFAULT_BUG_REPORT,
    build_bug_report_candidate,
    run_remote_bug_report_workload,
)


class BugReportWorkloadTests(unittest.TestCase):
    def test_build_bug_report_candidate_embeds_model_and_bug_report_fields(self) -> None:
        code = build_bug_report_candidate(
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            port=8000,
            duration_seconds=45,
            request_workers=3,
            workers=5,
            compute_workers=7,
            memory_stream_rounds=11,
        )
        self.assertIn("/home/user/models/Qwen2.5-Coder-32B-Instruct", code)
        self.assertIn("http://127.0.0.1:8000/v1/chat/completions", code)
        self.assertIn(DEFAULT_BUG_REPORT["title"], code)
        self.assertIn("AAB_MEMORY_BW_GBPS=", code)
        self.assertIn("workers = 5", code)
        self.assertIn("compute_workers = 7", code)
        self.assertIn("memory_stream_rounds = 11", code)
        self.assertIn("import multiprocessing", code)
        self.assertIn("import os", code)
        self.assertIn("os.sched_setaffinity", code)
        self.assertIn("request_count = min(request_workers, max(1, cpu_total // 12)) if request_workers > 0 else 0", code)
        self.assertIn("memory_count = min(max(workers, 1), max(1, cpu_total - request_count))", code)
        self.assertNotIn("cpu_total // 3", code)
        self.assertIn("multiprocessing.Process(", code)
        self.assertIn("process.join(timeout=grace_seconds)", code)
        self.assertNotIn("ThreadPoolExecutor", code)
        self.assertIn("worker_bandwidth = copied_bytes / elapsed / 1_000_000_000", code)
        self.assertIn("hashlib.blake2b", code)
        self.assertIn('"kind": "request"', code)
        self.assertIn('"latencies_seconds": latencies_seconds', code)
        self.assertIn('"prompt_tokens": prompt_tokens_total', code)
        self.assertIn('"completion_tokens": completion_tokens_total', code)
        self.assertIn('"context_bytes": context_bytes_total', code)
        self.assertIn("posting_lists = build_posting_lists()", code)
        self.assertIn("symbol_table = build_symbol_table()", code)
        self.assertIn("dependency_graph = build_dependency_graph()", code)
        self.assertIn("test_affinity_map = build_test_affinity_map()", code)
        self.assertIn("candidate_rerank", code)
        self.assertIn("context_packing", code)
        self.assertIn("impact_scan", code)
        self.assertIn("gathered_scores", code)
        self.assertIn("scatter_buffer", code)
        self.assertIn("shadow_buffer", code)
        self.assertIn("for _ in range(memory_stream_rounds)", code)
        self.assertIn("scatter_buffer[:] = target", code)
        self.assertIn("for posting in posting_lists.values()", code)
        self.assertIn("text[:768]", code)
        self.assertIn("min(remaining + grace_seconds, 60.0)", code)
        self.assertIn("runtime_seconds = max(5, duration_seconds - 8)", code)

    def test_run_remote_bug_report_workload_summarizes_execution(self) -> None:
        def fake_healthcheck(*_args, **_kwargs):
            return {
                "status": "ok",
                "ready": True,
                "container_name": "aab-vllm",
                "image": "vllm/vllm-openai:latest",
                "model": "/home/user/models/Qwen2.5-Coder-32B-Instruct",
                "port": 8000,
                "tensor_parallel_size": 8,
            }

        captured = {}

        def fake_executor(_remote_config, code, duration_seconds, sample_interval_seconds):
            captured["code"] = code
            captured["duration_seconds"] = duration_seconds
            captured["sample_interval_seconds"] = sample_interval_seconds
            return {
                "returncode": 0,
                "stdout": "AAB_MEMORY_BW_GBPS=212.40\n",
                "stderr": "",
                "memory_bandwidth_gbps": 212.4,
                "samples": [
                    {
                        "timestamp_s": 0,
                        "cpu_util_percent": 83.0,
                        "memory_util_percent": 61.0,
                        "gpu_util_percent": 72.0,
                    },
                    {
                        "timestamp_s": 5,
                        "cpu_util_percent": 85.0,
                        "memory_util_percent": 63.0,
                        "gpu_util_percent": 70.0,
                    },
                ],
            }

        result = run_remote_bug_report_workload(
            remote_config=object(),
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            healthcheck_runner=fake_healthcheck,
            execution_runner=fake_executor,
            duration_seconds=30,
            sample_interval_seconds=5,
            tensor_parallel_size=8,
        )

        self.assertTrue(result["healthcheck"]["ready"])
        self.assertEqual(result["summary"]["cpu_avg"], 84.0)
        self.assertEqual(result["summary"]["memory_bandwidth_avg_gbps"], 212.4)
        self.assertIn("Bug report", captured["code"])
        self.assertEqual(captured["duration_seconds"], 30)
        self.assertEqual(captured["sample_interval_seconds"], 5)

    def test_run_remote_bug_report_workload_reuses_existing_vllm_when_ready(self) -> None:
        def failing_healthcheck(*_args, **_kwargs):
            raise AssertionError("healthcheck should not restart a ready vLLM")

        def fake_probe(_remote_config, model, port):
            return {
                "status": "ok",
                "ready": True,
                "reused_existing": True,
                "model": model,
                "port": port,
            }

        def fake_executor(_remote_config, _code, duration_seconds, sample_interval_seconds):
            return {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "memory_bandwidth_gbps": 363.17,
                "memory_bandwidth_peak_gbps": 427.15,
                "samples": [],
            }

        result = run_remote_bug_report_workload(
            remote_config=object(),
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            workers=32,
            request_workers=2,
            healthcheck_runner=failing_healthcheck,
            existing_vllm_probe=fake_probe,
            execution_runner=fake_executor,
            duration_seconds=30,
            sample_interval_seconds=5,
        )

        self.assertTrue(result["healthcheck"]["reused_existing"])
        self.assertEqual(result["summary"]["memory_bandwidth_avg_gbps"], 363.17)

    def test_run_remote_bug_report_workload_skips_vllm_healthcheck_without_requests(self) -> None:
        def failing_healthcheck(*_args, **_kwargs):
            raise AssertionError("healthcheck should be skipped for CPU-only runs")

        def fake_executor(_remote_config, _code, duration_seconds, sample_interval_seconds):
            return {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "memory_bandwidth_gbps": 293.69,
                "memory_bandwidth_peak_gbps": 293.69,
                "samples": [],
            }

        result = run_remote_bug_report_workload(
            remote_config=object(),
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            request_workers=0,
            healthcheck_runner=failing_healthcheck,
            execution_runner=fake_executor,
            duration_seconds=30,
            sample_interval_seconds=5,
        )

        self.assertEqual(result["healthcheck"]["status"], "skipped")
        self.assertEqual(result["summary"]["memory_bandwidth_avg_gbps"], 293.69)

    def test_cli_defaults_match_requested_qwen_configuration(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_remote_bug_report_workload.py"
        namespace = {}
        exec(compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"), namespace)
        parser = namespace["build_parser"]()
        args = parser.parse_args([])
        self.assertEqual(args.image, "vllm/vllm-openai:latest")
        self.assertEqual(args.model, "/home/user/models/Qwen2.5-Coder-32B-Instruct")
        self.assertEqual(args.tensor_parallel_size, 8)
        self.assertEqual(args.request_workers, 2)
        self.assertEqual(args.workers, 32)
        self.assertIsNone(args.memory_workers)
        self.assertEqual(args.compute_workers, 64)
        self.assertEqual(args.memory_stream_rounds, 8)
        self.assertFalse(args.skip_vllm_healthcheck)


if __name__ == "__main__":
    unittest.main()
