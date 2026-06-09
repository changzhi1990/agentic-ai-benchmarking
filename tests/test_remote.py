import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_benchmarking import remote
from agentic_benchmarking.remote import RemoteConfig, build_ssh_command, run_remote_command
from agentic_benchmarking.remote_smoke import run_remote_protocol_smoke
from agentic_benchmarking.remote_workload import (
    _extract_amduprof_pcm_bandwidth_gbps,
    _extract_amduprof_pcm_bandwidth_samples_gbps,
    _extract_memory_bandwidth_gbps,
    _build_amduprof_pcm_command,
    execute_remote_candidate,
)
from agentic_benchmarking.vllm_remote import (
    build_vllm_docker_run_command,
    run_remote_vllm_healthcheck,
    summarize_vllm_health,
)


class RemoteTests(unittest.TestCase):
    def test_build_ssh_command_includes_password_auth_options(self) -> None:
        config = RemoteConfig(host="10.0.0.1", user="user", password="secret", port=2222)
        command = build_ssh_command(config, "hostname")
        self.assertEqual(command[0:3], ["sshpass", "-p", "secret"])
        self.assertIn("StrictHostKeyChecking=no", command)
        self.assertIn("UserKnownHostsFile=/dev/null", command)
        self.assertIn("-p", command)
        self.assertEqual(command[-2], "user@10.0.0.1")
        self.assertEqual(command[-1], "hostname")

    def test_run_remote_command_uses_paramiko_when_sshpass_is_unavailable(self) -> None:
        config = RemoteConfig(host="10.0.0.1", user="user", password="secret")

        class FakeStdout:
            def read(self) -> bytes:
                return b"gpu-5090-box\n"

        class FakeStderr:
            def read(self) -> bytes:
                return b""

        class FakeClient:
            def set_missing_host_key_policy(self, _policy) -> None:
                return None

            def connect(self, **kwargs) -> None:
                self.kwargs = kwargs

            def exec_command(self, command, timeout=None):
                self.command = command
                self.timeout = timeout
                return (None, FakeStdout(), FakeStderr())

            def close(self) -> None:
                return None

        class FakeParamikoModule:
            class AutoAddPolicy:
                pass

            SSHClient = FakeClient

        with patch.object(remote.shutil, "which", return_value=None):
            with patch.object(remote, "_load_paramiko", return_value=FakeParamikoModule):
                output = run_remote_command(config, "hostname", timeout=15)
        self.assertEqual(output, "gpu-5090-box")

    def test_summarize_vllm_health_derives_ready_state(self) -> None:
        summary = summarize_vllm_health(
            container_name="aab-vllm",
            image="vllm/vllm-openai:latest",
            model="meta-llama/Llama-3.1-8B-Instruct",
            port=8000,
            models_payload={"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]},
            startup_seconds=18.4,
        )
        self.assertTrue(summary["ready"])
        self.assertEqual(summary["served_model_count"], 1)

    def test_build_vllm_docker_run_command_mounts_local_model_path(self) -> None:
        command = build_vllm_docker_run_command(
            container_name="aab-vllm",
            image="vllm/vllm-openai:latest",
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            port=8000,
        )
        self.assertIn("-v '/home/user/models:/home/user/models:ro'", command)
        self.assertIn("--model '/home/user/models/Qwen2.5-Coder-32B-Instruct'", command)

    def test_build_vllm_docker_run_command_supports_tensor_parallel(self) -> None:
        command = build_vllm_docker_run_command(
            container_name="aab-vllm",
            image="vllm/vllm-openai:latest",
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            port=8000,
            tensor_parallel_size=8,
        )
        self.assertIn("--tensor-parallel-size 8", command)

    def test_remote_smoke_writes_remote_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        manifest_path = repo_root / "templates" / "run-manifest.example.json"
        profile_path = repo_root / "profiles" / "strict-repeatability.json"

        def fake_preflight(_config):
            return {
                "status": "ok",
                "hostname": "gpu-5090-box",
                "docker_available": True,
                "nvidia_smi_available": True,
                "gpu_count": 8,
                "gpu_names": ["NVIDIA GeForce RTX 5090"],
                "driver_versions": ["555.42"],
            }

        def fake_healthcheck(_config, model, port, image, tensor_parallel_size=1):
            return {
                "status": "ok",
                "ready": True,
                "container_name": "aab-vllm",
                "image": image,
                "model": model,
                "port": port,
                "tensor_parallel_size": tensor_parallel_size,
                "startup_seconds": 12.5,
                "served_model_count": 1,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "remote-artifacts"
            result = run_remote_protocol_smoke(
                manifest_path=manifest_path,
                profile_path=profile_path,
                output_dir=output_dir,
                remote_config=RemoteConfig(
                    host="10.83.32.172",
                    user="user",
                    password="secret",
                ),
                model="meta-llama/Llama-3.1-8B-Instruct",
                preflight_runner=fake_preflight,
                healthcheck_runner=fake_healthcheck,
            )

            self.assertEqual(result["status"], "success")
            self.assertTrue((output_dir / "remote-preflight.json").exists())
            self.assertTrue((output_dir / "remote-vllm-health.json").exists())
            self.assertTrue((output_dir / "final-report.json").exists())

    def test_healthcheck_can_leave_container_running(self) -> None:
        commands = []

        def fake_executor(_config, command, timeout):
            commands.append(command)
            if "curl -sf" in command:
                return '{"data":[{"id":"model"}]}'
            return ""

        result = run_remote_vllm_healthcheck(
            RemoteConfig(host="10.0.0.1", user="user", password="secret"),
            model="/home/user/models/Qwen2.5-Coder-32B-Instruct",
            tensor_parallel_size=8,
            executor=fake_executor,
            max_attempts=1,
            sleep_seconds=0.0,
            cleanup=False,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(
            len([command for command in commands if "docker rm -f aab-vllm" in command]),
            1,
        )

    def test_execute_remote_candidate_allows_grace_period_before_termination(self) -> None:
        captured = {}

        def fake_runner(_config, remote_command, timeout=0):
            captured["command"] = remote_command
            captured["timeout"] = timeout
            return '{"returncode":0,"stdout":"AAB_MEMORY_BW_GBPS=10.50","stderr":"","samples":[]}'

        payload = execute_remote_candidate(
            RemoteConfig(host="10.0.0.1", user="user", password="secret"),
            "print('hello')",
            duration_seconds=20,
            sample_interval_seconds=5,
            grace_period_seconds=7,
            runner=fake_runner,
        )

        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["memory_bandwidth_gbps"], 10.5)
        self.assertIn("process.wait(timeout=7)", captured["command"])
        self.assertIn("process.kill()", captured["command"])
        self.assertEqual(captured["timeout"], 207)
        self.assertIn("AMDuProfPcm", captured["command"])
        self.assertIn("--msr", captured["command"])
        self.assertIn("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", captured["command"])

    def test_extract_memory_bandwidth_sums_worker_reports_from_summary_json(self) -> None:
        combined_output = """
AAB_MEMORY_BW_GBPS=38.88
AAB_MEMORY_BW_GBPS=42.41
{
  "memory_reports": [
    {"worker_id": 0, "bandwidth_gbps": 38.88},
    {"worker_id": 1, "bandwidth_gbps": 42.41},
    {"worker_id": 2, "bandwidth_gbps": 39.00}
  ]
}
"""
        self.assertEqual(_extract_memory_bandwidth_gbps(combined_output), 120.29)

    def test_extract_amduprof_pcm_bandwidth_gbps_parses_total_mem_bw_samples(self) -> None:
        sample_output = """
Total Mem Bw (GB/s),Total Mem RdBw (GB/s),Total Mem WrBw (GB/s),
Profiling started
12.34,8.00,4.34,
56.78,40.00,16.78,
"""
        self.assertEqual(_extract_amduprof_pcm_bandwidth_gbps(sample_output), 34.56)

    def test_extract_amduprof_pcm_bandwidth_gbps_ignores_non_sample_rows_and_tiny_tail(self) -> None:
        sample_output = """
0,0, 0 1 2 3
Profile Time: 2025/08/26 23:44:44:294
DF METRICS,,,
System (Aggregated),,, 
Total Mem Bw (GB/s),Total Mem RdBw (GB/s),Total Mem WrBw (GB/s),
Profiling started
227.58,114.73,112.85,
230.82,116.12,114.70,
221.40,109.70,111.70,
0.05,0.03,0.02,
"""
        self.assertEqual(_extract_amduprof_pcm_bandwidth_gbps(sample_output), 226.6)
        self.assertEqual(
            _extract_amduprof_pcm_bandwidth_samples_gbps(sample_output),
            [227.58, 230.82, 221.4],
        )

    def test_build_amduprof_pcm_command_uses_requested_tool_and_sudo(self) -> None:
        command = _build_amduprof_pcm_command(
            tool_path="/home/user/zhi/AMDuProf_Nda_Linux_x64_5.0.1479/bin/AMDuProfPcm",
            sudo_password="000000",
            duration_seconds=24,
            sample_interval_seconds=5,
        )
        self.assertIn("sudo -S -p ''", command)
        self.assertIn("/home/user/zhi/AMDuProf_Nda_Linux_x64_5.0.1479/bin/AMDuProfPcm", command)
        self.assertIn("-m memory -a --msr -r", command)
        self.assertIn("-d 24", command)
        self.assertIn("-I 5000", command)


if __name__ == "__main__":
    unittest.main()
