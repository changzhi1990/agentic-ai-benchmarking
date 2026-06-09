# Bug Report Remote Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bug report-driven coding-agent workload that can launch `vllm/vllm-openai:latest` on the remote 5090 server and run a smoke benchmark against `/home/user/models/Qwen2.5-Coder-32B-Instruct` with tensor parallel size 8.

**Architecture:** Reuse the existing remote SSH and vLLM helpers, then add a workload-specific prompt builder, remote runner, and CLI that target a bug-report debugging loop instead of generic synthesis. Keep the implementation small: generation uses the remote OpenAI-compatible `vLLM` endpoint, execution stays on the remote host, and tests focus on command wiring plus prompt content.

**Tech Stack:** Python 3, `unittest`, Docker, remote SSH via Paramiko/sshpass fallback, local-path model mounting for `vLLM`

---

### Task 1: Add workload-generation tests

**Files:**
- Modify: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_bug_report_messages_include_workload_contract():
    ...

def test_default_remote_bug_report_arguments_match_qwen_tp8():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_generator tests.test_remote -v`
Expected: FAIL with missing workload prompt builder / CLI symbols

- [ ] **Step 3: Write minimal implementation**

```python
def build_bug_report_messages(...):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_generator tests.test_remote -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_generator.py tests/test_remote.py agentic_benchmarking/generator.py scripts/run_remote_bug_report_workload.py
git commit -m "feat: add remote bug report workload baseline"
```

### Task 2: Add workload runner and CLI

**Files:**
- Modify: `agentic_benchmarking/generator.py`
- Create: `agentic_benchmarking/bug_report_workload.py`
- Create: `scripts/run_remote_bug_report_workload.py`
- Modify: `tests/test_remote.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_remote_bug_report_workload_returns_iteration_summary():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_remote -v`
Expected: FAIL with missing runner module / CLI entrypoint

- [ ] **Step 3: Write minimal implementation**

```python
def run_remote_bug_report_workload(...):
    ...
```

- [ ] **Step 4: Run tests to verify it passes**

Run: `python3 -m unittest tests.test_remote -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_benchmarking/generator.py agentic_benchmarking/bug_report_workload.py scripts/run_remote_bug_report_workload.py tests/test_remote.py
git commit -m "feat: add remote bug report workload runner"
```

### Task 3: Verify on the remote 5090 server

**Files:**
- No code changes

- [ ] **Step 1: Run targeted unit tests**

Run: `python3 -m unittest tests.test_generator tests.test_remote -v`
Expected: PASS

- [ ] **Step 2: Launch remote vLLM smoke check**

Run: `AAB_REMOTE_HOST=10.83.32.172 AAB_REMOTE_USER=user AAB_REMOTE_PASSWORD=000000 python3 scripts/remote_vllm_healthcheck.py --model /home/user/models/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 8 --keep-running`
Expected: JSON output with `"ready": true`

- [ ] **Step 3: Run the remote bug report workload**

Run: `AAB_REMOTE_HOST=10.83.32.172 AAB_REMOTE_USER=user AAB_REMOTE_PASSWORD=000000 python3 scripts/run_remote_bug_report_workload.py --model /home/user/models/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 8`
Expected: JSON output with per-iteration metrics and a remote execution payload
