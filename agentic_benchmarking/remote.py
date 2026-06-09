from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    user: str
    password: str
    port: int = 22

    @classmethod
    def from_env(cls) -> "RemoteConfig":
        host = os.environ["AAB_REMOTE_HOST"]
        user = os.environ["AAB_REMOTE_USER"]
        password = os.environ["AAB_REMOTE_PASSWORD"]
        port = int(os.environ.get("AAB_REMOTE_PORT", "22"))
        return cls(host=host, user=user, password=password, port=port)

    def target(self) -> str:
        return f"{self.user}@{self.host}"


def build_ssh_command(config: RemoteConfig, remote_command: str) -> list[str]:
    return [
        "sshpass",
        "-p",
        config.password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(config.port),
        config.target(),
        remote_command,
    ]


def _load_paramiko():
    try:
        return importlib.import_module("paramiko")
    except ImportError:
        return None


def _run_remote_command_with_paramiko(
    config: RemoteConfig, remote_command: str, timeout: int
) -> str:
    paramiko = _load_paramiko()
    if paramiko is None:
        raise RuntimeError(
            "password-based remote execution requires either sshpass or paramiko"
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=config.host,
            port=config.port,
            username=config.user,
            password=config.password,
            timeout=timeout,
            look_for_keys=False,
            allow_agent=False,
        )
        _, stdout, stderr = client.exec_command(remote_command, timeout=timeout)
        output = stdout.read().decode("utf-8").strip()
        error_output = stderr.read().decode("utf-8").strip()
        if error_output:
            raise RuntimeError(error_output)
        return output
    finally:
        client.close()


def run_remote_command(
    config: RemoteConfig, remote_command: str, timeout: int = 60
) -> str:
    if shutil.which("sshpass") is None:
        return _run_remote_command_with_paramiko(config, remote_command, timeout)

    completed = subprocess.run(
        build_ssh_command(config, remote_command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def quote_remote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"
