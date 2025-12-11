"""
Sandbox type reference.

WBAL uses an external SandboxInterface implementation.
This module provides type aliases for reference.

The sandbox interface provides:
- async read_file(filepath) -> bytes
- async write_file(filepath, contents: bytes) -> bool
- async exec(command: list[str]) -> ExecResult

Where ExecResult has:
- stdout: bytes
- stderr: bytes
- returncode: int

The sandbox is an async context manager and handles its own lifecycle.
"""

from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass
from sandbox.interface import SandboxInterface, ExecResult
import asyncio

import shlex


# Helpers
async def run_cmd(cmd: list[str], sandbox: SandboxInterface) -> ExecResult:
    """
    Run a raw argv command (no shell) and raise on failure.
    """
    result = await sandbox.exec(cmd)
    if result.returncode != 0:
        stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
        stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}): {result.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return result


async def run_shell(script: str, sandbox: SandboxInterface) -> ExecResult:
    """
    Run a shell script via /bin/bash -lc and raise on failure.
    """
    result = await sandbox.exec(["bash", "-lc", script])
    if result.returncode != 0:
        stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
        stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
        raise RuntimeError(
            "Shell command failed (bash -lc): "
            f"{result.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return result


async def set_api_key_env(api_key_env_var: str, api_key: str, sandbox: SandboxInterface) -> None:
    """
    Persist {api_key_env_var} into ~/.bashrc inside the sandbox.
    @zamborg note that this is a hack of a tool
    """
    quoted = shlex.quote(api_key)
    await run_shell(f"echo 'export {api_key_env_var}={quoted}' >> ~/.bashrc", sandbox)


