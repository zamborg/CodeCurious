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


@dataclass
class ExecResult:
    """Result from sandbox exec operation."""
    stdout: bytes
    stderr: bytes
    returncode: int


@runtime_checkable
class SandboxProtocol(Protocol):
    """
    Protocol defining the expected sandbox interface.

    Your sandbox implementation should match this interface.
    """

    async def exec(
        self,
        command: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> ExecResult:
        """Execute a command in the sandbox."""
        ...

    async def read_file(
        self,
        filepath: str,
        *,
        timeout_seconds: int | None = None,
    ) -> bytes:
        """Read a file from the sandbox."""
        ...

    async def write_file(
        self,
        filepath: str,
        contents: bytes,
        *,
        timeout_seconds: int | None = None,
    ) -> bool:
        """Write a file to the sandbox."""
        ...


# Type alias for convenience
Sandbox = SandboxProtocol
