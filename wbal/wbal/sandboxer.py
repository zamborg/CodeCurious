"""
Helpers for starting a sandbox and wiring WBAL objects to it.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Iterable, Optional

from sandbox.interface import SandboxInterface
from wbal.object import WBALObject


def setup_objects(objects: Iterable[WBALObject], sandbox: SandboxInterface) -> None:
    """
    Setup all objects in the list, supporting both sync and async `setup`.
    """
    for obj in objects:
        result = obj.setup(sandbox)
        if inspect.isawaitable(result):
            asyncio.run(result)


def start_sandbox(
    sandbox: SandboxInterface,
    command: str = "sleep",
    args: Optional[list[str]] = None,
    container_image: str = "python:3.11-slim",
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    timeout: Optional[float] = 60.0,
) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """
    Start a sandbox, spinning up an event loop in a background thread if needed.

    Returns (loop, thread) so callers can manage lifecycle if desired.
    """
    if args is None:
        args = ["infinity"]

    if loop is None or not loop.is_running():
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="sandbox-loop", daemon=True)
        thread.start()
    else:
        thread = threading.current_thread()

    future = asyncio.run_coroutine_threadsafe(
        sandbox.start(command=command, args=args, container_image=container_image),
        loop,
    )
    future.result(timeout=timeout)

    return loop, thread

