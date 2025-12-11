import asyncio
import os
import shlex
from typing import Optional
from pydantic import Field

from wbal.agent import Agent
from wbal.lm import LM, GPT5MiniTester
from wbal.environment import Environment
from wbal.helper import weaveTool
from wbal.object import WBALObject
from sandbox.interface import SandboxInterface
from sandbox.docker import DockerSandbox
from wbal._tmp_.sandbox import run_cmd, run_shell, set_api_key_env

import textwrap

# MEAT AND POTATOES

class SandboxedDataEnvironment(Environment):
    task: str = "You are a data analyst. Analyze the data in the local filesystem please."
    sandbox_workdir: str = "./cc_sandbox"
    env: str = "This is a sandboxed data environment. Local files are loaded into `cc_sandbox` and are available to agents to analyze *inside* the sandbox:directory."

    files_to_mount: list[str] = Field(default_factory=list)
    """Files to mount into the sandbox. The agent will have read access to these files."""

    container_filepaths: list[str] = Field(default_factory=list)
    """Files that are mounted into the sandbox. This is set by the setup method and is READ-ONLY."""

    sandbox: SandboxInterface = Field(default=None)
    """The sandbox that the environment is running in. This is set by the setup method and is READ-ONLY."""

    async def setup(self, sandbox: SandboxInterface) -> None:
        self.sandbox = sandbox
        result = await sandbox.exec(["mkdir", "-p", self.sandbox_workdir])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sandbox workdir: {result.stderr.decode()}")
        for file in self.files_to_mount:
            fname = os.path.basename(file)
            target_path = os.path.join(self.sandbox_workdir, fname)
            self.container_filepaths.append(target_path)
            with open(file, "rb") as f:
                await sandbox.write_file(target_path, f.read())

    def observe(self) -> str:
        return "# Available Files: \n" + "\n- ".join([os.path.basename(fp) for fp in self.container_filepaths])


class RegionedClaudeCodeAgent(Agent):
    # this does not have an LM
    env: Environment = SandboxedDataEnvironment()
    api_key: str = Field(default=os.getenv("ANTHROPIC_API_KEY"))
    sandbox: SandboxInterface = Field(default=None)
    event_loop: Optional[asyncio.AbstractEventLoop] = Field(default=None)

    async def setup(self, sandbox: SandboxInterface) -> None:
        """Install Claude Code and prerequisites inside the sandbox."""
        self.event_loop = asyncio.get_running_loop()
        self.sandbox = sandbox
        await super().setup(sandbox) # this sets up the environment object
        if self.api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set in your local shell and therefore cannot be exported to the sandbox")
        # Base system deps (no shell needed).
        await run_shell("apt-get update", sandbox)
        await run_shell("apt-get install -y curl", sandbox)

        # This mirrors your bash script but runs as a single non-interactive bash -lc.
        install_script = textwrap.dedent(
            """
            set -e

            # Install nvm
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash

            # Load nvm into this shell
            export NVM_DIR="$HOME/.nvm"
            if [ -s "$NVM_DIR/nvm.sh" ]; then
            . "$NVM_DIR/nvm.sh"
            else
            echo "nvm.sh not found at $NVM_DIR/nvm.sh" >&2
            exit 1
            fi

            # Install Node 22 and verify npm
            nvm install 22
            npm -v

            # Install Claude Code globally
            npm install -g @anthropic-ai/claude-code@latest
            """
        ).strip()

        await run_shell(install_script, sandbox)
        # export api key into bashrc
        await set_api_key_env(self.api_key, sandbox)
        return True

    @weaveTool
    def invoke_claude_code(self, prompt: str) -> str:
        """
        Invoke Claude Code with a prompt.
        """
        return self._run_sync(self._invoke_claude_code_async(prompt))

    def _run_sync(self, coro: asyncio.Future) -> str:
        """
        Run an async coroutine in the captured event loop from setup,
        returning its result synchronously.
        """
        loop = self.event_loop
        if loop is not None and loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        # Fallback: run in a new loop if none captured (not ideal for shared sandbox)
        return asyncio.run(coro)

    async def _invoke_claude_code_async(self, prompt: str) -> str:
        if not getattr(self.env, "sandbox", None):
            raise RuntimeError("Sandbox not initialized; call setup(sandbox) first.")

        safe_prompt = shlex.quote(prompt)
        workdir = shlex.quote(self.env.sandbox_workdir)
        result = await self.env.sandbox.exec(
            ["bash", "-lc", f"cd {workdir} && claude -p {safe_prompt}"]
        )

        stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
        stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
        if result.returncode != 0:
            raise RuntimeError(f"claude failed: {result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}")
        return stdout

    