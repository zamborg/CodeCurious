"""
Simple example showing WBAL usage.

This demonstrates:
1. Writing a custom environment with @weaveTool methods
2. Writing a custom agent with @weaveTool methods
3. Running them together via a sandbox
"""

import asyncio
from wbal import OpenAIResponsesLM, Agent, Env, weaveTool


# -----------------------------------------------------------------------------
# 1. Define an Environment
# -----------------------------------------------------------------------------


class SimpleEnv(Env):
    """
    An environment that provides file access tools.

    This could be specialized for a particular data source, region,
    or service (e.g., DatadogEnv, GCPEnv, etc.)
    """

    task: str = "Do whatever the user asks of you"

    env_str: str = """You have access to a sandboxed file system.
Use the available tools to read and write files as needed.
The sandbox contains data files you need to analyze.

DO NOT ACCESS THE INTERNET.
"""

    @weaveTool
    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        try:
            content = await self.sandbox.read_file(path)
            return content.decode()
        except Exception as e:
            return f"Error: {e}"

    @weaveTool
    async def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        success = await self.sandbox.write_file(path, content.encode())
        return f"Successfully wrote to {path}" if success else f"Failed to write {path}"

    @weaveTool
    async def run_command(self, command: str) -> str:
        """Run a shell command in the sandbox."""
        result = await self.sandbox.exec(["sh", "-c", command])
        output = result.stdout.decode()
        if result.stderr:
            output += f"\nstderr: {result.stderr.decode()}"
        return output


# -----------------------------------------------------------------------------
# 2. Define an Agent
# -----------------------------------------------------------------------------


class AnalysisAgent(Agent):
    """
    An agent that can analyze files and maintain notes.
    """

    system_prompt: str = """You are a helpful analysis assistant.
When given a task, use the available tools to accomplish it.
Think step by step and use the note tool to track your reasoning."""

    _notes: list[str] = []

    @weaveTool
    def note(self, content: str) -> str:
        """Record a note or observation during analysis."""
        self._notes.append(content)
        return f"Note recorded: {content}"

    @weaveTool
    def get_notes(self) -> str:
        """Retrieve all recorded notes."""
        if not self._notes:
            return "No notes recorded yet."
        return "\n".join(f"- {n}" for n in self._notes)


# -----------------------------------------------------------------------------
# 3. Run it (with your sandbox implementation)
# -----------------------------------------------------------------------------


async def main(sandbox):
    """
    Run the agent with a sandbox.

    Args:
        sandbox: A SandboxInterface implementation
    """
    # Create environment and agent
    env = SimpleEnv()
    lm = OpenAIResponsesLM(model="gpt-5-mini")  # or claude-sonnet-4-5-20250514, etc.
    agent = AnalysisAgent(lm=lm, env=env, max_steps=10)

    # Set up with sandbox
    agent.setup(sandbox)

    # Run!
    print("Starting agent...")
    print(f"Task: {env.task}")
    print("-" * 50)

    result = await agent.run()

    print("-" * 50)
    print(f"Completed in {result['steps']} steps")
    if result['final_message']:
        print(f"Final response: {result['final_message'].get('content', '')[:500]}")


# Usage:
# async with YourSandbox() as sandbox:
#     await main(sandbox)

if __name__ == "__main__":
    asyncio.run(main("sandbox"))