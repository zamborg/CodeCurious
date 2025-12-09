"""
Env - Environment interface.

Environments define the control plane / surface layer that agents operate in.
They provide:
- Task definition
- Environment instructions
- Observable state
- Tools (via @tool decorator) for data access and service APIs
"""

from typing import Any

from wbal.object import WBALObject
from wbal.sandbox import SandboxProtocol
from wbal.tool import get_tools
from wbal.helpers import extract_tool_schema, to_openai_tool, to_anthropic_tool


class Env(WBALObject):
    """
    Base class for environments.

    Environments define the control plane that agents operate within.
    They encapsulate:
    - Service access (Datadog, GCP, databases, etc.)
    - Region-controlled data
    - Scripts/tools not directly accessible to agent code

    The agent runs in a separate context but calls into the environment
    through the sandbox boundary.

    Example:
        class DatadogEnv(Env):
            task: str = "Investigate the alert and find root cause"
            env_str: str = "You have access to Datadog logs and metrics."

            @tool
            async def get_logs(self, service: str) -> str:
                '''Fetch logs for a service.'''
                result = await self.sandbox.exec(["datadog", "logs", service])
                return result.stdout.decode()
    """

    task: str = ""
    """The task or goal for the agent to accomplish in this environment"""

    env_str: str = ""
    """Instructions/context about what the environment provides"""

    _sandbox: SandboxProtocol | None = None
    """The sandbox this environment operates in"""

    def setup(self, sandbox: SandboxProtocol) -> None:
        """Initialize the environment with a sandbox."""
        self._sandbox = sandbox

    @property
    def sandbox(self) -> SandboxProtocol:
        """Get the sandbox, raising if not set up."""
        if self._sandbox is None:
            raise RuntimeError("Environment not set up. Call setup(sandbox) first.")
        return self._sandbox

    def observe(self) -> str:
        """Return observable state of the environment."""
        return self.env_str

    def get_tools(self) -> dict[str, Any]:
        """Get all @tool decorated methods."""
        return get_tools(self)

    def get_tool_definitions(self, format: str = "openai") -> list[dict[str, Any]]:
        """
        Generate tool definitions from @tool methods.

        Args:
            format: "openai" or "anthropic"

        Returns:
            List of tool definitions in the specified format
        """
        tools = []
        converter = to_openai_tool if format == "openai" else to_anthropic_tool

        for name, method in self.get_tools().items():
            schema = extract_tool_schema(method)
            tools.append(converter(schema))

        return tools

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool method by name.

        Args:
            name: The method name
            arguments: Dict of arguments to pass

        Returns:
            The method's return value
        """
        methods = self.get_tools()
        if name not in methods:
            raise ValueError(f"Unknown tool: {name}")

        result = methods[name](**arguments)

        # Handle async methods
        if hasattr(result, "__await__"):
            result = await result

        return result
