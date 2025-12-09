"""
Environment - Base environment interface.

Environments define the control plane that agents operate within.
They provide task definition, environment context, and tools via @tool decorator.
"""

from typing import Any

from wbal.object import WBALObject
from wbal.helper import get_tools


class Environment(WBALObject):
    """
    Base class for environments.

    Environments define the control plane that agents operate within.
    They encapsulate:
    - Task definition
    - Environment instructions/context
    - Tools (via @tool decorator) for data access and service APIs

    The agent imports @tool-decorated methods from the environment
    into its own tool definitions.

    Example:
        class MyEnv(Environment):
            task: str = "Find the bug in the code"
            env: str = "You have access to file operations."

            @tool
            def read_file(self, path: str) -> str:
                '''Read contents of a file.'''
                with open(path) as f:
                    return f.read()
    """

    task: str = ""
    """The task or goal for the agent to accomplish"""

    env: str = ""
    """Instructions/context about what the environment provides"""

    def observe(self) -> str:
        """Return observable state of the environment."""
        return self.env

    def get_tools(self) -> dict[str, Any]:
        """Get all @tool decorated methods on this environment."""
        return get_tools(self)

    def execute_environment_fabric(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute an environment method by name.

        This is a helper that allows dynamic invocation of environment
        tools by name with provided arguments.

        Args:
            method_name: The name of the method to execute
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            The method's return value

        Raises:
            ValueError: If method_name is not a valid tool
        """
        tools = self.get_tools()
        if method_name not in tools:
            raise ValueError(f"Unknown tool: {method_name}")
        return tools[method_name](*args, **kwargs)
