"""
Tool decorator for marking methods as agent/environment tools.

Both Agents and Environments use @tool to expose callable methods.
"""

from typing import Callable


def tool(func: Callable) -> Callable:
    """
    Decorator marking a method as a tool.

    Tools are methods that can be called by the LLM. They're used in both
    Agents (internal capabilities like memory, reasoning) and Environments
    (external capabilities like service APIs, data access).

    The method's docstring becomes the tool description.
    Type annotations are used to generate the parameter schema.

    Example:
        class MyAgent(Agent):
            @tool
            def remember(self, key: str, value: str) -> str:
                '''Store a value in memory.'''
                self._memory[key] = value
                return f"Remembered {key}"

        class MyEnv(Env):
            @tool
            def get_logs(self, service: str, limit: int = 100) -> str:
                '''Fetch logs from a service.'''
                return await self.sandbox.exec(["fetch-logs", service, str(limit)])
    """
    func._is_tool = True
    return func


def get_tools(obj: object) -> dict[str, Callable]:
    """
    Get all @tool decorated methods from an object.

    Args:
        obj: The object to inspect

    Returns:
        Dict mapping method names to methods
    """
    methods = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            methods[name] = attr
    return methods
