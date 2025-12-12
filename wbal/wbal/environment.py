"""
Environment - Base environment interface.

Environments define the control plane that agents operate within.
They provide task definition, environment context, and tools via @tool decorator.
"""

import json
import os
import textwrap
from datetime import datetime
from typing import Any, Callable

from pydantic import Field

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

    output_handler: Callable[[str], None] = Field(default=lambda x: print(x))
    """Handler for agent text output. Override for custom output routing (e.g., WebUI, logging)."""

    include_tools_in_observe: bool = False
    """If True, observe() includes formatted tool descriptions."""

    def observe(self) -> str:
        """
        Return observable state of the environment.

        If include_tools_in_observe is True, appends formatted
        tool descriptions to the base observation.
        """
        base = self.env

        if self.include_tools_in_observe:
            tools_desc = self.get_tool_descriptions()
            if tools_desc:
                return f"{base}\n\n{tools_desc}" if base else tools_desc

        return base

    def get_tool_descriptions(self) -> str:
        """
        Generate formatted descriptions of all available tools.

        Extracts docstrings from @tool decorated methods and formats
        them for inclusion in prompts.

        Returns:
            Formatted string with all tool descriptions, or empty string if no tools
        """
        tools = self.get_tools()
        if not tools:
            return ""

        descriptions = []
        for name in sorted(tools.keys()):
            method = tools[name]
            doc = method.__doc__ or "No description available."
            # Clean up docstring indentation
            doc = textwrap.dedent(doc).strip()
            descriptions.append(f"## {name}\n{doc}")

        return "# Available Tools\n\n" + "\n\n".join(descriptions)

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


class StatefulEnvironment(Environment):
    """
    Environment with persistent state support.

    Provides automatic state persistence to a working directory.
    State is stored as JSON and can be loaded/saved between sessions.

    Attributes:
        working_directory: Path to directory for state persistence.
            If None, state is in-memory only.
        _state: Internal state dictionary. Access via state property.

    Example:
        class MyEnv(StatefulEnvironment):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Custom initialization after state is loaded

            @tool
            def remember(self, key: str, value: str) -> str:
                self._state["memory"][key] = value
                self.save_state()
                return f"Remembered {key}"
    """

    working_directory: str | None = None
    """Directory for state persistence. None = in-memory only."""

    _state: dict[str, Any] = {}
    """Internal state storage. Override _default_state() to customize structure."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = self._default_state()
        if self.working_directory:
            self._ensure_working_directory()
            self.load_state()

    def _default_state(self) -> dict[str, Any]:
        """
        Return the default state structure.

        Override this method to customize the initial state structure.

        Returns:
            Dict with default state structure
        """
        return {
            "data": {},
            "metadata": {
                "created_at": None,
                "last_updated": None,
            },
        }

    def _ensure_working_directory(self) -> None:
        """Create working directory if it doesn't exist."""
        if self.working_directory:
            os.makedirs(self.working_directory, exist_ok=True)

    def _state_file_path(self) -> str | None:
        """Get path to state file."""
        if self.working_directory:
            return os.path.join(self.working_directory, "environment_state.json")
        return None

    def load_state(self) -> bool:
        """
        Load state from working_directory.

        Returns:
            True if state was loaded, False if no state file exists
        """
        state_file = self._state_file_path()
        if not state_file or not os.path.exists(state_file):
            return False

        try:
            with open(state_file, "r") as f:
                loaded_state = json.load(f)
                # Merge with default state structure (preserves new keys)
                self._state.update(loaded_state)
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load state from {state_file}: {e}")
            return False

    def save_state(self) -> bool:
        """
        Persist state to working_directory.

        Returns:
            True if state was saved, False if no working_directory set
        """
        state_file = self._state_file_path()
        if not state_file:
            return False

        try:
            now = datetime.now().isoformat()
            self._state["metadata"]["last_updated"] = now
            if self._state["metadata"]["created_at"] is None:
                self._state["metadata"]["created_at"] = now

            with open(state_file, "w") as f:
                json.dump(self._state, f, indent=2)
            return True
        except OSError as e:
            print(f"Warning: Failed to save state to {state_file}: {e}")
            return False

    @property
    def state(self) -> dict[str, Any]:
        """Read-only access to state. Modify via _state directly."""
        return self._state.copy()
