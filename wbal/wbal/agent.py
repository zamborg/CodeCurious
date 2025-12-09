"""
Agent - Core agent interface.

Agents execute the perceive-invoke-do loop, maintaining state
and orchestrating LLM calls with tools from both the agent and environment.
"""

from typing import Any, Callable

from pydantic import Field, model_validator

from wbal.object import WBALObject
from wbal.lm import LM
from wbal.environment import Environment
from wbal.helper import get_tools, extract_tool_schema, to_openai_tool


# Type aliases for formatter callables
ToolCallDefinitionFormatter = Callable[[dict[str, Any]], dict[str, Any]]
ToolResultFormatter = Callable[[str, Any], dict[str, Any]]


class Agent(WBALObject):
    """
    Base class for agents.

    Agents orchestrate LLM calls in a perceive-invoke-do loop:
    - perceive(): Gather observations, update state
    - invoke(): Call LLM with messages and tools
    - do(): Execute tool calls from LLM response

    The loop continues until stopCondition returns True or max_steps is reached.

    Example:
        class MyAgent(Agent):
            lm: LM = GPT5Large()
            env: Environment = MyEnv()

            @tool
            def think(self, thought: str) -> str:
                '''Record a thought.'''
                return f"Thought: {thought}"

            def perceive(self) -> None:
                # Custom perception logic
                pass

        agent = MyAgent()
        result = agent.run(task="Solve the problem")
    """

    env: Environment
    """The environment the agent operates in"""

    maxSteps: int = 100
    """Maximum number of steps before stopping"""

    toolDefinitionFormatter: ToolCallDefinitionFormatter | None = None
    """Optional callable to format tool definitions for the LLM"""

    toolResultFormatter: ToolResultFormatter | None = None
    """Optional callable to format tool results for the LLM"""

    # Internal state
    _step_count: int = 0
    _tool_definitions: list[dict[str, Any]] = []
    _tool_callables: dict[str, Callable] = {}

    @model_validator(mode="after")
    def _setup_tools(self) -> "Agent":
        """Post-init: discover tools and validate no duplicates."""
        self._tool_definitions, self._tool_callables = self.getToolDefinitions()
        return self

    def observe(self) -> str:
        """Return observable state of the agent."""
        return f"Agent(step={self._step_count}, tools={list(self._tool_callables.keys())})"

    @property
    def stopCondition(self) -> bool:
        """
        Check if the agent should stop.

        Override this property in subclasses for custom stopping logic.
        By default, only stops when max_steps is reached.

        Returns:
            True if agent should stop
        """
        return self._step_count >= self.maxSteps

    def getToolDefinitions(self) -> tuple[list[dict[str, Any]], dict[str, Callable]]:
        """
        Discover and validate all @tool decorated methods.

        Traverses both the agent and environment, extracts tool schemas,
        and builds a callable lookup dictionary.

        Returns:
            Tuple of (tool_definitions_list, tool_name_to_callable_dict)

        Raises:
            ValueError: If duplicate tool names are found
        """
        definitions = []
        callables: dict[str, Callable] = {}
        formatter = self.toolDefinitionFormatter or to_openai_tool

        # Collect tools from agent
        agent_tools = get_tools(self)
        for name, method in agent_tools.items():
            if name in callables:
                raise ValueError(
                    f"Duplicate @tool name '{name}' found. "
                    "You are not allowed to have any @tool-decorated methods with the same name."
                )
            schema = extract_tool_schema(method)
            definitions.append(formatter(schema))
            callables[name] = method

        # Collect tools from environment
        env_tools = self.env.get_tools()
        for name, method in env_tools.items():
            if name in callables:
                raise ValueError(
                    f"Duplicate @tool name '{name}' found between agent and environment. "
                    "You are not allowed to have any @tool-decorated methods with the same name."
                )
            schema = extract_tool_schema(method)
            definitions.append(formatter(schema))
            callables[name] = method

        return definitions, callables

    def perceive(self) -> None:
        """
        Gather observations and update state.

        Override this in subclasses to add custom perception logic.
        """
        pass

    def invoke(self) -> Any:
        """
        Call the LLM with current state and tools.

        Override this in subclasses to implement LLM invocation.

        Returns:
            The LLM response (format depends on implementation)
        """
        pass

    def do(self) -> None:
        """
        Execute actions based on LLM response.

        Override this in subclasses to implement tool execution
        and action handling.
        """
        pass

    def step(self) -> None:
        """
        Execute one perceive-invoke-do cycle.

        Calls perceive(), invoke(), and do() in sequence,
        then increments the step counter.
        """
        self.perceive()
        self.invoke()
        self.do()
        self._step_count += 1

    def run(self, task: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """
        Run the agent until stop condition is met.

        Args:
            task: The task string. If not provided, uses env.task
            max_steps: Override maxSteps for this run. If not provided, uses self.maxSteps

        Returns:
            Dict with run results (can be extended by subclasses)
        """
        # Set task from argument or environment
        if task is not None:
            self.env.task = task

        # Override max_steps if provided
        if max_steps is not None:
            self.maxSteps = max_steps

        # Reset step counter
        self._step_count = 0

        # Main loop
        while not self.stopCondition:
            self.step()

        return {
            "steps": self._step_count,
            "task": self.env.task,
        }
