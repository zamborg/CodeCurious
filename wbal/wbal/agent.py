"""
Agent - Core agent interface.

Agents execute the perceive-invoke-do loop, maintaining state
and orchestrating LLM calls with tools from both the agent and environment.
"""

import json
from typing import Any, Callable

from pydantic import Field, model_validator

from wbal.object import WBALObject
from wbal.lm import LM
from wbal.environment import Environment
from wbal.helper import (
    get_tools,
    extract_tool_schema,
    to_openai_tool,
    format_openai_tool_response,
    TOOL_CALL_TYPE,
)

from sandbox.interface import SandboxInterface

import weave


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
    """Maximum number of steps before stopping. Alias: max_steps"""

    lm: LM | None = None
    """Language model for invoke(). Set in subclass or at instantiation."""

    @property
    def max_steps(self) -> int:
        """Pythonic alias for maxSteps."""
        return self.maxSteps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Pythonic alias for maxSteps."""
        self.maxSteps = value

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Conversation history. Populated by perceive(), extended by invoke() and do()."""

    toolDefinitionFormatter: ToolCallDefinitionFormatter | None = None
    """Optional callable to format tool definitions for the LLM"""

    toolResultFormatter: ToolResultFormatter | None = None
    """Optional callable to format tool results for the LLM"""

    # Internal state
    _step_count: int = 0
    _tool_definitions: list[dict[str, Any]] = []
    _tool_callables: dict[str, Callable] = {}
    _last_response: Any = None
    """Last response from LLM invoke(). Set by invoke(), used by do()."""

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
        return False

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

    def reset(self, clear_messages: bool = False) -> None:
        """
        Reset agent state for a new run.

        Args:
            clear_messages: If True, also clear the message history.
                Default is False to allow conversation continuation.

        Call this before re-running an agent to clear step count and
        any internal state. Subclasses should override this method
        and call super().reset() to add their own reset logic.
        """
        self._step_count = 0
        self._last_response = None
        if clear_messages:
            self.messages = []

    def invoke(self) -> Any:
        """
        Call the LLM with current messages and tools.

        Default implementation:
        1. Calls self.lm.invoke(messages=self.messages, tools=self._tool_definitions)
        2. Stores response in self._last_response
        3. Extends self.messages with response.output (OpenAI format)

        Override this method for custom LLM invocation logic or
        different response formats.

        Returns:
            The LLM response (stored in _last_response)

        Note:
            This default implementation assumes:
            - self.lm is set and has an invoke() method
            - self.messages is populated (by perceive())
            - Response has .output attribute (OpenAI Responses API)

            If these assumptions don't hold for your use case,
            override this method.
        """
        # Check prerequisites
        if not hasattr(self, 'lm') or self.lm is None:
            return None

        if not self.messages:
            return None

        # Prepare tools (None if empty)
        tools = self._tool_definitions if self._tool_definitions else None

        # Call LLM
        response = self.lm.invoke(messages=self.messages, tools=tools)
        self._last_response = response

        # Extend messages with response (OpenAI format)
        # Response.output is a list of message items
        if hasattr(response, 'output'):
            self.messages.extend(response.output)

        return response

    def do(self) -> None:
        """
        Execute tool calls from the LLM response.

        Default implementation:
        1. Extracts function_call items from _last_response.output
        2. Executes each tool via _tool_callables
        3. Formats results and appends to messages
        4. If no tool calls, sends output_text to env.output_handler

        Override this method for custom tool execution logic or
        different response formats.

        Note:
            This default implementation assumes:
            - self._last_response has .output (list) and .output_text (str)
            - Tool calls have .type == "function_call"
            - Tool calls have .name, .arguments (JSON string), .call_id

            If these assumptions don't hold, override this method.
        """
        if self._last_response is None:
            return

        # Get response output
        output = getattr(self._last_response, 'output', None)
        if output is None:
            return

        # Extract tool calls (OpenAI format: type == TOOL_CALL_TYPE)
        tool_calls = [
            item for item in output
            if getattr(item, 'type', None) == TOOL_CALL_TYPE
        ]

        # If no tool calls, handle text output
        if not tool_calls:
            output_text = getattr(self._last_response, 'output_text', '')
            if output_text and hasattr(self.env, 'output_handler'):
                self.env.output_handler(output_text)
            return

        # Execute each tool call
        for tc in tool_calls:
            tc_name = getattr(tc, 'name', '')
            tc_args_raw = getattr(tc, 'arguments', '{}')
            tc_id = getattr(tc, 'call_id', '')

            # Parse arguments
            if isinstance(tc_args_raw, str):
                try:
                    tc_args = json.loads(tc_args_raw)
                except json.JSONDecodeError:
                    tc_args = {}
            else:
                tc_args = tc_args_raw or {}

            # Execute tool
            if tc_name in self._tool_callables:
                try:
                    tc_output = self._tool_callables[tc_name](**tc_args)
                except Exception as e:
                    tc_output = f"Error executing {tc_name}: {e}"
            else:
                tc_output = f"Unknown tool: {tc_name}"

            # Format and append result
            result = format_openai_tool_response(tc_output, tc_id)
            self.messages.append(result)

    @weave.op()
    def step(self) -> None:
        """
        Execute one perceive-invoke-do cycle.

        Calls perceive(), invoke(), and do() in sequence,
        then increments the step counter.
        """
        # @zamborg FIX this please percieve should probably mutate state, but invoke should pass to do should return ...
        self.perceive()
        self.invoke()
        self.do()
        self._step_count += 1

    @weave.op()
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

        for _ in range(self.maxSteps):
            self.step()
            if self.stopCondition:
                break

        return {
            "steps": self._step_count,
            "task": self.env.task,
        }

    # NOTE: WE RECOMMEND USING THIS TO EXIT THE AGENT WHEN YOU WANT TO STOP THE AGENT FROM RUNNING LONG_N
    # @weaveTool
    # def exit(self, exit_message: str) -> str:
    #     """
    #     Exit your run loop.
    #         - please provide `exit_message` as a message to the user or developer. This can be your terminal result or your final summary or any content you'd like to leave your controller with after your run loop.
    #     """
    #     print(exit_message)
    #     return exit_message
