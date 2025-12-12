"""
Agent - Core agent interface.

Agents execute the perceive-invoke-do loop, maintaining message history
and orchestrating LLM calls with tools from both the agent and environment.
"""

from typing import Any

from pydantic import Field

from wbal.object import WBALObject
from wbal.lm import OpenAIResponsesLM, LM
from wbal.env import Env
from wbal.sandbox import SandboxProtocol
from wbal.tool import tool, get_tools
from wbal.helpers import (
    extract_tool_schema,
    to_openai_tool,
    to_anthropic_tool,
    parse_openai_tool_calls,
    format_openai_tool_result,
)


class Agent(WBALObject):
    """
    Base class for agents.

    Agents orchestrate LLM calls in a perceive-invoke-do loop:
    - perceive(): Gather observations from environment
    - invoke(): Call LLM with messages and tools
    - do(): Execute tool calls from LLM response

    The loop continues until the agent decides to stop (no tool calls)
    or hits max_steps.

    Example:
        class MyAgent(Agent):
            system_prompt: str = "You are a helpful assistant."

            @tool
            def think(self, thought: str) -> str:
                '''Record a thought.'''
                return f"Thought recorded: {thought}"

        agent = MyAgent(lm=LM(model="gpt-4"), env=my_env)
        result = await agent.run()
    """

    lm: LM
    """The language model to use for invocations"""

    env: Env
    """The environment the agent operates in"""

    system_prompt: str = "You are a helpful AI assistant."
    """System prompt for the LLM"""

    max_steps: int = 50
    """Maximum number of steps before stopping"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Conversation history"""

    step_count: int = 0
    """Current step number"""

    _sandbox: SandboxProtocol | None = None
    """The sandbox for execution"""

    def setup(self, sandbox: SandboxProtocol) -> None:
        """Initialize agent and environment with sandbox."""
        self._sandbox = sandbox
        self.env.setup(sandbox)

    @property
    def sandbox(self) -> SandboxProtocol:
        """Get the sandbox, raising if not set up."""
        if self._sandbox is None:
            raise RuntimeError("Agent not set up. Call setup(sandbox) first.")
        return self._sandbox

    def observe(self) -> str:
        """Return observable state of the agent."""
        return (
            f"Agent(step={self.step_count}, "
            f"messages={len(self.messages)}, "
            f"lm={self.lm.model})"
        )

    # -------------------------------------------------------------------------
    # Tool discovery
    # -------------------------------------------------------------------------

    def get_agent_tools(self) -> dict[str, Any]:
        """Get all @tool decorated methods on this agent."""
        return get_tools(self)

    def get_tool_definitions(self, format: str = "openai") -> list[dict[str, Any]]:
        """
        Generate tool definitions from agent @tool methods.

        Args:
            format: "openai" or "anthropic"

        Returns:
            List of tool definitions in the specified format
        """
        tools = []
        converter = to_openai_tool if format == "openai" else to_anthropic_tool

        for name, method in self.get_agent_tools().items():
            schema = extract_tool_schema(method)
            tools.append(converter(schema))

        return tools

    def get_all_tools(self, format: str = "openai") -> list[dict[str, Any]]:
        """Get combined tool definitions from agent and environment."""
        return self.get_tool_definitions(format) + self.env.get_tool_definitions(format)

    # -------------------------------------------------------------------------
    # Perceive-Invoke-Do loop
    # -------------------------------------------------------------------------

    def perceive(self) -> None:
        """
        Gather observations and update state.

        Override this to add custom perception logic.
        By default, adds environment observation to messages on first step.
        """
        if self.step_count == 0 or len(self.messages) == 0:
            # Initial perception: add task and environment context
            initial_message = (
                f"Task: {self.env.task}\n\n"
                f"Environment: {self.env.env_str}\n\n"
                f"Observation: {self.env.observe()}"
            )
            self.messages.append({"role": "user", "content": initial_message})

    def invoke(self) -> dict[str, Any]:
        """
        Call the LLM with current messages and tools.

        Returns:
            The assistant's response message
        """
        tools = self.get_all_tools()

        response = self.lm.invoke(
            messages=self.messages,
            tools=tools if tools else None,
        )

        # Add response to history
        self.messages.append(response)

        return response

    async def do(self, response: dict[str, Any]) -> bool:
        """
        Execute tool calls from the LLM response.

        Args:
            response: The assistant's response message

        Returns:
            True if there were tool calls to execute, False otherwise
        """
        tool_calls = parse_openai_tool_calls(response)

        if not tool_calls:
            return False

        for tc in tool_calls:
            result = await self._execute_tool(tc["name"], tc["arguments"])

            # Add tool result to messages
            self.messages.append(format_openai_tool_result(tc["id"], result))

        return True

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name, checking agent then environment."""
        agent_tools = self.get_agent_tools()

        if name in agent_tools:
            result = agent_tools[name](**arguments)
            # Handle async agent tools
            if hasattr(result, "__await__"):
                result = await result
            return result

        # Try environment tools
        return await self.env.execute(name, arguments)

    def stop_condition(self) -> bool:
        """
        Check if the agent should stop.

        Override for custom stopping logic.

        Returns:
            True if agent should stop
        """
        if self.step_count >= self.max_steps:
            return True

        # Stop if last response had no tool calls (agent is done)
        if self.messages and self.messages[-1].get("role") == "assistant":
            if not self.messages[-1].get("tool_calls"):
                return True

        return False

    async def step(self) -> dict[str, Any]:
        """
        Execute one perceive-invoke-do cycle.

        Returns:
            The LLM response from this step
        """
        self.perceive()
        response = self.invoke()
        await self.do(response)
        self.step_count += 1
        return response

    async def run(self) -> dict[str, Any]:
        """
        Run the agent until stop condition is met.

        Returns:
            Dict with 'success', 'steps', 'final_message'
        """
        while not self.stop_condition():
            response = await self.step()

        return {
            "success": True,
            "steps": self.step_count,
            "final_message": self.messages[-1] if self.messages else None,
        }


# Re-export tool decorator for convenience
__all__ = ["Agent", "tool"]
