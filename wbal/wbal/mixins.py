"""
Optional mixins for Agent functionality.

Mixins provide common patterns that can be composed into agents.
"""

from wbal.agent import Agent
from wbal.helper import weaveTool


class ExitableAgent(Agent):
    """
    Agent mixin that provides a built-in exit tool.

    Adds:
    - exit(exit_message) tool for graceful termination
    - _exit flag checked in stopCondition
    - _exit_message storing the final message

    The agent will stop when the LLM calls the exit tool.

    Example:
        class MyAgent(ExitableAgent):
            # Inherits exit tool automatically

            @tool
            def do_work(self) -> str:
                return "Working..."

        agent = MyAgent(env=env)
        result = agent.run(task="Do work then exit with summary")
        print(agent._exit_message)  # Final message from exit() call
    """

    _exit: bool = False
    """Flag set when exit() tool is called."""

    _exit_message: str = ""
    """Message provided to exit() tool."""

    @property
    def stopCondition(self) -> bool:
        """
        Stop when exit() is called or parent condition is met.

        Subclasses overriding stopCondition should call super():
            @property
            def stopCondition(self) -> bool:
                return self._custom_condition or super().stopCondition
        """
        return self._exit or super().stopCondition

    def reset(self, clear_messages: bool = False) -> None:
        """Reset exit state in addition to base reset."""
        super().reset(clear_messages=clear_messages)
        self._exit = False
        self._exit_message = ""

    @weaveTool
    def exit(self, exit_message: str) -> str:
        """
        Exit the agent run loop with a final message.

        Call this tool when:
        - The task is complete and no more actions are needed
        - The task is impossible with available tools
        - You want to return a final summary or result

        The agent will stop after this tool call completes.

        Args:
            exit_message: Final message to return. This becomes the
                agent's terminal output and is stored in _exit_message.

        Returns:
            The exit_message (for inclusion in conversation history)
        """
        self._exit = True
        self._exit_message = exit_message
        return exit_message
