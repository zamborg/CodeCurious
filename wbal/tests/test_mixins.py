import pytest

from wbal.environment import Environment
from wbal.mixins import ExitableAgent
from wbal.helper import tool


class TestExitableAgent:
    """Tests for ExitableAgent mixin."""

    def test_has_exit_tool(self):
        """ExitableAgent has exit tool."""
        agent = ExitableAgent(env=Environment())
        assert "exit" in agent._tool_callables

    def test_exit_sets_flag(self):
        """Calling exit() sets _exit flag."""
        agent = ExitableAgent(env=Environment())
        assert agent._exit is False

        agent.exit("Done!")

        assert agent._exit is True
        assert agent._exit_message == "Done!"

    def test_exit_stops_agent(self):
        """Agent stops after exit() is called."""
        class TestAgent(ExitableAgent):
            def perceive(self):
                if self._step_count == 2:
                    self.exit("Stopping at step 2")

        agent = TestAgent(env=Environment(), maxSteps=100)
        result = agent.run()

        assert result["steps"] == 3  # 0, 1, 2, then stop
        assert agent._exit_message == "Stopping at step 2"

    def test_reset_clears_exit_state(self):
        """reset() clears exit flag and message."""
        agent = ExitableAgent(env=Environment())
        agent.exit("Done!")

        agent.reset()

        assert agent._exit is False
        assert agent._exit_message == ""

    def test_stop_condition_composition(self):
        """Subclasses can add to stopCondition."""
        class CustomAgent(ExitableAgent):
            custom_stop: bool = False

            @property
            def stopCondition(self) -> bool:
                return self.custom_stop or super().stopCondition

        agent = CustomAgent(env=Environment())

        # Neither condition met
        assert agent.stopCondition is False

        # Custom condition
        agent.custom_stop = True
        assert agent.stopCondition is True

        # Exit condition
        agent.custom_stop = False
        agent._exit = True
        assert agent.stopCondition is True

    def test_exit_tool_definition(self):
        """Exit tool has proper schema."""
        agent = ExitableAgent(env=Environment())
        exit_def = next(
            (t for t in agent._tool_definitions if t["name"] == "exit"),
            None
        )

        assert exit_def is not None
        assert "exit_message" in exit_def["parameters"]["properties"]
        assert "exit_message" in exit_def["parameters"]["required"]
