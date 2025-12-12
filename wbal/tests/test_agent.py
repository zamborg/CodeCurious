import pytest
from unittest.mock import Mock

from wbal.agent import Agent
from wbal.environment import Environment
from wbal.lm import LM
from wbal.helper import tool


class MockLM(LM):
    """Mock LM for testing."""
    mock_response: object = None
    invoke_calls: list = []

    def observe(self) -> str:
        return "MockLM"

    def invoke(self, messages=None, tools=None, mcp_servers=None):
        self.invoke_calls.append({"messages": messages, "tools": tools})
        return self.mock_response


class TestAgent:
    """Unit tests for the Agent base class."""

    def test_requires_environment(self):
        """Agent requires an environment."""
        with pytest.raises(Exception):  # pydantic validation error
            Agent()

    def test_default_max_steps(self):
        """Default maxSteps is 100."""
        agent = Agent(env=Environment())
        assert agent.maxSteps == 100

    def test_observe(self):
        """observe() returns agent state string."""
        agent = Agent(env=Environment())
        obs = agent.observe()
        assert "Agent" in obs
        assert "step=0" in obs

    def test_stop_condition_default_false(self):
        """stopCondition returns False by default (run() uses maxSteps for limiting)."""
        agent = Agent(env=Environment(), maxSteps=5)
        assert agent.stopCondition is False

        # Note: The base Agent stopCondition always returns False.
        # The maxSteps limit is enforced by the run() method's for loop.
        agent._step_count = 5
        assert agent.stopCondition is False  # Still False, run() stops via loop

    def test_agent_discovers_own_tools(self):
        """Agent discovers its own @tool methods."""
        class MyAgent(Agent):
            @tool
            def think(self, thought: str) -> str:
                '''Record a thought.'''
                return f"Thought: {thought}"

        agent = MyAgent(env=Environment())
        assert "think" in agent._tool_callables
        assert agent._tool_callables["think"]("hello") == "Thought: hello"

    def test_agent_discovers_env_tools(self):
        """Agent discovers environment @tool methods."""
        class MyEnv(Environment):
            @tool
            def env_tool(self) -> str:
                '''Environment tool.'''
                return "from env"

        agent = Agent(env=MyEnv())
        assert "env_tool" in agent._tool_callables

    def test_combined_tools(self):
        """Agent combines its tools with environment tools."""
        class MyEnv(Environment):
            @tool
            def env_tool(self) -> str:
                '''Env tool.'''
                return "env"

        class MyAgent(Agent):
            @tool
            def agent_tool(self) -> str:
                '''Agent tool.'''
                return "agent"

        agent = MyAgent(env=MyEnv())

        assert "env_tool" in agent._tool_callables
        assert "agent_tool" in agent._tool_callables
        assert len(agent._tool_callables) == 2

    def test_duplicate_tool_names_raises_error(self):
        """Duplicate tool names between agent and env raise ValueError."""
        class MyEnv(Environment):
            @tool
            def duplicate(self) -> str:
                '''Env version.'''
                return "env"

        class MyAgent(Agent):
            @tool
            def duplicate(self) -> str:
                '''Agent version.'''
                return "agent"

        with pytest.raises(Exception, match="Duplicate @tool name 'duplicate'"):
            MyAgent(env=MyEnv())

    def test_step_increments_counter(self):
        """step() increments the step counter."""
        agent = Agent(env=Environment())
        assert agent._step_count == 0

        agent.step()
        assert agent._step_count == 1

        agent.step()
        assert agent._step_count == 2

    def test_run_executes_until_max_steps(self):
        """run() executes steps until maxSteps is reached."""
        agent = Agent(env=Environment(), maxSteps=3)
        result = agent.run()

        assert result["steps"] == 3
        assert agent._step_count == 3

    def test_run_with_task_argument(self):
        """run() accepts task argument and sets it on env."""
        env = Environment()
        agent = Agent(env=env, maxSteps=1)

        result = agent.run(task="Test task")

        assert env.task == "Test task"
        assert result["task"] == "Test task"

    def test_run_with_max_steps_override(self):
        """run() accepts max_steps argument to override maxSteps."""
        agent = Agent(env=Environment(), maxSteps=100)
        result = agent.run(max_steps=2)

        assert result["steps"] == 2

    def test_run_uses_env_task_by_default(self):
        """run() uses env.task if no task argument provided."""
        env = Environment(task="Default task")
        agent = Agent(env=env, maxSteps=1)

        result = agent.run()

        assert result["task"] == "Default task"

    def test_tool_definitions_generated(self):
        """Tool definitions are generated in correct format."""
        class MyAgent(Agent):
            @tool
            def my_tool(self, arg: str) -> str:
                '''My tool description.'''
                return arg

        agent = MyAgent(env=Environment())

        assert len(agent._tool_definitions) == 1
        defn = agent._tool_definitions[0]

        assert defn["name"] == "my_tool"
        assert defn["type"] == "function"
        assert "My tool description" in defn["description"]

    def test_custom_stop_condition(self):
        """Subclasses can override stopCondition."""
        class EarlyStopAgent(Agent):
            @property
            def stopCondition(self) -> bool:
                return self._step_count >= 2 or super().stopCondition

        agent = EarlyStopAgent(env=Environment(), maxSteps=100)
        result = agent.run()

        assert result["steps"] == 2

    def test_messages_field_exists(self):
        """Agent has messages field."""
        agent = Agent(env=Environment())
        assert hasattr(agent, "messages")
        assert agent.messages == []

    def test_messages_can_be_appended(self):
        """Messages list can be modified."""
        agent = Agent(env=Environment())
        agent.messages.append({"role": "user", "content": "hello"})
        assert len(agent.messages) == 1

    def test_last_response_initially_none(self):
        """_last_response is None before invoke."""
        agent = Agent(env=Environment())
        assert agent._last_response is None

    def test_reset_clears_step_count(self):
        """reset() clears the step counter."""
        agent = Agent(env=Environment(), maxSteps=5)
        agent.run()
        assert agent._step_count == 5

        agent.reset()
        assert agent._step_count == 0

    def test_reset_allows_rerun(self):
        """Agent can be run again after reset."""
        agent = Agent(env=Environment(), maxSteps=3)
        result1 = agent.run()
        assert result1["steps"] == 3

        agent.reset()
        result2 = agent.run()
        assert result2["steps"] == 3

    def test_reset_with_clear_messages(self):
        """reset(clear_messages=True) clears message history."""
        agent = Agent(env=Environment())
        agent.messages.append({"role": "user", "content": "test"})

        agent.reset(clear_messages=True)

        assert agent.messages == []

    def test_reset_preserves_messages_by_default(self):
        """reset() preserves messages by default."""
        agent = Agent(env=Environment())
        agent.messages.append({"role": "user", "content": "test"})

        agent.reset()

        assert len(agent.messages) == 1

    def test_invoke_without_lm_is_noop(self):
        """invoke() does nothing if lm is not set."""
        agent = Agent(env=Environment())
        result = agent.invoke()
        assert result is None
        assert agent._last_response is None

    def test_invoke_without_messages_is_noop(self):
        """invoke() does nothing if messages is empty."""
        mock_lm = MockLM(invoke_calls=[])
        agent = Agent(env=Environment(), lm=mock_lm)
        result = agent.invoke()
        assert result is None
        assert len(mock_lm.invoke_calls) == 0

    def test_invoke_calls_lm(self):
        """invoke() calls lm.invoke with messages and tools."""
        mock_response = Mock()
        mock_response.output = [{"role": "assistant", "content": "hi"}]

        mock_lm = MockLM(mock_response=mock_response, invoke_calls=[])

        agent = Agent(env=Environment(), lm=mock_lm)
        agent.messages = [{"role": "user", "content": "hello"}]

        agent.invoke()

        assert len(mock_lm.invoke_calls) == 1
        assert agent._last_response == mock_response
        assert len(agent.messages) == 2  # Original + response

    def test_invoke_can_be_overridden(self):
        """Subclasses can override invoke."""
        class CustomAgent(Agent):
            custom_invoke_called: bool = False

            def invoke(self):
                self.custom_invoke_called = True
                return "custom"

        agent = CustomAgent(env=Environment())
        result = agent.invoke()

        assert result == "custom"
        assert agent.custom_invoke_called

    def test_do_without_response_is_noop(self):
        """do() does nothing if _last_response is None."""
        agent = Agent(env=Environment())
        agent.do()  # Should not raise

    def test_do_executes_tool_calls(self):
        """do() executes tool calls and appends results."""
        class TestEnv(Environment):
            @tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

        agent = Agent(env=TestEnv())

        # Mock response with tool call
        mock_tc = Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "add"
        mock_tc.arguments = '{"a": 2, "b": 3}'
        mock_tc.call_id = "call_123"

        mock_response = Mock()
        mock_response.output = [mock_tc]

        agent._last_response = mock_response
        agent.do()

        # Check result was appended
        assert len(agent.messages) == 1
        assert agent.messages[0]["type"] == "function_call_output"
        assert "5" in agent.messages[0]["output"]

    def test_do_handles_text_output(self):
        """do() calls output_handler for text responses."""
        outputs = []
        env = Environment(output_handler=lambda x: outputs.append(x))
        agent = Agent(env=env)

        mock_response = Mock()
        mock_response.output = []  # No tool calls
        mock_response.output_text = "Hello, world!"

        agent._last_response = mock_response
        agent.do()

        assert outputs == ["Hello, world!"]

    def test_do_handles_tool_errors(self):
        """do() catches and reports tool execution errors."""
        class TestEnv(Environment):
            @tool
            def failing_tool(self) -> str:
                """A tool that fails."""
                raise ValueError("Tool failed!")

        agent = Agent(env=TestEnv())

        mock_tc = Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "failing_tool"
        mock_tc.arguments = '{}'
        mock_tc.call_id = "call_456"

        mock_response = Mock()
        mock_response.output = [mock_tc]

        agent._last_response = mock_response
        agent.do()  # Should not raise

        assert "Error" in agent.messages[0]["output"]
        assert "Tool failed!" in agent.messages[0]["output"]

    def test_do_handles_unknown_tool(self):
        """do() handles calls to unknown tools gracefully."""
        agent = Agent(env=Environment())

        mock_tc = Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "nonexistent_tool"
        mock_tc.arguments = '{}'
        mock_tc.call_id = "call_789"

        mock_response = Mock()
        mock_response.output = [mock_tc]

        agent._last_response = mock_response
        agent.do()

        assert "Unknown tool" in agent.messages[0]["output"]
