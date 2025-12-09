import pytest

from wbal import Agent, Environment, tool


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

    def test_stop_condition_respects_max_steps(self):
        """stopCondition returns True when step count reaches maxSteps."""
        agent = Agent(env=Environment(), maxSteps=5)
        assert agent.stopCondition is False

        agent._step_count = 5
        assert agent.stopCondition is True

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
