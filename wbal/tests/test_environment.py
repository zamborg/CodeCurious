import tempfile
import os
import json
import pytest

from wbal.environment import Environment, StatefulEnvironment
from wbal.helper import tool


class TestEnvironment:
    """Unit tests for the Environment base class."""

    def test_default_values(self):
        """Environment has sensible defaults."""
        env = Environment()
        assert env.task == ""
        assert env.env == ""

    def test_observe_returns_env_string(self):
        """observe() returns the env string."""
        env = Environment(env="You have access to files.")
        assert env.observe() == "You have access to files."

    def test_get_tools_empty_by_default(self):
        """Base Environment has no tools."""
        env = Environment()
        assert env.get_tools() == {}

    def test_subclass_with_tool(self):
        """Subclass @tool methods are discovered."""
        class MyEnv(Environment):
            @tool
            def read_file(self, path: str) -> str:
                '''Read a file.'''
                return f"contents of {path}"

        env = MyEnv()
        tools = env.get_tools()

        assert "read_file" in tools
        assert tools["read_file"]("test.txt") == "contents of test.txt"

    def test_execute_environment_fabric(self):
        """execute_environment_fabric calls tools by name."""
        class MyEnv(Environment):
            @tool
            def add(self, a: int, b: int) -> int:
                '''Add two numbers.'''
                return a + b

        env = MyEnv()
        result = env.execute_environment_fabric("add", 2, 3)
        assert result == 5

    def test_execute_environment_fabric_with_kwargs(self):
        """execute_environment_fabric works with kwargs."""
        class MyEnv(Environment):
            @tool
            def greet(self, name: str, greeting: str = "Hello") -> str:
                '''Greet someone.'''
                return f"{greeting}, {name}!"

        env = MyEnv()
        result = env.execute_environment_fabric("greet", name="World", greeting="Hi")
        assert result == "Hi, World!"

    def test_execute_environment_fabric_unknown_tool(self):
        """execute_environment_fabric raises ValueError for unknown tools."""
        env = Environment()
        with pytest.raises(ValueError, match="Unknown tool: nonexistent"):
            env.execute_environment_fabric("nonexistent")

    def test_multiple_tools(self):
        """Multiple @tool methods are all discovered."""
        class MyEnv(Environment):
            @tool
            def tool_a(self) -> str:
                '''Tool A.'''
                return "a"

            @tool
            def tool_b(self) -> str:
                '''Tool B.'''
                return "b"

            def not_a_tool(self) -> str:
                '''Regular method.'''
                return "not a tool"

        env = MyEnv()
        tools = env.get_tools()

        assert len(tools) == 2
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert "not_a_tool" not in tools

    def test_output_handler_default(self, capsys):
        """Default output_handler prints to stdout."""
        env = Environment()
        env.output_handler("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_output_handler_custom(self):
        """Custom output_handler is called."""
        messages = []
        env = Environment(output_handler=lambda x: messages.append(x))
        env.output_handler("hello")
        assert messages == ["hello"]

    def test_get_tool_descriptions_empty(self):
        """get_tool_descriptions returns empty string if no tools."""
        env = Environment()
        assert env.get_tool_descriptions() == ""

    def test_get_tool_descriptions_formats_tools(self):
        """get_tool_descriptions formats tool docstrings."""
        class MyEnv(Environment):
            @tool
            def my_tool(self, arg: str) -> str:
                """
                This is my tool.

                Args:
                    arg: An argument
                """
                return arg

        env = MyEnv()
        desc = env.get_tool_descriptions()

        assert "# Available Tools" in desc
        assert "## my_tool" in desc
        assert "This is my tool." in desc

    def test_observe_without_tools_flag(self):
        """observe() returns only env when flag is False."""
        class MyEnv(Environment):
            @tool
            def hidden_tool(self) -> str:
                """Hidden."""
                return "x"

        env = MyEnv(env="Base observation")
        assert env.observe() == "Base observation"
        assert "hidden_tool" not in env.observe()

    def test_observe_with_tools_flag(self):
        """observe() includes tools when flag is True."""
        class MyEnv(Environment):
            @tool
            def visible_tool(self) -> str:
                """Visible."""
                return "x"

        env = MyEnv(env="Base observation", include_tools_in_observe=True)
        obs = env.observe()

        assert "Base observation" in obs
        assert "## visible_tool" in obs
        assert "Visible." in obs


class TestStatefulEnvironment:
    """Tests for StatefulEnvironment."""

    def test_in_memory_state(self):
        """State works without working_directory."""
        env = StatefulEnvironment()
        env._state["data"]["key"] = "value"
        assert env._state["data"]["key"] == "value"

    def test_state_persistence(self):
        """State persists to working_directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            env1 = StatefulEnvironment(working_directory=tmpdir)
            env1._state["data"]["test"] = "hello"
            env1.save_state()

            # Load in new instance
            env2 = StatefulEnvironment(working_directory=tmpdir)
            assert env2._state["data"]["test"] == "hello"

    def test_creates_working_directory(self):
        """Working directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "new", "nested", "dir")
            env = StatefulEnvironment(working_directory=subdir)
            assert os.path.exists(subdir)

    def test_metadata_timestamps(self):
        """Metadata timestamps are set on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = StatefulEnvironment(working_directory=tmpdir)
            assert env._state["metadata"]["created_at"] is None

            env.save_state()

            assert env._state["metadata"]["created_at"] is not None
            assert env._state["metadata"]["last_updated"] is not None

    def test_default_state_override(self):
        """Subclasses can override _default_state."""
        class CustomEnv(StatefulEnvironment):
            def _default_state(self):
                state = super()._default_state()
                state["notes"] = []
                state["counter"] = 0
                return state

        env = CustomEnv()
        assert "notes" in env._state
        assert env._state["counter"] == 0
