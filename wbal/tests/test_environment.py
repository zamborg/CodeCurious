import pytest

from wbal import Environment, tool


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
