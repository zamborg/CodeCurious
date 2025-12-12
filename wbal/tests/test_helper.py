import time
import pytest

from wbal.helper import tool_timeout, ToolTimeoutError, extract_tool_schema, to_anthropic_tool


class TestToolTimeout:
    """Tests for tool_timeout context manager."""

    def test_no_timeout_when_fast(self):
        """Fast operations complete normally."""
        with tool_timeout(5, "fast_tool"):
            result = 1 + 1
        assert result == 2

    def test_timeout_raises_error(self):
        """Slow operations raise ToolTimeoutError."""
        with pytest.raises(ToolTimeoutError, match="slow_tool"):
            with tool_timeout(1, "slow_tool"):
                time.sleep(5)

    def test_timeout_error_includes_tool_name(self):
        """Error message includes the tool name."""
        try:
            with tool_timeout(1, "my_custom_tool"):
                time.sleep(5)
        except ToolTimeoutError as e:
            assert "my_custom_tool" in str(e)
            assert "1 seconds" in str(e)


class TestToAnthropicTool:
    """Tests for to_anthropic_tool function."""

    def test_to_anthropic_tool_format(self):
        """to_anthropic_tool produces correct Anthropic format."""
        def my_tool(arg: str) -> str:
            """My tool description."""
            return arg

        schema = extract_tool_schema(my_tool)
        anthropic_format = to_anthropic_tool(schema)

        assert anthropic_format["name"] == "my_tool"
        assert anthropic_format["description"] == "My tool description."
        assert "input_schema" in anthropic_format
        assert "parameters" not in anthropic_format
        assert anthropic_format["input_schema"]["type"] == "object"
