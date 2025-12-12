import time
from typing import Annotated, Literal, get_origin
import pytest
from pydantic import Field

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


class TestExtractToolSchema:
    """Tests for extract_tool_schema function."""

    def test_basic_types(self):
        """Schema extracts basic types correctly."""
        def my_func(name: str, count: int) -> str:
            """A function."""
            return name

        schema = extract_tool_schema(my_func)
        props = schema["parameters"]["properties"]

        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"

    def test_type_hints_field_present(self):
        """Each property includes type_hints dict with raw annotation (except plain str)."""
        def my_func(name: str, count: int) -> str:
            """A function."""
            return name

        schema = extract_tool_schema(my_func)
        props = schema["parameters"]["properties"]

        # Plain str without extras doesn't need type_hints
        assert "type_hints" not in props["name"]
        # int gets type_hints since it's not str
        assert props["count"]["type_hints"]["raw"] is int

    def test_literal_type_creates_enum(self):
        """Literal types are converted to enum in type_hints."""
        def simplefunc(unan: str, yesan: Literal['foo', 'bar']):
            """A function with Literal."""
            return

        schema = extract_tool_schema(simplefunc)
        props = schema["parameters"]["properties"]

        # unan is regular string - no type_hints needed
        assert props["unan"]["type"] == "string"
        assert "type_hints" not in props["unan"]

        # yesan is Literal - enum should be in type_hints
        assert props["yesan"]["type"] == "string"
        assert props["yesan"]["type_hints"]["enum"] == ["foo", "bar"]

    def test_literal_type_hints_preserved(self):
        """Literal type_hints is preserved for downstream detection."""
        def simplefunc(unan: str, yesan: Literal['foo']):
            """A function with Literal."""
            return

        schema = extract_tool_schema(simplefunc)
        props = schema["parameters"]["properties"]

        # unan is plain str - no type_hints
        assert "type_hints" not in props["unan"]

        # yesan type_hints raw should be the Literal type
        yesan_hint = props["yesan"]["type_hints"]["raw"]
        assert get_origin(yesan_hint) is Literal

    def test_single_literal_value(self):
        """Single Literal value works correctly."""
        def func(mode: Literal['only_option']):
            """Single option."""
            return

        schema = extract_tool_schema(func)
        props = schema["parameters"]["properties"]

        assert props["mode"]["type_hints"]["enum"] == ["only_option"]

    def test_default_arg_not_required(self):
        """Parameters with defaults are not in required list and default is captured."""
        def simplefunc(unan: str, yesan: Literal['foo'], default_arg: int = 372):
            return

        schema = extract_tool_schema(simplefunc)
        props = schema["parameters"]["properties"]
        required = schema["parameters"]["required"]

        # All three params should be in properties
        assert "unan" in props
        assert "yesan" in props
        assert "default_arg" in props

        # Only unan and yesan should be required (default_arg has a default)
        assert "unan" in required
        assert "yesan" in required
        assert "default_arg" not in required

        # default_arg type_hints raw should be int, default in type_hints
        assert props["default_arg"]["type_hints"]["raw"] is int
        assert props["default_arg"]["type"] == "integer"
        assert props["default_arg"]["type_hints"]["default"] == 372

    def test_annotated_field_description(self):
        """Annotated with Field extracts description."""
        def simplefunc(
            unan: str,
            yesan: Literal['foo'],
            default_arg: int = 372,
            ann: Annotated[int, Field(description="this is a desc")] = 99
        ):
            return

        schema = extract_tool_schema(simplefunc)
        props = schema["parameters"]["properties"]

        # ann should have description from Field at top level
        assert props["ann"]["description"] == "this is a desc"
        assert props["ann"]["type"] == "integer"
        # default and field_info should be in type_hints
        assert props["ann"]["type_hints"]["default"] == 99
        assert "field_info" in props["ann"]["type_hints"]

        # Other params should not have description
        assert "description" not in props["unan"]
        assert "description" not in props["yesan"]
        assert "description" not in props["default_arg"]
