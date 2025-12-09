"""
Helper utilities for WBAL.

Contains data marshalling functions for converting between different
tool definition formats (OpenAI, Claude/Anthropic, etc.)
"""

from re import L
from typing import Any, Callable, get_type_hints, get_origin, get_args, Union
import inspect


# -----------------------------------------------------------------------------
# Type conversion
# -----------------------------------------------------------------------------


def python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """
    Convert a Python type annotation to a JSON schema type.

    Handles basic types, Optional, Union, List, Dict, etc.

    Args:
        python_type: The Python type annotation

    Returns:
        JSON schema dict (e.g., {"type": "string"})
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    basic_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "binary"},
    }

    if python_type in basic_map:
        return basic_map[python_type]

    # Handle generic types (List, Dict, Optional, Union)
    origin = get_origin(python_type)
    args = get_args(python_type)

    # list[X] or List[X]
    if origin is list:
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # dict[K, V] or Dict[K, V]
    if origin is dict:
        schema: dict[str, Any] = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return schema

    # Optional[X] is Union[X, None]
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # This is Optional[X]
            return python_type_to_json_schema(non_none_args[0])
        # True union - use anyOf
        return {"anyOf": [python_type_to_json_schema(a) for a in args]}

    # Default fallback
    return {"type": "string"}


# -----------------------------------------------------------------------------
# Tool definition extraction from Python functions
# -----------------------------------------------------------------------------

def extract_tool_schema(func: Callable) -> dict[str, Any]:
    """
    Extract a tool schema from a Python function.

    This is a provider-agnostic intermediate representation.
    Use `to_openai_tool` or `to_anthropic_tool` to convert to specific formats.

    Args:
        func: The function to extract schema from

    Returns:
        Dict with 'name', 'description', 'parameters' (JSON schema)
    """
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = hints.get(param_name, str)
        properties[param_name] = python_type_to_json_schema(param_type)

        # Add description from docstring if we can parse it
        # (simplified - could use docstring_parser for more robust parsing)

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": func.__name__,
        "description": func.__doc__ or f"Call {func.__name__}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# -----------------------------------------------------------------------------
# OpenAI format
# -----------------------------------------------------------------------------

def to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a tool schema to OpenAI's tool format.

    Args:
        schema: Tool schema from extract_tool_schema()

    Returns:
        OpenAI-compatible tool definition
    """
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["parameters"],
    }


def to_openai_tools(funcs: list[Callable]) -> list[dict[str, Any]]:
    """Convert multiple functions to OpenAI tool format."""
    return [to_openai_tool(extract_tool_schema(f)) for f in funcs]


# -----------------------------------------------------------------------------
# Anthropic/Claude format
# -----------------------------------------------------------------------------

def to_anthropic_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a tool schema to Anthropic's tool format.

    Args:
        schema: Tool schema from extract_tool_schema()

    Returns:
        Anthropic-compatible tool definition
    """
    return {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],
    }


def to_anthropic_tools(funcs: list[Callable]) -> list[dict[str, Any]]:
    """Convert multiple functions to Anthropic tool format."""
    return [to_anthropic_tool(extract_tool_schema(f)) for f in funcs]


# -----------------------------------------------------------------------------
# Tool call parsing
# -----------------------------------------------------------------------------

def parse_openai_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse tool calls from an OpenAI response message.

    Args:
        message: The assistant message dict

    Returns:
        List of dicts with 'id', 'name', 'arguments' (as dict)
    """
    import json

    tool_calls = message.get("tool_calls", [])
    parsed = []

    for tc in tool_calls:
        parsed.append({
            "id": tc.get("id"),
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"]),
        })

    return parsed


def parse_anthropic_tool_calls(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Parse tool calls from Anthropic response content blocks.

    Args:
        content_blocks: The content array from an Anthropic response

    Returns:
        List of dicts with 'id', 'name', 'arguments' (as dict)
    """
    parsed = []

    for block in content_blocks:
        if block.get("type") == "tool_use":
            parsed.append({
                "id": block.get("id"),
                "name": block["name"],
                "arguments": block["input"],
            })

    return parsed


# -----------------------------------------------------------------------------
# Tool result formatting
# -----------------------------------------------------------------------------

def format_openai_tool_result(tool_call_id: str, result: Any) -> dict[str, Any]:
    """Format a tool result for OpenAI's messages format."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": str(result),
    }


def format_anthropic_tool_result(tool_use_id: str, result: Any) -> dict[str, Any]:
    """Format a tool result for Anthropic's messages format."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": str(result),
    }
