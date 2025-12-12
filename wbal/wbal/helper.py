"""
Helper utilities for WBAL.

Contains data marshalling functions for converting between different
tool definition formats (OpenAI, Anthropic, etc.)
"""

import json
import signal
from contextlib import contextmanager
from typing import Any, Callable, get_type_hints, get_origin, get_args, Union
import inspect
import weave


# -----------------------------------------------------------------------------
# Constants for OpenAI Response API format
# -----------------------------------------------------------------------------

# Tool call types
TOOL_TYPE_FUNCTION = "function"
TOOL_CALL_TYPE = "function_call"
TOOL_RESULT_TYPE = "function_call_output"


# -----------------------------------------------------------------------------
# Tool timeout utilities
# -----------------------------------------------------------------------------

class ToolTimeoutError(Exception):
    """Raised when a tool execution exceeds its timeout."""
    pass


@contextmanager
def tool_timeout(seconds: int, tool_name: str = "tool"):
    """
    Context manager for timing out tool executions.

    Args:
        seconds: Maximum execution time in seconds
        tool_name: Name of the tool (for error message)

    Raises:
        ToolTimeoutError: If execution exceeds timeout

    Example:
        with tool_timeout(30, "fetch_data"):
            result = slow_api_call()

    Warning:
        This uses SIGALRM which only works on Unix systems and
        only in the main thread. For Windows or threaded contexts,
        use alternative timeout mechanisms.
    """
    def _timeout_handler(signum, frame):
        raise ToolTimeoutError(
            f"Tool '{tool_name}' timed out after {seconds} seconds"
        )

    # Store old handler and set alarm
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

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
    except NameError:
        # Type hint references undefined name (e.g., forward reference not resolved)
        pass
    except AttributeError:
        # Function doesn't support type hints (e.g., built-in)
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
        "type": TOOL_TYPE_FUNCTION,
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["parameters"],
    }


def format_openai_tool_response(tc_output: dict[str, Any] | str, call_id: str) -> dict[str, Any]:
    """
    Marshal a value into the tool-response format expected by OpenAI Responses API.

    Args:
        tc_output: The tool's return value (will be JSON-serialized if dict)
        call_id: The call_id from the function_call item in the response

    Returns:
        Dict with call_id, output (as string), and type fields
    """
    output_str = json.dumps(tc_output) if isinstance(tc_output, dict) else str(tc_output)
    return {
        "call_id": call_id,
        "output": output_str,
        "type": TOOL_RESULT_TYPE,
    }

# -----------------------------------------------------------------------------
# Anthropic format
# -----------------------------------------------------------------------------

def to_anthropic_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a tool schema to Anthropic's tool format.

    Anthropic expects 'input_schema' instead of 'parameters'.
    See: https://docs.anthropic.com/en/docs/tool-use

    Args:
        schema: Tool schema from extract_tool_schema()

    Returns:
        Anthropic-compatible tool definition
    """
    return {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],  # Anthropic uses input_schema
    }

# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------

def tool(func: Callable) -> Callable:
    """
    Decorator to mark a method as a tool.

    Tools are methods that can be called by the LLM. They're used in both
    Agents (internal capabilities like memory, reasoning) and Environments
    (external capabilities like service APIs, data access).

    The method's docstring becomes the tool description.
    Type annotations are used to generate the parameter schema.
    """
    func._is_tool = True
    return func


def weaveTool(func: Callable) -> Callable:
    """
    Assigns func._is_tool = True.
    Also adds a weave.op() decorator to the function.
    """
    func._is_tool = True
    func = weave.op()(func)
    return func


# -----------------------------------------------------------------------------
# Tool discovery
# -----------------------------------------------------------------------------

def get_tools(obj: object) -> dict[str, Callable]:
    """
    Get all @tool decorated methods from an object.

    Args:
        obj: The object to inspect

    Returns:
        Dict mapping method names to bound methods
    """
    methods = {}
    # Skip pydantic internal attributes to avoid deprecation warnings
    skip_attrs = {"model_fields", "model_computed_fields", "model_config"}
    for name in dir(obj):
        if name.startswith("_") or name in skip_attrs:
            continue
        attr = getattr(obj, name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            methods[name] = attr
    return methods