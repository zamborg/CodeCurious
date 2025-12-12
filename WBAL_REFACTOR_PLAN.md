# WBAL Package Refactor Plan

## Overview

This document contains detailed development instructions for improving the WBAL (Weights & Biases Agent Library) package. The goal is to reduce boilerplate that every agent implementation currently requires while maintaining the framework's flexibility and simplicity.

**Current State:** WBAL is a minimal ~740 line framework providing `Agent`, `Environment`, and `LM` base classes. The perceive-invoke-do loop is well-designed, but subclasses must implement significant boilerplate for message management, tool execution, and exit handling.

**Target State:** Provide sensible default implementations for common patterns while allowing full override capability. A simple agent should require ~50% less code to implement.

---

## Phase 1: Safe Additive Changes (Very Low to Low Risk)

These changes add new functionality without modifying existing behavior. They cannot break existing code.

---

### 1.1 Add output_handler to Environment

**File:** `/wbal/environment.py`

**Risk Level:** Low

**Rationale:** Every real agent needs a way to handle text output from the LLM. Currently, implementations either call `print()` directly or define their own handler. This should be a first-class concept on the Environment since output handling is context-dependent (CLI vs WebUI vs headless).

**Current Code:**

    class Environment(WBALObject):
        task: str = ""
        env: str = ""

**New Code:**

    from typing import Callable
    from pydantic import Field

    class Environment(WBALObject):
        task: str = ""
        env: str = ""
        
        output_handler: Callable[[str], None] = Field(default=lambda x: print(x))
        """Handler for agent text output. Override for custom output routing (e.g., WebUI, logging)."""

**Usage Example:**

    # CLI usage (default)
    env = Environment(task="Do something")
    env.output_handler("Hello")  # prints to stdout

    # WebUI usage
    def send_to_ui(text: str):
        websocket.send({"type": "message", "content": text})

    env = Environment(task="Do something", output_handler=send_to_ui)

**Test to Add:** `tests/test_environment.py`

    def test_output_handler_default(capsys):
        """Default output_handler prints to stdout."""
        env = Environment()
        env.output_handler("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_output_handler_custom():
        """Custom output_handler is called."""
        messages = []
        env = Environment(output_handler=lambda x: messages.append(x))
        env.output_handler("hello")
        assert messages == ["hello"]

---

### 1.2 Add reset() Method to Agent

**File:** `/wbal/agent.py`

**Risk Level:** Low

**Rationale:** Agents need to be reusable across multiple runs without recreating them. Currently there's no standard way to reset agent state between runs.

**Add this method to the Agent class:**

    def reset(self) -> None:
        """
        Reset agent state for a new run.
        
        Call this before re-running an agent to clear step count and
        any internal state. Subclasses should override this method
        and call super().reset() to add their own reset logic.
        
        Note: This does NOT clear messages by default. Override in
        subclass if message clearing is desired.
        """
        self._step_count = 0

**Usage Example:**

    agent = MyAgent(env=env)
    result1 = agent.run(task="First task")

    agent.reset()  # Clear state for reuse
    result2 = agent.run(task="Second task")

**Test to Add:** `tests/test_agent.py`

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

---

### 1.3 Add Tool Timeout Utility

**File:** `/wbal/helper.py`

**Risk Level:** Low

**Rationale:** Tools can hang indefinitely (network calls, file operations, etc.). A timeout utility prevents agents from getting stuck. This is additive and opt-in.

**Add to `/wbal/helper.py`:**

    import signal
    from contextlib import contextmanager


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

**Update `/wbal/__init__.py`:**

    from wbal.helper import weaveTool, tool, get_tools, tool_timeout, ToolTimeoutError

    __all__ = [
        # ... existing exports ...
        "tool_timeout",
        "ToolTimeoutError",
    ]

**Test to Add:** `tests/test_helper.py` (new file)

    import time
    import pytest
    from wbal.helper import tool_timeout, ToolTimeoutError


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

---

### 1.4 Fix Anthropic Tool Format

**File:** `/wbal/helper.py`

**Risk Level:** Very Low

**Rationale:** The current `to_anthropic_tool` function has a comment saying "I DON'T THINK THIS WORKS". Either fix it or remove it. Since Anthropic uses `input_schema` instead of `parameters`, this is a simple fix.

**Current Code:**

    # I DON'T THINK THIS WORKS
    def to_anthropic_tool(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],  # WRONG KEY
        }

**Fixed Code:**

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

**Test to Add:** `tests/test_helper.py`

    from wbal.helper import extract_tool_schema, to_anthropic_tool

    def test_to_anthropic_tool_format():
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

---

### 1.5 Add StatefulEnvironment Class

**File:** `/wbal/environment.py` (add to existing file)

**Risk Level:** Very Low

**Rationale:** Many agents need persistent state that survives across sessions. This is a common pattern that should be provided as an optional base class.

**Add to `/wbal/environment.py`:**

    import json
    import os
    from datetime import datetime
    from typing import Any


    class StatefulEnvironment(Environment):
        """
        Environment with persistent state support.
        
        Provides automatic state persistence to a working directory.
        State is stored as JSON and can be loaded/saved between sessions.
        
        Attributes:
            working_directory: Path to directory for state persistence.
                If None, state is in-memory only.
            _state: Internal state dictionary. Access via state property.
        
        Example:
            class MyEnv(StatefulEnvironment):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    # Custom initialization after state is loaded
                
                @tool
                def remember(self, key: str, value: str) -> str:
                    self._state["memory"][key] = value
                    self.save_state()
                    return f"Remembered {key}"
        """
        
        working_directory: str | None = None
        """Directory for state persistence. None = in-memory only."""
        
        _state: dict[str, Any] = {}
        """Internal state storage. Override _default_state() to customize structure."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._state = self._default_state()
            if self.working_directory:
                self._ensure_working_directory()
                self.load_state()
        
        def _default_state(self) -> dict[str, Any]:
            """
            Return the default state structure.
            
            Override this method to customize the initial state structure.
            
            Returns:
                Dict with default state structure
            """
            return {
                "data": {},
                "metadata": {
                    "created_at": None,
                    "last_updated": None,
                },
            }
        
        def _ensure_working_directory(self) -> None:
            """Create working directory if it doesn't exist."""
            if self.working_directory:
                os.makedirs(self.working_directory, exist_ok=True)
        
        def _state_file_path(self) -> str | None:
            """Get path to state file."""
            if self.working_directory:
                return os.path.join(self.working_directory, "environment_state.json")
            return None
        
        def load_state(self) -> bool:
            """
            Load state from working_directory.
            
            Returns:
                True if state was loaded, False if no state file exists
            """
            state_file = self._state_file_path()
            if not state_file or not os.path.exists(state_file):
                return False
            
            try:
                with open(state_file, "r") as f:
                    loaded_state = json.load(f)
                    # Merge with default state structure (preserves new keys)
                    self._state.update(loaded_state)
                return True
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load state from {state_file}: {e}")
                return False
        
        def save_state(self) -> bool:
            """
            Persist state to working_directory.
            
            Returns:
                True if state was saved, False if no working_directory set
            """
            state_file = self._state_file_path()
            if not state_file:
                return False
            
            try:
                now = datetime.now().isoformat()
                self._state["metadata"]["last_updated"] = now
                if not self._state["metadata"]["created_at"]:
                    self._state["metadata"]["created_at"] = now
                
                with open(state_file, "w") as f:
                    json.dump(self._state, f, indent=2)
                return True
            except IOError as e:
                print(f"Warning: Failed to save state to {state_file}: {e}")
                return False
        
        @property
        def state(self) -> dict[str, Any]:
            """Read-only access to state. Modify via _state directly."""
            return self._state.copy()

**Update `/wbal/__init__.py`:**

    from wbal.environment import Environment, StatefulEnvironment

    __all__ = [
        # ... existing exports ...
        "StatefulEnvironment",
    ]

**Test to Add:** `tests/test_environment.py`

    import tempfile
    import os
    import json

    from wbal import StatefulEnvironment


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

---

## Phase 2: Careful Additions (Low to Medium Risk)

These changes add new fields or modify default behavior. Existing code may need minor updates.

---

### 2.1 Add messages Field to Agent

**File:** `/wbal/agent.py`

**Risk Level:** Low

**Rationale:** Every agent implementation needs to track message history. Currently this is left to subclasses, but it's such a universal need that it belongs in the base class.

**Existing Subclass Behavior:** Subclasses that already define `messages: list[dict] = Field(default_factory=list)` will simply override this field (Pydantic allows field overrides). No breakage.

**Add to Agent class attributes:**

    class Agent(WBALObject):
        env: Environment
        maxSteps: int = 100
        
        # NEW: Add messages field
        messages: list[dict[str, Any]] = Field(default_factory=list)
        """Conversation history. Populated by perceive(), extended by invoke() and do()."""
        
        # ... rest of class

**Update reset() method (from 1.2) to optionally clear messages:**

    def reset(self, clear_messages: bool = False) -> None:
        """
        Reset agent state for a new run.
        
        Args:
            clear_messages: If True, also clear the message history.
                Default is False to allow conversation continuation.
        """
        self._step_count = 0
        if clear_messages:
            self.messages = []

**Test to Add:** `tests/test_agent.py`

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

---

### 2.2 Add Built-in exit Tool via ExitableAgent Mixin

**File:** `/wbal/mixins.py` (NEW FILE)

**Risk Level:** Medium

**Rationale:** Nearly every agent needs an exit mechanism. The current base class has a commented-out `exit` tool with a note saying "WE RECOMMEND USING THIS". Make it opt-in via a mixin to avoid tool name collisions.

**Why a Mixin:** Using a mixin (`ExitableAgent`) rather than adding directly to `Agent` avoids:
1. Tool name collision if subclass defines its own `exit`
2. Adding `_exit` state to agents that don't need it
3. Changing `stopCondition` behavior unexpectedly

**Create new file `/wbal/mixins.py`:**

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

**Update `/wbal/__init__.py`:**

    from wbal.mixins import ExitableAgent

    __all__ = [
        # ... existing exports ...
        "ExitableAgent",
    ]

**Test to Add:** `tests/test_mixins.py` (new file)

    import pytest
    from wbal import Environment, ExitableAgent, tool


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

---

### 2.3 Add _last_response Field to Agent

**File:** `/wbal/agent.py`

**Risk Level:** Low

**Rationale:** The `do()` method needs access to the LLM response from `invoke()`. Currently subclasses must define this themselves. Adding it to the base class enables default implementations.

**Add to Agent class:**

    from typing import Any

    class Agent(WBALObject):
        # ... existing fields ...
        
        _last_response: Any = None
        """
        Last response from LLM invoke(). Set by invoke(), used by do().
        
        Type depends on LM implementation (typically OpenAI Response object).
        """

**Test to Add:** `tests/test_agent.py`

    def test_last_response_initially_none(self):
        """_last_response is None before invoke."""
        agent = Agent(env=Environment())
        assert agent._last_response is None

---

## Phase 3: Default Implementations (Medium to Medium-High Risk)

These changes provide default behavior for `invoke()` and `do()`. They require careful implementation to avoid breaking existing subclasses.

---

### 3.1 Default invoke() Implementation

**File:** `/wbal/agent.py`

**Risk Level:** Medium-High

**Rationale:** The invoke pattern is nearly identical across implementations: call LM with messages and tools, store response, extend messages. Providing a default reduces boilerplate.

**Constraints:**
1. Must not break subclasses that override `invoke()`
2. Must handle case where `lm` is not set
3. Must work with OpenAI Response API format

**Implementation Strategy:** 
- Check if `lm` attribute exists and is not the base `LM` class
- Check if `messages` is populated
- If either check fails, do nothing (let subclass handle it)

**Modify invoke() in Agent class:**

    def invoke(self) -> Any:
        """
        Call the LLM with current messages and tools.
        
        Default implementation:
        1. Calls self.lm.invoke(messages=self.messages, tools=self._tool_definitions)
        2. Stores response in self._last_response
        3. Extends self.messages with response.output (OpenAI format)
        
        Override this method for custom LLM invocation logic or
        different response formats.
        
        Returns:
            The LLM response (stored in _last_response)
        
        Note:
            This default implementation assumes:
            - self.lm is set and has an invoke() method
            - self.messages is populated (by perceive())
            - Response has .output attribute (OpenAI Responses API)
            
            If these assumptions don't hold for your use case,
            override this method.
        """
        # Check prerequisites
        if not hasattr(self, 'lm') or self.lm is None:
            return None
        
        if not self.messages:
            return None
        
        # Prepare tools (None if empty)
        tools = self._tool_definitions if self._tool_definitions else None
        
        # Call LLM
        response = self.lm.invoke(messages=self.messages, tools=tools)
        self._last_response = response
        
        # Extend messages with response (OpenAI format)
        # Response.output is a list of message items
        if hasattr(response, 'output'):
            self.messages.extend(response.output)
        
        return response

**IMPORTANT:** Update the Agent class to NOT require `lm` by default:

    class Agent(WBALObject):
        env: Environment
        maxSteps: int = 100
        
        lm: LM | None = None  # Make optional with None default
        """Language model for invoke(). Set in subclass or at instantiation."""

**Test to Add:** `tests/test_agent.py`

    def test_invoke_without_lm_is_noop(self):
        """invoke() does nothing if lm is not set."""
        agent = Agent(env=Environment())
        result = agent.invoke()
        assert result is None
        assert agent._last_response is None

    def test_invoke_without_messages_is_noop(self):
        """invoke() does nothing if messages is empty."""
        agent = Agent(env=Environment(), lm=GPT5MiniTester())
        result = agent.invoke()
        assert result is None

    def test_invoke_calls_lm(self, mocker):
        """invoke() calls lm.invoke with messages and tools."""
        mock_response = mocker.Mock()
        mock_response.output = [{"role": "assistant", "content": "hi"}]
        
        mock_lm = mocker.Mock()
        mock_lm.invoke.return_value = mock_response
        
        agent = Agent(env=Environment(), lm=mock_lm)
        agent.messages = [{"role": "user", "content": "hello"}]
        
        agent.invoke()
        
        mock_lm.invoke.assert_called_once()
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

---

### 3.2 Default do() Implementation

**File:** `/wbal/agent.py`

**Risk Level:** Medium

**Rationale:** Tool execution logic is identical across implementations: parse tool calls from response, execute each one, format results, append to messages. This is significant boilerplate (~40 lines).

**Constraints:**
1. Must not break subclasses that override `do()`
2. Must handle case where `_last_response` is None
3. Must work with OpenAI Response API format
4. Should use `output_handler` for text output

**Add helper import at top of agent.py:**

    from wbal.helper import format_openai_tool_response
    import json

**Modify do() in Agent class:**

    def do(self) -> None:
        """
        Execute tool calls from the LLM response.
        
        Default implementation:
        1. Extracts function_call items from _last_response.output
        2. Executes each tool via _tool_callables
        3. Formats results and appends to messages
        4. If no tool calls, sends output_text to env.output_handler
        
        Override this method for custom tool execution logic or
        different response formats.
        
        Note:
            This default implementation assumes:
            - self._last_response has .output (list) and .output_text (str)
            - Tool calls have .type == "function_call"
            - Tool calls have .name, .arguments (JSON string), .call_id
            
            If these assumptions don't hold, override this method.
        """
        if self._last_response is None:
            return
        
        # Get response output
        output = getattr(self._last_response, 'output', None)
        if output is None:
            return
        
        # Extract tool calls (OpenAI format: type == "function_call")
        tool_calls = [
            item for item in output 
            if getattr(item, 'type', None) == 'function_call'
        ]
        
        # If no tool calls, handle text output
        if not tool_calls:
            output_text = getattr(self._last_response, 'output_text', '')
            if output_text and hasattr(self.env, 'output_handler'):
                self.env.output_handler(output_text)
            return
        
        # Execute each tool call
        for tc in tool_calls:
            tc_name = getattr(tc, 'name', '')
            tc_args_raw = getattr(tc, 'arguments', '{}')
            tc_id = getattr(tc, 'call_id', '')
            
            # Parse arguments
            if isinstance(tc_args_raw, str):
                try:
                    tc_args = json.loads(tc_args_raw)
                except json.JSONDecodeError:
                    tc_args = {}
            else:
                tc_args = tc_args_raw or {}
            
            # Execute tool
            if tc_name in self._tool_callables:
                try:
                    tc_output = self._tool_callables[tc_name](**tc_args)
                except Exception as e:
                    tc_output = f"Error executing {tc_name}: {e}"
            else:
                tc_output = f"Unknown tool: {tc_name}"
            
            # Format and append result
            result = format_openai_tool_response(tc_output, tc_id)
            self.messages.append(result)

**Test to Add:** `tests/test_agent.py`

    def test_do_without_response_is_noop(self):
        """do() does nothing if _last_response is None."""
        agent = Agent(env=Environment())
        agent.do()  # Should not raise

    def test_do_executes_tool_calls(self, mocker):
        """do() executes tool calls and appends results."""
        class TestEnv(Environment):
            @tool
            def add(self, a: int, b: int) -> int:
                return a + b
        
        agent = Agent(env=TestEnv())
        
        # Mock response with tool call
        mock_tc = mocker.Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "add"
        mock_tc.arguments = '{"a": 2, "b": 3}'
        mock_tc.call_id = "call_123"
        
        mock_response = mocker.Mock()
        mock_response.output = [mock_tc]
        
        agent._last_response = mock_response
        agent.do()
        
        # Check result was appended
        assert len(agent.messages) == 1
        assert agent.messages[0]["type"] == "function_call_output"
        assert "5" in agent.messages[0]["output"]

    def test_do_handles_text_output(self, mocker):
        """do() calls output_handler for text responses."""
        outputs = []
        env = Environment(output_handler=lambda x: outputs.append(x))
        agent = Agent(env=env)
        
        mock_response = mocker.Mock()
        mock_response.output = []  # No tool calls
        mock_response.output_text = "Hello, world!"
        
        agent._last_response = mock_response
        agent.do()
        
        assert outputs == ["Hello, world!"]

    def test_do_handles_tool_errors(self, mocker):
        """do() catches and reports tool execution errors."""
        class TestEnv(Environment):
            @tool
            def failing_tool(self) -> str:
                raise ValueError("Tool failed!")
        
        agent = Agent(env=TestEnv())
        
        mock_tc = mocker.Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "failing_tool"
        mock_tc.arguments = '{}'
        mock_tc.call_id = "call_456"
        
        mock_response = mocker.Mock()
        mock_response.output = [mock_tc]
        
        agent._last_response = mock_response
        agent.do()  # Should not raise
        
        assert "Error" in agent.messages[0]["output"]
        assert "Tool failed!" in agent.messages[0]["output"]

    def test_do_handles_unknown_tool(self, mocker):
        """do() handles calls to unknown tools gracefully."""
        agent = Agent(env=Environment())
        
        mock_tc = mocker.Mock()
        mock_tc.type = "function_call"
        mock_tc.name = "nonexistent_tool"
        mock_tc.arguments = '{}'
        mock_tc.call_id = "call_789"
        
        mock_response = mocker.Mock()
        mock_response.output = [mock_tc]
        
        agent._last_response = mock_response
        agent.do()
        
        assert "Unknown tool" in agent.messages[0]["output"]

---

### 3.3 Dynamic Tool Descriptions in Environment

**File:** `/wbal/environment.py`

**Risk Level:** Medium

**Rationale:** Agents often need to include tool descriptions in their system prompts. Currently this requires manual construction. The environment should be able to generate formatted tool documentation.

**Why Medium Risk:** Changes `observe()` behavior. Existing code relying on `observe()` returning exactly `self.env` may break.

**Mitigation:** Add a separate method `get_tool_descriptions()` and make the enhanced `observe()` opt-in via a flag.

**Add to Environment class:**

    import textwrap

    class Environment(WBALObject):
        # ... existing fields ...
        
        include_tools_in_observe: bool = False
        """If True, observe() includes formatted tool descriptions."""
        
        def get_tool_descriptions(self) -> str:
            """
            Generate formatted descriptions of all available tools.
            
            Extracts docstrings from @tool decorated methods and formats
            them for inclusion in prompts.
            
            Returns:
                Formatted string with all tool descriptions, or empty string if no tools
            """
            tools = self.get_tools()
            if not tools:
                return ""
            
            descriptions = []
            for name in sorted(tools.keys()):
                method = tools[name]
                doc = method.__doc__ or "No description available."
                # Clean up docstring indentation
                doc = textwrap.dedent(doc).strip()
                descriptions.append(f"## {name}\n{doc}")
            
            return "# Available Tools\n\n" + "\n\n".join(descriptions)
        
        def observe(self) -> str:
            """
            Return observable state of the environment.
            
            If include_tools_in_observe is True, appends formatted
            tool descriptions to the base observation.
            """
            base = self.env
            
            if self.include_tools_in_observe:
                tools_desc = self.get_tool_descriptions()
                if tools_desc:
                    return f"{base}\n\n{tools_desc}" if base else tools_desc
            
            return base

**Test to Add:** `tests/test_environment.py`

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

---

## Phase 4: LM Enhancements (Low Risk)

---

### 4.1 Add @weave.op() to LM invoke methods

**File:** `/wbal/lm.py`

**Risk Level:** Low

**Rationale:** LLM calls should be traced for observability. Adding `@weave.op()` makes them visible in weave traces.

**Modify GPT5Large and GPT5MiniTester:**

    import weave

    class GPT5Large(LM):
        # ... existing fields ...
        
        @weave.op()
        def invoke(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            mcp_servers: list[dict[str, Any]] | None = None,
        ) -> Response:
            """Invoke the language model."""
            # ... existing implementation ...


    class GPT5MiniTester(LM):
        # ... existing fields ...
        
        @weave.op()
        def invoke(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            mcp_servers: list[dict[str, Any]] | None = None,
        ) -> Response:
            """Invoke the language model."""
            # ... existing implementation ...

---

## Summary: Implementation Order

Execute phases in order. Each phase builds on the previous.

### Phase 1 (Start Here - Very Low/Low Risk)
1. `output_handler` on Environment
2. `reset()` method on Agent
3. Tool timeout utility in helper.py
4. Fix Anthropic tool format
5. `StatefulEnvironment` class

### Phase 2 (After Phase 1 - Low/Medium Risk)
1. `messages` field on Agent
2. `ExitableAgent` mixin (new file: mixins.py)
3. `_last_response` field on Agent

### Phase 3 (After Phase 2 - Medium/Medium-High Risk)
1. Default `invoke()` implementation
2. Default `do()` implementation
3. Dynamic tool descriptions in Environment

### Phase 4 (Any Time - Low Risk)
1. `@weave.op()` on LM invoke methods

---

## Testing Strategy

1. **Run existing tests after each change:** `pytest tests/`
2. **Add new tests as specified in each section**
3. **Test the story_summarizer example:** Ensure it still works after each phase
4. **Integration test:** After all phases, create a minimal agent using only base classes with default implementations

**Minimal Agent Test (after all phases):**

    from wbal import ExitableAgent, Environment, GPT5MiniTester, tool

    class MinimalEnv(Environment):
        @tool
        def get_answer(self) -> str:
            """Get the answer to everything."""
            return "42"

    class MinimalAgent(ExitableAgent):
        lm = GPT5MiniTester()
        
        def perceive(self):
            if self._step_count == 0:
                self.messages = [
                    {"role": "system", "content": "You answer questions. Use exit() when done."},
                    {"role": "user", "content": self.env.task},
                ]

    # This should work with just the above code:
    agent = MinimalAgent(env=MinimalEnv(task="What is the answer?"))
    result = agent.run()
    print(agent._exit_message)

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `/wbal/environment.py` | Add `output_handler`, `include_tools_in_observe`, `get_tool_descriptions()`, `StatefulEnvironment` class |
| `/wbal/agent.py` | Add `messages`, `_last_response`, `reset()`, default `invoke()`, default `do()` |
| `/wbal/helper.py` | Add `tool_timeout()`, `ToolTimeoutError`, fix `to_anthropic_tool()` |
| `/wbal/lm.py` | Add `@weave.op()` to invoke methods |
| `/wbal/mixins.py` | New file with `ExitableAgent` |
| `/wbal/__init__.py` | Export new classes and functions |
| `/tests/test_environment.py` | Add tests for new Environment features |
| `/tests/test_agent.py` | Add tests for new Agent features |
| `/tests/test_helper.py` | New file for helper tests |
| `/tests/test_mixins.py` | New file for mixin tests |
