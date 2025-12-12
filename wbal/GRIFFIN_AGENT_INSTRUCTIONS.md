# WBAL Framework - Developer Hotstart Guide

## Overview

WBAL is a minimal agent framework built on three primitives:

1. **Agent** - Orchestrates the perceive-invoke-do loop
2. **Environment** - Provides tools and context for the agent
3. **LM** - Language model interface

All components inherit from `WBALObject` (Pydantic BaseModel + `observe()` method).

## Quick Start

```python
import weave
from wbal import Agent, Environment, weaveTool, GPT5MiniTester, LM
from wbal.helper import format_openai_tool_response

weave.init('your-project-name')

# 1. Define your Environment
class MyEnv(Environment):
    task: str = "Your task description"
    env: str = "Context about available tools"

    @weaveTool
    def my_tool(self, arg: str) -> str:
        '''Tool description for the LLM.'''
        return f"Result: {arg}"

# 2. Define your Agent
class MyAgent(Agent):
    lm: LM = Field(default_factory=GPT5MiniTester)
    messages: list[dict] = Field(default_factory=list)
    _exit: bool = False

    @property
    def stopCondition(self) -> bool:
        return self._exit

    @weaveTool
    def exit(self, exit_message: str) -> str:
        '''Exit the run loop with a final message.'''
        self._exit = True
        return exit_message

    def perceive(self) -> None:
        # Set up messages on first step
        pass

    def invoke(self) -> None:
        # Call LLM, append response to messages
        pass

    def do(self) -> None:
        # Execute tool calls, append results to messages
        pass

# 3. Run it
agent = MyAgent(env=MyEnv(), maxSteps=20)
result = agent.run()
```

## The Perceive-Invoke-Do Loop

Each `step()` calls three methods in sequence:

### `perceive()` - Gather observations, update state

- Runs at the start of each step
- Typically sets up initial messages on step 0
- Can update state based on environment observations

```python
def perceive(self) -> None:
    if self._step_count == 0:
        self.messages.append({"role": "system", "content": self.system_prompt})
        self.messages.append({"role": "user", "content": f"Task: {self.env.task}\n\n{self.env.observe()}"})
```

### `invoke()` - Call the LLM

- Calls `self.lm.invoke()` with messages and tool definitions
- Appends LLM response to message history

```python
def invoke(self) -> None:
    tools = self._tool_definitions if self._tool_definitions else None
    self._last_response = self.lm.invoke(messages=self.messages, tools=tools)
    self.messages.extend(self._last_response.output)  # OpenAI format
```

### `do()` - Execute tool calls

- Parses tool calls from LLM response
- Executes tools via `self._tool_callables[name](**args)`
- Appends tool results to messages

```python
def do(self) -> None:
    output = self._last_response.output
    tcs = [item for item in output if item.type == "function_call"]
    tc_results = handle_oai_tcs(self, tcs)  # Helper function
    self.messages.extend(tc_results)
```

## Decorators

### `@weaveTool` (Recommended)

Marks a method as a tool AND adds Weave tracing:

```python
@weaveTool
def ReadFile(self, file_path: str) -> str:
    '''Read contents of a file.'''
    return open(file_path).read()
```

### `@tool`

Basic decorator without Weave tracing:

```python
@tool
def simple_tool(self, arg: str) -> str:
    '''Tool description.'''
    return arg
```

## Tool Discovery

Tools are automatically discovered from both Agent and Environment:

- `self._tool_definitions` - List of tool schemas for LLM
- `self._tool_callables` - Dict mapping tool name to callable

**Important:** No duplicate tool names allowed between agent and environment.

## Stop Conditions

Override `stopCondition` property to control when the agent stops:

```python
@property
def stopCondition(self) -> bool:
    return self._exit  # Custom flag set by exit() tool
```

The recommended pattern is to provide an `exit()` tool:

```python
@weaveTool
def exit(self, exit_message: str) -> str:
    '''Exit your run loop with a final message.'''
    self._exit = True
    return exit_message
```

## Handling OpenAI Tool Calls

Helper function for executing tool calls (put in your agent file or factor into helpers):

```python
from wbal.helper import format_openai_tool_response
import json

def handle_oai_tcs(self, tc_items: list) -> list:
    tc_results = []
    for item in tc_items:
        tc_name = item.name
        tc_args = item.arguments
        tc_id = item.call_id

        if isinstance(tc_args, str):
            tc_args = json.loads(tc_args)

        if tc_name in self._tool_callables:
            try:
                tc_output = self._tool_callables[tc_name](**tc_args)
            except Exception as e:
                tc_output = f"Error: {e}"
        else:
            tc_output = f"Unknown tool: {tc_name}"

        tc_results.append(format_openai_tool_response(tc_output, tc_id))
    return tc_results
```

## Language Models

Available LM implementations:

```python
from wbal import GPT5Large, GPT5MiniTester

# Full GPT-5 with reasoning
lm = GPT5Large()

# Mini version for testing (cheaper, faster)
lm = GPT5MiniTester()
```

Both use OpenAI's Responses API format.

## Environment Best Practices

1. **`env` string** - Describe what tools are available
2. **`task` string** - The goal for the agent
3. **`observe()`** - Return current state (called by agent in perceive)

```python
class StoryEnvironment(Environment):
    file_paths: list[str] = Field(default_factory=list)
    env: str = "You have access to story files."

    @weave.op()
    def observe(self) -> str:
        files_list = "\n".join(f"- {fp}" for fp in self.file_paths)
        return f"Available files:\n{files_list}"

    @weaveTool
    def ReadFile(self, file_path: str) -> str:
        '''Read a file. Only allowed files can be read.'''
        if file_path not in self.file_paths:
            raise ValueError(f"File not allowed: {file_path}")
        return open(file_path).read()
```

## Agent Best Practices

1. **Memory** - Store state in agent fields (e.g., `memory: list[str]`)
2. **Messages** - Maintain conversation history in `messages: list[dict]`
3. **Exit tool** - Always provide a way for the agent to stop gracefully

```python
class MyAgent(Agent):
    lm: LM = Field(default_factory=GPT5MiniTester)
    memory: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    _exit: bool = False

    @weaveTool
    def AddToMemory(self, content: str) -> str:
        '''Store information in memory.'''
        self.memory.append(content)
        return f"Added: {content}"

    @weaveTool
    def exit(self, exit_message: str) -> str:
        '''Exit with final result.'''
        self._exit = True
        return exit_message

    @property
    def stopCondition(self) -> bool:
        return self._exit
```

## Full Reference Implementation

See `examples/story_summarizer.py` for a complete working example with:

- `StoryEnvironment` with `ReadFile` tool
- `StoryAgent` with `AddToMemory` and `exit` tools
- Full perceive-invoke-do implementation
- Weave tracing throughout

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
wbal/
├── wbal/
│   ├── __init__.py      # Exports
│   ├── object.py        # WBALObject base class
│   ├── agent.py         # Agent base class
│   ├── environment.py   # Environment base class
│   ├── lm.py            # Language model implementations
│   └── helper.py        # Tool decorators, schema extraction, formatters
├── examples/
│   └── story_summarizer.py  # Reference implementation
└── tests/
    ├── test_agent.py
    ├── test_environment.py
    └── test_story_summarizer.py
```
