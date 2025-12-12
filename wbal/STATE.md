# WBAL - Weights & Biases Agent Launch

## What This Is

An easy-to-use framework for building and deploying agents with sandboxed environments.

## Core Architecture

```
Agent (generic executor)          Env (control plane)
├── LM(s) for invocation          ├── task, env_str
├── @tool methods (memory, etc)   ├── @tool methods (service APIs)
├── perceive/invoke/do loop       ├── sandbox access
└── message history               └── region-locked data
        │                                 │
        └────────── Sandbox ──────────────┘
                    (exec, read_file, write_file)
```

**Key separation**: Agent runs in different context than Env. Agent calls into Env tools which execute in the sandbox. This enables region locking and service access control.

## Files Created

```
wbal/
├── __init__.py      # Exports: WBALObject, LM, Agent, Env, tool, helpers...
├── object.py        # WBALObject base class (observe + setup)
├── lm.py            # LM - litellm wrapper for model calls
├── agent.py         # Agent - async perceive/invoke/do loop
├── env.py           # Env - control plane with task, env_str, tools
├── tool.py          # Unified @tool decorator
├── sandbox.py       # SandboxProtocol type reference (no impl - uses coworker's)
├── helpers.py       # Tool definition marshalling (OpenAI/Anthropic formats)
└── examples/
    └── simple_example.py
```

## Design Decisions Made

1. **Unified `@tool` decorator** - Both Agent and Env use same decorator
2. **Async-native** - Agent.run(), step(), do() and Env.execute() are async to match sandbox interface
3. **WBALObject base class** - All components have `observe() -> str` and `setup(sandbox)`
4. **helpers.py for marshalling** - `extract_tool_schema()` → `to_openai_tool()` / `to_anthropic_tool()`
5. **SandboxProtocol** - Just a Protocol type; actual impl comes from coworker's package with `exec`, `read_file`, `write_file` (all async, bytes-based)

## Sandbox Interface (from coworker)

```python
async def exec(command: list[str]) -> ExecResult  # stdout/stderr: bytes, returncode: int
async def read_file(filepath: str) -> bytes
async def write_file(filepath: str, contents: bytes) -> bool
```

Sandbox is async context manager with start/stop lifecycle.

## Open Questions / TODO

1. **setup() interface** - User said they'd define this more precisely
2. **MCP servers** - Mentioned for both agent and env, not yet implemented
3. **Multiple LMs** - User mentioned agents might have many LMs, current impl has single `lm: LM`
4. **Deploy story** - No `wbal.deploy()` or `wbal.launch()` yet
5. **Tracing/observability** - Original zen had Trace/TraceEvent, not ported yet

## Usage Pattern

```python
from wbal import LM, Agent, Env, tool

class MyEnv(Env):
    task = "Do the thing"
    env_str = "You have access to X"

    @tool
    async def get_data(self, key: str) -> str:
        result = await self.sandbox.exec(["fetch", key])
        return result.stdout.decode()

class MyAgent(Agent):
    system_prompt = "You are helpful"

    @tool
    def remember(self, note: str) -> str:
        return f"Noted: {note}"

async with SomeSandbox() as sandbox:
    agent = MyAgent(lm=LM(model="gpt-4o"), env=MyEnv())
    agent.setup(sandbox)
    result = await agent.run()
```
