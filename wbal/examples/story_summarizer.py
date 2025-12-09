"""
Story Summarizer Example

Demonstrates:
1. An agent with memory (AddToMemory tool)
2. An environment with file access (ReadFile tool)
3. The perceive-invoke-do loop with actual LLM calls
"""

from typing import Any

from pydantic import Field

import json

from wbal import Agent, Environment, weaveTool, GPT5MiniTester, LM

import weave

from wbal.helper import format_openai_tool_response
weave.init('zubin-dump')

## ZUBIN YOU SHOULD FACTOR THIS INTO HELPERS

def handle_oai_tcs(self, tc_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Handle OpenAI tool calls. self is injected by the agent
    """
    # self should be injected by the agent
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

# -----------------------------------------------------------------------------
# Story Environment
# -----------------------------------------------------------------------------


class StoryEnvironment(Environment):
    """
    Environment that provides access to story files.

    The agent can only read files from the allowed file_paths list.
    """

    file_paths: list[str] = Field(default_factory=list)
    """List of allowed file paths the agent can read."""

    env: str = "You have access to story files. Use ReadFile to read them one at a time."

    @weave.op()
    def observe(self) -> str:
        """Return the list of available files."""
        if not self.file_paths:
            return "No files available."
        files_list = "\n".join(f"- {fp}" for fp in self.file_paths)
        return f"Available files:\n{files_list}"

    @weaveTool
    def ReadFile(self, file_path: str) -> str:
        """
        Read the contents of a file.

        Args:
            file_path: Path to the file to read

        Returns:
            The file contents as a string

        Raises:
            ValueError: If file_path is not in the allowed list
            FileNotFoundError: If the file does not exist
        """
        if file_path not in self.file_paths:
            raise ValueError(
                f"File '{file_path}' is not in the allowed file list. "
                f"Allowed files: {self.file_paths}"
            )

        with open(file_path, "r") as f:
            return f.read()


# -----------------------------------------------------------------------------
# Story Summarizer Agent
# -----------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a story summarizer agent.

Your job is to read a story and produce a comprehensive summary. However, you must:
1. Read the story ONE FILE/CHUNK at a time using the ReadFile tool
2. After reading each chunk, use AddToMemory to store key information, events, characters, and plot points
3. Only after reading ALL files should you produce your final summary

Your memory is your scratchpad - use it to track:
- Main characters and their traits
- Key plot points and events
- Important themes or motifs
- The narrative arc

Be thorough but concise in your memory notes. When you're done reading all files,
synthesize your memory into a coherent summary."""


class StoryAgent(Agent):
    """
    Agent that summarizes stories by reading files and maintaining memory.
    """

    lm: LM = Field(default_factory=GPT5MiniTester)
    """The language model to use."""

    system_prompt: str = SYSTEM_PROMPT
    """System prompt for the agent."""

    memory: list[str] = Field(default_factory=list)
    """Agent's memory storage."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Conversation history."""

    _last_response: dict[str, Any] | None = None
    """Last LLM response for use in do()."""

    _exit: bool = False

    @weaveTool
    def AddToMemory(self, content: str) -> str:
        """
        Add a piece of information to memory.

        Use this to store key information from each file chunk you read.

        Args:
            content: The information to remember

        Returns:
            Confirmation message
        """
        self.memory.append(content)
        return f"Added to memory: {content}"

    @property
    def stopCondition(self) -> bool:
        """Stop when max steps reached or LLM produces no tool calls."""
        return  self._exit

    @weave.op()
    def perceive(self) -> None:
        """Set up initial messages on first step."""
        if self._step_count == 0:
            # First message: system prompt
            self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })

            # Second message: environment observation (file list)
            self.messages.append({
                "role": "user",
                "content": f"Task: {self.env.task}\n\n{self.env.observe()}"
            })
            
            self.messages.append({
                "role": "assistant",
                "content": f"Memory Block:\n\n```\n{'\n'.join(self.memory)}\n```"
            })

        else:
            self.messages[2]["content"] = f"Memory Block:\n\n```\n{'\n'.join(self.memory)}\n```" # set our memory!

    @weave.op()
    def invoke(self) -> None:
        """Call the LLM with current messages and tools."""
        tools = self._tool_definitions if self._tool_definitions else None

        self._last_response = self.lm.invoke(
            messages=self.messages,
            tools=tools,
        )
        # extend last response:
        self.messages.extend(self._last_response.output) # just append it!

    @weave.op()
    def do(self) -> None:
        """Execute any tool calls from the LLM response."""
        if self._last_response is None:
            return

        output = self._last_response.output

        # NOTE: @zamborg @gtarpenning @AGENT -- this should probably be factored into a function in @helper.py in the OPENAI block because it is openai specific but an executor of the tool call
        tcs = [item for item in output if item.type == "function_call"]
        tc_results = handle_oai_tcs(self, tcs)
        self.messages.extend(tc_results)

    @weaveTool # ref implementation
    def exit(self, exit_message: str) -> str:
        """
        Exit your run loop.
            - please provide `exit_message` as a message to the user or developer. This can be your terminal result or your final summary or any content you'd like to leave your controller with after your run loop.
        """
        self._exit = True
        return exit_message



# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------


def run_story_summarizer(file_paths: list[str], task: str = "Summarize this story.") -> dict:
    """
    Run the story summarizer on a list of files.

    Args:
        file_paths: List of file paths containing story chunks
        task: The task description

    Returns:
        Dict with summary, memory, and step count
    """
    env = StoryEnvironment(
        file_paths=file_paths,
        task=task,
    )

    agent = StoryAgent(
        env=env,
        maxSteps=20,  # Allow enough steps for reading + summarizing
    )

    result = agent.run()

    return {
        "steps": result["steps"],
        "messages": agent.messages,
        'agent': agent,
    }


if __name__ == "__main__":
    # Example usage - fill in your file paths here
    FILE_PATHS = [
        '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/and_1.txt', 
        '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/as_3.txt', 
        '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/Blessed_13.txt', 
        '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/face_8.txt', 
        '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/flight_9.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/has_6.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/He_2.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/immediately_0.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/inexpressible_10.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/much_4.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/on_14.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/the_7.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/trust_12.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/were_11.txt', 
        # '/Users/zaysola/Documents/wandb/CodeCurious/wbal/examples/story_summarizer_files/Yet_5.txt'
    ]

    if not FILE_PATHS:
        print("Please add file paths to FILE_PATHS list")
    else:
        result = run_story_summarizer(FILE_PATHS)

        print("=== Memory ===")
        for i, mem in enumerate(result["agent"].memory):
            print(f"{i}. {mem}")

        print(f"\n=== Completed in {result['steps']} steps === using agent: {len(result['messages'])} messages")
