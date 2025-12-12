"""Tests for the story summarizer example."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add examples to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from story_summarizer import StoryEnvironment, StoryAgent


class TestStoryEnvironment:
    """Unit tests for StoryEnvironment."""

    def test_observe_with_files(self):
        """observe() returns formatted file list."""
        env = StoryEnvironment(file_paths=["/a.txt", "/b.txt"])
        obs = env.observe()

        assert "Available files:" in obs
        assert "/a.txt" in obs
        assert "/b.txt" in obs

    def test_observe_empty(self):
        """observe() handles empty file list."""
        env = StoryEnvironment()
        assert env.observe() == "No files available."

    def test_read_file_allowed(self):
        """ReadFile works for allowed files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            path = f.name

        env = StoryEnvironment(file_paths=[path])
        result = env.ReadFile(path)

        assert result == "Test content"

    def test_read_file_not_allowed(self):
        """ReadFile raises ValueError for files not in list."""
        env = StoryEnvironment(file_paths=["/allowed.txt"])

        with pytest.raises(ValueError, match="not in the allowed file list"):
            env.ReadFile("/not_allowed.txt")

    def test_read_file_not_exists(self):
        """ReadFile raises FileNotFoundError for missing files."""
        env = StoryEnvironment(file_paths=["/nonexistent.txt"])

        with pytest.raises(FileNotFoundError):
            env.ReadFile("/nonexistent.txt")


class TestStoryAgent:
    """Unit tests for StoryAgent."""

    def test_add_to_memory(self):
        """AddToMemory appends to memory list."""
        agent = StoryAgent(env=StoryEnvironment())

        result = agent.AddToMemory("First item")
        assert "Added to memory" in result
        assert agent.memory == ["First item"]

        agent.AddToMemory("Second item")
        assert agent.memory == ["First item", "Second item"]

    def test_get_memory(self):
        """get_memory() returns a copy of memory."""
        agent = StoryAgent(env=StoryEnvironment())
        agent.AddToMemory("Test")

        mem = agent.get_memory()
        assert mem == ["Test"]

        # Verify it's a copy
        mem.append("Modified")
        assert agent.memory == ["Test"]

    def test_combined_tools(self):
        """Agent has both AddToMemory and ReadFile tools."""
        env = StoryEnvironment(file_paths=["/test.txt"])
        agent = StoryAgent(env=env)

        tools = agent._tool_callables
        assert "AddToMemory" in tools
        assert "ReadFile" in tools

    def test_perceive_sets_initial_messages(self):
        """perceive() sets up system prompt and env observation."""
        env = StoryEnvironment(
            file_paths=["/story.txt"],
            task="Summarize the story"
        )
        agent = StoryAgent(env=env)

        assert len(agent.messages) == 0

        agent.perceive()

        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "system"
        assert "story summarizer" in agent.messages[0]["content"].lower()
        assert agent.messages[1]["role"] == "user"
        assert "Summarize the story" in agent.messages[1]["content"]
        assert "/story.txt" in agent.messages[1]["content"]

    def test_perceive_only_runs_once(self):
        """perceive() only adds messages on first step."""
        agent = StoryAgent(env=StoryEnvironment())

        agent.perceive()
        assert len(agent.messages) == 2

        # Simulate moving to next step
        agent._step_count = 1
        agent.perceive()
        assert len(agent.messages) == 2  # No new messages added

    def test_stop_condition_max_steps(self):
        """stopCondition returns True at maxSteps."""
        agent = StoryAgent(env=StoryEnvironment(), maxSteps=5)

        agent._step_count = 4
        assert agent.stopCondition is False

        agent._step_count = 5
        assert agent.stopCondition is True

    def test_system_prompt_content(self):
        """System prompt contains key instructions."""
        agent = StoryAgent(env=StoryEnvironment())

        prompt = agent.system_prompt.lower()
        assert "memory" in prompt
        assert "one file" in prompt or "chunk" in prompt
        assert "summary" in prompt
