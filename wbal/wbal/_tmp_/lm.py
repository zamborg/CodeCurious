"""
LM - Language Model interface.

A simple interface for calling a specific language model.
"""

from typing import Any

import litellm

from wbal.object import WBALObject

class LM(WBALObject):
    pass


class OpenAIResponsesLM(LM):
    """
    Interface for calling a specific language model.

    Example:
        lm = OpenAIResponsesLM(model="o3-mini")
        response = lm.invoke(messages=[{"role": "user", "content": "Hello!"}])
    """

    model: str
    """The model identifier (e.g., 'gpt-5', 'o3-mini')"""

    temperature: float = 1.0
    """Sampling temperature"""

    max_tokens: int = 4096
    """Maximum tokens in response"""

    def observe(self) -> str:
        return f"LM(model={self.model}, temperature={self.temperature})"

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Call the language model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (OpenAI format)
            system: Optional system prompt

        Returns:
            The assistant's response message dict
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "include": ["reasoning.encrypted_content"],
        }

        if tools:
            kwargs["tools"] = tools

        response = litellm.responses(**kwargs)

        return response.model_dump()
