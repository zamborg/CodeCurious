import os
from typing import Any

import weave
from openai import OpenAI
from openai.types.responses import Response
from wbal.object import WBALObject


class LM(WBALObject):
    """
    Base class for language models.

    Subclasses must implement the invoke() method to call the underlying LLM API.
    """

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Invoke the language model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions in provider format.
            mcp_servers: Optional list of MCP server tool definitions.

        Returns:
            The LLM response object (format depends on provider).

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement invoke method")

class GPT5Large(LM):
    model: str = "gpt-5"
    include: list[str] = ["reasoning.encrypted_content"]
    temperature: float = 1.0
    client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def observe(self) -> str:
        """
        Return a concise description of the model configuration.
        """
        return f"GPT5Large(model={self.model}, temperature={self.temperature})"

    @weave.op()
    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> Response:
        """
        Invoke the language model.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "include": self.include,
        }
        # Combine tools and mcp_servers without mutating input lists
        if tools or mcp_servers:
            combined_tools = list(tools) if tools else []
            if mcp_servers:
                combined_tools.extend(mcp_servers)
            kwargs["tools"] = combined_tools
        response: Response = self.client.responses.create(**kwargs)
        return response


class GPT5MiniTester(LM):
    model: str = "gpt-5-mini"
    client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    reasoning: dict[str, Any] = {'effort': 'minimal'}
    temperature: float = 1.0

    def observe(self) -> str:
        """
        Return a concise description of the model configuration.
        """
        return f"GPT5MiniTester(model={self.model}, temperature={self.temperature})"

    @weave.op()
    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> Response:
        """
        Invoke the language model.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "reasoning": self.reasoning,
        }
        # Combine tools and mcp_servers without mutating input lists
        if tools or mcp_servers:
            combined_tools = list(tools) if tools else []
            if mcp_servers:
                combined_tools.extend(mcp_servers)
            kwargs["tools"] = combined_tools
        return self.client.responses.create(**kwargs)