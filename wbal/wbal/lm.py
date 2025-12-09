import os
from typing import Any

from openai import OpenAI
from openai.types.responses import Response
from wbal.object import WBALObject

# BASECLASS
class LM(WBALObject):
    def invoke(self) -> dict[str, Any]:
        """
        Invoke the language model.
        """
        raise NotImplementedError("Subclasses must implement invoke method")

class GPT5Large(WBALObject):
    model: str = "gpt-5"
    include: list[str] = ["reasoning.encrypted_content"]
    temperature: float = 1.0
    client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def observe(self) -> str:
        """
        Return a concise description of the model configuration.
        """
        return f"GPT5Large(model={self.model}, temperature={self.temperature})"

    def invoke(
        self, 
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke the language model.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "include": self.include,
        }
        if tools:
            kwargs["tools"] = tools
        if mcp_servers:
            kwargs["tools"].extend(mcp_servers)
        response: Response = self.client.responses.create(**kwargs)
        return response # return raw response
