import pytest

from wbal.lm import GPT5Large, LM


class DummyLM(LM):
    def observe(self) -> str:
        return "dummy"


def test_lm_invoke_not_implemented():
    dummy = DummyLM()
    with pytest.raises(NotImplementedError):
        dummy.invoke()


def test_gpt5large_invoke_without_tools(monkeypatch):
    recorded_calls: list[dict] = []

    class DummyResponses:
        def create(self, **kwargs):
            recorded_calls.append(kwargs)
            return {"ok": True}

    class DummyClient:
        def __init__(self):
            self.responses = DummyResponses()

    lm = GPT5Large()
    lm.client = DummyClient()

    messages = [{"role": "user", "content": "hello"}]
    result = lm.invoke(messages=messages)

    assert result == {"ok": True}
    assert recorded_calls == [
        {
            "model": "gpt-5",
            "input": messages,
            "temperature": 1.0,
            "include": ["reasoning.encrypted_content"],
        }
    ]


def test_gpt5large_invoke_with_tools_and_mcp(monkeypatch):
    recorded_calls: list[dict] = []

    class DummyResponses:
        def create(self, **kwargs):
            recorded_calls.append(kwargs)
            return {"ok": True}

    class DummyClient:
        def __init__(self):
            self.responses = DummyResponses()

    lm = GPT5Large()
    lm.client = DummyClient()

    tools = [{"name": "tool1"}]
    mcp_servers = [{"name": "mcp1"}]

    result = lm.invoke(messages=[], tools=tools, mcp_servers=mcp_servers)

    assert result == {"ok": True}
    assert recorded_calls[0]["tools"] == [{"name": "tool1"}, {"name": "mcp1"}]
    # Ensure we extended the original list (current implementation mutates).
    assert tools == [{"name": "tool1"}, {"name": "mcp1"}]

