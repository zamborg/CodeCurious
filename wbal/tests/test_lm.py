import pytest

from wbal.lm import GPT5Large, GPT5MiniTester, LM


class DummyLM(LM):
    def observe(self) -> str:
        return "dummy"


class TestLMBase:
    """Tests for LM base class."""

    def test_lm_invoke_not_implemented(self):
        """Base LM.invoke() raises NotImplementedError."""
        dummy = DummyLM()
        with pytest.raises(NotImplementedError):
            dummy.invoke(messages=[])

    def test_lm_invoke_signature(self):
        """Base LM.invoke() has correct signature."""
        dummy = DummyLM()
        # Should accept these parameters without error (even though it raises)
        with pytest.raises(NotImplementedError):
            dummy.invoke(messages=[], tools=None, mcp_servers=None)


class TestGPT5Large:
    """Tests for GPT5Large."""

    def test_invoke_without_tools(self):
        """invoke() works without tools."""
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

    def test_invoke_with_tools_and_mcp(self):
        """invoke() combines tools and mcp_servers."""
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

    def test_invoke_does_not_mutate_tools_list(self):
        """invoke() should NOT mutate the input tools list."""
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

        lm.invoke(messages=[], tools=tools, mcp_servers=mcp_servers)

        # Original tools list should NOT be mutated
        assert tools == [{"name": "tool1"}], "tools list was mutated!"

    def test_invoke_with_only_mcp_servers(self):
        """invoke() works with only mcp_servers (no tools)."""
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

        mcp_servers = [{"name": "mcp1"}]
        lm.invoke(messages=[], tools=None, mcp_servers=mcp_servers)

        assert recorded_calls[0]["tools"] == [{"name": "mcp1"}]


class TestGPT5MiniTester:
    """Tests for GPT5MiniTester."""

    def test_invoke_basic(self):
        """invoke() works with basic parameters."""
        recorded_calls: list[dict] = []

        class DummyResponses:
            def create(self, **kwargs):
                recorded_calls.append(kwargs)
                return {"ok": True}

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()

        lm = GPT5MiniTester()
        lm.client = DummyClient()

        messages = [{"role": "user", "content": "test"}]
        result = lm.invoke(messages=messages)

        assert result == {"ok": True}
        assert recorded_calls[0]["model"] == "gpt-5-mini"
        assert recorded_calls[0]["reasoning"] == {"effort": "minimal"}

    def test_invoke_does_not_mutate_tools_list(self):
        """GPT5MiniTester.invoke() should NOT mutate the input tools list."""
        recorded_calls: list[dict] = []

        class DummyResponses:
            def create(self, **kwargs):
                recorded_calls.append(kwargs)
                return {"ok": True}

        class DummyClient:
            def __init__(self):
                self.responses = DummyResponses()

        lm = GPT5MiniTester()
        lm.client = DummyClient()

        tools = [{"name": "tool1"}]
        mcp_servers = [{"name": "mcp1"}]

        lm.invoke(messages=[], tools=tools, mcp_servers=mcp_servers)

        # Original tools list should NOT be mutated
        assert tools == [{"name": "tool1"}], "tools list was mutated!"

    def test_observe(self):
        """observe() returns model description."""
        lm = GPT5MiniTester()
        obs = lm.observe()
        assert "GPT5MiniTester" in obs
        assert "gpt-5-mini" in obs

