import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import weave
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_proxy")

# Track requests with weave; this gives visibility without calling upstream.
weave.init("openrouter-proxy-mock")

app = FastAPI(
    title="OpenAI/OpenRouter-compatible mock",
    version="0.2.0",
    description="Mock backend that returns OpenAI/OpenRouter-shaped responses, instrumented with weave.",
)


def _build_choice(content: str, finish_reason: Literal["stop"] = "stop") -> Dict[str, Any]:
    return {
        "index": 0,
        "finish_reason": finish_reason,
        "message": {
            "role": "assistant",
            "content": content,
        },
    }


def _build_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _mock_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize a minimal OpenAI/OpenRouter-compatible completion response."""
    content = payload.get("messages", [{}])[-1].get("content", "Hello from the mock backend.")
    now = int(time.time())
    model = payload.get("model", "mock-model")

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [_build_choice(f"(mock) echo: {content}")],
        "usage": _build_usage(prompt_tokens=10, completion_tokens=10),
    }


@weave.op()
def record_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record the inbound request and return a mock response."""
    return _mock_response(payload)


@weave.op()
def dummy_messages(
    model: str,
    messages: List[Dict[str, Any]],
    system: Optional[Any] = None,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Anthropic-style /v1/messages mock; signature mirrors the request body so weave can render inputs.
    """
    last = messages[-1] if messages else {}
    content = last.get("content", "")
    if isinstance(content, list) and content:
        # Try to pull text out of the first text block.
        first_block = content[0]
        if isinstance(first_block, dict):
            content = first_block.get("text", "")
    elif not isinstance(content, str):
        content = str(content)

    raw_payload = {
        "model": model,
        "messages": messages,
        "system": system,
        "max_tokens": max_tokens,
        "stop_sequences": stop_sequences,
        "metadata": metadata,
        **kwargs,
    }

    return {
        "id": f"msg-{uuid.uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": f"(mock anthropic) echo: {content}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _build_usage(prompt_tokens=10, completion_tokens=max_tokens or 10),
        "debug_payload": raw_payload,
    }


@weave.op()
def record_anthropic_count_tokens(payload: Dict[str, Any]) -> Dict[str, int]:
    """Record a count-tokens request and return a mock token count."""
    text_parts = []
    for msg in payload.get("messages", []):
        if isinstance(msg, dict):
            val = msg.get("content")
            if isinstance(val, str):
                text_parts.append(val)
            elif isinstance(val, list):
                text_parts.extend(str(x) for x in val)
    token_estimate = sum(len(part.split()) for part in text_parts)

    return {
        "input_tokens": token_estimate,
        "output_tokens": 0,
        "total_tokens": token_estimate,
        # Echo back payload for debugging/visibility.
        "debug_payload": payload,
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload: Dict[str, Any] = await request.json()
    logger.info("Received /v1/chat/completions payload: %s", json.dumps(payload))

    if payload.get("stream"):
        # This mock does not implement SSE streaming.
        raise HTTPException(status_code=400, detail="stream=True is not supported by this mock proxy.")

    try:
        result = record_request(payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mock handling failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content=result)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic/Claude-style messages endpoint used by Claude Code."""
    payload: Dict[str, Any] = await request.json()
    logger.info("Received /v1/messages payload: %s", json.dumps(payload))

    try:
        resp = dummy_messages(**payload)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc
    return JSONResponse(content=resp)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Return a mock token count for Claude Code tooling."""
    payload: Dict[str, Any] = await request.json()
    logger.info("Received /v1/messages/count_tokens payload: %s", json.dumps(payload))

    resp = record_anthropic_count_tokens(payload)
    return JSONResponse(content=resp)


def main():
    import uvicorn

    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting mock OpenAI/OpenRouter server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

