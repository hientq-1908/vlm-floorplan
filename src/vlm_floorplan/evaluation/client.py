"""A minimal OpenAI-compatible chat client (standard library only).

It speaks `POST {base_url}/chat/completions`, which vLLM, llama.cpp, and the hosted
APIs all accept, so the same harness runs the base model, our LoRA, or a frontier
model. The API key, if any, is read from an environment variable and never written to
a run directory.

`ChatClient` is a protocol so tests can pass a fake instead of opening sockets.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


class ApiError(RuntimeError):
    """The endpoint failed or returned something that is not a chat completion."""


@dataclass(frozen=True)
class ChatResponse:
    content: str
    reasoning: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    extra: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Structured tool calls as {"id": ..., "name": ..., "arguments": <JSON text>}, in order."""


class ChatClient(Protocol):
    def complete(self, request: dict[str, Any]) -> ChatResponse: ...


def parse_chat_response(body: dict[str, Any], latency_s: float) -> ChatResponse:
    """Read one completion. vLLM names the reasoning field `reasoning_content` or `reasoning`."""
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as err:
        raise ApiError(f"response has no choices[0].message: {str(body)[:200]}") from err
    usage = body.get("usage") or {}
    tool_calls = [
        {
            "id": call.get("id"),
            "name": (call.get("function") or {}).get("name"),
            "arguments": (call.get("function") or {}).get("arguments"),
        }
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    return ChatResponse(
        content=message.get("content") or "",
        reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_s=latency_s,
        tool_calls=tool_calls,
    )


def _error_body(err: urllib.error.HTTPError) -> bytes:
    """The start of an error reply's body, or nothing if the connection drops while reading it."""
    try:
        return err.read()[:300]
    except (OSError, http.client.HTTPException):
        return b""


class HttpChatClient:
    def __init__(self, base_url: str, api_key_env: str | None = None, timeout_s: float = 600.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ApiError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(self, request: dict[str, Any]) -> ChatResponse:
        data = json.dumps(request).encode()
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            started = time.monotonic()
            http_request = urllib.request.Request(
                self.url, data=data, headers=self._headers(), method="POST"
            )
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout_s) as reply:
                    body = json.loads(reply.read())
                return parse_chat_response(body, time.monotonic() - started)
            except urllib.error.HTTPError as err:
                last_error = ApiError(f"HTTP {err.code}: {_error_body(err)!r}")
                if err.code not in RETRY_STATUS:
                    break
            # urllib wraps only errors raised while sending; a dropped connection while
            # waiting for or reading the reply arrives as a raw OSError or HTTPException.
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as err:
                last_error = ApiError(f"{type(err).__name__}: {err}")
            time.sleep(BACKOFF_SECONDS * 2**attempt)
        raise ApiError(f"{self.url} failed after retries: {last_error}")
