"""HttpChatClient against a throwaway server on localhost (no external network)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vlm_floorplan.evaluation import client as client_module
from vlm_floorplan.evaluation.client import ApiError, HttpChatClient

OK_BODY = {
    "choices": [{"message": {"content": "hi", "reasoning_content": "hm"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
}


DROP_BEFORE_REPLY = -1
"""Close the socket without answering, as a crashed or restarted server does."""
DROP_MID_BODY = -2
"""Promise a longer body than is sent, then close: the reply is cut off mid-read."""


class Handler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    seen: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        Handler.seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length)),
            }
        )
        status = Handler.statuses.pop(0) if Handler.statuses else 200
        if status == DROP_BEFORE_REPLY:
            self.close_connection = True
            return
        if status == DROP_MID_BODY:
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"choices": ')
            self.close_connection = True
            return
        payload = json.dumps(OK_BODY if status == 200 else {"error": "x"}).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def base_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(client_module, "BACKOFF_SECONDS", 0.0)
    Handler.statuses, Handler.seen = [], []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_posts_to_chat_completions_with_bearer_key_from_env(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    reply = HttpChatClient(base_url, api_key_env="MY_KEY").complete({"model": "m"})
    assert (reply.content, reply.reasoning) == ("hi", "hm")
    assert Handler.seen[0]["path"] == "/v1/chat/completions"
    assert Handler.seen[0]["auth"] == "Bearer secret"


def test_transient_server_errors_are_retried(base_url: str) -> None:
    Handler.statuses = [503, 429]
    assert HttpChatClient(base_url).complete({}).content == "hi"
    assert len(Handler.seen) == 3


@pytest.mark.parametrize("drop", [DROP_BEFORE_REPLY, DROP_MID_BODY])
def test_dropped_connections_are_retried(base_url: str, drop: int) -> None:
    Handler.statuses = [drop]
    assert HttpChatClient(base_url).complete({}).content == "hi"
    assert len(Handler.seen) == 2


def test_dropped_connections_that_persist_become_api_error(base_url: str) -> None:
    Handler.statuses = [DROP_BEFORE_REPLY] * client_module.MAX_ATTEMPTS
    with pytest.raises(ApiError, match="failed after retries"):
        HttpChatClient(base_url).complete({})


def test_client_errors_are_not_retried(base_url: str) -> None:
    Handler.statuses = [400]
    with pytest.raises(ApiError, match="HTTP 400"):
        HttpChatClient(base_url).complete({})
    assert len(Handler.seen) == 1


def test_missing_key_variable_fails_loudly(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(ApiError, match="MY_KEY"):
        HttpChatClient(base_url, api_key_env="MY_KEY").complete({})
