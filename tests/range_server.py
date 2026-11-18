"""A tiny local HTTP server that supports `Range:` requests, for offline fetch tests."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _handler(
    payload: bytes, state: dict[str, int], fail_first: int
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            state["requests"] += 1
            if state["requests"] <= fail_first:
                self.send_error(503)
                return
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            if match is None:
                self.send_response(200)  # a server that ignores Range
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            start = int(match.group(1))
            end = min(int(match.group(2) or len(payload) - 1), len(payload) - 1)
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            state["bytes_sent"] += len(body)

        def log_message(self, *args: object) -> None:
            pass

    return Handler


@contextmanager
def serve(payload: bytes, fail_first: int = 0) -> Iterator[tuple[str, dict[str, int]]]:
    """Serve `payload` at the returned URL; `state` counts requests and bytes sent."""
    state = {"requests": 0, "bytes_sent": 0}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(payload, state, fail_first))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/file.zip", state
    finally:
        server.shutdown()
        server.server_close()
