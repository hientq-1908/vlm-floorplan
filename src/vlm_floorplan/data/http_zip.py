"""Open a remote zip file over HTTP range requests, without downloading all of it.

CubiCasa5K is one 5.5 GB zip on Zenodo, but one plan is only three small files.
A zip stores its table of contents at the end of the file, and every member
at a known byte offset, so a client can read just those parts:

1. ask for the last bytes of the file (the table of contents),
2. ask for the bytes of each member it wants.

Python's `zipfile` already knows how to do this on any seekable file.
`HttpRangeFile` is a seekable file whose bytes come from `Range:` requests,
so `zipfile.ZipFile(HttpRangeFile(url))` just works.
"""

from __future__ import annotations

import http.client
import io
import re
import time
import urllib.error
import urllib.request
from typing import Final

READ_BLOCK_BYTES: Final = 256 * 1024
"""Every network read fetches at least this much; zipfile makes many tiny reads."""

RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
"""Zenodo intermittently answers 500 and 504; a retry succeeds."""

MAX_ATTEMPTS: Final = 5
REQUEST_TIMEOUT_SECONDS: Final = 60
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class RangeNotSupportedError(OSError):
    """The server ignored the Range header (it answered 200 instead of 206)."""


def _request_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    """Fetch bytes [start, end] (inclusive) and return them with the total file size."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                if response.status != 206:
                    raise RangeNotSupportedError(f"{url}: expected 206, got {response.status}")
                match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise RangeNotSupportedError(f"{url}: missing Content-Range header")
                return response.read(), int(match.group(3))
        except RangeNotSupportedError:
            raise  # retrying cannot help
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                raise
        except (OSError, http.client.HTTPException):
            # Dropped connections, timeouts and truncated bodies (IncompleteRead) are
            # all transient on Zenodo.
            if attempt == MAX_ATTEMPTS:
                raise
        time.sleep(2**attempt)  # 2, 4, 8, 16 seconds
    raise AssertionError("unreachable")  # pragma: no cover


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._position = 0
        self._block_start = 0
        self._block = b""
        # A one-byte range request is the cheapest way to learn the file size.
        _, self._size = _request_range(url, 0, 0)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        elif whence == io.SEEK_END:
            self._position = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._position = max(0, self._position)
        return self._position

    def _cached(self, start: int, end: int) -> bool:
        return self._block_start <= start and end <= self._block_start + len(self._block)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._position
        start = self._position
        end = min(start + size, self._size)  # exclusive
        if start >= end:
            return b""
        if not self._cached(start, end):
            fetch_end = min(max(end, start + READ_BLOCK_BYTES), self._size)
            self._block, _ = _request_range(self.url, start, fetch_end - 1)
            self._block_start = start
        offset = start - self._block_start
        self._position = end
        return self._block[offset : offset + (end - start)]

    def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
