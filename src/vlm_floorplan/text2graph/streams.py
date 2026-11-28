"""Independent random streams for the held-out set build.

One shared `random.Random` makes every draw depend on all the draws before it, so a change to
one source's candidate pool re-rolls every later source and set. A stream is derived from the
base seed and a path of string keys instead (set, source, purpose, plan), so a draw depends on
those keys alone. The derivation is a SHA-256 of the keys, never Python's salted `hash()`, so
the same seed gives the same streams in every process.
"""

from __future__ import annotations

import hashlib
import json
import random


class Streams:
    """Derives a `random.Random` per key path from one base seed."""

    def __init__(self, seed: int):
        self.seed = seed

    def sort_key(self, *keys: str) -> int:
        """A number for `keys`; sorting by it orders items without depending on the others."""
        digest = hashlib.sha256(json.dumps([self.seed, *keys]).encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def rng(self, *keys: str) -> random.Random:
        """The stream for `keys`; the same keys always give the same sequence."""
        return random.Random(self.sort_key(*keys))
