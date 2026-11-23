"""The leak check: nothing a model trains on may appear in a held-out set.

A held-out plan leaks when any of these hold against the train plans (and train prompts, if
given):

- its record id is a train record id,
- (Swiss Dwellings) its site holds a train apartment,
- its structure, a Weisfeiler-Lehman hash over typed rooms, typed links and entrance flags,
  equals a train structure, so the set measures generalization and not recall,
- its prompt text equals a train prompt.

The split definitions already keep ids and sites apart; the check proves it on the actual files
and adds the structure and text checks.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from vlm_floorplan.schema import FloorplanGraph
from vlm_floorplan.text2graph.splits import swiss_site

WL_ROUNDS = 3


def structure_hash(graph: FloorplanGraph) -> str:
    """Weisfeiler-Lehman hash of the typed graph; positions and sizes do not matter."""
    labels = {room.id: f"{room.type.value}|{int(room.entrance)}" for room in graph.rooms()}
    edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for floor in graph.floors:
        for link in floor.links:
            edges[link.a].append((link.b, link.kind.value))
            edges[link.b].append((link.a, link.kind.value))
    for _round in range(WL_ROUNDS):
        labels = {
            room_id: hashlib.sha1(
                (
                    label
                    + "#"
                    + ",".join(sorted(f"{kind}:{labels[o]}" for o, kind in edges[room_id]))
                ).encode()
            ).hexdigest()[:16]
            for room_id, label in labels.items()
        }
    return hashlib.sha1(",".join(sorted(labels.values())).encode()).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


class LeakError(RuntimeError):
    """A held-out prompt overlaps the train data."""


@dataclass
class LeakIndex:
    """What the train data contains, for checking held-out plans against it."""

    ids: set[str] = field(default_factory=set)
    sites: set[str] = field(default_factory=set)
    structures: set[str] = field(default_factory=set)
    texts: set[str] = field(default_factory=set)

    def add_train_plan(self, record_id: str, graph: FloorplanGraph) -> None:
        self.ids.add(record_id)
        if record_id.startswith("swiss_dwellings/"):
            self.sites.add(swiss_site(record_id))
        self.structures.add(structure_hash(graph))

    def add_train_texts(self, texts: Iterable[str]) -> None:
        self.texts.update(normalize_text(text) for text in texts)

    def violations(
        self, record_id: str, graph: FloorplanGraph, text: str | None = None
    ) -> list[str]:
        found = []
        if record_id in self.ids:
            found.append("record id is a train record")
        if record_id.startswith("swiss_dwellings/") and swiss_site(record_id) in self.sites:
            found.append("site holds a train apartment")
        if structure_hash(graph) in self.structures:
            found.append("structure seen in train")
        if text is not None and normalize_text(text) in self.texts:
            found.append("text equals a train prompt")
        return found
