"""Compact JSON: the text form of a graph that the model reads and writes.

Every token the model emits costs time, so the training target drops what it can:

- no whitespace,
- links as `[a, b, kind]` triples instead of objects,
- optional room fields (`label`, `entrance`) only when set,
- no image size (grid coordinates do not depend on it).

Example (one floor, two rooms, one door):

    {"floors":[{"floor":1,"rooms":[{"id":"r0","type":"entry","xy":[474,735]},
    {"id":"r1","type":"living","label":"OH","xy":[583,295]}],"links":[["r0","r1","door"]]}]}

`from_compact(to_compact(graph), graph.image) == graph` holds for every valid graph,
which the tests check.
Parsing goes through the same pydantic validators, so a model output that parses
is also a structurally valid graph.
"""

from __future__ import annotations

import json
from typing import Any

from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room


def _room_to_compact(room: Room) -> dict[str, Any]:
    item: dict[str, Any] = {"id": room.id, "type": room.type.value}
    if room.label is not None:
        item["label"] = room.label
    item["xy"] = list(room.xy)
    if room.entrance:
        item["entrance"] = True
    return item


def to_compact(graph: FloorplanGraph) -> str:
    """Serialize a graph to compact JSON text."""
    floors = [
        {
            "floor": floor.floor,
            "rooms": [_room_to_compact(room) for room in floor.rooms],
            "links": [[link.a, link.b, link.kind.value] for link in floor.links],
        }
        for floor in graph.floors
    ]
    # ensure_ascii=False keeps Finnish labels such as "KÄYTÄVÄ" readable and short.
    return json.dumps({"floors": floors}, separators=(",", ":"), ensure_ascii=False)


def _floor_from_compact(data: dict[str, Any]) -> Floor:
    links = [_link_from_compact(item) for item in data.get("links", [])]
    return Floor(floor=data["floor"], rooms=data["rooms"], links=links)


def _link_from_compact(item: Any) -> Link:
    if not isinstance(item, list) or len(item) != 3:
        raise ValueError(f"a compact link must be [a, b, kind], got {item!r}")
    a, b, kind = item
    return Link(a=a, b=b, kind=kind)


def from_compact(text: str, image: ImageSize) -> FloorplanGraph:
    """Parse compact JSON text back into a validated graph.

    Raises `ValueError` (pydantic's `ValidationError` is a subclass) when the text is
    not valid JSON or does not describe a valid graph.
    """
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("floors"), list):
        raise ValueError('compact graph must be a JSON object with a "floors" list')
    if not all(isinstance(item, dict) for item in data["floors"]):
        raise ValueError("every compact floor must be a JSON object")
    try:
        floors = [_floor_from_compact(item) for item in data["floors"]]
    except (KeyError, TypeError) as err:
        raise ValueError(f"malformed compact floor: {err!r}") from err
    return FloorplanGraph(floors=floors, image=image)
