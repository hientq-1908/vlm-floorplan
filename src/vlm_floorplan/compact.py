"""Compact JSON: the text form of a plan that the model writes.

Every token the model emits costs time, so the target drops what it can:

- no whitespace,
- links as `[a, b, kind]` triples instead of objects,
- optional room fields (`label`, `entrance`) only when set,
- no image size (grid coordinates do not depend on it),
- no floor: the plan is flat, `{"rooms":[...],"links":[...]}`.

The exterior wall is input only: it is read from the plan text, never written, so it is
not part of this form. `from_compact` takes it from the caller.

Example (two rooms, one door):

    {"rooms":[{"id":"r0","type":"entry","xy":[474,735],"size":52,"entrance":true},
    {"id":"r1","type":"living","label":"OH","xy":[583,295],"size":160}],
    "links":[["r0","r1","door"]]}

Internally a graph still holds a list of floors; this form is always exactly one,
numbered 1. `from_compact(to_compact(graph), graph.image, graph.exterior_wall) == graph`
holds for every valid single-floor graph whose floor is numbered 1 and whose rooms all
have a size, which the tests check. `from_compact` rejects a room without a size.
Parsing goes through the same pydantic validators, so a model output that parses
is also a structurally valid graph.
"""

from __future__ import annotations

import json
from typing import Any

from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

FLOOR_NUMBER = 1
"""The number of the one floor a compact plan describes."""


def _room_to_compact(room: Room) -> dict[str, Any]:
    item: dict[str, Any] = {"id": room.id, "type": room.type.value}
    if room.label is not None:
        item["label"] = room.label
    item["xy"] = list(room.xy)
    if room.size is not None:
        item["size"] = room.size
    if room.entrance:
        item["entrance"] = True
    return item


def to_compact(graph: FloorplanGraph) -> str:
    """Serialize a single-floor graph to compact JSON text.

    Raises `ValueError` for a plan with several floors: the form has no way to name one.
    """
    if len(graph.floors) != 1:
        raise ValueError(f"compact JSON is single-floor; this plan has {len(graph.floors)} floors")
    floor = graph.floors[0]
    plan = {
        "rooms": [_room_to_compact(room) for room in floor.rooms],
        "links": [[link.a, link.b, link.kind.value] for link in floor.links],
    }
    # ensure_ascii=False keeps Finnish labels such as "KÄYTÄVÄ" readable and short.
    return json.dumps(plan, separators=(",", ":"), ensure_ascii=False)


def _link_from_compact(item: Any) -> Link:
    if not isinstance(item, list) or len(item) != 3:
        raise ValueError(f"a compact link must be [a, b, kind], got {item!r}")
    a, b, kind = item
    return Link(a=a, b=b, kind=kind)


def from_compact(
    text: str,
    image: ImageSize,
    exterior_wall: list[tuple[int, int]] | None = None,
) -> FloorplanGraph:
    """Parse compact JSON text into a validated one-floor graph.

    `exterior_wall` is carried over unchanged: the model never writes it.
    Raises `ValueError` (pydantic's `ValidationError` is a subclass) when the text is
    not valid JSON or does not describe a valid graph.
    """
    data = json.loads(text)
    if isinstance(data, dict) and "floors" in data:
        raise ValueError('compact plan has no "floors": write {"rooms":[...],"links":[...]}')
    if not isinstance(data, dict) or not isinstance(data.get("rooms"), list):
        raise ValueError('compact plan must be a JSON object with a "rooms" list')
    for item in data["rooms"]:
        if isinstance(item, dict) and item.get("size") is None:
            raise ValueError(f'room {item.get("id")!r} has no "size": every room needs one')
    links = data.get("links", [])
    if not isinstance(links, list):
        raise ValueError('"links" must be a list of [a, b, kind]')
    floor = Floor(
        floor=FLOOR_NUMBER,
        rooms=data["rooms"],
        links=[_link_from_compact(item) for item in links],
    )
    return FloorplanGraph(floors=[floor], image=image, exterior_wall=exterior_wall)
