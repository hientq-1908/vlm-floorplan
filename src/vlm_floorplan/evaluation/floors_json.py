"""The `floors` JSON form the zero-shot perception baseline is asked to write.

The baseline reads an image and may describe several floors, so its prompt
(`prompt.py`, `zero-shot-v1`) and parser still use the wrapper `{"floors":[{"floor":1,...}]}`.
The editing agent does not: its plan text and JSON are single-floor (`compact.py`).
This module keeps the baseline's form working on its own, so changing the agent's format
does not change what the baseline prompt asks for or how its output is scored.
"""

from __future__ import annotations

import json
from typing import Any

from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room


def _room_to_json(room: Room) -> dict[str, Any]:
    item: dict[str, Any] = {"id": room.id, "type": room.type.value}
    if room.label is not None:
        item["label"] = room.label
    item["xy"] = list(room.xy)
    if room.size is not None:
        item["size"] = room.size
    if room.entrance:
        item["entrance"] = True
    return item


def to_floors_json(graph: FloorplanGraph) -> str:
    """Serialize a graph, any number of floors, to compact JSON with the `floors` wrapper."""
    floors = [
        {
            "floor": floor.floor,
            "rooms": [_room_to_json(room) for room in floor.rooms],
            "links": [[link.a, link.b, link.kind.value] for link in floor.links],
        }
        for floor in graph.floors
    ]
    return json.dumps({"floors": floors}, separators=(",", ":"), ensure_ascii=False)


def _link_from_json(item: Any) -> Link:
    if not isinstance(item, list) or len(item) != 3:
        raise ValueError(f"a compact link must be [a, b, kind], got {item!r}")
    a, b, kind = item
    return Link(a=a, b=b, kind=kind)


def _floor_from_json(data: dict[str, Any]) -> Floor:
    links = [_link_from_json(item) for item in data.get("links", [])]
    return Floor(floor=data["floor"], rooms=data["rooms"], links=links)


def from_floors_json(text: str, image: ImageSize) -> FloorplanGraph:
    """Parse the `floors` form into a validated graph. Raises `ValueError` when it is not one."""
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("floors"), list):
        raise ValueError('compact graph must be a JSON object with a "floors" list')
    if not all(isinstance(item, dict) for item in data["floors"]):
        raise ValueError("every compact floor must be a JSON object")
    try:
        floors = [_floor_from_json(item) for item in data["floors"]]
    except (KeyError, TypeError) as err:
        raise ValueError(f"malformed compact floor: {err!r}") from err
    return FloorplanGraph(floors=floors, image=image)
