"""A small six-room plan with polygons, for tests and as the example script's fallback.

    +--------+------+----------+
    | r0     | r1   | r2       |     r0 bedroom, r1 bathroom, r2 kitchen
    | bed    | bath | kitchen  |
    +--------+------+----------+
    | r3 living         | r4   |     r3 living, r4 entry (the entrance), r5 storage
    |                   | entry|
    |                   +------+
    |                   | r5   |
    +-------------------+------+

Links: r0-r1 wall, r1-r2 wall, r0-r3 door, r1-r3 door, r2-r3 opening,
r2-r4 door, r3-r4 door, r3-r5 wall, r4-r5 door.
"""

from __future__ import annotations

import math

from vlm_floorplan.record import PlanQuality, PlanRecord, RoomSource
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

_BOXES = {
    "r0": ("bedroom", (0, 0, 400, 400)),
    "r1": ("bathroom", (400, 0, 600, 400)),
    "r2": ("kitchen", (600, 0, 1000, 400)),
    "r3": ("living", (0, 400, 700, 1000)),
    "r4": ("entry", (700, 400, 1000, 700)),
    "r5": ("storage", (700, 700, 1000, 1000)),
}
_LINKS = [
    ("r0", "r1", "wall"),
    ("r1", "r2", "wall"),
    ("r0", "r3", "door"),
    ("r1", "r3", "door"),
    ("r2", "r3", "opening"),
    ("r2", "r4", "door"),
    ("r3", "r4", "door"),
    ("r3", "r5", "wall"),
    ("r4", "r5", "door"),
]


def sample_record() -> PlanRecord:
    rooms = [
        Room(
            id=room_id,
            type=room_type,
            xy=((x1 + x2) // 2, (y1 + y2) // 2),
            size=round(math.sqrt((x2 - x1) * (y2 - y1) / math.pi)),
            entrance=room_id == "r4",
        )
        for room_id, (room_type, (x1, y1, x2, y2)) in _BOXES.items()
    ]
    links = [Link(a=a, b=b, kind=kind) for a, b, kind in _LINKS]
    graph = FloorplanGraph(
        floors=[Floor(floor=1, rooms=rooms, links=links)],
        image=ImageSize(w=800, h=800),
        exterior_wall=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
    )
    sources = {
        room_id: RoomSource(
            source_type=room_type,
            polygon=[[(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]],
        )
        for room_id, (room_type, (x1, y1, x2, y2)) in _BOXES.items()
    }
    return PlanRecord(
        id="sample/six-rooms",
        source="sample",
        image="six-rooms.png",
        graph=graph,
        rooms=sources,
        quality=PlanQuality.from_graph(graph, []),
    )
