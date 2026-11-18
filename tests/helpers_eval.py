"""Synthetic plans for evaluation tests: a 1000x1000 grid cut into four squares.

+--------+--------+
| r0     | r1     |      r0 living, r1 kitchen, r2 unknown (unlabeled), r3 entry
| living | kitchen|
+--------+--------+      links: r0-r1 opening, r0-r2 door, r0-r3 wall, r1-r3 door
| r2     | r3     |      r3 is the entrance room
| ???    | entry  |
+--------+--------+
"""

from __future__ import annotations

from vlm_floorplan.record import PlanQuality, PlanRecord, RoomSource
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

IMAGE = ImageSize(w=800, h=600)

SQUARES = {
    "r0": (0, 0, 500, 500),
    "r1": (500, 0, 1000, 500),
    "r2": (0, 500, 500, 1000),
    "r3": (500, 500, 1000, 1000),
}


def ring(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def gt_record(with_polygons: bool = True) -> PlanRecord:
    types = {"r0": "living", "r1": "kitchen", "r2": "unknown", "r3": "entry"}
    rooms = [
        Room(
            id=room_id,
            type=types[room_id],
            xy=((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
            entrance=room_id == "r3",
        )
        for room_id, box in SQUARES.items()
    ]
    links = [
        Link(a="r0", b="r1", kind="opening"),
        Link(a="r0", b="r2", kind="door"),
        Link(a="r0", b="r3", kind="wall"),
        Link(a="r1", b="r3", kind="door"),
    ]
    graph = FloorplanGraph(floors=[Floor(floor=1, rooms=rooms, links=links)], image=IMAGE)
    sources = {
        room_id: RoomSource(
            source_type=types[room_id], polygon=[ring(box)] if with_polygons else []
        )
        for room_id, box in SQUARES.items()
    }
    return PlanRecord(
        id="synthetic/plan0",
        source="synthetic",
        image="plan0.png",
        graph=graph,
        rooms=sources,
        quality=PlanQuality.from_graph(graph, []),
    )


def prediction(rooms: list[dict], links: list[list[str]] | None = None) -> FloorplanGraph:
    """Build a one-floor prediction from plain dicts (id, type, xy, optional entrance)."""
    return FloorplanGraph(
        floors=[
            Floor(
                floor=1,
                rooms=[Room(**room) for room in rooms],
                links=[Link(a=a, b=b, kind=kind) for a, b, kind in (links or [])],
            )
        ],
        image=IMAGE,
    )


def perfect_prediction() -> FloorplanGraph:
    """The ground truth itself, with the same ids."""
    return gt_record().graph


def shifted_ids_prediction() -> FloorplanGraph:
    """Correct rooms and links, but with different ids and points off the GT points."""
    return prediction(
        [
            {"id": "r10", "type": "living", "xy": (100, 100)},
            {"id": "r11", "type": "kitchen", "xy": (900, 100)},
            {"id": "r12", "type": "bedroom", "xy": (100, 900)},
            {"id": "r13", "type": "entry", "xy": (900, 900), "entrance": True},
        ],
        [
            ["r10", "r11", "opening"],
            ["r10", "r12", "door"],
            ["r10", "r13", "wall"],
            ["r11", "r13", "door"],
        ],
    )
