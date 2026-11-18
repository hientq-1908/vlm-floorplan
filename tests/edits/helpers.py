"""A small two-floor plan shared by the edit tests.

Floor 1:  r0 entry (entrance) -door- r1 living -opening- r2 kitchen
          r1 -wall- r3 bedroom -wall- r4 bathroom, and r2 touches nothing else.
Floor 2:  r5 bedroom -door- r6 stairs.
"""

from __future__ import annotations

from vlm_floorplan.edits import EditApplied, EditOp, EditRejected, apply_op
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room, RoomType


def sample_graph() -> FloorplanGraph:
    floor1 = Floor(
        floor=1,
        rooms=[
            Room(id="r0", type=RoomType.ENTRY, xy=(100, 500), entrance=True),
            Room(
                id="r1", type=RoomType.LIVING, label="OH", xy=(400, 500), bbox=(300, 400, 500, 600)
            ),
            Room(id="r2", type=RoomType.KITCHEN, xy=(700, 500)),
            Room(id="r3", type=RoomType.BEDROOM, xy=(400, 800)),
            Room(id="r4", type=RoomType.BATHROOM, xy=(700, 800)),
        ],
        links=[
            Link(a="r0", b="r1", kind="door"),
            Link(a="r1", b="r2", kind="opening"),
            Link(a="r1", b="r3", kind="wall"),
            Link(a="r3", b="r4", kind="wall"),
        ],
    )
    floor2 = Floor(
        floor=2,
        rooms=[
            Room(id="r5", type=RoomType.BEDROOM, xy=(300, 300)),
            Room(id="r6", type=RoomType.STAIRS, xy=(600, 300)),
        ],
        links=[Link(a="r5", b="r6", kind="door")],
    )
    return FloorplanGraph(floors=[floor1, floor2], image=ImageSize(w=800, h=600))


def applied(graph: FloorplanGraph, op: EditOp) -> EditApplied:
    outcome = apply_op(graph, op)
    assert isinstance(outcome, EditApplied), outcome
    return outcome


def rejected(graph: FloorplanGraph, op: EditOp) -> EditRejected:
    outcome = apply_op(graph, op)
    assert isinstance(outcome, EditRejected), outcome
    return outcome


def link_kinds(graph: FloorplanGraph) -> dict[frozenset[str], str]:
    return {link.pair: link.kind.value for floor in graph.floors for link in floor.links}
