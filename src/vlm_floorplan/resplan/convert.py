"""Convert one ResPlan plan into a `FloorplanGraph` plus the record's ground truth.

The graph's coordinates come from `PlanView`, the same object the renderer draws with,
so a room's point lands on that room in the rendered image.

Choices worth knowing:

- Room ids follow reading order (top to bottom, then left to right, by the room's point).
  Ids then do not depend on how ResPlan groups rooms by type, and a model learns
  "r0 is the top-left room" more easily than an arbitrary order.
- ResPlan's rooms have no printed names, so `label` stays empty.
- ResPlan plans are single-floor: the graph has one floor, numbered 1.
- The yard layers (`garden`, `parking`, `pool`) are not rooms. ResPlan's own graph leaves them
  out, they have no doors, and they would crowd the plan with unlinked nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlm_floorplan.record import RoomSource
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.resplan.links import RoomShape, derive_links, find_entrances, room_shapes
from vlm_floorplan.room_types import RESPLAN_TYPES
from vlm_floorplan.schema import Floor, FloorplanGraph, Link, Room


@dataclass(frozen=True)
class ConvertedPlan:
    graph: FloorplanGraph
    sources: dict[str, RoomSource]
    """Ground truth per room id: ResPlan's category and the room outline in grid space."""
    warnings: list[str]


def convert_plan(plan: dict[str, Any], view: PlanView | None = None) -> ConvertedPlan:
    """Build the graph for `plan`. Raises `ValueError` for a plan with no rooms."""
    shapes = room_shapes(plan)
    if not shapes:
        raise ValueError("plan has no rooms")
    view = view or PlanView.for_plan(plan)
    frame = view.grid

    points = [frame.point(shape.geometry) for shape in shapes]
    # Reading order: sort by (y, x) of the point. `order[k]` is the shape that becomes r{k}.
    order = sorted(range(len(shapes)), key=lambda i: (points[i][1], points[i][0]))
    new_index = {old: new for new, old in enumerate(order)}

    entrances, warnings = find_entrances(plan, shapes)
    if not entrances and not warnings:
        warnings.append("plan has no front door")

    rooms = [_room(shapes[i], f"r{new_index[i]}", points[i], i in entrances, view) for i in order]
    renumbered = {
        tuple(sorted((new_index[i], new_index[j]))): kind
        for (i, j), kind in derive_links(plan, shapes).items()
    }
    links = [Link(a=f"r{a}", b=f"r{b}", kind=kind) for (a, b), kind in sorted(renumbered.items())]
    linked = {end for link in links for end in (link.a, link.b)}
    if len(linked) < len(rooms):
        warnings.append(f"{len(rooms) - len(linked)} room(s) have no link")

    graph = FloorplanGraph(floors=[Floor(floor=1, rooms=rooms, links=links)], image=view.image)
    sources = {
        f"r{new_index[i]}": RoomSource(
            source_type=shapes[i].layer, polygon=frame.rings(shapes[i].geometry)
        )
        for i in order
    }
    return ConvertedPlan(graph=graph, sources=sources, warnings=warnings)


def _room(
    shape: RoomShape, room_id: str, point: tuple[int, int], entrance: bool, view: PlanView
) -> Room:
    return Room(
        id=room_id,
        type=RESPLAN_TYPES[shape.layer],
        xy=point,
        bbox=view.grid.bbox(shape.geometry),
        entrance=entrance,
    )
