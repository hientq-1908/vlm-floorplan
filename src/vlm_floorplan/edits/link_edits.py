"""Handlers for the edits that change a link: add, remove, set kind.

The rule that matters most here: a door or an opening can only go where two rooms
already touch, which the graph records as an existing `wall` (or `opening`) link.
The model never sees walls after perception, so this is the check that stops
"add a door between the garage and the bathroom" when they are far apart.
"""

from __future__ import annotations

from vlm_floorplan.edits.change import Change, make_error, no_change
from vlm_floorplan.edits.lookup import find_link, locate_room, replace_floor, unknown_room_error
from vlm_floorplan.edits.ops import AddLink, RemoveLink, SetLinkKind
from vlm_floorplan.edits.outcome import EditError, EditErrorCode
from vlm_floorplan.schema import Floor, FloorplanGraph, Link, LinkKind, Room


def _link_floor(
    graph: FloorplanGraph, room_a: str, room_b: str
) -> tuple[int, Room, Room] | EditError:
    """The floor both rooms are on, or the error that explains why a link is impossible."""
    if room_a == room_b:
        return make_error(
            EditErrorCode.SELF_LINK, f"A room cannot be linked to itself ({room_a}).", room_a
        )
    found = []
    for room_id in (room_a, room_b):
        entry = locate_room(graph, room_id)
        if entry is None:
            return unknown_room_error(graph, room_id)
        found.append(entry)
    (index_a, a), (index_b, b) = found
    if index_a != index_b:
        return make_error(
            EditErrorCode.CROSS_FLOOR,
            f"{a.id} is on floor {graph.floors[index_a].floor} and {b.id} is on floor"
            f" {graph.floors[index_b].floor}; links never join different floors.",
            a.id,
            b.id,
        )
    return index_a, a, b


def _with_links(graph: FloorplanGraph, index: int, links: list[Link]) -> FloorplanGraph:
    floor = graph.floors[index]
    return replace_floor(graph, index, Floor(floor=floor.floor, rooms=floor.rooms, links=links))


def add_link(graph: FloorplanGraph, op: AddLink) -> Change | EditError:
    resolved = _link_floor(graph, op.room_a, op.room_b)
    if isinstance(resolved, EditError):
        return resolved
    index, a, b = resolved
    floor = graph.floors[index]
    existing = find_link(floor, a.id, b.id)
    new_link = Link(a=a.id, b=b.id, kind=op.kind)
    if existing is None:
        if op.kind != LinkKind.WALL:
            return make_error(
                EditErrorCode.NOT_ADJACENT,
                f"Cannot add a {op.kind.value} between {a.id} and {b.id}: they share no wall or"
                f" opening, so there is nowhere to put it. If they do touch, first add a link"
                f" of kind 'wall' between them.",
                a.id,
                b.id,
            )
        description = f"linked {a.id} and {b.id} with a wall"
        return Change(_with_links(graph, index, [*floor.links, new_link]), op, description)
    position, old = existing
    if old.kind == LinkKind.WALL and op.kind != LinkKind.WALL:
        links = [*floor.links[:position], new_link, *floor.links[position + 1 :]]
        description = f"put a {op.kind.value} into the wall between {a.id} and {b.id}"
        return Change(_with_links(graph, index, links), op, description)
    return make_error(
        EditErrorCode.DUPLICATE_LINK,
        f"{a.id} and {b.id} are already linked by a {old.kind.value}."
        " To change how they connect, use set_link_kind.",
        a.id,
        b.id,
    )


def remove_link(graph: FloorplanGraph, op: RemoveLink) -> Change | EditError:
    resolved = _link_floor(graph, op.room_a, op.room_b)
    if isinstance(resolved, EditError):
        return resolved
    index, a, b = resolved
    floor = graph.floors[index]
    existing = find_link(floor, a.id, b.id)
    if existing is None:
        return make_error(
            EditErrorCode.UNKNOWN_LINK, f"{a.id} and {b.id} are not linked.", a.id, b.id
        )
    position, old = existing
    links = [*floor.links[:position], *floor.links[position + 1 :]]
    description = f"removed the {old.kind.value} link between {a.id} and {b.id}"
    return Change(_with_links(graph, index, links), op, description)


def set_link_kind(graph: FloorplanGraph, op: SetLinkKind) -> Change | EditError:
    resolved = _link_floor(graph, op.room_a, op.room_b)
    if isinstance(resolved, EditError):
        return resolved
    index, a, b = resolved
    floor = graph.floors[index]
    existing = find_link(floor, a.id, b.id)
    if existing is None:
        return make_error(
            EditErrorCode.UNKNOWN_LINK,
            f"{a.id} and {b.id} are not linked, so there is no link to change."
            " If they touch, add a link of kind 'wall' first.",
            a.id,
            b.id,
        )
    position, old = existing
    if old.kind == op.kind:
        return no_change(
            f"{a.id} and {b.id} are already connected by a {old.kind.value}.", a.id, b.id
        )
    links = [
        *floor.links[:position],
        Link(a=old.a, b=old.b, kind=op.kind),
        *floor.links[position + 1 :],
    ]
    description = (
        f"changed the link between {a.id} and {b.id} from {old.kind.value} to {op.kind.value}"
    )
    return Change(_with_links(graph, index, links), op, description)
