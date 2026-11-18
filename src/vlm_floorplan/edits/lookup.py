"""Small read-only helpers over a `FloorplanGraph`, shared by the edit handlers.

Graphs are immutable by convention: nothing in `vlm_floorplan.edits` mutates one.
`replace_floor` builds a new graph through the schema constructors, so the schema's
own validators run on every edit result as a second safety net.
"""

from __future__ import annotations

from vlm_floorplan.edits.outcome import EditError, EditErrorCode
from vlm_floorplan.schema import Floor, FloorplanGraph, Link, Room

MAX_LISTED_IDS = 30
"""Cap on ids quoted in an error message, so a big plan cannot flood the model's context."""


def locate_room(graph: FloorplanGraph, room_id: str) -> tuple[int, Room] | None:
    """The floor index and room with this id, or None."""
    for index, floor in enumerate(graph.floors):
        for room in floor.rooms:
            if room.id == room_id:
                return index, room
    return None


def find_link(floor: Floor, room_a: str, room_b: str) -> tuple[int, Link] | None:
    """The link joining two rooms (either direction), with its index in `floor.links`."""
    pair = frozenset((room_a, room_b))
    for index, link in enumerate(floor.links):
        if link.pair == pair:
            return index, link
    return None


def next_room_id(graph: FloorplanGraph) -> str:
    """The id one above the highest in use, so a new room never reuses a live id."""
    numbers = [int(room.id[1:]) for room in graph.rooms()]
    return f"r{max(numbers) + 1 if numbers else 0}"


def replace_floor(graph: FloorplanGraph, index: int, floor: Floor) -> FloorplanGraph:
    """A new graph with one floor swapped. Re-runs the schema's validators."""
    floors = [*graph.floors[:index], floor, *graph.floors[index + 1 :]]
    return FloorplanGraph(floors=floors, image=graph.image)


def rebuilt_room(room: Room, **changes: object) -> Room:
    """A copy of `room` with some fields changed, validated by the schema."""
    return Room.model_validate({**room.model_dump(), **changes})


def unknown_room_error(graph: FloorplanGraph, room_id: str) -> EditError:
    ids = [room.id for room in graph.rooms()]
    listed = ", ".join(ids[:MAX_LISTED_IDS]) + (", ..." if len(ids) > MAX_LISTED_IDS else "")
    known = f"The plan's rooms are: {listed}." if ids else "The plan has no rooms yet."
    return EditError(EditErrorCode.UNKNOWN_ROOM, f"There is no room {room_id}. {known}", (room_id,))
