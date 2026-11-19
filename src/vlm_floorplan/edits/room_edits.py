"""Handlers for the edits that change one room: add, move, delete, retype, rename, entrance.

Each handler takes a graph and an op and returns a `Change` or an `EditError`.
Handlers check the rules that need the graph; `apply_op` validates the result afterwards.
"""

from __future__ import annotations

from vlm_floorplan.edits.change import Change, make_error, no_change
from vlm_floorplan.edits.lookup import (
    locate_room,
    next_room_id,
    rebuilt_room,
    replace_floor,
    unknown_room_error,
)
from vlm_floorplan.edits.ops import (
    AddRoom,
    DeleteRoom,
    MoveRoom,
    SetEntrance,
    SetRoomLabel,
    SetRoomType,
)
from vlm_floorplan.edits.outcome import EditError, EditErrorCode
from vlm_floorplan.schema import Floor, FloorplanGraph, Room


def swap_room(graph: FloorplanGraph, floor_index: int, new_room: Room) -> FloorplanGraph:
    floor = graph.floors[floor_index]
    rooms = [new_room if room.id == new_room.id else room for room in floor.rooms]
    return replace_floor(
        graph, floor_index, Floor(floor=floor.floor, rooms=rooms, links=floor.links)
    )


def _resolve_floor(graph: FloorplanGraph, number: int | None) -> int | EditError:
    numbers = [floor.floor for floor in graph.floors]
    if number is None:
        if len(numbers) == 1:
            return 0
        return make_error(
            EditErrorCode.AMBIGUOUS_FLOOR,
            f"The plan has floors {numbers}; say which floor the room goes on.",
        )
    if number not in numbers:
        return make_error(
            EditErrorCode.UNKNOWN_FLOOR, f"There is no floor {number}; floors are {numbers}."
        )
    return numbers.index(number)


def add_room(graph: FloorplanGraph, op: AddRoom) -> Change | EditError:
    floor_index = _resolve_floor(graph, op.floor)
    if isinstance(floor_index, EditError):
        return floor_index
    floor = graph.floors[floor_index]
    room_id = op.room_id or next_room_id(graph)
    if locate_room(graph, room_id) is not None:
        return make_error(EditErrorCode.ID_IN_USE, f"Room id {room_id} already exists.", room_id)
    room = Room(
        id=room_id,
        type=op.type,
        label=op.label,
        xy=(op.x, op.y),
        entrance=op.entrance,
    )
    new_floor = Floor(floor=floor.floor, rooms=[*floor.rooms, room], links=floor.links)
    applied = op.model_copy(update={"room_id": room_id, "floor": floor.floor})
    name = f" '{room.label}'" if room.label else ""
    description = (
        f"added {room.type.value} room {room_id}{name} at ({op.x},{op.y}) on floor {floor.floor}"
    )
    return Change(replace_floor(graph, floor_index, new_floor), applied, description)


def move_room(graph: FloorplanGraph, op: MoveRoom) -> Change | EditError:
    found = locate_room(graph, op.room_id)
    if found is None:
        return unknown_room_error(graph, op.room_id)
    floor_index, room = found
    if room.xy == (op.x, op.y):
        return no_change(f"Room {room.id} is already at ({op.x},{op.y}).", room.id)
    new_graph = swap_room(graph, floor_index, rebuilt_room(room, xy=(op.x, op.y), bbox=None))
    return Change(new_graph, op, f"moved {room.id} from {room.xy} to ({op.x},{op.y})")


def delete_room(graph: FloorplanGraph, op: DeleteRoom) -> Change | EditError:
    found = locate_room(graph, op.room_id)
    if found is None:
        return unknown_room_error(graph, op.room_id)
    floor_index, room = found
    floor = graph.floors[floor_index]
    kept = [link for link in floor.links if room.id not in (link.a, link.b)]
    dropped = [link for link in floor.links if room.id in (link.a, link.b)]
    rooms = [other for other in floor.rooms if other.id != room.id]
    new_graph = replace_floor(graph, floor_index, Floor(floor=floor.floor, rooms=rooms, links=kept))
    description = f"deleted {room.type.value} room {room.id}"
    if dropped:
        others = ", ".join(link.b if link.a == room.id else link.a for link in dropped)
        description += f" and its {len(dropped)} link(s) to {others}"
    return Change(new_graph, op, description)


def set_room_type(graph: FloorplanGraph, op: SetRoomType) -> Change | EditError:
    found = locate_room(graph, op.room_id)
    if found is None:
        return unknown_room_error(graph, op.room_id)
    floor_index, room = found
    if room.type == op.type:
        return no_change(f"Room {room.id} is already of type {room.type.value}.", room.id)
    new_graph = swap_room(graph, floor_index, rebuilt_room(room, type=op.type))
    return Change(new_graph, op, f"changed {room.id} from {room.type.value} to {op.type.value}")


def set_room_label(graph: FloorplanGraph, op: SetRoomLabel) -> Change | EditError:
    found = locate_room(graph, op.room_id)
    if found is None:
        return unknown_room_error(graph, op.room_id)
    floor_index, room = found
    label = op.label or None
    if room.label == label:
        said = f"already named '{label}'" if label else "already has no name"
        return no_change(f"Room {room.id} is {said}.", room.id)
    new_graph = swap_room(graph, floor_index, rebuilt_room(room, label=label))
    description = f"renamed {room.id} to '{label}'" if label else f"removed the name of {room.id}"
    return Change(new_graph, op, description)


def set_entrance(graph: FloorplanGraph, op: SetEntrance) -> Change | EditError:
    found = locate_room(graph, op.room_id)
    if found is None:
        return unknown_room_error(graph, op.room_id)
    floor_index, room = found
    if room.entrance == op.entrance:
        state = "already is" if op.entrance else "already is not"
        return no_change(f"Room {room.id} {state} an entrance.", room.id)
    new_graph = swap_room(graph, floor_index, rebuilt_room(room, entrance=op.entrance))
    verb = "marked" if op.entrance else "unmarked"
    return Change(new_graph, op, f"{verb} {room.id} as an entrance")
