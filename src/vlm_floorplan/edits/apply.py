"""Apply one edit op to a graph: the only function that changes a plan.

`apply_op(graph, op)` never mutates `graph` and never raises for a bad request.
It returns `EditApplied` with a new graph, or `EditRejected` with a typed error.

How an edit is judged, in order:

1. The handler (`room_edits.py`, `link_edits.py`) checks the rules that need the graph
   (the room exists, the pair is linked, a door only goes into a shared wall)
   and builds the new graph.
2. `validate_graph` re-checks the result. A handler bug becomes `INVALID_RESULT`
   instead of a broken plan.
3. Warnings the edit introduced are attached to the result.

Edits that change nothing are refused with `NO_CHANGE`.
A history full of no-ops would teach a model that they are fine, and the message
tells the model its request was already true.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vlm_floorplan.edits import link_edits, room_edits
from vlm_floorplan.edits.change import Change
from vlm_floorplan.edits.ops import (
    AddExteriorDoor,
    AddLink,
    AddRoom,
    DeleteRoom,
    EditOp,
    MoveRoom,
    RemoveExteriorDoor,
    RemoveLink,
    SetLinkKind,
    SetRoomLabel,
    SetRoomSize,
    SetRoomType,
)
from vlm_floorplan.edits.outcome import (
    EditApplied,
    EditError,
    EditErrorCode,
    EditOutcome,
    EditRejected,
)
from vlm_floorplan.edits.validator import new_warnings, validate_graph
from vlm_floorplan.schema import FloorplanGraph

_HANDLERS: dict[type, Callable[[FloorplanGraph, Any], Change | EditError]] = {
    AddRoom: room_edits.add_room,
    MoveRoom: room_edits.move_room,
    SetRoomSize: room_edits.set_room_size,
    DeleteRoom: room_edits.delete_room,
    SetRoomType: room_edits.set_room_type,
    SetRoomLabel: room_edits.set_room_label,
    AddExteriorDoor: room_edits.add_exterior_door,
    RemoveExteriorDoor: room_edits.remove_exterior_door,
    AddLink: link_edits.add_link,
    RemoveLink: link_edits.remove_link,
    SetLinkKind: link_edits.set_link_kind,
}


def apply_op(graph: FloorplanGraph, op: EditOp) -> EditOutcome:
    """Apply one edit. Returns the new graph, or the reason it was refused."""
    result = _HANDLERS[type(op)](graph, op)
    if isinstance(result, EditError):
        return EditRejected(result, op)
    errors = validate_graph(result.graph).errors
    if errors:
        reasons = "; ".join(issue.message for issue in errors)
        room_ids = tuple(dict.fromkeys(r for issue in errors for r in issue.room_ids))
        message = f"The edit would break the plan: {reasons}."
        return EditRejected(EditError(EditErrorCode.INVALID_RESULT, message, room_ids), op)
    return EditApplied(
        graph=result.graph,
        op=result.op,
        description=result.description,
        warnings=new_warnings(graph, result.graph),
    )
