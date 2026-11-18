"""The success value and error helpers shared by the edit handlers."""

from __future__ import annotations

from dataclasses import dataclass

from vlm_floorplan.edits.ops import EditOp
from vlm_floorplan.edits.outcome import EditError, EditErrorCode
from vlm_floorplan.schema import FloorplanGraph


@dataclass(frozen=True)
class Change:
    """A handler's success value: the new graph, the op as applied, and what happened."""

    graph: FloorplanGraph
    op: EditOp
    description: str


def make_error(code: EditErrorCode, message: str, *room_ids: str) -> EditError:
    return EditError(code, message, room_ids)


def no_change(message: str, *room_ids: str) -> EditError:
    return make_error(EditErrorCode.NO_CHANGE, message, *room_ids)
