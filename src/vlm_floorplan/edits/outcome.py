"""What applying an edit returns: the new graph, or a typed error.

Edits never raise for a bad request.
A model that asks for something impossible ("add a door between rooms that do not touch")
is an expected event, and the reason must go back to the model as text.
So `apply_op` returns one of two frozen values, and callers branch on the type:

    outcome = apply_op(graph, op)
    if isinstance(outcome, EditApplied):
        graph = outcome.graph
    else:
        print(outcome.error.message)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vlm_floorplan.edits.ops import EditOp
    from vlm_floorplan.edits.validator import Issue
    from vlm_floorplan.schema import FloorplanGraph


class EditErrorCode(StrEnum):
    """Why an edit was refused. Stable strings: tool responses and training data quote them."""

    INVALID_ARGUMENTS = "invalid_arguments"  # the tool call itself is malformed
    UNKNOWN_TOOL = "unknown_tool"
    UNKNOWN_ROOM = "unknown_room"
    UNKNOWN_FLOOR = "unknown_floor"
    AMBIGUOUS_FLOOR = "ambiguous_floor"  # several floors and the op did not say which
    ID_IN_USE = "id_in_use"
    CROSS_FLOOR = "cross_floor"  # links never join rooms of different floors
    SELF_LINK = "self_link"
    UNKNOWN_LINK = "unknown_link"  # the two rooms are not linked at all
    DUPLICATE_LINK = "duplicate_link"
    NOT_ADJACENT = "not_adjacent"  # a door or opening between rooms that are not linked
    NO_CHANGE = "no_change"  # the edit would leave the graph exactly as it is
    INVALID_RESULT = "invalid_result"  # safety net: the edit would break a graph rule


@dataclass(frozen=True)
class EditError:
    """A refused edit, written for the model to read and act on."""

    code: EditErrorCode
    message: str
    room_ids: tuple[str, ...] = ()
    """The rooms the problem is about, so a UI can highlight them."""


@dataclass(frozen=True)
class EditApplied:
    """A successful edit."""

    graph: FloorplanGraph
    """The new graph. The graph that was passed in is never modified."""
    op: EditOp
    """The op as it was applied: defaults are filled in (the new room's id, the floor),
    so replaying `op` on the same starting graph gives the same result."""
    description: str
    """One sentence saying what changed, including side effects such as dropped links."""
    warnings: tuple[Issue, ...] = ()
    """Problems this edit introduced that do not make the graph invalid (an isolated room)."""


@dataclass(frozen=True)
class EditRejected:
    """A refused edit. The graph is unchanged."""

    error: EditError
    op: EditOp | None = None
    """The op that was refused, or None when the tool call could not even be parsed."""


EditOutcome = EditApplied | EditRejected
