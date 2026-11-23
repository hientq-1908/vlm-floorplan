"""Typed edit operations: the single way a plan changes.

A chat request ("add a bathroom next to the bedroom"), a drag in the UI, a synthetic
training episode and an RL rollout all produce the same `EditOp` values and apply them
with `vlm_floorplan.edits.apply.apply_op`.
One code path means the model is trained on exactly the behavior the app runs.

Design notes:

- Every op is a flat record of scalars (room ids, enums, ints, bools).
  Qwen3.5 renders tool arguments as XML text, so nested objects and lists are fragile,
  and `"75"` and `75` look identical there.
  Flat arguments also make the JSON tool schemas (`tools.py`) trivial.
- The `op` field is the discriminator of a pydantic tagged union, so `EDIT_OP_ADAPTER`
  turns `{"op": "move_room", ...}` into the right class, and the tool name is the `op` value.
- The docstrings and field descriptions are the text the model reads in its tool list,
  so they say what the op does and when to use it, not how it is implemented.
- Ops are frozen. Op-local rules (a coordinate is on the grid) are checked here;
  rules that need the graph (does the room exist?) are checked when the op is applied.
- A room is a point with a size (a radius), so ops place a room by its point and size alone.
- The plan is single-floor at this boundary: ops carry no floor.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from vlm_floorplan.schema import GridCoord, LinkKind, RoomId, RoomSize, RoomType

MAX_LABEL_LENGTH = 40

_X = "Horizontal position on the 0-1000 grid, 0 = left edge of the image."
_SIZE = "the radius of a circle with the room's area."
_Y = "Vertical position on the 0-1000 grid, 0 = top edge of the image."


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _clean_label(value: str | None) -> str | None:
    """Strip a label; the empty string means "no label"."""
    if value is None:
        return None
    return value.strip() or None


def _strip_label(value: str) -> str:
    """Strip a label; it stays a string, and "" still means "remove the name"."""
    return value.strip()


class AddRoom(_Op):
    """Add a new room. The room starts with no links; link it with add_link."""

    op: Literal["add_room"] = "add_room"
    type: RoomType = Field(description="The room type.")
    x: GridCoord = Field(description=f"A point inside the room. {_X}")
    y: GridCoord = Field(description=f"A point inside the room. {_Y}")
    label: str | None = Field(
        default=None,
        max_length=MAX_LABEL_LENGTH,
        description="The room's name as it should be shown, for example 'Master bedroom'.",
    )
    size: RoomSize = Field(description=f"Room size on the 0-1000 grid: {_SIZE}")
    room_id: RoomId | None = Field(
        default=None,
        description="Leave out: the next free id is assigned. Replays and tests set it.",
    )

    _label = field_validator("label")(_clean_label)


class MoveRoom(_Op):
    """Move a room to a new position. Its links and type stay as they are."""

    op: Literal["move_room"] = "move_room"
    room_id: RoomId = Field(description="The room to move.")
    x: GridCoord = Field(description=f"New position. {_X}")
    y: GridCoord = Field(description=f"New position. {_Y}")


class SetRoomSize(_Op):
    """Resize a room. Its position, links and type stay as they are."""

    op: Literal["set_room_size"] = "set_room_size"
    room_id: RoomId = Field(description="The room to resize.")
    size: RoomSize = Field(description=f"The new size on the 0-1000 grid: {_SIZE}")


class DeleteRoom(_Op):
    """Delete a room and every link that touches it."""

    op: Literal["delete_room"] = "delete_room"
    room_id: RoomId = Field(description="The room to delete.")


class SetRoomType(_Op):
    """Change a room's type, for example from bedroom to office (type 'other')."""

    op: Literal["set_room_type"] = "set_room_type"
    room_id: RoomId = Field(description="The room to change.")
    type: RoomType = Field(description="The new room type.")


class SetRoomLabel(_Op):
    """Rename a room. An empty label removes the name."""

    op: Literal["set_room_label"] = "set_room_label"
    room_id: RoomId = Field(description="The room to rename.")
    label: str = Field(
        max_length=MAX_LABEL_LENGTH,
        description="The new name. Use an empty string to remove the name.",
    )

    _label = field_validator("label")(_strip_label)


class AddLink(_Op):
    """Link two rooms on the same floor that are not linked yet.

    Use kind 'wall' to say that two rooms are next to each other with a wall between them.
    A door or opening needs that wall first: add a 'wall' link, then change it with
    set_link_kind. Rooms that are already linked are changed only with set_link_kind.
    """

    op: Literal["add_link"] = "add_link"
    room_a: RoomId = Field(description="One of the two rooms.")
    room_b: RoomId = Field(description="The other room.")
    kind: LinkKind = Field(
        description="Use 'wall'. For a door or opening, add a wall, then use set_link_kind."
    )


class RemoveLink(_Op):
    """Remove the link between two rooms completely, so they are no longer adjacent.

    To close a door in a wall that stays, use set_link_kind with kind 'wall' instead.
    """

    op: Literal["remove_link"] = "remove_link"
    room_a: RoomId = Field(description="One of the two rooms.")
    room_b: RoomId = Field(description="The other room.")


class SetLinkKind(_Op):
    """Change how two already linked rooms connect (door, opening, wall)."""

    op: Literal["set_link_kind"] = "set_link_kind"
    room_a: RoomId = Field(description="One of the two rooms.")
    room_b: RoomId = Field(description="The other room.")
    kind: LinkKind = Field(description="The new kind: door, opening, or wall.")


class AddExteriorDoor(_Op):
    """Mark that an exterior door opens into a room (any outside door, not only the front door)."""

    op: Literal["add_exterior_door"] = "add_exterior_door"
    room_id: RoomId = Field(description="The room the outside door opens into.")


class RemoveExteriorDoor(_Op):
    """Remove the exterior door from a room (it no longer has a door to the outside)."""

    op: Literal["remove_exterior_door"] = "remove_exterior_door"
    room_id: RoomId = Field(description="The room that has the outside door.")


EditOp = Annotated[
    AddRoom
    | MoveRoom
    | SetRoomSize
    | DeleteRoom
    | SetRoomType
    | SetRoomLabel
    | AddLink
    | RemoveLink
    | SetLinkKind
    | AddExteriorDoor
    | RemoveExteriorDoor,
    Field(discriminator="op"),
]

OP_CLASSES: dict[str, type[_Op]] = {
    cls.model_fields["op"].default: cls
    for cls in (
        AddRoom,
        MoveRoom,
        SetRoomSize,
        DeleteRoom,
        SetRoomType,
        SetRoomLabel,
        AddLink,
        RemoveLink,
        SetLinkKind,
        AddExteriorDoor,
        RemoveExteriorDoor,
    )
}
"""Op name to class, in the order tools are listed to the model."""

EDIT_OP_ADAPTER: TypeAdapter[EditOp] = TypeAdapter(EditOp)
EDIT_OPS_ADAPTER: TypeAdapter[list[EditOp]] = TypeAdapter(list[EditOp])


def dump_ops(ops: list[EditOp] | tuple[EditOp, ...]) -> str:
    """Serialize ops to JSON text (an edit history on disk, or a training episode)."""
    return EDIT_OPS_ADAPTER.dump_json(list(ops), exclude_none=True).decode()


def load_ops(text: str) -> list[EditOp]:
    """Parse the output of `dump_ops`. Raises pydantic's `ValidationError` on bad input."""
    return EDIT_OPS_ADAPTER.validate_json(text)


__all__ = [
    "EDIT_OPS_ADAPTER",
    "EDIT_OP_ADAPTER",
    "MAX_LABEL_LENGTH",
    "OP_CLASSES",
    "AddExteriorDoor",
    "AddLink",
    "AddRoom",
    "DeleteRoom",
    "EditOp",
    "MoveRoom",
    "RemoveExteriorDoor",
    "RemoveLink",
    "SetLinkKind",
    "SetRoomLabel",
    "SetRoomSize",
    "SetRoomType",
    "dump_ops",
    "load_ops",
]
