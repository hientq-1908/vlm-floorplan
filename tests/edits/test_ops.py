from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_floorplan.edits import (
    EDIT_OP_ADAPTER,
    AddRoom,
    MoveRoom,
    SetRoomLabel,
    dump_ops,
    load_ops,
)


def test_discriminator_picks_the_op_class() -> None:
    assert EDIT_OP_ADAPTER.validate_python(
        {"op": "move_room", "room_id": "r1", "x": 1, "y": 2}
    ) == (MoveRoom(room_id="r1", x=1, y=2))


def test_ops_are_frozen() -> None:
    op = MoveRoom(room_id="r1", x=1, y=2)
    with pytest.raises(ValidationError):
        op.x = 5  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        MoveRoom(room_id="r1", x=1, y=2, z=3)  # type: ignore[call-arg]


@pytest.mark.parametrize("coord", [-1, 1001])
def test_coordinates_must_be_on_the_grid(coord: int) -> None:
    with pytest.raises(ValidationError):
        MoveRoom(room_id="r1", x=coord, y=0)


def test_blank_label_means_no_label_and_long_labels_are_rejected() -> None:
    assert AddRoom(type="living", x=1, y=1, label="   ").label is None
    with pytest.raises(ValidationError):
        AddRoom(type="living", x=1, y=1, label="x" * 41)


def test_set_room_label_keeps_a_string_so_it_survives_json() -> None:
    op = SetRoomLabel(room_id="r1", label="   ")
    assert op.label == ""
    assert load_ops(dump_ops([op])) == [op]
