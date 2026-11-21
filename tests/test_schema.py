from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room, RoomType

IMAGE = ImageSize(w=100, h=80)


def room(room_id: str, x: int = 100, y: int = 100, **extra: Any) -> Room:
    return Room(id=room_id, type=RoomType.LIVING, xy=(x, y), **extra)


def graph(rooms: list[Room], links: list[Link] | None = None) -> FloorplanGraph:
    return FloorplanGraph(floors=[Floor(floor=1, rooms=rooms, links=links or [])], image=IMAGE)


def test_valid_graph_is_accepted() -> None:
    result = graph([room("r0"), room("r1")], [Link(a="r0", b="r1", kind="door")])
    assert len(result.rooms()) == 2


def test_duplicate_room_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="used more than once"):
        graph([room("r0"), room("r0", 200, 200)])


def test_room_ids_must_be_unique_across_floors() -> None:
    floors = [Floor(floor=1, rooms=[room("r0")]), Floor(floor=2, rooms=[room("r0")])]
    with pytest.raises(ValidationError, match="used more than once"):
        FloorplanGraph(floors=floors, image=IMAGE)


def test_floor_numbers_must_be_unique() -> None:
    floors = [Floor(floor=1, rooms=[room("r0")]), Floor(floor=1, rooms=[room("r1")])]
    with pytest.raises(ValidationError, match="floor numbers must be unique"):
        FloorplanGraph(floors=floors, image=IMAGE)


def test_dangling_link_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a room on this floor"):
        graph([room("r0")], [Link(a="r0", b="r9", kind="door")])


def test_link_across_floors_is_rejected() -> None:
    """A floor's links may only name rooms of that floor, so nothing links floors."""
    with pytest.raises(ValidationError, match="not a room on this floor"):
        Floor(floor=2, rooms=[room("r1")], links=[Link(a="r1", b="r0", kind="door")])


def test_self_link_is_rejected() -> None:
    with pytest.raises(ValidationError, match="to itself"):
        graph([room("r0")], [Link(a="r0", b="r0", kind="door")])


@pytest.mark.parametrize("second", [("r1", "r0", "door"), ("r0", "r1", "wall")])
def test_duplicate_link_is_rejected_in_either_direction_or_kind(
    second: tuple[str, str, str],
) -> None:
    links = [Link(a="r0", b="r1", kind="door"), Link(a=second[0], b=second[1], kind=second[2])]
    with pytest.raises(ValidationError, match="duplicate link"):
        graph([room("r0"), room("r1")], links)


@pytest.mark.parametrize("xy", [(-1, 5), (5, -1), (1001, 5), (5, 1001)])
def test_coordinates_out_of_range_are_rejected(xy: tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        room("r0", *xy)


def test_grid_corners_are_accepted() -> None:
    assert room("r0", 0, 1000).xy == (0, 1000)


def test_coordinates_must_be_integers() -> None:
    with pytest.raises(ValidationError):
        Room(id="r0", type=RoomType.LIVING, xy=(10.5, 20))


@pytest.mark.parametrize("room_id", ["0", "room1", "r", "r-1", "R1"])
def test_room_id_format_is_enforced(room_id: str) -> None:
    with pytest.raises(ValidationError):
        room(room_id)


def test_a_room_is_a_point_and_rejects_a_bbox_from_an_old_record() -> None:
    assert "bbox" not in Room.model_fields
    old_record = {"id": "r0", "type": "living", "xy": [100, 100], "bbox": [50, 50, 200, 200]}
    with pytest.raises(ValidationError, match="bbox"):
        Room.model_validate(old_record)


def test_unknown_room_type_is_rejected_but_unknown_is_a_member() -> None:
    with pytest.raises(ValidationError):
        Room(id="r0", type="spaceship", xy=(1, 1))
    assert Room(id="r0", type="unknown", xy=(1, 1)).type == RoomType.UNKNOWN


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Room(id="r0", type="living", xy=(1, 1), colour="red")


def test_a_graph_needs_at_least_one_floor() -> None:
    with pytest.raises(ValidationError):
        FloorplanGraph(floors=[], image=IMAGE)
