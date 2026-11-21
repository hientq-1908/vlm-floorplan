"""Entrance detection in the CubiCasa5K converter, one small plan per cause.

Each plan is a room (or two) with one exterior door, written so that the door's
surroundings reproduce a pattern found in the real data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cubicasa_fixture import _door, _rect, _space, _wall, write_custom_plan

from vlm_floorplan.converters.cubicasa import convert_plan
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import LinkKind

SIZE = (420, 420)


def _convert(tmp_path: Path, parts: list[str]) -> PlanRecord:
    plan_dir = write_custom_plan(tmp_path / "plan", parts, SIZE)
    return convert_plan(plan_dir, "cubicasa5k/test/1")


def _entrance_types(record: PlanRecord) -> list[str]:
    return [room.type.value for room in record.graph.rooms() if room.entrance]


def _outer_door_plan(outdoor_class: str) -> list[str]:
    """A living room whose bottom door opens onto an outdoor space."""
    return [
        _space("Space LivingRoom", _rect(20, 20, 200, 200)),
        _space(outdoor_class, _rect(20, 220, 200, 300)),
        _wall(_rect(20, 200, 200, 220), external=True, doors=_door(_rect(80, 200, 140, 220))),
    ]


@pytest.mark.parametrize(
    "outdoor_class",
    ["Space Outdoor", "Space Outdoor Porch", "Space Outdoor Terrace", "Space CarPort"],
)
def test_door_onto_an_outdoor_space_marks_the_room_inside(
    tmp_path: Path, outdoor_class: str
) -> None:
    record = _convert(tmp_path, _outer_door_plan(outdoor_class))
    assert _entrance_types(record) == ["living"]


def test_door_onto_an_outdoor_space_keeps_its_door_link(tmp_path: Path) -> None:
    record = _convert(tmp_path, _outer_door_plan("Space Outdoor Porch"))
    assert [link.kind for link in record.graph.floors[0].links] == [LinkKind.DOOR]


@pytest.mark.parametrize("balcony_class", ["Space Outdoor Balcony", "Space Outdoor Balcony Glazed"])
def test_balcony_door_is_part_of_the_dwelling_not_an_entrance(
    tmp_path: Path, balcony_class: str
) -> None:
    assert _entrance_types(_convert(tmp_path, _outer_door_plan(balcony_class))) == []


def test_balcony_named_in_finnish_is_not_an_entrance(tmp_path: Path) -> None:
    parts = _outer_door_plan("Space Outdoor")
    parts[1] = _space("Space Outdoor", _rect(20, 220, 200, 300), "PARVEKE")
    assert _entrance_types(_convert(tmp_path, parts)) == []


def test_door_between_two_interior_rooms_on_an_exterior_wall_is_no_entrance(
    tmp_path: Path,
) -> None:
    """A neighbouring room beyond the wall is not outdoors, so the door stays a link."""
    parts = [
        _space("Space LivingRoom", _rect(20, 20, 200, 200)),
        _space("Space Storage", _rect(20, 220, 200, 300)),
        _wall(_rect(20, 200, 200, 220), external=True, doors=_door(_rect(80, 200, 140, 220))),
    ]
    assert _entrance_types(_convert(tmp_path, parts)) == []


def test_door_whose_two_rooms_lie_on_the_same_side_opens_outside(tmp_path: Path) -> None:
    """An entry hall and a closet beside it both touch the door's growth zone, but only
    the hall is across the wall from the street."""
    parts = [
        _space("Space Entry Lobby", _rect(20, 20, 200, 380)),
        _space("Space Closet WalkIn", _rect(210, 20, 300, 380)),
        _wall(_rect(20, 380, 300, 400), external=True, doors=_door(_rect(160, 380, 212, 400))),
        _wall(_rect(200, 20, 210, 380)),
    ]
    record = _convert(tmp_path, parts)
    assert _entrance_types(record) == ["entry"]
    assert LinkKind.DOOR not in {link.kind for link in record.graph.floors[0].links}


def test_door_drawn_just_past_the_room_outline_is_found(tmp_path: Path) -> None:
    parts = [
        _space("Space Entry Lobby", _rect(20, 20, 200, 200)),
        _wall(_rect(20, 208, 200, 228), external=True, doors=_door(_rect(80, 208, 140, 228))),
    ]
    assert _entrance_types(_convert(tmp_path, parts)) == ["entry"]


def test_single_room_door_on_a_plan_with_no_exterior_wall_tag_is_an_entrance(
    tmp_path: Path,
) -> None:
    parts = [
        _space("Space Entry Lobby", _rect(20, 20, 200, 200)),
        _wall(_rect(20, 200, 200, 220), doors=_door(_rect(80, 200, 140, 220))),
    ]
    assert _entrance_types(_convert(tmp_path, parts)) == ["entry"]


def test_single_room_door_on_an_interior_wall_is_skipped_when_walls_are_tagged(
    tmp_path: Path,
) -> None:
    """Without evidence that the wall is exterior, the door may lead to an unannotated room."""
    parts = [
        _space("Space Entry Lobby", _rect(20, 20, 200, 200)),
        _wall(_rect(20, 200, 200, 220), doors=_door(_rect(80, 200, 140, 220))),
        _wall(_rect(0, 0, 400, 10), external=True),
    ]
    record = _convert(tmp_path, parts)
    assert _entrance_types(record) == []
    assert any("skipped" in warning for warning in record.quality.warnings)


def test_plan_with_no_door_in_the_source_gets_no_entrance(tmp_path: Path) -> None:
    parts = [
        _space("Space Entry Lobby", _rect(20, 20, 200, 200)),
        _wall(_rect(20, 200, 200, 220), external=True),
    ]
    assert _entrance_types(_convert(tmp_path, parts)) == []
