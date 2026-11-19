"""Swiss Dwellings converter: rooms, door and entrance detection, wall/opening links."""

from __future__ import annotations

import pytest
from swiss_fixture import apartment_rows, row

from vlm_floorplan.converters.swiss import convert_apartment
from vlm_floorplan.converters.swiss_rows import group_apartments
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import LinkKind, Room, RoomType


def _by_source(record: PlanRecord) -> dict[str, Room]:
    return {record.rooms[room.id].source_type: room for room in record.graph.rooms()}


def _kind(record: PlanRecord, a: str, b: str) -> LinkKind | None:
    rooms = _by_source(record)
    wanted = frozenset((rooms[a].id, rooms[b].id))
    for floor in record.graph.floors:
        for link in floor.links:
            if link.pair == wanted:
                return link.kind
    return None


def test_areas_become_rooms_and_shafts_are_dropped() -> None:
    record = convert_apartment(apartment_rows())

    assert record.id == "swiss_dwellings/s1/apt1"
    assert record.source == "swiss_dwellings" and record.image == ""
    rooms = _by_source(record)
    assert set(rooms) == {"CORRIDOR", "BEDROOM", "KITCHEN", "LIVING_ROOM", "DINING"}
    assert rooms["CORRIDOR"].type == RoomType.ENTRY
    assert rooms["LIVING_ROOM"].type == RoomType.LIVING


def test_a_door_polygon_links_the_two_rooms_it_overlaps() -> None:
    record = convert_apartment(apartment_rows())

    assert _kind(record, "CORRIDOR", "BEDROOM") == LinkKind.DOOR


def test_entrance_door_flags_the_room_it_opens_into() -> None:
    record = convert_apartment(apartment_rows())
    flagged = [record.rooms[room.id].source_type for room in record.graph.rooms() if room.entrance]

    assert flagged == ["CORRIDOR"]


def test_rooms_with_a_wall_between_get_a_wall_link() -> None:
    record = convert_apartment(apartment_rows())

    assert _kind(record, "CORRIDOR", "KITCHEN") == LinkKind.WALL
    assert _kind(record, "KITCHEN", "LIVING_ROOM") == LinkKind.WALL


def test_rooms_that_touch_get_an_opening_link() -> None:
    record = convert_apartment(apartment_rows())

    assert _kind(record, "LIVING_ROOM", "DINING") == LinkKind.OPENING


def test_rooms_a_gap_apart_with_no_wall_in_it_are_not_linked() -> None:
    rows = apartment_rows()
    # Remove the wall strip between hall and kitchen; other walls remain in the layer.
    rows = [r for r in rows if r["geometry"] != "POLYGON ((0 2, 3 2, 3 2.2, 0 2.2, 0 2))"]

    assert _kind(convert_apartment(rows), "CORRIDOR", "KITCHEN") is None


def test_windows_and_fixtures_are_ignored() -> None:
    record = convert_apartment(apartment_rows())

    assert len(record.graph.rooms()) == 5
    assert not record.quality.warnings


def test_entrance_door_touching_two_rooms_links_them_and_flags_the_non_stairwell() -> None:
    rows = apartment_rows()
    rows.append(row("area", "STAIRCASE", (-3.2, 0, -0.2, 2)))
    # A wider entrance door that reaches the apartment's own staircase.
    rows = [r for r in rows if r["entity_subtype"] != "ENTRANCE_DOOR"]
    rows.append(row("opening", "ENTRANCE_DOOR", (-0.4, 0.5, 0.2, 1.3)))
    record = convert_apartment(rows)
    flagged = [record.rooms[room.id].source_type for room in record.graph.rooms() if room.entrance]

    assert _kind(record, "CORRIDOR", "STAIRCASE") == LinkKind.DOOR
    assert flagged == ["CORRIDOR"]


def test_a_door_that_reaches_one_room_only_is_skipped_with_a_warning() -> None:
    rows = apartment_rows() + [row("opening", "DOOR", (20, 20, 20.4, 21))]
    record = convert_apartment(rows)

    assert any("door touching 0 room" in warning for warning in record.quality.warnings)


def test_duplex_floors_are_numbered_by_elevation_and_never_linked_to_each_other() -> None:
    lower = apartment_rows(floor="f-low")
    upper = [
        row("area", "BEDROOM", (0, 0, 3, 2), floor="f-up", elevation=2.7),
        row("area", "BATHROOM", (3.2, 0, 6, 2), floor="f-up", elevation=2.7),
    ]
    record = convert_apartment(upper + lower)  # file order must not decide the floor order

    assert [floor.floor for floor in record.graph.floors] == [1, 2]
    assert len(record.graph.floors[0].rooms) == 5 and len(record.graph.floors[1].rooms) == 2
    ids = [room.id for room in record.graph.rooms()]
    assert ids == [f"r{i}" for i in range(7)]  # ids continue across floors


def test_coordinates_fit_the_grid_with_y_flipped() -> None:
    record = convert_apartment(apartment_rows())
    rooms = _by_source(record)

    assert all(0 <= v <= 1000 for room in record.graph.rooms() for v in room.xy)
    assert max(record.graph.image.w, record.graph.image.h) == 1024
    assert rooms["KITCHEN"].xy[1] < rooms["CORRIDOR"].xy[1]  # kitchen is further up the plan


def test_unknown_area_subtype_is_kept_as_other_with_a_warning() -> None:
    rows = apartment_rows() + [row("area", "SPA", (10, 0, 13, 2))]
    record = convert_apartment(rows)

    assert _by_source(record)["SPA"].type == RoomType.OTHER
    assert any("SPA" in warning for warning in record.quality.warnings)


def test_unreadable_geometry_is_a_value_error() -> None:
    rows = apartment_rows() + [row("area", "BEDROOM", (0, 0, 1, 1), wkt="POLYGON ((broken")]

    with pytest.raises(ValueError, match="unreadable geometry"):
        convert_apartment(rows)


def test_an_apartment_without_rooms_is_an_error() -> None:
    with pytest.raises(ValueError, match="no rooms"):
        convert_apartment([row("opening", "WINDOW", (0, 0, 1, 1))])


def test_shared_spaces_belong_to_no_apartment() -> None:
    rows = apartment_rows() + [row("area", "STAIRCASE", (-3.2, 0, -0.2, 2), apartment="")]
    grouped = group_apartments(rows)

    assert list(grouped) == ["apt1"]
    assert all(r["entity_subtype"] != "STAIRCASE" for r in grouped["apt1"])
