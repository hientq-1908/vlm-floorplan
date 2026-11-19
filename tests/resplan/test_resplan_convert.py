from __future__ import annotations

from typing import Any

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from vlm_floorplan.resplan.convert import ConvertedPlan, convert_plan
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.schema import GRID_MAX, LinkKind, RoomType


def _room_of_type(converted: ConvertedPlan, room_type: RoomType) -> str:
    matches = [room.id for room in converted.graph.rooms() if room.type == room_type]
    assert len(matches) == 1, f"expected one {room_type} room, got {matches}"
    return matches[0]


def _link_kind(converted: ConvertedPlan, a: RoomType, b: RoomType) -> LinkKind | None:
    ids = {_room_of_type(converted, a), _room_of_type(converted, b)}
    for link in converted.graph.floors[0].links:
        if {link.a, link.b} == ids:
            return link.kind
    return None


def test_rooms_are_mapped_to_coarse_types(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    types = sorted(room.type for room in converted.graph.rooms())
    assert types == sorted(
        [RoomType.LIVING, RoomType.KITCHEN, RoomType.BEDROOM, RoomType.BATHROOM, RoomType.OUTDOOR]
    )


def test_front_door_is_not_a_room(plan: dict[str, Any]) -> None:
    assert len(convert_plan(plan).graph.rooms()) == 5


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (RoomType.BEDROOM, RoomType.LIVING, LinkKind.DOOR),
        (RoomType.LIVING, RoomType.KITCHEN, LinkKind.OPENING),
        (RoomType.BEDROOM, RoomType.BATHROOM, LinkKind.WALL),
        (RoomType.BATHROOM, RoomType.LIVING, LinkKind.WALL),
        (RoomType.BATHROOM, RoomType.KITCHEN, LinkKind.WALL),
        # A window is not a passage, so the balcony is only walled off from the living room.
        (RoomType.OUTDOOR, RoomType.LIVING, LinkKind.WALL),
        # Far apart: nothing to link.
        (RoomType.BEDROOM, RoomType.KITCHEN, None),
    ],
)
def test_link_kind_follows_the_geometry(
    plan: dict[str, Any], a: RoomType, b: RoomType, expected: LinkKind | None
) -> None:
    assert _link_kind(convert_plan(plan), a, b) == expected


def test_front_door_marks_the_room_it_opens_into(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    entrances = [room.type for room in converted.graph.rooms() if room.entrance]
    assert entrances == [RoomType.LIVING]


def test_front_door_touching_two_rooms_marks_only_the_larger_overlap(plan: dict[str, Any]) -> None:
    # A door in the wall band, mostly facing the living room, that also nicks a balcony corner.
    plan["front_door"] = box(-4.5, -0.3, 0, 8)
    plan["balcony"] = MultiPolygon([box(-10, -13, 30, -0.5)])
    assert plan["front_door"].buffer(1.0).intersects(plan["balcony"])
    entrances = [room.type for room in convert_plan(plan).graph.rooms() if room.entrance]
    assert entrances == [RoomType.LIVING]


def test_door_touching_a_single_room_adds_no_link(plan: dict[str, Any]) -> None:
    before = convert_plan(plan).graph.floors[0].links
    plan["door"] = MultiPolygon([*plan["door"].geoms, box(100, 20, 104.5, 30)])
    assert convert_plan(plan).graph.floors[0].links == before


def test_window_between_two_interior_rooms_becomes_a_door(plan: dict[str, Any]) -> None:
    # ResPlan sometimes draws a door as a window; its own converter relabels those as doors.
    plan["window"] = MultiPolygon([*plan["window"].geoms, box(40, 60, 43, 70)])
    assert _link_kind(convert_plan(plan), RoomType.BEDROOM, RoomType.BATHROOM) == LinkKind.DOOR


def test_living_parts_are_merged_into_one_room(plan: dict[str, Any]) -> None:
    plan["living"] = MultiPolygon([box(0, 0, 30, 50), box(30, 0, 60, 50)])
    converted = convert_plan(plan)
    assert len([r for r in converted.graph.rooms() if r.type == RoomType.LIVING]) == 1


def test_split_non_living_rooms_stay_separate(plan: dict[str, Any]) -> None:
    plan["bathroom"] = MultiPolygon([box(43, 53, 55, 80), box(58, 53, 70, 80)])
    converted = convert_plan(plan)
    assert len([r for r in converted.graph.rooms() if r.type == RoomType.BATHROOM]) == 2


def test_each_room_point_lies_inside_its_polygon(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    for room in converted.graph.rooms():
        rings = converted.sources[room.id].polygon
        assert any(Polygon(ring).contains(Point(room.xy)) for ring in rings), room


def test_grid_y_points_down_like_the_image(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    y_of = {room.type: room.xy[1] for room in converted.graph.rooms()}
    assert y_of[RoomType.BEDROOM] < y_of[RoomType.LIVING] < y_of[RoomType.OUTDOOR]


def test_coordinates_stay_on_the_grid(plan: dict[str, Any]) -> None:
    for room in convert_plan(plan).graph.rooms():
        assert room.bbox is not None
        assert all(0 <= value <= GRID_MAX for value in (*room.xy, *room.bbox))


def test_ids_follow_reading_order(plan: dict[str, Any]) -> None:
    rooms = convert_plan(plan).graph.rooms()
    keys = [(room.xy[1], room.xy[0]) for room in rooms]
    assert keys == sorted(keys)
    assert [room.id for room in rooms] == [f"r{i}" for i in range(len(rooms))]


def test_image_size_comes_from_the_shared_view(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    assert converted.graph.image == PlanView.for_plan(plan).image


def test_sources_keep_the_resplan_category(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    bedroom = _room_of_type(converted, RoomType.BEDROOM)
    assert converted.sources[bedroom].source_type == "bedroom"


def test_plan_without_rooms_is_rejected() -> None:
    with pytest.raises(ValueError, match="no rooms"):
        convert_plan({"id": 9, "inner": box(0, 0, 10, 10)})


def test_missing_front_door_is_a_warning(plan: dict[str, Any]) -> None:
    del plan["front_door"]
    converted = convert_plan(plan)
    assert "plan has no front door" in converted.warnings
    assert not any(room.entrance for room in converted.graph.rooms())
