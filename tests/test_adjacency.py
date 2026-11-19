"""Geometry-only opening and wall links between rooms of one floor."""

from __future__ import annotations

import pytest
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from vlm_floorplan.converters.adjacency import link_adjacent_rooms
from vlm_floorplan.converters.assemble import SourceFloor, SourceRoom
from vlm_floorplan.converters.msd import MSD_ADJACENCY
from vlm_floorplan.converters.swiss import SWISS_ADJACENCY
from vlm_floorplan.schema import LinkKind, RoomType


def _floor(*geometries: BaseGeometry) -> SourceFloor:
    rooms = [
        SourceRoom(geometry=g, type=RoomType.BEDROOM, source_type=str(i))
        for i, g in enumerate(geometries)
    ]
    return SourceFloor(number=0, rooms=rooms)


def test_a_corner_nub_does_not_hide_the_wall_two_rooms_run_along() -> None:
    # A and B run 3 m alongside a 0.2 m wall; a 0.1 m nub of B comes within 0.02 m
    # of A's corner, too short to be an opening.
    floor = _floor(box(0, 0, 4, 3), box(4.2, 0, 8, 3).union(box(4.02, 2.9, 4.2, 3)))

    link_adjacent_rooms(floor, SWISS_ADJACENCY, walls=box(4, 0, 4.2, 3))

    assert floor.links == {(0, 1): LinkKind.WALL}


def test_diagonal_rooms_a_wall_apart_are_not_linked_without_a_wall_layer() -> None:
    floor = _floor(box(0, 0, 4, 3), box(4.2, 3.2, 8, 6))

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {}


def test_rooms_side_by_side_a_wall_apart_are_linked_without_a_wall_layer() -> None:
    floor = _floor(box(0, 0, 4, 3), box(4.2, 0, 8, 3))

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {(0, 1): LinkKind.WALL}


@pytest.mark.parametrize(
    "rooms",
    [
        (box(0, 0, 4, 3), box(4, 0, 8, 3)),
        (box(0, 0, 4, 3), box(3.95, 0, 8, 3)),
        (box(0, 0, 0.1, 3), box(0.5, 0, 1, 3)),
    ],
    ids=["touching", "overlapping-5cm", "a-full-wall-gap-apart"],
)
def test_msd_rooms_sharing_a_3m_side_get_a_wall_link(
    rooms: tuple[BaseGeometry, BaseGeometry],
) -> None:
    floor = _floor(*rooms)

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {(0, 1): LinkKind.WALL}


def test_rooms_overlapping_10cm_at_an_offset_corner_of_a_wall_junction_are_not_linked() -> None:
    # B sits across the wall junction from A and overlaps A's side by only 0.1 m,
    # below the 0.5 m the rooms must run alongside each other.
    rooms = (box(0, 0, 4, 3), box(4.2, 2.9, 8, 6))
    walls = box(4, 0, 4.2, 3.2).union(box(0, 3, 8, 3.2))

    msd_floor, swiss_floor = _floor(*rooms), _floor(*rooms)
    link_adjacent_rooms(msd_floor, MSD_ADJACENCY)
    link_adjacent_rooms(swiss_floor, SWISS_ADJACENCY, walls=walls)

    assert (msd_floor.links, swiss_floor.links) == ({}, {})


def test_rooms_touching_only_at_a_corner_are_not_linked() -> None:
    floor = _floor(box(0, 0, 4, 3), box(4, 3, 8, 6))

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {}


def test_a_room_reaching_another_with_two_short_legs_is_not_linked() -> None:
    # Two 0.15 m legs of a U-shaped room, 0.5 m apart, end a wall above A:
    # 0.3 m of contact in total, so the 0.5 m between the legs must not count.
    legs = box(1, 3.2, 1.15, 6).union(box(1.65, 3.2, 1.8, 6)).union(box(0, 6, 4, 7))
    floor = _floor(box(0, 0, 4, 3), legs)

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {}


def test_rooms_along_a_curved_wall_built_from_short_facets_are_linked() -> None:
    # A circle of radius 3 and a ring from radius 3.2, both made of ~0.3 m facets,
    # run about 4.9 m alongside each other in the first quadrant.
    inner = Point(0, 0).buffer(3)
    ring = Point(0, 0).buffer(6).difference(Point(0, 0).buffer(3.2)).intersection(box(0, 0, 7, 7))
    floor = _floor(inner, ring)

    link_adjacent_rooms(floor, MSD_ADJACENCY)

    assert floor.links == {(0, 1): LinkKind.WALL}


def test_swiss_rooms_exactly_a_full_wall_gap_apart_with_wall_between_are_linked() -> None:
    floor = _floor(box(0, 0, 4, 3), box(4.4, 0, 8, 3))

    link_adjacent_rooms(floor, SWISS_ADJACENCY, walls=box(4, 0, 4.4, 3))

    assert floor.links == {(0, 1): LinkKind.WALL}
