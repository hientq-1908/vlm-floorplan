"""Geometry-only opening and wall links between rooms of one floor."""

from __future__ import annotations

import pytest
from shapely.geometry import box
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
