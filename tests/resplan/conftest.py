"""A small synthetic ResPlan plan (no dataset files are committed).

Plan units, y pointing up. Interior walls are 3 thick, exterior walls 4.5.

    y=80  +----------+-------------+
          | bedroom  |  bathroom   |      bedroom and living: a door in the wall between them
    y=53  +----------+-------------+      bedroom and bathroom: wall only
    y=50  +--------+-+-------------+--+   living and kitchen: a 20-unit gap in the wall (opening)
          | living   | kitchen        |   front door on the living room's left wall
    y=0   +--------+-+----------------+
    y=-3  |  window |                      balcony below the living room, a window but no door
    y=-13 +--------+

Every layer is a shapely geometry, as in the real pickle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

WALL_DEPTH = 4.5


def _multi(*boxes: Polygon) -> MultiPolygon:
    return MultiPolygon(list(boxes))


def build_plan(plan_id: int = 1) -> dict[str, Any]:
    living = box(0, 0, 60, 50)
    kitchen = box(63, 0, 100, 50)
    bedroom = box(0, 53, 40, 80)
    bathroom = box(43, 53, 70, 80)
    balcony = box(0, -13, 30, -3)
    rooms = [living, kitchen, bedroom, bathroom]

    door = box(10, 50, 20, 53)
    window = box(10, -3, 20, 0)
    front_door = box(-WALL_DEPTH, 20, 0, 30)
    opening = box(60, 15, 63, 35)
    footprint = box(-WALL_DEPTH, -3, 100 + WALL_DEPTH, 80 + WALL_DEPTH)
    walls = footprint.difference(unary_union([*rooms, door, window, front_door, opening]))

    return {
        "id": plan_id,
        "wall_depth": WALL_DEPTH,
        "inner": footprint,
        "living": living,
        "kitchen": kitchen,
        "bedroom": _multi(bedroom),
        "bathroom": _multi(bathroom),
        "balcony": _multi(balcony),
        "wall": walls,
        "door": _multi(door),
        "window": _multi(window),
        "front_door": front_door,
        "land": box(-50, -50, 150, 130),
        "neighbor": (),
    }


@pytest.fixture
def plan() -> dict[str, Any]:
    return build_plan()


@pytest.fixture
def make_plan() -> Callable[[int], dict[str, Any]]:
    return build_plan
