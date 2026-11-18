from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import Point

from vlm_floorplan.resplan.convert import convert_plan
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.resplan.render import (
    PLAIN_ROOM_FILL,
    ROOM_FILLS,
    WALL,
    RenderStyle,
    RoomLabel,
    render_plan,
)
from vlm_floorplan.room_types import RESPLAN_TYPES
from vlm_floorplan.schema import GRID_MAX

COLOR_TOLERANCE = 3
"""Flat areas keep their color through resampling; allow a little for rounding."""


def _close(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    return all(abs(a - e) <= COLOR_TOLERANCE for a, e in zip(actual, expected, strict=True))


def _pixel_at_grid(image: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    x = min(image.width - 1, round(xy[0] / GRID_MAX * image.width))
    y = min(image.height - 1, round(xy[1] / GRID_MAX * image.height))
    return image.getpixel((x, y))  # type: ignore[return-value]


def test_image_size_matches_the_graph(plan: dict[str, Any]) -> None:
    graph = convert_plan(plan).graph
    image = render_plan(plan)
    assert image.size == (graph.image.w, graph.image.h)


def test_longer_side_follows_the_style(plan: dict[str, Any]) -> None:
    image = render_plan(plan, style=RenderStyle(long_side=300))
    assert max(image.size) == 300


def test_each_room_point_lands_on_that_room_in_the_image(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    image = render_plan(plan)
    for room in converted.graph.rooms():
        layer = converted.sources[room.id].source_type
        assert RESPLAN_TYPES[layer] == room.type
        assert _close(_pixel_at_grid(image, room.xy), ROOM_FILLS[layer]), room


def test_plain_style_uses_one_room_color(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    image = render_plan(plan, style=RenderStyle(fill_rooms=False))
    for room in converted.graph.rooms():
        assert _close(_pixel_at_grid(image, room.xy), PLAIN_ROOM_FILL)


def test_walls_are_drawn_where_the_plan_has_them(plan: dict[str, Any]) -> None:
    view = PlanView.for_plan(plan)
    image = render_plan(plan, view)
    # The middle of the left exterior wall (x from -4.5 to 0), at y = 40.
    x, y = view.pixel_frame().apply(Point(-2.25, 40)).coords[0]
    assert _close(image.getpixel((round(x), round(y))), WALL)  # type: ignore[arg-type]


def test_labels_are_drawn_over_the_plan(plan: dict[str, Any]) -> None:
    converted = convert_plan(plan)
    room = converted.graph.rooms()[0]
    plain = render_plan(plan)
    labeled = render_plan(plan, labels=[RoomLabel("Bedroom", room.xy)])
    assert plain.tobytes() != labeled.tobytes()
    # Text is dark: somewhere near the label's center a pixel differs strongly from the fill.
    cx = round(room.xy[0] / GRID_MAX * labeled.width)
    cy = round(room.xy[1] / GRID_MAX * labeled.height)
    box = labeled.crop((cx - 40, cy - 12, cx + 40, cy + 12))
    assert np.asarray(box).sum(axis=2).min() < 3 * 120
