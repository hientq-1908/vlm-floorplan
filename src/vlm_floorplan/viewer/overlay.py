"""Draw a room graph on top of its floorplan image.

This is the main way to check a converter by eye:

- each room is a dot at its `xy` point, colored by room type and captioned with
  its id and type (hollow grey for `unknown`), plus a thin box if it has a `bbox`;
- each link is a line between the two room points, styled by kind:
  doors are thick orange lines, openings are dashed blue, walls are thin dotted grey;
- a room with an exterior door gets a green ring.

A dot that sits in the wrong room, or a door line crossing three rooms,
is a converter bug that no unit test would catch as quickly as a glance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from vlm_floorplan.schema import GRID_MAX, FloorplanGraph, LinkKind, Room, RoomType

Color = tuple[int, int, int]

TYPE_COLORS: Final[dict[RoomType, Color]] = {
    RoomType.LIVING: (230, 80, 80),
    RoomType.KITCHEN: (245, 160, 40),
    RoomType.DINING: (200, 170, 40),
    RoomType.BEDROOM: (70, 110, 220),
    RoomType.BATHROOM: (40, 180, 190),
    RoomType.ENTRY: (150, 90, 200),
    RoomType.STORAGE: (140, 100, 60),
    RoomType.UTILITY: (100, 130, 100),
    RoomType.OUTDOOR: (60, 170, 80),
    RoomType.GARAGE: (90, 90, 90),
    RoomType.STAIRS: (210, 100, 170),
    RoomType.OTHER: (60, 60, 60),
    RoomType.UNKNOWN: (160, 160, 160),
}


@dataclass(frozen=True)
class LinkStyle:
    """How one link kind is drawn. Lengths are in drawing units (see `UNITS_PER_IMAGE_WIDTH`)."""

    color: Color
    width: float
    dash: tuple[float, float] | None
    """(dash length, gap length), or None for a solid line."""


LINK_STYLES: Final[dict[LinkKind, LinkStyle]] = {
    LinkKind.DOOR: LinkStyle((235, 110, 20), 0.5, None),
    LinkKind.OPENING: LinkStyle((30, 120, 230), 0.35, (3.0, 2.0)),
    LinkKind.WALL: LinkStyle((120, 120, 120), 0.2, (0.6, 1.4)),
}
ENTRANCE_COLOR: Final[Color] = (20, 160, 60)

UNITS_PER_IMAGE_WIDTH: Final = 400
"""One drawing unit is image width / 400, so overlays look alike at any resolution."""


def _to_pixels(room: Room, size: tuple[int, int]) -> tuple[float, float]:
    return room.xy[0] * size[0] / GRID_MAX, room.xy[1] * size[1] / GRID_MAX


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    style: LinkStyle,
    scale: float,
) -> None:
    """Draw a line from start to end; `style.dash` is (dash length, gap length) in scale units."""
    width = max(1, round(style.width * scale))
    if style.dash is None:
        draw.line([start, end], fill=style.color, width=width)
        return
    dash, gap = style.dash[0] * scale, style.dash[1] * scale
    length = math.dist(start, end)
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        a = (
            start[0] + (end[0] - start[0]) * position / length,
            start[1] + (end[1] - start[1]) * position / length,
        )
        b = (
            start[0] + (end[0] - start[0]) * stop / length,
            start[1] + (end[1] - start[1]) * stop / length,
        )
        draw.line([a, b], fill=style.color, width=width)
        position += dash + gap


def _draw_links(
    draw: ImageDraw.ImageDraw, graph: FloorplanGraph, size: tuple[int, int], scale: float
) -> None:
    # Weakest first, so the strong door lines end up on top.
    order = [LinkKind.WALL, LinkKind.OPENING, LinkKind.DOOR]
    for kind in order:
        for floor in graph.floors:
            rooms = {room.id: room for room in floor.rooms}
            for link in floor.links:
                if link.kind == kind:
                    _dashed_line(
                        draw,
                        _to_pixels(rooms[link.a], size),
                        _to_pixels(rooms[link.b], size),
                        LINK_STYLES[kind],
                        scale,
                    )


def _draw_room(
    draw: ImageDraw.ImageDraw,
    room: Room,
    size: tuple[int, int],
    scale: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = _to_pixels(room, size)
    color = TYPE_COLORS[room.type]
    if room.bbox is not None:
        x1, y1, x2, y2 = (v * s / GRID_MAX for v, s in zip(room.bbox, size * 2, strict=True))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, round(0.2 * scale)))
    radius = 1.6 * scale
    if room.entrance:
        ring = radius + 1.2 * scale
        draw.ellipse(
            [x - ring, y - ring, x + ring, y + ring],
            outline=ENTRANCE_COLOR,
            width=max(2, round(0.5 * scale)),
        )
    fill = None if room.type == RoomType.UNKNOWN else color
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=fill,
        outline=color,
        width=max(2, round(0.4 * scale)),
    )
    caption = f"{room.id} {room.type.value}"
    anchor = (x + radius + 0.6 * scale, y - radius)
    draw.text(
        anchor,
        caption,
        fill=(255, 255, 255),
        font=font,
        stroke_width=max(1, round(0.4 * scale)),
        stroke_fill=color,
    )


def draw_overlay(image: Image.Image, graph: FloorplanGraph) -> Image.Image:
    """Return a copy of `image` with the graph drawn on it."""
    canvas = image.convert("RGB")
    scale = canvas.width / UNITS_PER_IMAGE_WIDTH
    font = ImageFont.load_default(size=max(10, round(2.4 * scale)))
    draw = ImageDraw.Draw(canvas)
    _draw_links(draw, graph, canvas.size, scale)
    for room in graph.rooms():
        _draw_room(draw, room, canvas.size, scale, font)
    return canvas
