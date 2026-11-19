"""Draw a ResPlan plan as a plain floorplan PNG.

ResPlan has no images, so this is where the rendered pictures come from.
They are synthetic: no door swings, text, furniture or paper texture.
Use them as perception augmentation next to real plans (CubiCasa5K), not instead of them.

The picture is fitted with a `PlanView`, the same object the converter uses for its
grid coordinates. A graph point (x, y) on the 0-1000 grid is therefore at pixel
(x / 1000 * width, y / 1000 * height) in the image this module returns.

Polygons are painted through masks because shapely polygons have holes
(the wall layer is a ring around the rooms) and Pillow's polygon call has none.
The canvas is drawn at twice the size and scaled down once, for smooth edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.resplan.links import ROOM_LAYERS
from vlm_floorplan.resplan.load import polygons
from vlm_floorplan.schema import GRID_MAX

Color = tuple[int, int, int]

SUPERSAMPLE = 2

BACKGROUND: Color = (255, 255, 255)
WALL: Color = (45, 45, 45)
WINDOW: Color = (176, 214, 235)
DOOR: Color = (255, 255, 255)
DOOR_OUTLINE: Color = (120, 120, 120)
FRONT_DOOR_OUTLINE: Color = (150, 90, 40)
OUTLINE_WIDTH = 2
"""Door and window outline width in pixels of the supersampled canvas."""
LABEL_COLOR: Color = (60, 60, 60)
LABEL_SIZE_FRACTION = 1 / 50
"""Label font size as a fraction of the image's longer side."""

# Soft fills so that room boundaries read clearly without competing with the walls.
ROOM_FILLS: dict[str, Color] = {
    "living": (247, 241, 226),
    "kitchen": (230, 238, 247),
    "bedroom": (232, 244, 235),
    "bathroom": (250, 232, 224),
    "balcony": (236, 236, 236),
    "storage": (245, 233, 240),
    "stair": (232, 230, 245),
}
PLAIN_ROOM_FILL: Color = (250, 250, 250)


@dataclass(frozen=True)
class RenderStyle:
    long_side: int = 1024
    """Pixels along the longer image side."""
    fill_rooms: bool = True
    """Tint rooms by type. False draws every room in the same near-white."""


@dataclass(frozen=True)
class RoomLabel:
    text: str
    xy: tuple[int, int]
    """Grid position (0-1000) of the text's center."""


def render_plan(
    plan: dict[str, Any],
    view: PlanView | None = None,
    style: RenderStyle | None = None,
    labels: list[RoomLabel] | None = None,
) -> Image.Image:
    """Render `plan` to an RGB image. `labels` are drawn on top when given."""
    style = style or RenderStyle()
    view = view or PlanView.for_plan(plan, long_side=style.long_side)
    size = (view.image.w * SUPERSAMPLE, view.image.h * SUPERSAMPLE)
    canvas = Image.new("RGB", size, BACKGROUND)

    for layer in ROOM_LAYERS:
        fill = ROOM_FILLS[layer] if style.fill_rooms else PLAIN_ROOM_FILL
        _paint(canvas, view, plan.get(layer), fill)
    _paint(canvas, view, plan.get("wall"), WALL)
    _paint(canvas, view, plan.get("window"), WINDOW, outline=WALL)
    _paint(canvas, view, plan.get("door"), DOOR, outline=DOOR_OUTLINE)
    _paint(canvas, view, plan.get("front_door"), DOOR, outline=FRONT_DOOR_OUTLINE)

    image = canvas.reduce(SUPERSAMPLE)
    if labels:
        _draw_labels(image, labels)
    return image


def _canvas_ring(ring_coords: Any, view: PlanView) -> list[tuple[float, float]]:
    """A ring in plan units -> coordinates on the supersampled canvas."""
    frame = view.pixel_frame()
    return [
        (
            (frame.scale_x * x + frame.offset_x) * SUPERSAMPLE,
            (frame.scale_y * y + frame.offset_y) * SUPERSAMPLE,
        )
        for x, y in ring_coords
    ]


def _paint(
    canvas: Image.Image,
    view: PlanView,
    layer: Any,
    fill: Color,
    outline: Color | None = None,
) -> None:
    """Fill every polygon of a layer, holes included, and optionally outline it."""
    parts = sorted(polygons(layer), key=lambda part: part.area, reverse=True)
    if not parts:
        return
    shapes = [
        (
            _canvas_ring(part.exterior.coords, view),
            [_canvas_ring(hole.coords, view) for hole in part.interiors],
        )
        for part in parts
    ]
    # The mask only covers the layer's bounding box: a room is a small part of the canvas.
    xs = [x for exterior, _ in shapes for x, _ in exterior]
    ys = [y for exterior, _ in shapes for _, y in exterior]
    left, top = max(0, floor(min(xs))), max(0, floor(min(ys)))
    right, bottom = min(canvas.width, ceil(max(xs)) + 1), min(canvas.height, ceil(max(ys)) + 1)
    if right <= left or bottom <= top:
        return
    mask = Image.new("L", (right - left, bottom - top), 0)
    mask_draw = ImageDraw.Draw(mask)
    for exterior, holes in shapes:
        mask_draw.polygon([(x - left, y - top) for x, y in exterior], fill=255)
        for hole in holes:
            mask_draw.polygon([(x - left, y - top) for x, y in hole], fill=0)
    canvas.paste(fill, box=(left, top), mask=mask)
    if outline is not None:
        draw = ImageDraw.Draw(canvas)
        for exterior, _ in shapes:
            draw.line(exterior, fill=outline, width=OUTLINE_WIDTH)


def _draw_labels(image: Image.Image, labels: list[RoomLabel]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(8, round(max(image.size) * LABEL_SIZE_FRACTION)))
    for label in labels:
        x = label.xy[0] / GRID_MAX * image.width
        y = label.xy[1] / GRID_MAX * image.height
        draw.text((x, y), label.text, fill=LABEL_COLOR, font=font, anchor="mm")
