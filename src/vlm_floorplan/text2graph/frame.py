"""The canonical frame: one source plan turned into the plan a text request is about.

A converter's plan lives in the frame of its source image: a different scale per axis, often
rotated, with source ids and labels. A user describing an apartment in words has none of
that, so every text-to-graph plan is first put into one frame:

1. polygons and points go to pixel space (the grid is per axis, so one unit of x is not one
   unit of y on a 1024 x 738 image),
2. the plan is rotated so its dominant wall direction is axis-aligned,
3. it is scaled uniformly into a 1000 x 1000 canvas with a margin and centered,
4. rooms are renumbered in reading order (top to bottom, then left to right) and links are
   sorted,
5. every room gets a size (the radius of a circle with its area), labels are dropped, and
   the exterior wall is the outline of the indoor rooms.

The result is a `PlanRecord` again, so the existing matching (`plan_score`, `evaluate_plan`)
works on it unchanged. It is a rigid motion plus a uniform scale, so every point stays inside
its room and the typed graph is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity, make_valid
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from vlm_floorplan.geometry import polygon_parts
from vlm_floorplan.record import PlanQuality, PlanRecord, RoomSource
from vlm_floorplan.schema import (
    GRID_MAX,
    MAX_ROOM_SIZE,
    Floor,
    FloorplanGraph,
    ImageSize,
    Link,
    Room,
    RoomType,
)

CANVAS_MARGIN = 60
"""Grid units kept free around the plan, so a terrace beside the wall still fits."""
WALL_CLOSING = 25.0
"""Gaps up to twice this between indoor rooms (walls, thin shafts) are closed in the outline."""
WALL_SIMPLIFY = 6.0
"""Outline vertices closer than this to a straight line are dropped."""
MIN_ROOMS = 3
MAX_ROOMS = 16
"""Plans the model is asked to draw: from a flat's core rooms up to a large house."""


class FrameError(ValueError):
    """A plan cannot be put into the canonical frame (for example it has no polygons)."""


@dataclass(frozen=True)
class Canonical:
    """A plan in the canonical frame, and what was done to get there."""

    record: PlanRecord
    rotation_deg: float
    """Rotation applied, degrees counter-clockwise on screen (y down), in (-45, 45]."""
    scale: float
    """Canvas units per source pixel."""


def unusable_reason(record: PlanRecord) -> str | None:
    """Why a plan cannot be a text-to-graph target, or None when it can.

    A usable plan is one floor with 3 to 16 rooms, every room typed, no room without a link,
    at least one entrance, and a living room or a bedroom plus a kitchen or a living room, so
    a description of it names a home and not a storage-plus-garage shed.
    """
    if len(record.graph.floors) != 1:
        return "several floors"
    floor = record.graph.floors[0]
    types = [room.type for room in floor.rooms]
    if not MIN_ROOMS <= len(types) <= MAX_ROOMS:
        return "room count outside 3 to 16"
    if RoomType.UNKNOWN in types:
        return "an unknown room"
    linked = {end for link in floor.links for end in (link.a, link.b)}
    if any(room.id not in linked for room in floor.rooms):
        return "an isolated room"
    if not any(room.entrance for room in floor.rooms):
        return "no entrance"
    if RoomType.LIVING not in types and RoomType.BEDROOM not in types:
        return "no living room or bedroom"
    if RoomType.KITCHEN not in types and RoomType.LIVING not in types:
        return "no kitchen or living room"
    if record.source == "swiss_dwellings" and RoomType.LIVING not in types:
        return "no living room (Swiss Dwellings types undesignated rooms as bedrooms)"
    if any(room_id not in record.rooms for room_id in (room.id for room in floor.rooms)):
        return "a room without a polygon"
    return None


def _room_geometry(record: PlanRecord, room_id: str) -> BaseGeometry:
    """A room's rings as one valid geometry, in pixel space of the source image."""
    image = record.graph.image
    parts = []
    for ring in record.rooms[room_id].polygon:
        points = [(x * image.w / GRID_MAX, y * image.h / GRID_MAX) for x, y in ring]
        if len(points) >= 4:
            parts.append(make_valid(Polygon(points)))
    geometry = unary_union(parts) if parts else Polygon()
    if geometry.is_empty or geometry.area <= 0:
        raise FrameError(f"{record.id}: room {room_id} has no area")
    return geometry


def dominant_angle(geometries: list[BaseGeometry]) -> float:
    """The wall direction modulo 90 degrees, in (-45, 45], weighted by edge length.

    Angles are averaged as 4 theta on the unit circle, so 0 and 90 degrees (the same wall
    direction) agree and a plan with walls at both 10 and 100 degrees reads as 10.
    """
    sin_sum = cos_sum = 0.0
    for geometry in geometries:
        for part in polygon_parts(geometry):
            coords = list(part.exterior.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
                length = math.hypot(x1 - x0, y1 - y0)
                theta = math.atan2(y1 - y0, x1 - x0)
                sin_sum += length * math.sin(4 * theta)
                cos_sum += length * math.cos(4 * theta)
    if sin_sum == 0.0 and cos_sum == 0.0:
        return 0.0
    degrees = math.degrees(math.atan2(sin_sum, cos_sum) / 4)
    return 45.0 if degrees == -45.0 else degrees


def _point_in(geometry: BaseGeometry, x: float, y: float) -> tuple[int, int]:
    """The point (x, y) rounded, or the geometry's representative point if rounding leaves it."""
    rounded = (round(x), round(y))
    if geometry.contains(Point(rounded)):
        return rounded
    inside = geometry.representative_point()
    return round(inside.x), round(inside.y)


def _round_rings(geometry: BaseGeometry) -> list[list[tuple[float, float]]]:
    return [
        [(round(x, 1), round(y, 1)) for x, y in part.exterior.coords]
        for part in polygon_parts(geometry)
    ]


def exterior_wall(indoor: list[BaseGeometry]) -> list[tuple[int, int]]:
    """The outline of the indoor rooms as a short polygon of grid points.

    Rooms are merged with a closing so walls between them vanish; a plan that stays in
    several pieces gets the outline of its convex hull, so every indoor room is inside.
    """
    merged = unary_union(indoor).buffer(WALL_CLOSING).buffer(-WALL_CLOSING)
    parts = polygon_parts(merged)
    outline = parts[0] if len(parts) == 1 else MultiPolygon(parts).convex_hull
    simple = outline.simplify(WALL_SIMPLIFY, preserve_topology=True)
    points: list[tuple[int, int]] = []
    for x, y in list(simple.exterior.coords)[:-1]:
        point = (min(GRID_MAX, max(0, round(x))), min(GRID_MAX, max(0, round(y))))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise FrameError("the exterior wall collapsed to fewer than 3 points")
    return points


def canonicalize(record: PlanRecord) -> Canonical:
    """Put a usable plan into the canonical frame. Raises `FrameError` when it cannot."""
    reason = unusable_reason(record)
    if reason is not None:
        raise FrameError(f"{record.id}: {reason}")
    floor = record.graph.floors[0]
    pixel = {room.id: _room_geometry(record, room.id) for room in floor.rooms}
    degrees = dominant_angle(list(pixel.values()))
    image = record.graph.image

    def move(geometry: BaseGeometry) -> BaseGeometry:
        return affinity.rotate(geometry, -degrees, origin=(0, 0))

    rotated = {room_id: move(geometry) for room_id, geometry in pixel.items()}
    min_x, min_y, max_x, max_y = unary_union(list(rotated.values())).bounds
    extent = max(max_x - min_x, max_y - min_y)
    if extent <= 0:
        raise FrameError(f"{record.id}: the plan has no extent")
    scale = (GRID_MAX - 2 * CANVAS_MARGIN) / extent
    shift_x = (GRID_MAX - (max_x - min_x) * scale) / 2 - min_x * scale
    shift_y = (GRID_MAX - (max_y - min_y) * scale) / 2 - min_y * scale
    matrix = [scale, 0.0, 0.0, scale, shift_x, shift_y]
    placed = {room_id: affinity.affine_transform(g, matrix) for room_id, g in rotated.items()}

    drafts = []
    for room in floor.rooms:
        geometry = placed[room.id]
        # The source point moves with the plan; the point is not recomputed from the shape.
        px, py = room.xy[0] * image.w / GRID_MAX, room.xy[1] * image.h / GRID_MAX
        moved = move(Point(px, py))
        x, y = moved.x * scale + shift_x, moved.y * scale + shift_y
        point = _point_in(geometry, x, y)
        size = min(MAX_ROOM_SIZE, max(1, round(math.sqrt(geometry.area / math.pi))))
        drafts.append((room, geometry, point, size))
    drafts.sort(key=lambda item: (item[2][1], item[2][0], item[0].id))
    new_id = {room.id: f"r{index}" for index, (room, *_rest) in enumerate(drafts)}

    rooms = [
        Room(
            id=new_id[room.id],
            type=room.type,
            xy=point,
            size=size,
            entrance=room.entrance,
        )
        for room, _geometry, point, size in drafts
    ]
    links = sorted(
        (
            Link(
                a=min(new_id[link.a], new_id[link.b], key=_index),
                b=max(new_id[link.a], new_id[link.b], key=_index),
                kind=link.kind,
            )
            for link in floor.links
        ),
        key=lambda link: (_index(link.a), _index(link.b)),
    )
    indoor = [geometry for room, geometry, _p, _s in drafts if room.type != RoomType.TERRACE]
    wall = exterior_wall(indoor or [geometry for _r, geometry, _p, _s in drafts])
    graph = FloorplanGraph(
        floors=[Floor(floor=1, rooms=rooms, links=links)],
        image=ImageSize(w=GRID_MAX, h=GRID_MAX),
        exterior_wall=wall,
    )
    sources = {
        new_id[room.id]: RoomSource(
            source_type=record.rooms[room.id].source_type, polygon=_round_rings(geometry)
        )
        for room, geometry, _p, _s in drafts
    }
    canonical = PlanRecord(
        id=record.id,
        source=record.source,
        split=record.split,
        image="",
        graph=graph,
        rooms=sources,
        quality=PlanQuality.from_graph(graph, []),
    )
    return Canonical(record=canonical, rotation_deg=round(degrees, 2), scale=scale)


def _index(room_id: str) -> int:
    return int(room_id[1:])


def indoor_points_inside_wall(graph: FloorplanGraph) -> bool:
    """True when every non-terrace room's point lies inside the exterior wall."""
    if graph.exterior_wall is None:
        return True
    wall = Polygon(graph.exterior_wall)
    return all(wall.contains(Point(room.xy)) for room in graph.rooms() if room.type != "terrace")
