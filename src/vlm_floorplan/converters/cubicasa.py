"""CubiCasa5K plan (model.svg + F1_scaled.png) to `PlanRecord`.

CubiCasa5K draws rooms, walls and doors as polygons but stores no graph, so the
links are recovered from geometry. The rules (all distances in scaled-image
pixels, where walls are about 15 px thick):

- door link: grow the door polygon by 6 px and intersect it with every room;
  the two rooms with the largest overlap are linked. If only one room is hit
  and the door's wall is exterior, that room is an entrance.
- opening link: two rooms closer than 3 px (no wall between them) that face
  each other along at least 20 px of boundary, e.g. an open-plan kitchen and
  living room.
- wall link: two rooms within 30 px of each other, facing each other along at
  least 20 px of boundary, with walls covering at least 20 % of the gap between them.

"Facing" means straight across: a stretch of one room's outline counts only if
the other room lies ahead of it along its outward normal. Four rooms around a
wall cross are close to their diagonal neighbour too, but face it along nothing,
so the diagonal pairs stay unlinked.

Only doors whose parent group is a Wall count: CubiCasa5K also draws fridge and
cabinet doors, and 86 % of plans have at least one.

Multi-floor plans draw all floors side by side in one image, so one record
holds one graph per floor and never links rooms across floors.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

from vlm_floorplan.converters.assemble import SourceFloor, SourceRoom, assemble
from vlm_floorplan.converters.cubicasa_svg import SvgFloor, parse_svg
from vlm_floorplan.converters.facing import facing_length
from vlm_floorplan.geometry import GridFrame, largest_polygon
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.room_types import cubicasa_room_type
from vlm_floorplan.schema import ImageSize, LinkKind

DOOR_BUFFER_PX = 6.0
OPENING_MAX_GAP_PX = 3.0
WALL_MAX_GAP_PX = 30.0
MIN_SHARED_PX = 20.0
MIN_WALL_COVER = 0.2


def _rooms(svg_floor: SvgFloor, warnings: list[str]) -> list[SourceRoom]:
    rooms: list[SourceRoom] = []
    for space in svg_floor.spaces:
        polygon = space.polygon
        if not polygon.is_valid:
            repaired = largest_polygon(polygon)
            if repaired is None:
                warnings.append(f"floor {svg_floor.number}: dropped empty {space.space_class}")
                continue
            warnings.append(f"floor {svg_floor.number}: repaired invalid {space.space_class}")
            polygon = repaired
        label = _printed_label(space.label)
        rooms.append(
            SourceRoom(
                geometry=polygon,
                type=cubicasa_room_type(space.space_class, label),
                source_type=space.space_class,
                label=label,
            )
        )
    return rooms


def _printed_label(label: str | None) -> str | None:
    """The room name as printed, or None.

    Unlabeled rooms carry the annotation tool's placeholder "Undefined", which is
    not printed on the plan, so it must not become a label the model learns to read.
    """
    if label is None or label.strip().lower() == "undefined":
        return None
    return label


def _link_doors(floor: SourceFloor, svg_floor: SvgFloor, warnings: list[str]) -> None:
    for door in svg_floor.doors:
        zone = door.polygon.buffer(DOOR_BUFFER_PX)
        overlaps = [
            (zone.intersection(room.geometry).area, index) for index, room in enumerate(floor.rooms)
        ]
        hits = sorted((pair for pair in overlaps if pair[0] > 0), reverse=True)
        if len(hits) >= 2:
            floor.link(hits[0][1], hits[1][1], LinkKind.DOOR)
        elif len(hits) == 1 and door.wall.is_external:
            floor.rooms[hits[0][1]].entrance = True
        else:
            warnings.append(f"floor {svg_floor.number}: door touching {len(hits)} room(s) skipped")


def _wall_cover(a: Polygon, b: Polygon, walls: Polygon) -> float:
    """Share of the strip between two rooms that is covered by wall."""
    half = WALL_MAX_GAP_PX / 2
    strip = a.buffer(half).intersection(b.buffer(half)).difference(a.union(b))
    if strip.is_empty or strip.area == 0:
        return 0.0
    return strip.intersection(walls).area / strip.area


def _link_neighbours(floor: SourceFloor, svg_floor: SvgFloor) -> None:
    walls = unary_union([wall.polygon.buffer(0) for wall in svg_floor.walls])
    for i, j in combinations(range(len(floor.rooms)), 2):
        a, b = floor.rooms[i].geometry, floor.rooms[j].geometry
        distance = a.distance(b)
        if distance > WALL_MAX_GAP_PX:
            continue
        if (
            distance <= OPENING_MAX_GAP_PX
            and facing_length(a, b, OPENING_MAX_GAP_PX) >= MIN_SHARED_PX
        ):
            floor.link(i, j, LinkKind.OPENING)
        elif (
            facing_length(a, b, WALL_MAX_GAP_PX) >= MIN_SHARED_PX
            and _wall_cover(a, b, walls) >= MIN_WALL_COVER
        ):
            floor.link(i, j, LinkKind.WALL)


def _convert_floor(svg_floor: SvgFloor, warnings: list[str]) -> SourceFloor:
    floor = SourceFloor(number=svg_floor.number, rooms=_rooms(svg_floor, warnings))
    _link_doors(floor, svg_floor, warnings)  # doors first: a door outranks a shared wall
    _link_neighbours(floor, svg_floor)
    return floor


def convert_plan(plan_dir: Path, plan_id: str, split: str | None = None) -> PlanRecord:
    """Convert one CubiCasa5K plan directory (holding model.svg and F1_scaled.png).

    The record's `image` is the absolute path of `F1_scaled.png`; the build
    script rewrites it relative to the JSONL file.
    """
    image_path = plan_dir / "F1_scaled.png"
    with Image.open(image_path) as image:
        width, height = image.size
    warnings: list[str] = []
    floors = [
        _convert_floor(svg_floor, warnings) for svg_floor in parse_svg(plan_dir / "model.svg")
    ]
    floors = [floor for floor in floors if floor.rooms]
    if not floors:
        raise ValueError(f"{plan_id}: no rooms found in model.svg")
    assembled = assemble(
        floors, GridFrame.for_image(width, height), ImageSize(w=width, h=height), warnings
    )
    return PlanRecord(
        id=plan_id,
        source="cubicasa5k",
        split=split,
        image=str(image_path),
        graph=assembled.graph,
        rooms=assembled.rooms,
        quality=assembled.quality,
    )
