"""One Swiss Dwellings apartment to a `PlanRecord`.

Swiss Dwellings stores geometry, not a graph, so links are recovered the same
way as for CubiCasa5K: from the doors' and rooms' polygons.

- A `DOOR` polygon overlaps the two rooms it connects: those become a `door` link.
- An `ENTRANCE_DOOR` touches one room of the apartment: that room gets the
  `entrance` flag. The other side is the shared stairwell, which is not part of
  any apartment. If it touches two rooms of the apartment, they are linked and the
  flag goes on the one that is not a stairwell.
- Rooms that touch with no wall between them get an `opening` link, rooms a wall apart a
  `wall` link (see `adjacency.py`); the wall layer must actually fill the gap.
- Windows, fixtures (`feature`), columns and railings are ignored: the graph
  has no use for them.

One apartment is one record. Duplexes span several `floor_id`s and become several
floors, numbered 1, 2, ... from the lowest, because the dataset gives elevations
(in metres) but no storey numbers. The floors share one coordinate frame, so a
renderer should draw each floor on its own image.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from vlm_floorplan.converters.adjacency import AdjacencyRules, link_adjacent_rooms
from vlm_floorplan.converters.assemble import SourceFloor, SourceRoom, assemble
from vlm_floorplan.converters.swiss_rows import Row
from vlm_floorplan.geometry import largest_polygon
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.room_types import SWISS_NON_ROOM_AREAS, SWISS_TYPES
from vlm_floorplan.schema import LinkKind, RoomType

SOURCE_NAME = "swiss_dwellings"

DOOR_REACHES_M = (0.05, 0.15, 0.3, 0.5)
"""How far a door polygon is grown to find its rooms, tried in order until it finds enough.

Most door polygons already overlap both rooms, but in thick walls (entrance doors especially)
the polygon stops 0.1-0.5 m short of a room. Growing in steps keeps the first, tightest
match, so a door in a thin wall is not linked to a room one wall further away.
"""
MIN_DOOR_OVERLAP_M2 = 1e-6
# Thresholds in metres. Door polygons are 0.17 m thick (median); a wall between rooms is
# 0.1-0.4 m. Rooms of one apartment that touch are at most 4 cm apart.
SWISS_ADJACENCY = AdjacencyRules(opening_gap=0.04, wall_gap=0.4, min_shared=0.5, min_wall_cover=0.2)


def _polygon(row: Row, warnings: list[str]) -> BaseGeometry | None:
    try:
        geometry = largest_polygon(wkt.loads(row["geometry"]))
    except GEOSException as err:
        raise ValueError(f"unreadable geometry of {row['entity_subtype']}: {err}") from err
    if geometry is None:
        warnings.append(f"{row['entity_subtype']} {row['area_id']}: dropped empty geometry")
    return geometry


def _rooms(rows: Sequence[Row], warnings: list[str]) -> list[SourceRoom]:
    rooms: list[SourceRoom] = []
    for row in rows:
        subtype = row["entity_subtype"]
        if row["entity_type"] != "area" or subtype in SWISS_NON_ROOM_AREAS:
            continue
        geometry = _polygon(row, warnings)
        if geometry is None:
            continue
        if subtype not in SWISS_TYPES:
            warnings.append(f"unknown area subtype {subtype!r} kept as 'other'")
        room_type = SWISS_TYPES.get(subtype, RoomType.OTHER)
        rooms.append(SourceRoom(geometry=geometry, type=room_type, source_type=subtype))
    return rooms


def _hits(door: BaseGeometry, rooms: list[SourceRoom], needed: int) -> list[int]:
    """Indices of the rooms the door reaches, most overlap first, at least `needed` if possible."""
    hits: list[int] = []
    for reach in DOOR_REACHES_M:
        zone = door.buffer(reach)
        overlaps = [(zone.intersection(room.geometry).area, i) for i, room in enumerate(rooms)]
        hits = [i for area, i in sorted(overlaps, reverse=True) if area > MIN_DOOR_OVERLAP_M2]
        if len(hits) >= needed:
            break
    return hits


def _link_doors(floor: SourceFloor, rows: Sequence[Row], warnings: list[str]) -> None:
    for row in rows:
        subtype = row["entity_subtype"]
        if row["entity_type"] != "opening" or subtype not in ("DOOR", "ENTRANCE_DOOR"):
            continue
        door = _polygon(row, warnings)
        if door is None:
            continue
        if subtype == "ENTRANCE_DOOR":
            _apply_entrance(floor, _hits(door, floor.rooms, needed=1), warnings)
            continue
        hits = _hits(door, floor.rooms, needed=2)
        if len(hits) >= 2:
            floor.link(hits[0], hits[1], LinkKind.DOOR)
        else:
            warnings.append(f"door touching {len(hits)} room(s) skipped")


def _apply_entrance(floor: SourceFloor, hits: list[int], warnings: list[str]) -> None:
    if not hits:
        warnings.append("entrance door touching no room skipped")
        return
    if len(hits) >= 2:
        floor.link(hits[0], hits[1], LinkKind.DOOR)
        hits = [i for i in hits[:2] if floor.rooms[i].type != RoomType.STAIRS] or hits[:1]
    floor.rooms[hits[0]].entrance = True


def _wall_layer(rows: Sequence[Row]) -> BaseGeometry | None:
    """All wall polygons of a floor as one geometry, or None if the floor has none."""
    try:
        shapes = [
            wkt.loads(row["geometry"]).buffer(0) for row in rows if row["entity_subtype"] == "WALL"
        ]
    except GEOSException as err:
        raise ValueError(f"unreadable wall geometry: {err}") from err
    return unary_union(shapes) if shapes else None


def _convert_floor(
    number: int, rows: Sequence[Row], warnings: list[str]
) -> tuple[SourceFloor, BaseGeometry | None]:
    floor = SourceFloor(number=number, rooms=_rooms(rows, warnings))
    _link_doors(floor, rows, warnings)  # doors first: a door outranks a shared wall
    walls = _wall_layer(rows)
    link_adjacent_rooms(floor, SWISS_ADJACENCY, walls)
    return floor, walls


def _floors_in_elevation_order(rows: Sequence[Row]) -> list[list[Row]]:
    """Rows grouped by floor, lowest floor first (by the lowest room elevation)."""
    by_floor: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_floor[row["floor_id"]].append(row)

    def elevation(floor_rows: list[Row]) -> float:
        areas = [float(row["elevation"]) for row in floor_rows if row["entity_type"] == "area"]
        return min(areas, default=0.0)

    return sorted(by_floor.values(), key=elevation)


def convert_apartment(rows: Sequence[Row]) -> PlanRecord:
    """Convert all rows of one apartment. Raises `ValueError` if it has no rooms."""
    apartment_id, site_id = rows[0]["apartment_id"], rows[0]["site_id"]
    warnings: list[str] = []
    converted = [
        _convert_floor(number, floor_rows, warnings)
        for number, floor_rows in enumerate(_floors_in_elevation_order(rows), start=1)
    ]
    floors = [floor for floor, _ in converted if floor.rooms]
    if not floors:
        raise ValueError(f"apartment {apartment_id}: no rooms")
    # Fit the view around rooms and walls, so the outer walls are inside the picture.
    shapes = [room.geometry for floor in floors for room in floor.rooms]
    shapes += [walls for _, walls in converted if walls is not None]
    view = PlanView.for_geometries(shapes)
    assembled = assemble(floors, view.grid, view.image, warnings)
    return PlanRecord(
        id=f"{SOURCE_NAME}/{site_id}/{apartment_id}",
        source=SOURCE_NAME,
        image="",
        graph=assembled.graph,
        rooms=assembled.rooms,
        quality=assembled.quality,
    )
