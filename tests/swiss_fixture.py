"""Synthetic Swiss Dwellings `geometries.csv` rows (the real file is 1.1 GB and not committed)."""

from __future__ import annotations

import csv
from pathlib import Path

Rect = tuple[float, float, float, float]  # x1, y1, x2, y2 in metres

COLUMNS = [
    "apartment_id", "site_id", "building_id", "plan_id", "floor_id", "unit_id", "area_id",
    "unit_usage", "entity_type", "entity_subtype", "geometry", "elevation", "height",
]  # fmt: skip


def wkt_rect(rect: Rect) -> str:
    x1, y1, x2, y2 = rect
    return f"POLYGON (({x1} {y1}, {x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}))"


def row(
    entity_type: str,
    subtype: str,
    rect: Rect,
    *,
    apartment: str = "apt1",
    site: str = "s1",
    floor: str = "f1",
    elevation: float = 0.0,
    wkt: str | None = None,
) -> dict[str, str]:
    return {
        "apartment_id": apartment,
        "site_id": site,
        "building_id": "b1",
        "plan_id": "p1",
        "floor_id": floor,
        "unit_id": "u1",
        "area_id": "1.0",
        "unit_usage": "RESIDENTIAL" if apartment else "PUBLIC",
        "entity_type": entity_type,
        "entity_subtype": subtype,
        "geometry": wkt or wkt_rect(rect),
        "elevation": str(elevation),
        "height": "2.6",
    }


def apartment_rows(apartment: str = "apt1", site: str = "s1", floor: str = "f1") -> list[dict]:
    """A hall, bedroom, kitchen and living room, with walls and doors between them.

    Rooms are 0.2 m apart (a wall) except living room and dining area, which touch.

        kitchen  | living  dining      y 2.2-5
        ---------+--------
        hall     | bedroom             y 0-2
    """
    kw = {"apartment": apartment, "site": site, "floor": floor}
    return [
        row("area", "CORRIDOR", (0, 0, 3, 2), **kw),
        row("area", "BEDROOM", (3.2, 0, 6, 2), **kw),
        row("area", "KITCHEN", (0, 2.2, 3, 5), **kw),
        row("area", "LIVING_ROOM", (3.2, 2.2, 6, 5), **kw),
        row("area", "DINING", (6, 2.2, 8, 5), **kw),  # touches the living room: no wall
        row("area", "SHAFT", (8.5, 0, 9, 1), **kw),  # not a room
        # Walls: the strips between rooms.
        row("separator", "WALL", (0, 2, 3, 2.2), **kw),  # hall | kitchen
        row("separator", "WALL", (3, 0, 3.2, 2), **kw),  # hall | bedroom
        row("separator", "WALL", (3.2, 2, 6, 2.2), **kw),  # bedroom | living room
        row("separator", "WALL", (3, 2.2, 3.2, 5), **kw),  # kitchen | living room
        # A door in the hall | bedroom wall, and the front door in the hall's left wall.
        row("opening", "DOOR", (2.9, 0.5, 3.3, 1.3), **kw),
        row("opening", "ENTRANCE_DOOR", (-0.3, 0.5, 0.1, 1.3), **kw),
        row("opening", "WINDOW", (4, 4.9, 5, 5.1), **kw),
        row("feature", "TOILET", (0.1, 0.1, 0.5, 0.5), **kw),
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path
