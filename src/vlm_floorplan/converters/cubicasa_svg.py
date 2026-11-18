"""Read the parts of a CubiCasa5K `model.svg` that the room graph needs.

The SVG is a tree of `<g>` groups whose `class` attribute says what they are:

    <g class="Floorplan Floor-1">                      one per floor
      <g class="Space Bedroom">                        a room
        <polygon points="..."/>                        its outline (first polygon)
        ... <g class="TextLabel NameLabel"><text>MH</text></g>   printed name
      <g class="Wall External">                        an exterior wall
        <polygon .../>
        <g class="Door Swing Beside"><polygon .../></g>   a door sits inside its wall
      <g class="FixedFurniture ... Refrigerator">
        <g class="Door ...">                           a fridge door: NOT a real door

Coordinates are pixels of `F1_scaled.png` (not `F1_original.png`).
This module only extracts shapes; `cubicasa.py` decides what they mean.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

SVG_NS = "{http://www.w3.org/2000/svg}"
_FLOOR_CLASS = re.compile(r"^Floorplan\s+(Floor|Basement)-(\d+)")
_MATRIX = re.compile(r"matrix\(([^)]*)\)")


@dataclass
class SvgSpace:
    """A room outline with its class and printed name."""

    space_class: str
    polygon: Polygon
    label: str | None


@dataclass
class SvgWall:
    polygon: Polygon
    is_external: bool


@dataclass
class SvgDoor:
    polygon: Polygon
    wall: SvgWall


@dataclass
class SvgFloor:
    number: int
    spaces: list[SvgSpace] = field(default_factory=list)
    walls: list[SvgWall] = field(default_factory=list)
    doors: list[SvgDoor] = field(default_factory=list)
    skipped_doors: int = 0
    """Door groups not inside a wall (fridge and cabinet doors)."""


def _classes(element: ET.Element) -> str:
    return element.get("class", "").strip()


def _parse_points(text: str) -> list[tuple[float, float]]:
    numbers = [float(n) for n in re.split(r"[\s,]+", text.strip()) if n]
    return list(zip(numbers[0::2], numbers[1::2], strict=False))


def _matrix(element: ET.Element) -> list[float] | None:
    """The element's own `matrix(a,b,c,d,e,f)` transform, as shapely's affine list."""
    match = _MATRIX.search(element.get("transform", ""))
    if not match:
        return None
    a, b, c, d, e, f = (float(v) for v in re.split(r"[\s,]+", match.group(1).strip()))
    return [a, c, b, d, e, f]


def _first_polygon(group: ET.Element, ancestors: list[ET.Element]) -> Polygon | None:
    """The group's own outline: its first direct `<polygon>` child, in page coordinates."""
    node = group.find(f"{SVG_NS}polygon")
    if node is None:
        return None
    points = _parse_points(node.get("points", ""))
    if len(points) < 3:
        return None
    geom: BaseGeometry = Polygon(points)
    # Structural groups have identity transforms in practice, but apply any we
    # find (innermost first) rather than silently assume it.
    for element in reversed([*ancestors, group]):
        matrix = _matrix(element)
        if matrix is not None:
            geom = affinity.affine_transform(geom, matrix)
    return geom


def _name_label(space: ET.Element) -> str | None:
    for group in space.iter(f"{SVG_NS}g"):
        if _classes(group) == "TextLabel NameLabel":
            text = "".join(group.itertext()).strip()
            return text or None
    return None


def _walk(element: ET.Element, ancestors: list[ET.Element], floor: SvgFloor) -> None:
    """Collect spaces, walls and wall doors below one floor group."""
    for child in element:
        if child.tag != f"{SVG_NS}g":
            continue
        classes = _classes(child)
        lineage = [*ancestors, element]
        if classes.startswith("Space "):
            polygon = _first_polygon(child, lineage)
            if polygon is not None:
                floor.spaces.append(SvgSpace(classes, polygon, _name_label(child)))
            continue  # nothing structural is nested inside a space
        if classes.startswith("Wall"):
            polygon = _first_polygon(child, lineage)
            if polygon is not None:
                wall = SvgWall(polygon, is_external="External" in classes.split())
                floor.walls.append(wall)
                _collect_wall_doors(child, lineage, wall, floor)
            continue
        if classes.split()[:1] == ["Door"]:
            floor.skipped_doors += 1  # a door whose parent is not a wall
        _walk(child, lineage, floor)


def _collect_wall_doors(
    wall_group: ET.Element, ancestors: list[ET.Element], wall: SvgWall, floor: SvgFloor
) -> None:
    for child in wall_group:
        if child.tag == f"{SVG_NS}g" and _classes(child).split()[:1] == ["Door"]:
            polygon = _first_polygon(child, [*ancestors, wall_group])
            if polygon is not None:
                floor.doors.append(SvgDoor(polygon, wall))


def _floor_number(classes: str) -> int | None:
    match = _FLOOR_CLASS.match(classes)
    if not match:
        return None
    kind, number = match.groups()
    return -int(number) if kind == "Basement" else int(number)


def parse_svg(path: Path) -> list[SvgFloor]:
    """Parse a CubiCasa5K `model.svg` into floors of spaces, walls and doors."""
    root = ET.parse(path).getroot()
    floors: list[SvgFloor] = []
    parents = {child: parent for parent in root.iter() for child in parent}
    for group in root.iter(f"{SVG_NS}g"):
        number = _floor_number(_classes(group))
        if number is None:
            continue
        ancestors: list[ET.Element] = []
        node = parents.get(group)
        while node is not None:
            ancestors.insert(0, node)
            node = parents.get(node)
        floor = SvgFloor(number=number)
        _walk(group, ancestors, floor)
        floors.append(floor)
    return floors
