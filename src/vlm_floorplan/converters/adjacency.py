"""Derive opening and wall links from room geometry alone.

MSD and Swiss Dwellings say where the doors are, but neither says which rooms
merely touch. The schema's `wall` link ("adjacent, but a wall separates them")
and `opening` link ("no wall between them") are recovered here from how close
two room polygons are:

- gap <= `opening_gap`: the polygons touch, so nothing separates them: `opening`.
- gap <= `wall_gap`: a wall-thickness apart: `wall`.
- either way the rooms must run alongside each other for at least `min_shared`
  metres. A pair that nearly touches at a corner but also runs alongside a wall
  still gets its `wall` link: the corner does not hide it.

"Alongside" is the length of one room's outline that lies within the gap of the
other room.
This is the same measure the CubiCasa5K converter uses, so the three
converters link rooms by one rule.

Known limit: rooms that meet only at a corner are measured over the 0.4 m reach
around that corner, so a corner contact can pass `min_shared`, and then it gets
a `wall` link although the rooms do not run alongside each other.
Swiss Dwellings is guarded by its wall-cover check.
MSD has no wall layer, so there a corner contact can produce a wall link.
A sharper measure (projecting both rooms onto their facing edges) was tried and
dropped: each variant fixed one geometry and broke another (offset corners, rooms
touching twice, curved walls built from short facets), for a link that is a weak
signal in the first place.

With a wall layer, the strip between the rooms (the area outside both that lies
within half the wall gap of each) also tells how much of the gap is wall.

Doors are linked first by the converters, and `SourceFloor.link` keeps the
strongest link per pair, so a pair with a door stays a `door` link.
Thresholds are in plan metres; both datasets use metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from shapely.geometry.base import BaseGeometry

from vlm_floorplan.converters.assemble import SourceFloor
from vlm_floorplan.schema import LinkKind


@dataclass(frozen=True)
class AdjacencyRules:
    """Distances, in metres, that decide how two nearby rooms are linked."""

    opening_gap: float | None
    """Closest distance still counted as "no wall"; None turns opening detection off."""
    wall_gap: float
    """Largest gap that still counts as one wall between two rooms."""
    min_shared: float
    """Length over which the two rooms must run alongside each other, at least."""
    min_wall_cover: float = 0.0
    """With a wall layer: share of the gap strip that must be wall (0 disables the check)."""


def _shared_length(a: BaseGeometry, b: BaseGeometry, gap: float) -> float:
    """Length of a's outline that lies within `gap` of b."""
    return a.boundary.intersection(b.buffer(gap)).length


def _strip_between(a: BaseGeometry, b: BaseGeometry, gap: float) -> BaseGeometry:
    """The area outside both rooms that lies within half of `gap` of each."""
    half = gap / 2
    return a.buffer(half).intersection(b.buffer(half)).difference(a.union(b))


def _wall_cover(strip: BaseGeometry, walls: BaseGeometry) -> float:
    """Share of the strip between two rooms that is covered by wall."""
    if strip.area == 0:
        return 0.0
    return strip.intersection(walls).area / strip.area


def link_adjacent_rooms(
    floor: SourceFloor, rules: AdjacencyRules, walls: BaseGeometry | None = None
) -> None:
    """Add `opening` and `wall` links between rooms of one floor, in place."""
    for i, j in combinations(range(len(floor.rooms)), 2):
        a, b = floor.rooms[i].geometry, floor.rooms[j].geometry
        distance = a.distance(b)
        if distance > rules.wall_gap:
            continue
        if (
            rules.opening_gap is not None
            and distance <= rules.opening_gap
            and _shared_length(a, b, rules.opening_gap) >= rules.min_shared
        ):
            floor.link(i, j, LinkKind.OPENING)
            continue
        if _shared_length(a, b, rules.wall_gap) < rules.min_shared:
            continue
        if walls is not None:
            strip = _strip_between(a, b, rules.wall_gap)
            if _wall_cover(strip, walls) < rules.min_wall_cover:
                continue
        floor.link(i, j, LinkKind.WALL)
