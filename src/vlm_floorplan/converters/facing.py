"""How far two room outlines run straight across from each other.

Every converter that derives `opening` and `wall` links from geometry asks the
same question: along how much of room a's outline does room b lie straight
ahead, within some gap? Measuring "within the gap" alone (a's outline inside a
buffer of b) also counts the round reach around a corner, so two rooms that
meet only at a corner would look like neighbours. Measuring "straight ahead"
does not: four rooms around a wall cross face their diagonal neighbour along
nothing, and two rooms whose corners overlap by 10 cm face each other along
10 cm.

The measure is unit-free: CubiCasa5K calls it in image pixels, MSD and Swiss
Dwellings in metres.
"""

from __future__ import annotations

import math
from itertools import pairwise

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from vlm_floorplan.geometry import polygon_parts


def facing_length(a: Polygon, b: Polygon, gap: float) -> float:
    """Length of a's outline that faces b across at most `gap`.

    For each edge of a, the rectangle `gap` deep in front of it is intersected
    with b, and the hit is projected back onto the edge. Projections on one edge
    are merged, so a room that reaches an edge twice is counted only where it is.
    """
    coords = list(orient(a, sign=1.0).exterior.coords)  # counter-clockwise
    total = 0.0
    for (x0, y0), (x1, y1) in pairwise(coords):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        nx, ny = uy, -ux  # outward normal: right of a counter-clockwise edge
        front = Polygon(
            [(x0, y0), (x1, y1), (x1 + nx * gap, y1 + ny * gap), (x0 + nx * gap, y0 + ny * gap)]
        )
        spans = []
        for piece in polygon_parts(front.intersection(b)):
            ts = [(x - x0) * ux + (y - y0) * uy for x, y in piece.exterior.coords]
            spans.append((min(ts), max(ts)))
        total += _union_length(spans)
    return total


def _union_length(spans: list[tuple[float, float]]) -> float:
    """Total length covered by possibly overlapping intervals."""
    total, reach = 0.0, -math.inf
    for start, end in sorted(spans):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total
