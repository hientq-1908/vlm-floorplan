"""Which rooms connect, and how: door, opening, or just a wall.

Adapted from `plan_to_graph` and `add_adjacency_edges` in ResPlan's `resplan_utils.py`
(Copyright (c) 2025 The ResPlan Authors, MIT licence; see THIRD_PARTY_NOTICES.md).
The geometric tests and their thresholds are theirs.
The pieces that differ, and why:

- shapely only: upstream's module imports OpenCV, matplotlib and networkx at the top,
  none of which a converter needs.
- The front door is not a graph node. It becomes the `entrance` flag on the room it opens
  into (`find_entrances`), so a door or window next to it cannot create a bogus edge.
  This avoids the `via_window` front-door edge the roadmap report found.
- A door joins exactly the two rooms with the most contact, never three.
- Upstream's `fallback` edges (nearest neighbour, only to keep the graph in one piece)
  are dropped. They link rooms that need not touch, and a `wall` link in our schema
  promises "adjacent, separated by a wall".

Upstream edge type -> `LinkKind`:

    via_door                       -> door
    via_window, interior rooms     -> door   (upstream relabels these the same way)
    via_window, a balcony involved -> no evidence (a window is not a passage);
                                      the pair falls through to opening or wall, and
                                      windows count as wall when looking for an opening
    via_opening                    -> opening
    adjacency (within a wall gap)  -> wall

A pair keeps its strongest link: door beats opening beats wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from vlm_floorplan.resplan.load import polygons
from vlm_floorplan.schema import LinkKind

# Room layers in ResPlan, in a fixed order. `living` is merged into one room (below).
ROOM_LAYERS = ("living", "kitchen", "bedroom", "bathroom", "balcony", "storage", "stair")

# Rooms a person walks through. A window between two of these is really a door.
INTERIOR_LAYERS = frozenset({"living", "kitchen", "bedroom", "bathroom", "storage", "stair"})

BUFFER_FACTOR = 0.007
"""Spatial tolerance as a fraction of the plan's extent (upstream default)."""

OPEN_PASSAGE_MIN_FACTOR = 2.0
"""An opening must be at least this many wall depths wide to count as a passage."""

WALL_GAP = 4.0
"""Rooms closer than this (plan units) are adjacent, separated by a wall."""

MIN_WALL_OVERLAP_AREA = 5.0
"""Smallest overlap of two rooms' dilated outlines that counts as sharing a wall."""

DEFAULT_WALL_DEPTH = 4.0

_STRENGTH = {LinkKind.WALL: 0, LinkKind.OPENING: 1, LinkKind.DOOR: 2}


@dataclass(frozen=True)
class RoomShape:
    """One room of a plan, before ids and grid coordinates are assigned."""

    layer: str
    """ResPlan's own category, e.g. "bedroom"."""
    geometry: BaseGeometry


def room_shapes(plan: dict[str, Any]) -> list[RoomShape]:
    """The plan's rooms: all `living` parts as one room, every other part on its own.

    Merging `living` follows ResPlan's own convention. It is also the honest choice,
    because ResPlan labels corridors and open-plan areas "living" without telling them apart.
    """
    shapes: list[RoomShape] = []
    living = polygons(plan.get("living"))
    if living:
        merged = unary_union(living)
        if not merged.is_empty:
            shapes.append(RoomShape("living", merged))
    for layer in ROOM_LAYERS[1:]:
        shapes.extend(RoomShape(layer, part) for part in polygons(plan.get(layer)))
    return shapes


def plan_tolerance(plan: dict[str, Any]) -> float:
    """Upstream's scale-aware tolerance: a fraction of the plan's longer side."""
    inner = plan.get("inner")
    if inner is not None and not inner.is_empty:
        x1, y1, x2, y2 = inner.bounds
        extent = max(x2 - x1, y2 - y1)
    else:
        extent = 256.0
    return max(extent * BUFFER_FACTOR, 0.5)


def wall_depth(plan: dict[str, Any]) -> float:
    return float(plan.get("wall_depth") or DEFAULT_WALL_DEPTH)


def _contact(connector: BaseGeometry, room: BaseGeometry, tolerance: float) -> float:
    """Length of the connector's outline that lies inside the (slightly grown) room."""
    return connector.boundary.intersection(room.buffer(tolerance * 0.6)).length


def _rooms_joined_by(
    connector: BaseGeometry, shapes: list[RoomShape], grown: list[BaseGeometry], tolerance: float
) -> tuple[int, int] | None:
    """The two rooms a door or window sits between, or None if it touches fewer than two.

    A door spans one wall, so it joins exactly two rooms.
    When it sits in a corner where three rooms meet, the two with the longest contact win.
    """
    touching = [i for i, area in enumerate(grown) if connector.intersects(area)]
    if len(touching) < 2:
        return None
    if len(touching) > 2:
        touching.sort(
            key=lambda i: _contact(connector, shapes[i].geometry, tolerance), reverse=True
        )
    first, second = sorted(touching[:2])
    return first, second


def _is_open(
    a: BaseGeometry,
    b: BaseGeometry,
    walls: tuple[BaseGeometry, BaseGeometry] | None,
    depth: float,
) -> bool:
    """True when the shared boundary of two rooms has a wall-free stretch wide enough to walk."""
    open_min = OPEN_PASSAGE_MIN_FACTOR * depth
    # Test 1: grow both rooms by about half a wall, subtract the walls, and see
    # whether a region wide enough to walk through is left where they overlap.
    gap = a.buffer(depth * 0.55).intersection(b.buffer(depth * 0.55))
    if walls is not None:
        gap = gap.difference(walls[0])
    for piece in getattr(gap, "geoms", [gap]):
        if piece.is_empty:
            continue
        x1, y1, x2, y2 = piece.bounds
        if max(x2 - x1, y2 - y1) >= open_min and piece.area >= depth * depth * 0.25:
            return True
    # Test 2: the part of a's outline that touches b and is not covered by wall.
    contact = a.boundary.intersection(b.buffer(depth * 0.6))
    if contact.is_empty or contact.length < open_min:
        return False
    open_segments = contact.difference(walls[1]) if walls is not None else contact
    if open_segments.is_empty:
        return False
    lengths = [seg.length for seg in getattr(open_segments, "geoms", [open_segments])]
    return max(lengths, default=0.0) >= open_min


def _stronger(
    links: dict[tuple[int, int], LinkKind], pair: tuple[int, int], kind: LinkKind
) -> None:
    current = links.get(pair)
    if current is None or _STRENGTH[kind] > _STRENGTH[current]:
        links[pair] = kind


def derive_links(plan: dict[str, Any], shapes: list[RoomShape]) -> dict[tuple[int, int], LinkKind]:
    """Links between rooms, keyed by (lower index, higher index) into `shapes`."""
    tolerance = plan_tolerance(plan)
    depth = wall_depth(plan)
    grown = [shape.geometry.buffer(tolerance) for shape in shapes]
    links: dict[tuple[int, int], LinkKind] = {}

    for door in polygons(plan.get("door")):
        pair = _rooms_joined_by(door, shapes, grown, tolerance)
        if pair is not None:
            _stronger(links, pair, LinkKind.DOOR)

    for window in polygons(plan.get("window")):
        pair = _rooms_joined_by(window, shapes, grown, tolerance)
        if pair is not None and all(shapes[i].layer in INTERIOR_LAYERS for i in pair):
            _stronger(links, pair, LinkKind.DOOR)

    # Walls and windows both stop a person, so both count as wall when looking for an opening.
    # ResPlan cuts door and window areas out of the wall layer, so without this a window
    # would look like a wall-free boundary (upstream makes that call; we do not).
    # Kept as (repaired geometry, a slightly thicker copy for the outline test).
    blockers = [plan.get(layer) for layer in ("wall", "window")]
    blockers = [geom for geom in blockers if geom is not None and not geom.is_empty]
    walls = None
    if blockers:
        repaired = unary_union(blockers).buffer(0.0)
        walls = (repaired, repaired.buffer(0.5))
    for i, j in combinations(range(len(shapes)), 2):
        if links.get((i, j)) == LinkKind.DOOR:
            continue
        if _is_open(shapes[i].geometry, shapes[j].geometry, walls, depth):
            _stronger(links, (i, j), LinkKind.OPENING)

    for i, j in combinations(range(len(shapes)), 2):
        if (i, j) in links:
            continue
        a, b = shapes[i].geometry, shapes[j].geometry
        if a.distance(b) > WALL_GAP:
            continue
        shared = a.buffer(WALL_GAP / 2 + 0.01).intersection(b.buffer(WALL_GAP / 2 + 0.01))
        if shared.area > MIN_WALL_OVERLAP_AREA:
            links[(i, j)] = LinkKind.WALL
    return links


def find_entrances(plan: dict[str, Any], shapes: list[RoomShape]) -> tuple[set[int], list[str]]:
    """Indices of the rooms an exterior door opens into, plus warnings.

    Each front door opens into one room: the one it overlaps most.
    A door in a corner that touches a second room (a balcony, a stair) still counts once.
    """
    tolerance = plan_tolerance(plan)
    entrances: set[int] = set()
    warnings: list[str] = []
    for door in polygons(plan.get("front_door")):
        reach = door.buffer(tolerance)
        overlaps = [reach.intersection(shape.geometry).area for shape in shapes]
        best = max(range(len(shapes)), key=overlaps.__getitem__, default=None)
        if best is None or overlaps[best] <= 0:
            warnings.append("front door touches no room")
        else:
            entrances.add(best)
    return entrances, warnings
