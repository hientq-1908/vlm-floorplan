"""One MSD plan (its `graph_out` access graph) to a `PlanRecord`.

MSD (Modified Swiss Dwellings) already contains the connectivity we need, so
unlike CubiCasa5K nothing has to be detected, only translated:

| MSD edge `connectivity` | graph link | extra |
|---|---|---|
| door     | `door`    | |
| passage  | `opening` | a doorless gap, e.g. open-plan kitchen and living room |
| entrance | `door`    | the apartment's front door: `entrance` flag, see below |

Pairs of rooms that touch without an edge get a derived `wall` link
(see `adjacency.py`).

An MSD plan is a whole storey of a building complex: several apartments plus
the shared stairs and corridors. That is why a plan has 16 to 75 rooms and its
graph often splits into several components. It is stored as one floor, numbered 1,
because MSD does not say which storey it is.

Entrance flag. An `entrance` edge joins an apartment's hall to the shared
circulation space. The flag belongs on the apartment side. To find it, remove
the entrance edges; each endpoint then sits in a component, and the side whose
component holds more dwelling rooms (bedrooms, kitchens, ...) is the apartment.
If neither side has more, both endpoints are flagged: an entrance door
does open into both rooms, and guessing would hide the other half.
"""

from __future__ import annotations

from collections import defaultdict

from shapely.geometry import Polygon

from vlm_floorplan.converters.adjacency import AdjacencyRules, link_adjacent_rooms
from vlm_floorplan.converters.assemble import SourceFloor, SourceRoom, assemble
from vlm_floorplan.converters.msd_pickle import MsdGraph, parse_msd_graph
from vlm_floorplan.geometry import largest_polygon
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.room_types import MSD_ROOM_NAMES, MSD_TYPES
from vlm_floorplan.schema import LinkKind, RoomType

SOURCE_NAME = "msd"
FLOOR_NUMBER = 1

# Thresholds in metres, from the distances between linked and unlinked room pairs in 40 plans:
# door-linked rooms are a median 0.15 m apart (at most 0.66), passage-linked ones 0.01 m,
# and the many unlinked pairs that share a wall are up to about 0.4 m apart.
MSD_ADJACENCY = AdjacencyRules(opening_gap=None, wall_gap=0.4, min_shared=0.5)
"""`opening_gap` is None: MSD's own passage edges decide openings."""

_EDGE_KINDS = {
    "door": LinkKind.DOOR,
    "passage": LinkKind.OPENING,
    "entrance": LinkKind.DOOR,
}
_DWELLING_TYPES = frozenset(
    {
        RoomType.LIVING,
        RoomType.KITCHEN,
        RoomType.DINING,
        RoomType.BEDROOM,
        RoomType.BATHROOM,
        RoomType.STORAGE,
    }
)


def _rooms(graph: MsdGraph, warnings: list[str]) -> tuple[list[SourceRoom], dict[int, int]]:
    """Source rooms, plus a map from MSD node key to room index (dropped nodes are absent)."""
    rooms: list[SourceRoom] = []
    index_of: dict[int, int] = {}
    for key, node in sorted(graph.nodes.items()):
        if not 0 <= node.room_type < len(MSD_ROOM_NAMES):
            raise ValueError(f"node {key}: room_type {node.room_type} is not a room class")
        polygon = Polygon(node.polygon)
        if not polygon.is_valid:
            repaired = largest_polygon(polygon)
            if repaired is None:
                warnings.append(f"node {key}: dropped empty polygon")
                continue
            warnings.append(f"node {key}: repaired invalid polygon")
            polygon = repaired
        name = MSD_ROOM_NAMES[node.room_type]
        index_of[key] = len(rooms)
        rooms.append(SourceRoom(geometry=polygon, type=MSD_TYPES[name], source_type=name))
    return rooms, index_of


def _components(count: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Component label of each room, given the links that connect rooms."""
    neighbours: dict[int, list[int]] = defaultdict(list)
    for i, j in pairs:
        neighbours[i].append(j)
        neighbours[j].append(i)
    label = [-1] * count
    for start in range(count):
        if label[start] != -1:
            continue
        label[start] = start
        stack = [start]
        while stack:
            for other in neighbours[stack.pop()]:
                if label[other] == -1:
                    label[other] = start
                    stack.append(other)
    return label


def _flag_entrances(floor: SourceFloor, entrance_pairs: list[tuple[int, int]]) -> None:
    """Set `entrance` on the apartment side of each entrance edge (see module docstring)."""
    entrance_set = set(entrance_pairs)
    inner_pairs = [pair for pair in floor.links if pair not in entrance_set]
    label = _components(len(floor.rooms), inner_pairs)
    dwellings: dict[int, int] = defaultdict(int)
    for room, component in zip(floor.rooms, label, strict=True):
        dwellings[component] += room.type in _DWELLING_TYPES
    for i, j in entrance_pairs:
        score_i, score_j = dwellings[label[i]], dwellings[label[j]]
        if score_i >= score_j:
            floor.rooms[i].entrance = True
        if score_j >= score_i:
            floor.rooms[j].entrance = True


def _link_edges(
    floor: SourceFloor, graph: MsdGraph, index_of: dict[int, int], warnings: list[str]
) -> None:
    entrance_pairs: list[tuple[int, int]] = []
    for edge in graph.edges:
        kind = _EDGE_KINDS.get(edge.connectivity)
        if kind is None:
            warnings.append(f"edge {edge.a}-{edge.b}: unknown connectivity {edge.connectivity!r}")
        elif edge.a in index_of and edge.b in index_of:
            i, j = index_of[edge.a], index_of[edge.b]
            floor.link(i, j, kind)
            if edge.connectivity == "entrance":
                entrance_pairs.append((min(i, j), max(i, j)))
    _flag_entrances(floor, entrance_pairs)


def convert_graph(graph: MsdGraph, plan_id: str) -> PlanRecord:
    """Convert a parsed MSD graph. Raises `ValueError` for a plan with no usable rooms."""
    warnings: list[str] = []
    rooms, index_of = _rooms(graph, warnings)
    if not rooms:
        raise ValueError(f"{plan_id}: no rooms")
    floor = SourceFloor(number=FLOOR_NUMBER, rooms=rooms)
    _link_edges(floor, graph, index_of, warnings)
    link_adjacent_rooms(floor, MSD_ADJACENCY)
    view = PlanView.for_geometries([room.geometry for room in rooms])
    assembled = assemble([floor], view.grid, view.image, warnings)
    return PlanRecord(
        id=f"{SOURCE_NAME}/{plan_id}",
        source=SOURCE_NAME,
        image="",
        graph=assembled.graph,
        rooms=assembled.rooms,
        quality=assembled.quality,
    )


def convert_plan(data: bytes, plan_id: str) -> PlanRecord:
    """Convert the bytes of one `graph_out/<plan id>.pickle` to a graph-only record."""
    return convert_graph(parse_msd_graph(data), plan_id)
