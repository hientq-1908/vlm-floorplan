"""Facts about one plan, computed once from its graph and used by the checker and the deriver.

Everything here reads only the graph (points, types, links, entrance flags), never polygons, so
the same function decides a requirement for a source plan and for a plan a model drew.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import cached_property

from vlm_floorplan.requirements.types import Region
from vlm_floorplan.schema import FloorplanGraph, LinkKind, Room, RoomType

ENSUITE_NEIGHBOR_TYPES = frozenset({RoomType.BEDROOM, RoomType.STORAGE, RoomType.BATHROOM})
"""A bathroom is ensuite when every door or opening it has leads to one of these."""
PASSAGE_KINDS = frozenset({LinkKind.DOOR, LinkKind.OPENING})
"""Link kinds a person can walk through."""
REGION_SHARE = 1 / 3


@dataclass(frozen=True)
class Facts:
    """The numbers and relations a requirement is decided from."""

    graph: FloorplanGraph
    rooms: dict[str, Room] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rooms", {room.id: room for room in self.graph.rooms()})

    @cached_property
    def by_type(self) -> dict[RoomType, list[Room]]:
        grouped: dict[RoomType, list[Room]] = defaultdict(list)
        for room in self.graph.rooms():
            grouped[room.type].append(room)
        return dict(grouped)

    def count(self, room_type: RoomType) -> int:
        return len(self.by_type.get(room_type, []))

    @cached_property
    def neighbors(self) -> dict[str, list[tuple[str, LinkKind]]]:
        """Each room's links as (other room id, kind)."""
        result: dict[str, list[tuple[str, LinkKind]]] = defaultdict(list)
        for floor in self.graph.floors:
            for link in floor.links:
                result[link.a].append((link.b, link.kind))
                result[link.b].append((link.a, link.kind))
        return dict(result)

    @cached_property
    def entrance_rooms(self) -> list[Room]:
        return [room for room in self.graph.rooms() if room.entrance]

    @cached_property
    def ensuite_ids(self) -> list[str]:
        """Bathrooms that belong to a bedroom.

        A bathroom counts when it has at least one door or opening, at least one of them
        leads to a bedroom, and none leads to a room outside bedrooms, storage and
        bathrooms. A bathroom that also opens to the hall is shared, so "ensuite" would be false.
        """
        found = []
        for room in self.by_type.get(RoomType.BATHROOM, []):
            through = [
                self.rooms[other].type
                for other, kind in self.neighbors.get(room.id, [])
                if kind in PASSAGE_KINDS
            ]
            if (
                through
                and RoomType.BEDROOM in through
                and all(kind in ENSUITE_NEIGHBOR_TYPES for kind in through)
            ):
                found.append(room.id)
        return found

    @cached_property
    def box(self) -> tuple[int, int, int, int]:
        """Bounding box of the room points: min x, min y, max x, max y."""
        xs = [room.xy[0] for room in self.graph.rooms()]
        ys = [room.xy[1] for room in self.graph.rooms()]
        return min(xs), min(ys), max(xs), max(ys)

    def regions_of(self, room: Room) -> frozenset[Region]:
        """The regions a room's point is in: thirds of the box of all room points."""
        min_x, min_y, max_x, max_y = self.box
        x, y = room.xy
        top = y <= min_y + (max_y - min_y) * REGION_SHARE
        bottom = y >= max_y - (max_y - min_y) * REGION_SHARE
        left = x <= min_x + (max_x - min_x) * REGION_SHARE
        right = x >= max_x - (max_x - min_x) * REGION_SHARE
        regions = set()
        for flag, region in (
            (top, Region.TOP),
            (bottom, Region.BOTTOM),
            (left, Region.LEFT),
            (right, Region.RIGHT),
            (top and left, Region.TOP_LEFT),
            (top and right, Region.TOP_RIGHT),
            (bottom and left, Region.BOTTOM_LEFT),
            (bottom and right, Region.BOTTOM_RIGHT),
        ):
            if flag:
                regions.add(region)
        return frozenset(regions)

    def reachable(self, avoid: RoomType | None = None) -> set[str]:
        """Room ids reachable from any entrance over doors and openings.

        Rooms of type `avoid` are never entered (an entrance of that type starts nothing).
        """
        start = [room.id for room in self.entrance_rooms if room.type != avoid]
        seen = set(start)
        queue = deque(start)
        while queue:
            here = queue.popleft()
            for other, kind in self.neighbors.get(here, []):
                if kind in PASSAGE_KINDS and other not in seen and self.rooms[other].type != avoid:
                    seen.add(other)
                    queue.append(other)
        return seen


def compute_facts(graph: FloorplanGraph) -> Facts:
    return Facts(graph)
