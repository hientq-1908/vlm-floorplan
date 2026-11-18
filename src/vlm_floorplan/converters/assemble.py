"""Turn converter output (rooms with geometry, typed links) into a graph and record.

Each converter does the dataset-specific part: find rooms, decide their types,
and decide which room pairs are linked and how. This module does the shared
part, so both datasets produce graphs with the same conventions:

- rooms are numbered in reading order (top to bottom, then left to right),
  continuing across floors, so the model sees a predictable order;
- each pair of rooms gets at most one link, the strongest one found
  (door beats opening, opening beats wall);
- points, boxes and polygons are moved into the 0-1000 grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry

from vlm_floorplan.geometry import GridFrame
from vlm_floorplan.record import PlanQuality, RoomSource
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, LinkKind, Room, RoomType

# Higher wins when two rules link the same pair.
LINK_STRENGTH = {LinkKind.DOOR: 3, LinkKind.OPENING: 2, LinkKind.WALL: 1}


@dataclass
class SourceRoom:
    """A room in source coordinates, before ids and grid coordinates exist."""

    geometry: BaseGeometry
    type: RoomType
    source_type: str
    label: str | None = None
    entrance: bool = False


@dataclass
class SourceFloor:
    """One floor: rooms plus links keyed by (room index, room index)."""

    number: int
    rooms: list[SourceRoom]
    links: dict[tuple[int, int], LinkKind] = field(default_factory=dict)

    def link(self, i: int, j: int, kind: LinkKind) -> None:
        """Record a link, keeping the strongest kind if the pair is already linked."""
        if i == j:
            return
        key = (min(i, j), max(i, j))
        current = self.links.get(key)
        if current is None or LINK_STRENGTH[kind] > LINK_STRENGTH[current]:
            self.links[key] = kind


@dataclass
class Assembled:
    """The finished graph plus the per-room ground truth and quality numbers."""

    graph: FloorplanGraph
    rooms: dict[str, RoomSource]
    quality: PlanQuality


def _assemble_floor(
    source_floor: SourceFloor, frame: GridFrame, first_id: int
) -> tuple[Floor, dict[str, RoomSource]]:
    points = [frame.point(room.geometry) for room in source_floor.rooms]
    # Reading order: by y, then x, on the grid.
    order = sorted(range(len(source_floor.rooms)), key=lambda i: (points[i][1], points[i][0]))
    # position[i] = reading-order position of source room i; its id is r<first_id + position>.
    position = {index: rank for rank, index in enumerate(order)}
    rooms: list[Room] = []
    sources: dict[str, RoomSource] = {}
    for index in order:
        room = source_floor.rooms[index]
        room_id = f"r{first_id + position[index]}"
        rooms.append(
            Room(
                id=room_id,
                type=room.type,
                label=room.label,
                xy=points[index],
                bbox=frame.bbox(room.geometry),
                entrance=room.entrance,
            )
        )
        sources[room_id] = RoomSource(
            source_type=room.source_type, polygon=frame.rings(room.geometry)
        )
    pairs = sorted(
        (sorted((position[i], position[j])), kind) for (i, j), kind in source_floor.links.items()
    )
    links = [Link(a=f"r{first_id + a}", b=f"r{first_id + b}", kind=kind) for (a, b), kind in pairs]
    return Floor(floor=source_floor.number, rooms=rooms, links=links), sources


def assemble(
    floors: list[SourceFloor], frame: GridFrame, image: ImageSize, warnings: list[str]
) -> Assembled:
    """Number rooms, convert coordinates and validate the result."""
    graph_floors: list[Floor] = []
    sources: dict[str, RoomSource] = {}
    for source_floor in sorted(floors, key=lambda f: f.number):
        floor, floor_sources = _assemble_floor(source_floor, frame, first_id=len(sources))
        graph_floors.append(floor)
        sources.update(floor_sources)
    graph = FloorplanGraph(floors=graph_floors, image=image)
    return Assembled(graph=graph, rooms=sources, quality=PlanQuality.from_graph(graph, warnings))
