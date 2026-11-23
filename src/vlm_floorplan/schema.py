"""The room graph: the one data structure every part of the project shares.

A floorplan image becomes a `FloorplanGraph`.
Rooms are nodes, and links between rooms are edges.
Walls and other geometry are deliberately absent: the model reads them once,
during perception, and the user only ever edits the graph.

Design notes (see docs/architecture.md for the full reasoning):

- Coordinates are integers on a 0-1000 grid over the image the model saw
  (x to the right, y down). This is the grounding convention Qwen-VL models use,
  and it does not depend on the image's pixel size.
- `xy` is a point guaranteed to lie inside the room, not its centroid
  (the centroid of an L-shaped room can fall outside it).
- One image can show several floors side by side, so a graph holds a list of
  per-floor graphs. Links never cross floors. What the model reads and writes is single-floor:
  the text and JSON forms (`summary.py`, `compact.py`) carry no floor at all.
- The validators here are the contract that converters, the model's output
  parser, and (later) the edit tools all rely on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

GRID_MAX = 1000
"""Coordinates run from 0 to GRID_MAX inclusive on both axes."""

GridCoord = Annotated[int, Field(ge=0, le=GRID_MAX)]
MAX_ROOM_SIZE = GRID_MAX // 2
"""A room's size is a radius on the grid, so half the grid is the most that fits."""
RoomSize = Annotated[int, Field(ge=1, le=MAX_ROOM_SIZE)]
RoomId = Annotated[str, Field(pattern=r"^r\d+$")]


class RoomType(StrEnum):
    """Coarse room types that CubiCasa5K, ResPlan and MSD all map into.

    The set is intentionally small: a 4B model has to predict it from an image,
    and retrieval (M6) has to compare plans across datasets.
    The dataset's own, finer label is kept outside the graph (see `PlanRecord`).
    """

    LIVING = "living"
    KITCHEN = "kitchen"
    DINING = "dining"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"  # also WC, shower, sauna (wet rooms)
    ENTRY = "entry"  # entrance, lobby, hall, corridor (circulation)
    STORAGE = "storage"  # also closets and walk-in wardrobes
    UTILITY = "utility"  # laundry, technical and boiler rooms
    TERRACE = "terrace"  # balcony, terrace, porch, patio
    GARAGE = "garage"  # also carports
    STAIRS = "stairs"
    OTHER = "other"  # a labeled room that fits none of the above (office, attic)
    UNKNOWN = "unknown"  # the source did not say; excluded from type loss and metrics


class LinkKind(StrEnum):
    """How two rooms connect.

    `wall` means "adjacent, but a wall separates them". The model never sees walls
    again after perception, so the edit validator needs this to reject requests
    like "add a door between two rooms that do not touch".
    """

    DOOR = "door"
    OPENING = "opening"  # no wall between them, e.g. an open-plan kitchen
    WALL = "wall"


class Room(BaseModel):
    """One room (graph node)."""

    model_config = ConfigDict(extra="forbid")

    id: RoomId
    type: RoomType
    label: str | None = None
    """The room's name as printed on the plan, if any (often a Finnish abbreviation)."""
    xy: tuple[GridCoord, GridCoord]
    size: RoomSize | None = None
    """Radius of a circle with the room's area, on the same 0-1000 grid as `xy`.

    Relative to the canvas, so it needs no metre scale. None when the source has no size.
    """
    entrance: bool = False
    """True when an exterior door opens into this room."""


class Link(BaseModel):
    """An undirected connection between two rooms on the same floor."""

    model_config = ConfigDict(extra="forbid")

    a: RoomId
    b: RoomId
    kind: LinkKind

    @property
    def pair(self) -> frozenset[str]:
        """The unordered room pair, used to detect duplicates."""
        return frozenset((self.a, self.b))


class Floor(BaseModel):
    """The room graph of one floor."""

    model_config = ConfigDict(extra="forbid")

    floor: int
    """Floor number as the source names it (1 = first floor, -1 = basement)."""
    rooms: list[Room]
    links: list[Link] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_links(self) -> Floor:
        room_ids = {room.id for room in self.rooms}
        seen_pairs: set[frozenset[str]] = set()
        for link in self.links:
            if link.a == link.b:
                raise ValueError(f"link {link.a}-{link.b} links a room to itself")
            for end in (link.a, link.b):
                if end not in room_ids:
                    raise ValueError(
                        f"link {link.a}-{link.b} refers to {end}, which is not a room in the plan"
                    )
            # One link per room pair: a pair is either door-, opening- or wall-connected.
            if link.pair in seen_pairs:
                raise ValueError(f"duplicate link between {link.a} and {link.b}")
            seen_pairs.add(link.pair)
        return self


class ImageSize(BaseModel):
    """Pixel size of the image the graph was drawn on."""

    model_config = ConfigDict(extra="forbid")

    w: Annotated[int, Field(gt=0)]
    h: Annotated[int, Field(gt=0)]


class FloorplanGraph(BaseModel):
    """All floors shown in one floorplan image."""

    model_config = ConfigDict(extra="forbid")

    floors: Annotated[list[Floor], Field(min_length=1)]
    image: ImageSize
    exterior_wall: Annotated[list[tuple[GridCoord, GridCoord]], Field(min_length=3)] | None = None
    """The outline of the building as a closed polygon of grid points, or None when unknown.

    Input only: the model reads it and never writes it, so edits and `new_plan` keep it.
    Indoor rooms lie inside it and terraces outside, against it.
    """

    @model_validator(mode="after")
    def _check_ids(self) -> FloorplanGraph:
        floor_numbers = [floor.floor for floor in self.floors]
        if len(set(floor_numbers)) != len(floor_numbers):
            raise ValueError(f"floor numbers must be unique, got {floor_numbers}")
        # Ids are unique across the whole plan, so "r7" is unambiguous in a chat edit.
        seen: set[str] = set()
        for floor in self.floors:
            for room in floor.rooms:
                if room.id in seen:
                    raise ValueError(f"room id {room.id} is used more than once")
                seen.add(room.id)
        return self

    def rooms(self) -> list[Room]:
        """All rooms on all floors, in floor order."""
        return [room for floor in self.floors for room in floor.rooms]
