"""A dataset row: one image plus its graph and the ground-truth extras.

The `FloorplanGraph` is the model's contract: only what the model reads or writes.
A `PlanRecord` wraps it with what training and evaluation need but the model
never produces:

- where the image is,
- the dataset's original room class (the graph only has the coarse type),
- room polygons, so evaluation can match a predicted point to the room whose
  polygon contains it, with no matching heuristic,
- quality flags, so mostly-unlabeled plans can be dropped or down-weighted.

Records are stored one per line in JSONL files (see `vlm_floorplan.data.build`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_floorplan.geometry import Ring
from vlm_floorplan.schema import FloorplanGraph, RoomType

MOSTLY_UNLABELED_THRESHOLD = 0.5
"""A plan whose share of `unknown` rooms exceeds this is flagged `mostly_unlabeled`."""


class RoomSource(BaseModel):
    """Ground truth for one room that is not part of the model target."""

    model_config = ConfigDict(extra="forbid")

    source_type: str
    """The dataset's own class, e.g. "Space Bath Shower" or "bedroom"."""
    polygon: list[Ring]
    """Room outline in grid coordinates (0-1000 floats); several rings if split."""


class PlanQuality(BaseModel):
    """Per-plan quality numbers."""

    model_config = ConfigDict(extra="forbid")

    unlabeled_fraction: float = Field(ge=0.0, le=1.0)
    mostly_unlabeled: bool
    warnings: list[str] = Field(default_factory=list)
    """Things the converter skipped or repaired (dangling doors, invalid polygons)."""

    @classmethod
    def from_graph(cls, graph: FloorplanGraph, warnings: list[str]) -> PlanQuality:
        rooms = graph.rooms()
        unknown = sum(1 for room in rooms if room.type == RoomType.UNKNOWN)
        fraction = unknown / len(rooms) if rooms else 1.0
        return cls(
            unlabeled_fraction=round(fraction, 3),
            mostly_unlabeled=fraction > MOSTLY_UNLABELED_THRESHOLD,
            warnings=warnings,
        )


class PlanRecord(BaseModel):
    """One training or evaluation example."""

    model_config = ConfigDict(extra="forbid")

    id: str
    """Unique across datasets, e.g. "cubicasa5k/colorful/8759" or "resplan/2814"."""
    source: str
    """Dataset name: "cubicasa5k" or "resplan"."""
    split: str | None = None
    image: str
    """Image path, relative to the directory of the JSONL file that holds the record."""
    graph: FloorplanGraph
    rooms: dict[str, RoomSource]
    """Ground truth per room id."""
    quality: PlanQuality

    @model_validator(mode="after")
    def _check_room_sources(self) -> PlanRecord:
        graph_ids = {room.id for room in self.graph.rooms()}
        if set(self.rooms) != graph_ids:
            raise ValueError(f"record {self.id}: room sources do not match the graph's room ids")
        return self
