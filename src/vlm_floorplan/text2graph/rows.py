"""One held-out prompt as stored in a set's JSONL, and how its files sit on disk.

A set directory holds `<set>.jsonl` files of `PromptRow`, `references.jsonl` (the canonical
source plan of every record id, with polygons, for L4 closeness and the real reference),
`reference_tables.json` (real link shares and room-count bands) and `manifest.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.record import PlanRecord
from vlm_floorplan.requirements import Requirement, requirements_from_json
from vlm_floorplan.schema import FloorplanGraph

SET_NAMES = (
    "template_test",
    "template_val",
    "paraphrase_test",
    "behavior_test",
    "handwritten_test",
)
REFERENCES_FILE = "references.jsonl"
TABLES_FILE = "reference_tables.json"
MANIFEST_FILE = "manifest.json"
Behavior = Literal["draw", "edit", "restart", "contradiction", "unsupported"]


class PromptRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    set: str
    source: str
    record_id: str | None = None
    level: int | None = None
    """Specificity level 0-4 of a draw prompt; None for behavior prompts."""
    kind: str
    """Input kind (free, list, ...), or the behavior kind for behavior prompts."""
    behavior: Behavior = "draw"
    text: str
    text_source: Literal["template", "paraphrase", "handwritten"] = "template"
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    start: dict[str, Any]
    """The plan the model is shown: empty plus the exterior wall, or an existing plan."""
    paraphrase_of: str | None = None
    expect: dict[str, Any] = Field(default_factory=dict)
    """What a correct behavior looks like (see `behavior.py`)."""

    def parsed_requirements(self) -> list[Requirement]:
        return requirements_from_json(self.requirements)

    def start_graph(self) -> FloorplanGraph:
        return FloorplanGraph.model_validate(self.start)


def empty_start(graph: FloorplanGraph) -> dict[str, Any]:
    """The start state of a draw prompt: no rooms, and the exterior wall of the target plan."""
    start = graph.model_copy(deep=True)
    start.floors[0].rooms, start.floors[0].links = [], []
    return start.model_dump(mode="json")


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))


def read_rows(path: Path) -> list[PromptRow]:
    if not path.exists():
        return []
    return [PromptRow.model_validate_json(line) for line in path.read_text().splitlines() if line]


def read_references(directory: Path) -> dict[str, PlanRecord]:
    path = directory / REFERENCES_FILE
    if not path.exists():
        return {}
    records = [PlanRecord.model_validate_json(line) for line in path.read_text().splitlines()]
    return {record.id: record for record in records}
