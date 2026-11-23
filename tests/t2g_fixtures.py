"""Synthetic plans for the text-to-graph tests: small grids of rooms with polygons."""

from __future__ import annotations

import json
import random
from pathlib import Path

from vlm_floorplan.record import PlanQuality, PlanRecord, RoomSource
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

_TYPES = ["living", "kitchen", "bedroom", "bedroom", "bathroom", "entry", "bedroom", "storage"]


def make_record(seed: int, source: str = "resplan", split: str | None = "train", cols: int = 4):
    """A 4 x 2 grid of rooms with random types and links, deterministic in `seed`."""
    rng = random.Random(seed)
    types = _TYPES[:]
    rng.shuffle(types)
    cell_w, cell_h = 1000 // cols, 500
    rooms, sources, boxes = [], {}, {}
    for index, room_type in enumerate(types):
        col, row = index % cols, index // cols
        x0, y0 = col * cell_w, 100 + row * cell_h
        box = (x0, y0, x0 + cell_w, y0 + cell_h)
        room_id = f"r{index}"
        boxes[(col, row)] = room_id
        rooms.append(
            Room(
                id=room_id,
                type=room_type,
                xy=((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                size=100,
                entrance=room_type in ("entry",),
            )
        )
        ring = [
            (box[0], box[1]),
            (box[2], box[1]),
            (box[2], box[3]),
            (box[0], box[3]),
            (box[0], box[1]),
        ]
        sources[room_id] = RoomSource(source_type=room_type, polygon=[ring])
    links = []
    for (col, row), room_id in boxes.items():
        for other in ((col + 1, row), (col, row + 1)):
            if other in boxes:
                kind = rng.choice(["door", "door", "opening", "wall"])
                links.append(Link(a=room_id, b=boxes[other], kind=kind))
    graph = FloorplanGraph(
        floors=[Floor(floor=1, rooms=rooms, links=links)], image=ImageSize(w=1000, h=1000)
    )
    return PlanRecord(
        id=f"{source}/{seed}",
        source=source,
        split=split,
        image="",
        graph=graph,
        rooms=sources,
        quality=PlanQuality.from_graph(graph, []),
    )


def write_split(path: Path, records: list[PlanRecord]) -> Path:
    path.write_text("".join(json.dumps(json.loads(r.model_dump_json())) + "\n" for r in records))
    return path
