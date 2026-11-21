"""Match predicted rooms to ground-truth rooms.

The model only outputs a point per room, so there is no polygon to compute an
IoU from. Instead a predicted room is matched to the ground-truth room whose polygon
contains its point. That needs no tuning and is easy to explain: "the point landed
inside the kitchen".

Rules:

1. Matching is one-to-one. If two predicted points land in the same ground-truth room,
   the one closest to the ground-truth room's own point wins and the other counts as a
   false positive (a duplicate room).
2. A ground-truth room without a polygon (not every source keeps one) is matched by
   nearest point instead, using only the predictions that no polygon claimed, and only
   within `NEAREST_MAX_DISTANCE` grid units.
3. Floors are ignored: one image has one coordinate frame, so a point identifies a room
   no matter which floor graph the model put it in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import FloorplanGraph, Room

NEAREST_MAX_DISTANCE = 80.0
"""Grid units (of 1000). About one small room wide; farther than this is a different room."""


@dataclass(frozen=True)
class RoomMatching:
    """Result of matching one prediction against one ground truth."""

    pred_to_gt: dict[str, str]
    """Predicted room id -> ground-truth room id, for matched rooms only."""
    method: dict[str, str]
    """Predicted room id -> "polygon" or "nearest", for matched rooms only."""

    def gt_to_pred(self) -> dict[str, str]:
        return {gt: pred for pred, gt in self.pred_to_gt.items()}


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _prepared_polygons(record: PlanRecord) -> dict[str, object]:
    """Prepared polygon (all rings of a room merged) for each room that has any."""
    prepared: dict[str, object] = {}
    for room_id, source in record.rooms.items():
        parts = [Polygon(ring) for ring in source.polygon if len(ring) >= 4]
        parts = [part if part.is_valid else part.buffer(0) for part in parts]
        parts = [part for part in parts if not part.is_empty]
        if parts:
            prepared[room_id] = [prep(part) for part in parts]
    return prepared


def _match_by_polygon(
    preds: list[Room], gt_rooms: dict[str, Room], polygons: dict[str, object]
) -> dict[str, str]:
    """Each ground-truth room claims the closest predicted point that lies inside it."""
    claims: dict[str, tuple[float, str]] = {}
    for pred in preds:
        point = Point(*pred.xy)
        for gt_id, parts in polygons.items():
            # `covers` (not `contains`) so a point exactly on a shared wall still counts.
            if any(part.covers(point) for part in parts):  # type: ignore[attr-defined]
                dist = _distance(pred.xy, gt_rooms[gt_id].xy)
                if gt_id not in claims or dist < claims[gt_id][0]:
                    claims[gt_id] = (dist, pred.id)
    # Overlapping ground-truth polygons could let one prediction be claimed twice.
    # Resolve in favour of the closer claim so the result stays one-to-one.
    best_for_pred: dict[str, tuple[float, str]] = {}
    for gt_id, (dist, pred_id) in claims.items():
        if pred_id not in best_for_pred or dist < best_for_pred[pred_id][0]:
            best_for_pred[pred_id] = (dist, gt_id)
    return {pred_id: gt_id for pred_id, (_, gt_id) in best_for_pred.items()}


def _match_by_nearest(
    preds: list[Room], gt_rooms: list[Room], max_distance: float
) -> dict[str, str]:
    """Greedy nearest-point matching: closest pairs first, each room used once."""
    pairs = sorted(
        (_distance(pred.xy, gt.xy), pred.id, gt.id)
        for pred in preds
        for gt in gt_rooms
        if _distance(pred.xy, gt.xy) <= max_distance
    )
    matched: dict[str, str] = {}
    used_gt: set[str] = set()
    for _, pred_id, gt_id in pairs:
        if pred_id not in matched and gt_id not in used_gt:
            matched[pred_id] = gt_id
            used_gt.add(gt_id)
    return matched


def match_rooms(
    prediction: FloorplanGraph,
    record: PlanRecord,
    max_distance: float = NEAREST_MAX_DISTANCE,
) -> RoomMatching:
    """Match every predicted room to at most one ground-truth room."""
    gt_rooms = {room.id: room for room in record.graph.rooms()}
    preds = prediction.rooms()
    polygons = _prepared_polygons(record)

    pred_to_gt = _match_by_polygon(preds, gt_rooms, polygons)
    method = dict.fromkeys(pred_to_gt, "polygon")

    free_preds = [pred for pred in preds if pred.id not in pred_to_gt]
    taken = set(pred_to_gt.values())
    free_gt = [
        room for gt_id, room in gt_rooms.items() if gt_id not in polygons and gt_id not in taken
    ]
    nearest = _match_by_nearest(free_preds, free_gt, max_distance)
    pred_to_gt.update(nearest)
    method.update(dict.fromkeys(nearest, "nearest"))
    return RoomMatching(pred_to_gt=pred_to_gt, method=method)
