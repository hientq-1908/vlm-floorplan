"""Per-plan metrics: raw counts that can be summed into dataset-level scores.

Each plan yields counts, not ratios. Aggregating then means summing counts and dividing
once ("micro" averaging), so a 30-room plan weighs more than a 3-room plan, the same
way it does in the data. Ratios for one plan are computed on demand with `ratio`.

What is measured, and why:

- Room recall / precision: did the model find the rooms, and only the rooms?
- Room-type accuracy: among matched rooms whose true type is known. Rooms the source
  labeled `unknown` are excluded, because the "truth" there is missing, not wrong.
- Link precision / recall / F1 per kind (door, opening, wall) and for "any" kind
  (is there a connection at all, whatever its kind). Two views are kept:
  `end_to_end` counts every ground-truth link, so a link between a missed room is a
  miss; `matched` only counts links between rooms that both sides found, which
  isolates connection quality from room detection quality.
- Entrance: accuracy of the flag on matched rooms, plus precision/recall because only
  a few rooms are entrances and accuracy alone is inflated by the many `False`.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from vlm_floorplan.evaluation.matching import RoomMatching, match_rooms
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import FloorplanGraph, LinkKind, RoomType

ANY_KIND = "any"
LINK_KINDS = [kind.value for kind in LinkKind] + [ANY_KIND]


class PrfCounts(BaseModel):
    """True positives, false positives and false negatives."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: PrfCounts) -> PrfCounts:
        return PrfCounts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def ratio(numerator: float, denominator: float) -> float | None:
    """Division that reports "undefined" (None) instead of guessing 0 or 1."""
    return None if denominator == 0 else numerator / denominator


class LinkCounts(BaseModel):
    """Link counts per kind (plus "any"), in the two views described above."""

    end_to_end: dict[str, PrfCounts] = Field(default_factory=dict)
    matched: dict[str, PrfCounts] = Field(default_factory=dict)


class PlanMetrics(BaseModel):
    """Everything measured on one plan. A plan with an invalid output has `valid=False`."""

    plan_id: str
    valid: bool
    gt_rooms: int
    pred_rooms: int = 0
    matched_rooms: int = 0
    gt_floors: int
    pred_floors: int = 0
    type_labeled: int = 0
    """Matched rooms whose ground-truth type is known."""
    type_correct: int = 0
    entrance_matched: int = 0
    entrance_correct: int = 0
    entrance: PrfCounts = Field(default_factory=PrfCounts)
    links: LinkCounts = Field(default_factory=LinkCounts)

    @property
    def room_count_error(self) -> int:
        """Signed: positive means the model predicted too many rooms."""
        return self.pred_rooms - self.gt_rooms


def invalid_plan_metrics(record: PlanRecord) -> PlanMetrics:
    """A plan whose output could not be parsed: every ground-truth room is a miss."""
    gt_links = {kind: 0 for kind in LINK_KINDS}
    for floor in record.graph.floors:
        for link in floor.links:
            gt_links[link.kind.value] += 1
            gt_links[ANY_KIND] += 1
    end_to_end = {kind: PrfCounts(fn=count) for kind, count in gt_links.items()}
    return PlanMetrics(
        plan_id=record.id,
        valid=False,
        gt_rooms=len(record.graph.rooms()),
        gt_floors=len(record.graph.floors),
        links=LinkCounts(end_to_end=end_to_end, matched={k: PrfCounts() for k in LINK_KINDS}),
    )


def _link_keys(
    graph: FloorplanGraph, to_gt: dict[str, str] | None
) -> set[tuple[frozenset[str], str]]:
    """(pair, kind) keys of a graph's links, optionally renamed through `to_gt`.

    Links with an end that `to_gt` does not know are left out; `_unmapped_by_kind`
    counts those.
    """
    keys: set[tuple[frozenset[str], str]] = set()
    for floor in graph.floors:
        for link in floor.links:
            if to_gt is None:
                pair = link.pair
            elif link.a in to_gt and link.b in to_gt:
                pair = frozenset((to_gt[link.a], to_gt[link.b]))
            else:
                continue
            keys.add((pair, link.kind.value))
    return keys


def _count_kind(
    gt_keys: Iterable[tuple[frozenset[str], str]],
    pred_keys: set[tuple[frozenset[str], str]],
    kind: str,
) -> PrfCounts:
    def select(keys: Iterable[tuple[frozenset[str], str]]) -> set[object]:
        if kind == ANY_KIND:
            return {pair for pair, _ in keys}
        return {pair for pair, k in keys if k == kind}

    gt, pred = select(gt_keys), select(pred_keys)
    return PrfCounts(tp=len(gt & pred), fp=len(pred - gt), fn=len(gt - pred))


def _link_counts(
    prediction: FloorplanGraph, record: PlanRecord, matching: RoomMatching
) -> LinkCounts:
    gt_keys = _link_keys(record.graph, None)
    pred_keys = _link_keys(prediction, matching.pred_to_gt)
    unmapped = _unmapped_by_kind(prediction, matching.pred_to_gt)
    matched_gt = set(matching.pred_to_gt.values())
    gt_between_matched = {(pair, k) for pair, k in gt_keys if pair <= matched_gt}

    end_to_end: dict[str, PrfCounts] = {}
    matched: dict[str, PrfCounts] = {}
    for kind in LINK_KINDS:
        counts = _count_kind(gt_keys, pred_keys, kind)
        # Predicted links with an unmatched end can never be right: false positives.
        end_to_end[kind] = PrfCounts(
            tp=counts.tp, fp=counts.fp + unmapped.get(kind, 0), fn=counts.fn
        )
        matched[kind] = _count_kind(gt_between_matched, pred_keys, kind)
    return LinkCounts(end_to_end=end_to_end, matched=matched)


def _unmapped_by_kind(prediction: FloorplanGraph, to_gt: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for floor in prediction.floors:
        for link in floor.links:
            if link.a not in to_gt or link.b not in to_gt:
                counts[link.kind.value] = counts.get(link.kind.value, 0) + 1
                counts[ANY_KIND] = counts.get(ANY_KIND, 0) + 1
    return counts


def _entrance_counts(
    prediction: FloorplanGraph, record: PlanRecord, matching: RoomMatching
) -> tuple[int, int, PrfCounts]:
    pred_rooms = {room.id: room for room in prediction.rooms()}
    gt_rooms = {room.id: room for room in record.graph.rooms()}
    correct = 0
    prf = PrfCounts()
    for pred_id, gt_id in matching.pred_to_gt.items():
        predicted, actual = pred_rooms[pred_id].entrance, gt_rooms[gt_id].entrance
        correct += predicted == actual
        prf = prf + PrfCounts(
            tp=int(predicted and actual),
            fp=int(predicted and not actual),
            fn=int(actual and not predicted),
        )
    return len(matching.pred_to_gt), correct, prf


def evaluate_plan(prediction: FloorplanGraph | None, record: PlanRecord) -> PlanMetrics:
    """Compare a predicted graph (None if the output was invalid) to the ground truth."""
    if prediction is None:
        return invalid_plan_metrics(record)

    matching = match_rooms(prediction, record)
    pred_rooms = {room.id: room for room in prediction.rooms()}
    gt_rooms = {room.id: room for room in record.graph.rooms()}

    type_labeled = type_correct = 0
    for pred_id, gt_id in matching.pred_to_gt.items():
        if gt_rooms[gt_id].type == RoomType.UNKNOWN:
            continue
        type_labeled += 1
        type_correct += pred_rooms[pred_id].type == gt_rooms[gt_id].type

    entrance_matched, entrance_correct, entrance = _entrance_counts(prediction, record, matching)
    return PlanMetrics(
        plan_id=record.id,
        valid=True,
        gt_rooms=len(gt_rooms),
        pred_rooms=len(pred_rooms),
        matched_rooms=len(matching.pred_to_gt),
        gt_floors=len(record.graph.floors),
        pred_floors=len(prediction.floors),
        type_labeled=type_labeled,
        type_correct=type_correct,
        entrance_matched=entrance_matched,
        entrance_correct=entrance_correct,
        entrance=entrance,
        links=_link_counts(prediction, record, matching),
    )
