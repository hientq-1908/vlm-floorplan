"""Aggregate per-plan counts into dataset-level scores, overall and by plan size.

Scores are micro-averaged: counts are summed over plans, then divided once.
Ratios whose denominator is zero are `None` (undefined), never 0 or 1, so a missing
score cannot be mistaken for a real one.

Room scores are computed over valid outputs only (an invalid output has no rooms to
score), and `room_recall_incl_invalid` shows the stricter view where an invalid output
misses every room. Read the two together with `validity_rate`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from vlm_floorplan.evaluation.metrics import LINK_KINDS, PlanMetrics, PrfCounts, ratio

SIZE_BINS: list[tuple[str, int, int | None]] = [
    ("1-6 rooms", 1, 6),
    ("7-10 rooms", 7, 10),
    ("11-15 rooms", 11, 15),
    ("16+ rooms", 16, None),
]
"""Plan-size bins by ground-truth room count. CubiCasa5K floors have a median of 9 rooms."""


class PrfScores(BaseModel):
    """Precision, recall and F1 with the counts they come from."""

    precision: float | None
    recall: float | None
    f1: float | None
    tp: int
    fp: int
    fn: int

    @classmethod
    def from_counts(cls, counts: PrfCounts) -> PrfScores:
        return cls(
            precision=counts.precision,
            recall=counts.recall,
            f1=counts.f1,
            tp=counts.tp,
            fp=counts.fp,
            fn=counts.fn,
        )


class LinkScores(BaseModel):
    end_to_end: PrfScores
    matched: PrfScores


class AggregateMetrics(BaseModel):
    n_plans: int
    n_valid: int
    validity_rate: float | None
    room_recall: float | None
    room_precision: float | None
    room_recall_incl_invalid: float | None
    room_count_mae: float | None
    room_count_bias: float | None
    """Mean signed error: positive means the model predicts too many rooms."""
    floor_count_accuracy: float | None
    type_accuracy: float | None
    type_labeled_rooms: int
    entrance_accuracy: float | None
    entrance: PrfScores
    links: dict[str, LinkScores] = Field(default_factory=dict)


def aggregate(plans: Sequence[PlanMetrics]) -> AggregateMetrics:
    valid = [plan for plan in plans if plan.valid]
    gt_valid = sum(plan.gt_rooms for plan in valid)
    gt_all = sum(plan.gt_rooms for plan in plans)
    matched = sum(plan.matched_rooms for plan in valid)
    pred = sum(plan.pred_rooms for plan in valid)
    type_labeled = sum(plan.type_labeled for plan in valid)

    return AggregateMetrics(
        n_plans=len(plans),
        n_valid=len(valid),
        validity_rate=ratio(len(valid), len(plans)),
        room_recall=ratio(matched, gt_valid),
        room_precision=ratio(matched, pred),
        room_recall_incl_invalid=ratio(matched, gt_all),
        room_count_mae=ratio(sum(abs(p.room_count_error) for p in valid), len(valid)),
        room_count_bias=ratio(sum(p.room_count_error for p in valid), len(valid)),
        floor_count_accuracy=ratio(sum(p.gt_floors == p.pred_floors for p in valid), len(valid)),
        type_accuracy=ratio(sum(p.type_correct for p in valid), type_labeled),
        type_labeled_rooms=type_labeled,
        entrance_accuracy=ratio(
            sum(p.entrance_correct for p in valid), sum(p.entrance_matched for p in valid)
        ),
        entrance=PrfScores.from_counts(_sum_counts(p.entrance for p in valid)),
        links={kind: _link_scores(valid, kind) for kind in LINK_KINDS},
    )


def _sum_counts(counts: Iterable[PrfCounts]) -> PrfCounts:
    total = PrfCounts()
    for item in counts:
        total = total + item
    return total


def _link_scores(valid: Sequence[PlanMetrics], kind: str) -> LinkScores:
    return LinkScores(
        end_to_end=PrfScores.from_counts(_sum_counts(p.links.end_to_end[kind] for p in valid)),
        matched=PrfScores.from_counts(_sum_counts(p.links.matched[kind] for p in valid)),
    )


def size_bin(gt_rooms: int) -> str:
    for name, low, high in SIZE_BINS:
        if gt_rooms >= low and (high is None or gt_rooms <= high):
            return name
    return SIZE_BINS[0][0]  # a plan with no rooms at all is "small"


def aggregate_by_size(plans: Sequence[PlanMetrics]) -> dict[str, AggregateMetrics]:
    """One aggregate per plan-size bin; bins with no plans are omitted."""
    grouped: dict[str, list[PlanMetrics]] = {}
    for plan in plans:
        grouped.setdefault(size_bin(plan.gt_rooms), []).append(plan)
    return {name: aggregate(grouped[name]) for name, _, _ in SIZE_BINS if name in grouped}
