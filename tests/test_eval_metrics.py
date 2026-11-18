from __future__ import annotations

import pytest
from helpers_eval import gt_record, perfect_prediction, prediction, shifted_ids_prediction

from vlm_floorplan.evaluation.aggregate import aggregate, aggregate_by_size, size_bin
from vlm_floorplan.evaluation.metrics import ANY_KIND, PrfCounts, evaluate_plan
from vlm_floorplan.schema import LinkKind


def test_perfect_prediction_scores_one_everywhere() -> None:
    metrics = evaluate_plan(perfect_prediction(), gt_record())
    agg = aggregate([metrics])
    assert agg.room_recall == 1.0
    assert agg.room_precision == 1.0
    assert agg.room_count_mae == 0.0
    assert agg.type_accuracy == 1.0
    assert agg.entrance_accuracy == 1.0
    for kind in ("door", "opening", "wall", ANY_KIND):
        assert agg.links[kind].end_to_end.f1 == 1.0, kind
        assert agg.links[kind].matched.f1 == 1.0, kind


def test_matching_does_not_depend_on_predicted_ids() -> None:
    agg = aggregate([evaluate_plan(shifted_ids_prediction(), gt_record())])
    assert agg.room_recall == 1.0
    assert agg.links[ANY_KIND].end_to_end.f1 == 1.0


def test_unlabeled_rooms_are_excluded_from_type_accuracy() -> None:
    # GT r2 is "unknown"; the prediction calls it bedroom. That must not count as an error.
    metrics = evaluate_plan(shifted_ids_prediction(), gt_record())
    assert metrics.type_labeled == 3
    assert metrics.type_correct == 3


def test_wrong_type_is_counted_on_labeled_rooms() -> None:
    pred = prediction(
        [
            {"id": "r0", "type": "bedroom", "xy": (100, 100)},  # GT living
            {"id": "r1", "type": "kitchen", "xy": (900, 100)},
        ]
    )
    metrics = evaluate_plan(pred, gt_record())
    assert (metrics.type_labeled, metrics.type_correct) == (2, 1)


def test_missing_and_extra_rooms_affect_recall_precision_and_count_error() -> None:
    pred = prediction(
        [
            {"id": "r0", "type": "living", "xy": (100, 100)},
            {"id": "r1", "type": "kitchen", "xy": (900, 100)},
            {"id": "r2", "type": "kitchen", "xy": (120, 120)},  # duplicate in the living room
        ]
    )
    metrics = evaluate_plan(pred, gt_record())
    assert (metrics.gt_rooms, metrics.pred_rooms, metrics.matched_rooms) == (4, 3, 2)
    assert metrics.room_count_error == -1
    agg = aggregate([metrics])
    assert agg.room_recall == 0.5
    assert agg.room_precision == pytest.approx(2 / 3)
    assert agg.room_count_bias == -1.0


def test_link_with_wrong_kind_is_a_miss_for_that_kind_but_counts_for_any() -> None:
    pred = shifted_ids_prediction()
    first = pred.floors[0].links[0]  # was an opening
    pred.floors[0].links[0] = first.model_copy(update={"kind": LinkKind.DOOR})
    agg = aggregate([evaluate_plan(pred, gt_record())])
    assert agg.links["opening"].end_to_end.recall == 0.0
    assert agg.links["opening"].end_to_end.precision is None  # nothing predicted as opening
    assert agg.links["door"].end_to_end.precision == pytest.approx(2 / 3)
    assert agg.links[ANY_KIND].end_to_end.f1 == 1.0


def test_link_to_an_unmatched_room_is_a_false_positive_only_end_to_end() -> None:
    pred = prediction(
        [
            {"id": "r0", "type": "living", "xy": (100, 100)},
            {"id": "r1", "type": "kitchen", "xy": (900, 100)},
            {
                "id": "r9",
                "type": "other",
                "xy": (2, 2),
            },  # r9 lands in the living room too: duplicate
        ],
        [["r0", "r1", "opening"], ["r0", "r9", "door"]],
    )
    metrics = evaluate_plan(pred, gt_record())
    door_e2e = metrics.links.end_to_end["door"]
    assert (door_e2e.tp, door_e2e.fp, door_e2e.fn) == (0, 1, 2)
    door_matched = metrics.links.matched["door"]
    assert (door_matched.tp, door_matched.fp, door_matched.fn) == (0, 0, 0)
    opening = metrics.links.matched["opening"]
    assert (opening.tp, opening.fp, opening.fn) == (1, 0, 0)


def test_entrance_precision_recall_and_accuracy() -> None:
    pred = shifted_ids_prediction()
    rooms = pred.floors[0].rooms
    rooms[0] = rooms[0].model_copy(update={"entrance": True})  # wrongly flagged
    rooms[3] = rooms[3].model_copy(update={"entrance": False})  # missed
    metrics = evaluate_plan(pred, gt_record())
    assert metrics.entrance == PrfCounts(tp=0, fp=1, fn=1)
    assert (metrics.entrance_matched, metrics.entrance_correct) == (4, 2)


def test_invalid_output_counts_every_room_and_link_as_missed() -> None:
    metrics = evaluate_plan(None, gt_record())
    assert not metrics.valid
    assert metrics.links.end_to_end["door"].fn == 2
    agg = aggregate([metrics, evaluate_plan(perfect_prediction(), gt_record())])
    assert agg.validity_rate == 0.5
    assert agg.room_recall == 1.0  # over valid outputs only
    assert agg.room_recall_incl_invalid == 0.5


def test_empty_aggregate_reports_undefined_not_zero() -> None:
    agg = aggregate([])
    assert agg.validity_rate is None
    assert agg.room_recall is None


@pytest.mark.parametrize(
    ("rooms", "expected"),
    [(1, "1-6 rooms"), (6, "1-6 rooms"), (7, "7-10 rooms"), (15, "11-15 rooms"), (40, "16+ rooms")],
)
def test_size_bins(rooms: int, expected: str) -> None:
    assert size_bin(rooms) == expected


def test_breakdown_by_size_only_lists_populated_bins() -> None:
    by_size = aggregate_by_size([evaluate_plan(perfect_prediction(), gt_record())])
    assert list(by_size) == ["1-6 rooms"]
    assert by_size["1-6 rooms"].n_plans == 1
