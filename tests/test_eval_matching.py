from __future__ import annotations

from helpers_eval import gt_record, prediction, shifted_ids_prediction

from vlm_floorplan.evaluation.matching import match_rooms


def test_points_match_the_polygon_that_contains_them() -> None:
    matching = match_rooms(shifted_ids_prediction(), gt_record())
    assert matching.pred_to_gt == {"r10": "r0", "r11": "r1", "r12": "r2", "r13": "r3"}
    assert set(matching.method.values()) == {"polygon"}


def test_second_point_in_the_same_room_is_a_duplicate_not_a_match() -> None:
    pred = prediction(
        [
            {"id": "r0", "type": "living", "xy": (480, 480)},  # far from the room's own point
            {"id": "r1", "type": "living", "xy": (260, 240)},  # closest to the GT point (250, 250)
        ]
    )
    matching = match_rooms(pred, gt_record())
    assert matching.pred_to_gt == {"r1": "r0"}


def test_point_outside_every_polygon_is_unmatched() -> None:
    # Polygons cover the whole grid, so use a record whose polygons leave a gap.
    record = gt_record()
    record.rooms["r0"].polygon = [[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]]
    pred = prediction([{"id": "r0", "type": "living", "xy": (400, 400)}])
    assert match_rooms(pred, record).pred_to_gt == {}


def test_point_on_a_shared_wall_still_matches_one_room() -> None:
    pred = prediction([{"id": "r0", "type": "living", "xy": (500, 250)}])
    assert len(match_rooms(pred, gt_record()).pred_to_gt) == 1


def test_without_polygons_nearest_point_is_used_within_the_distance_limit() -> None:
    record = gt_record(with_polygons=False)
    pred = prediction(
        [
            {"id": "r0", "type": "living", "xy": (270, 260)},  # 22 from GT r0 at (250, 250)
            {"id": "r1", "type": "kitchen", "xy": (0, 0)},  # 353+ from every GT point
        ]
    )
    matching = match_rooms(pred, record)
    assert matching.pred_to_gt == {"r0": "r0"}
    assert matching.method == {"r0": "nearest"}


def test_nearest_matching_is_one_to_one() -> None:
    record = gt_record(with_polygons=False)
    pred = prediction(
        [
            {"id": "r0", "type": "living", "xy": (260, 250)},
            {"id": "r1", "type": "living", "xy": (300, 250)},
        ]
    )
    matching = match_rooms(pred, record)
    assert matching.pred_to_gt == {"r0": "r0"}  # r1 is also near GT r0, but r0 is taken


def test_polygon_rooms_are_not_matched_by_nearest_fallback() -> None:
    pred = prediction([{"id": "r0", "type": "living", "xy": (255, 255)}])
    record = gt_record()
    record.rooms["r0"].polygon = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
    assert match_rooms(pred, record).pred_to_gt == {}
