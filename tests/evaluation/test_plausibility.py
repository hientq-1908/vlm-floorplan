import pytest

from vlm_floorplan.agent.sample_plan import sample_record
from vlm_floorplan.evaluation.plausibility import (
    ReferenceTables,
    aggregate_plausibility,
    build_reference_tables,
    percentile,
    plan_plausibility,
    segments_cross,
)
from vlm_floorplan.schema import Link


def test_segments_cross_only_when_they_properly_intersect():
    assert segments_cross((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_cross((0, 0), (10, 0), (0, 5), (10, 5))
    assert not segments_cross((0, 0), (10, 10), (10, 10), (20, 0))


def test_percentile_interpolates_and_handles_nothing():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert percentile([], 95) is None


def test_a_real_plan_is_circulable_with_no_crossings():
    plan = plan_plausibility(sample_record().graph, ReferenceTables())
    assert plan.unreachable_rooms == 0 and not plan.has_crossing and not plan.warnings
    assert plan.in_band is None  # no reference tables given


def test_a_room_reachable_only_through_a_wall_is_not_circulable():
    graph = sample_record().graph.model_copy(deep=True)
    floor = graph.floors[0]
    floor.links = [
        Link(a=link.a, b=link.b, kind="wall") if "r5" in (link.a, link.b) else link
        for link in floor.links
    ]
    assert plan_plausibility(graph, ReferenceTables()).unreachable_rooms == 1


def test_crossing_links_are_found():
    graph = sample_record().graph.model_copy(deep=True)
    floor = graph.floors[0]
    floor.links = [*floor.links, Link(a="r0", b="r4", kind="wall")]  # across r3's links
    assert plan_plausibility(graph, ReferenceTables()).has_crossing


def test_reference_tables_mark_unseen_link_types_as_rare_and_set_a_room_band():
    graph = sample_record().graph
    tables = build_reference_tables([graph] * 25)
    assert tables.room_count_band["1"] == (6, 6)
    plan = plan_plausibility(graph, tables)
    assert plan.rare_links == 0 and plan.in_band is True
    odd = graph.model_copy(deep=True)
    odd.floors[0].links[0] = Link(a="r0", b="r1", kind="door")
    assert plan_plausibility(odd, tables).rare_links == 1


def test_aggregate_counts_first_and_returns_none_for_nothing():
    plans = [plan_plausibility(sample_record().graph, ReferenceTables())] * 2
    summary = aggregate_plausibility(plans)
    assert summary["circulation_rate"] == 1.0 and summary["crossing_rate"] == 0.0
    empty = aggregate_plausibility([])
    assert empty["circulation_rate"] is None and empty["link_length_ratio_p95"] is None
