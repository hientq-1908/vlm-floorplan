import math

import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon
from t2g_fixtures import make_record

from vlm_floorplan.agent.env import plan_score
from vlm_floorplan.record import RoomSource
from vlm_floorplan.schema import RoomType
from vlm_floorplan.text2graph.frame import (
    FrameError,
    canonicalize,
    dominant_angle,
    indoor_points_inside_wall,
    unusable_reason,
)


def rotated(record, degrees):
    """The same plan with every polygon and point turned by `degrees` about (500, 500)."""
    new = record.model_copy(deep=True)
    for room_id, source in record.rooms.items():
        rings = [
            [
                tuple(affinity.rotate(Point(x, y), degrees, origin=(500, 500)).coords[0])
                for x, y in ring
            ]
            for ring in source.polygon
        ]
        new.rooms[room_id] = RoomSource(source_type=source.source_type, polygon=rings)
    for room in new.graph.floors[0].rooms:
        point = affinity.rotate(Point(room.xy), degrees, origin=(500, 500))
        room.xy = (round(point.x), round(point.y))
    return new


def test_the_frame_is_a_square_canvas_with_sizes_and_a_wall():
    canon = canonicalize(make_record(1)).record
    graph = canon.graph
    assert (graph.image.w, graph.image.h) == (1000, 1000)
    assert all(room.size and room.label is None for room in graph.rooms())
    xs = [x for room in graph.rooms() for x, _ in [room.xy]]
    assert min(xs) >= 60 and max(xs) <= 940
    assert graph.exterior_wall and len(graph.exterior_wall) >= 3
    assert indoor_points_inside_wall(graph)


def test_ids_follow_reading_order_and_links_are_sorted():
    graph = canonicalize(make_record(2)).record.graph
    points = [(room.xy[1], room.xy[0]) for room in graph.rooms()]
    assert points == sorted(points)
    assert [room.id for room in graph.rooms()] == [f"r{i}" for i in range(len(points))]
    pairs = [(int(link.a[1:]), int(link.b[1:])) for link in graph.floors[0].links]
    assert pairs == sorted(pairs) and all(a < b for a, b in pairs)


def test_every_point_stays_inside_its_room_polygon():
    canon = canonicalize(rotated(make_record(4), 27))
    for room in canon.record.graph.rooms():
        ring = canon.record.rooms[room.id].polygon[0]
        assert Polygon(ring).buffer(1).contains(Point(room.xy))


def test_a_rotated_plan_is_straightened_and_keeps_its_graph():
    original = canonicalize(make_record(5)).record.graph
    turned = canonicalize(rotated(make_record(5), 27))
    assert abs(turned.rotation_deg) == pytest.approx(27, abs=1)
    rings = [Polygon(source.polygon[0]) for source in turned.record.rooms.values()]
    assert dominant_angle(rings) == pytest.approx(0, abs=1)
    again = turned.record.graph
    assert sorted(r.type.value for r in again.rooms()) == sorted(
        r.type.value for r in original.rooms()
    )
    assert len(again.floors[0].links) == len(original.floors[0].links)
    assert sorted(link.kind.value for link in again.floors[0].links) == sorted(
        link.kind.value for link in original.floors[0].links
    )


def test_the_canonical_plan_scores_one_against_itself():
    canon = canonicalize(make_record(6)).record
    assert plan_score(canon.graph, canon) == pytest.approx(1.0)


def test_dominant_angle_ignores_quarter_turns():
    box = Polygon([(0, 0), (40, 0), (40, 10), (0, 10)])
    for degrees in (0, 90, 180):
        assert dominant_angle([affinity.rotate(box, degrees, origin=(0, 0))]) == pytest.approx(
            0, abs=1e-6
        )
    assert dominant_angle([affinity.rotate(box, 20, origin=(0, 0))]) == pytest.approx(20, abs=1e-6)
    assert math.isfinite(dominant_angle([]))


def test_unusable_plans_say_why():
    record = make_record(7)
    record.graph.floors[0].rooms[0].type = RoomType.UNKNOWN
    assert unusable_reason(record) == "an unknown room"
    with pytest.raises(FrameError):
        canonicalize(record)
