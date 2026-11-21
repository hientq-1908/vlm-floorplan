from __future__ import annotations

import json

import pytest

from vlm_floorplan.compact import from_compact, to_compact
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

WALL = [(10, 10), (10, 900), (900, 900)]


def sample_graph() -> FloorplanGraph:
    rooms = [
        Room(id="r0", type="entry", xy=(474, 735), size=52, entrance=True),
        Room(id="r1", type="living", xy=(583, 295), size=160, label="KÄYTÄVÄ"),
        Room(id="r2", type="bedroom", xy=(10, 20)),
    ]
    floor = Floor(floor=1, rooms=rooms, links=[Link(a="r0", b="r1", kind="door")])
    return FloorplanGraph(floors=[floor], image=ImageSize(w=640, h=480), exterior_wall=WALL)


def test_round_trip_is_lossless() -> None:
    graph = sample_graph()
    assert from_compact(to_compact(graph), graph.image, graph.exterior_wall) == graph


def test_compact_form_is_flat_and_minimal() -> None:
    text = to_compact(sample_graph())
    assert " " not in text
    assert "floor" not in text
    data = json.loads(text)
    assert list(data) == ["rooms", "links"]
    assert data["links"] == [["r0", "r1", "door"]]
    assert "entrance" not in data["rooms"][1]  # defaults are omitted
    assert data["rooms"][0]["entrance"] is True
    assert "KÄYTÄVÄ" in text  # not escaped to Ä


def test_a_room_is_written_as_a_point_with_a_size() -> None:
    rooms = json.loads(to_compact(sample_graph()))["rooms"]
    assert list(rooms[0]) == ["id", "type", "xy", "size", "entrance"]
    assert set(rooms[1]) == {"id", "type", "label", "xy", "size"}
    assert set(rooms[2]) == {"id", "type", "xy"}  # a room without a size writes none


def test_the_exterior_wall_is_never_written() -> None:
    assert "wall" not in to_compact(sample_graph())


def test_the_exterior_wall_comes_from_the_caller() -> None:
    text = to_compact(sample_graph())
    assert from_compact(text, ImageSize(w=1, h=1)).exterior_wall is None
    assert from_compact(text, ImageSize(w=1, h=1), WALL).exterior_wall == WALL


def test_a_plan_with_several_floors_has_no_compact_form() -> None:
    graph = sample_graph()
    second = Floor(floor=2, rooms=[Room(id="r3", type="bedroom", xy=(1, 1))])
    with pytest.raises(ValueError, match="single-floor"):
        to_compact(graph.model_copy(update={"floors": [*graph.floors, second]}))


def test_the_floors_wrapper_is_rejected_with_the_flat_form_named() -> None:
    wrapped = '{"floors":[{"floor":1,"rooms":[],"links":[]}]}'
    with pytest.raises(ValueError, match=r'write \{"rooms"'):
        from_compact(wrapped, ImageSize(w=640, h=480))


def test_a_compact_room_with_a_bbox_is_rejected() -> None:
    old = '{"rooms":[{"id":"r0","type":"living","xy":[5,5],"bbox":[0,0,10,10]}],"links":[]}'
    with pytest.raises(ValueError, match="bbox"):
        from_compact(old, ImageSize(w=640, h=480))


@pytest.mark.parametrize("size", [0, -3, 501, 2.5, "big"])
def test_a_room_size_outside_the_grid_radius_is_rejected(size: object) -> None:
    text = json.dumps({"rooms": [{"id": "r0", "type": "living", "xy": [5, 5], "size": size}]})
    with pytest.raises(ValueError, match="size"):
        from_compact(text, ImageSize(w=10, h=10))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"rooms":1}',
        '{"rooms":"ab"}',
        '{"rooms":[1]}',
        '{"links": []}',
        '{"rooms":[],"links":[["r0","r1"]]}',  # malformed link
        '{"rooms":[],"links":"x"}',
        '{"rooms":[{"id":"r0","type":"living","xy":[1,2]}],"links":[["r0","r9","door"]]}',
    ],
)
def test_malformed_model_output_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        from_compact(text, ImageSize(w=10, h=10))
