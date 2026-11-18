from __future__ import annotations

import json

import pytest

from vlm_floorplan.compact import from_compact, to_compact
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room


def sample_graph() -> FloorplanGraph:
    rooms = [
        Room(id="r0", type="entry", xy=(474, 735), entrance=True),
        Room(id="r1", type="living", xy=(583, 295), label="KÄYTÄVÄ", bbox=(400, 100, 900, 500)),
    ]
    second = Floor(floor=2, rooms=[Room(id="r2", type="bedroom", xy=(10, 20))])
    first = Floor(floor=1, rooms=rooms, links=[Link(a="r0", b="r1", kind="door")])
    return FloorplanGraph(floors=[first, second], image=ImageSize(w=640, h=480))


def test_round_trip_is_lossless() -> None:
    graph = sample_graph()
    assert from_compact(to_compact(graph), graph.image) == graph


def test_compact_form_is_minimal() -> None:
    text = to_compact(sample_graph())
    assert " " not in text
    data = json.loads(text)
    assert data["floors"][0]["links"] == [["r0", "r1", "door"]]
    assert "entrance" not in data["floors"][0]["rooms"][1]  # defaults are omitted
    assert data["floors"][0]["rooms"][0]["entrance"] is True
    assert "KÄYTÄVÄ" in text  # not escaped to Ä


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"floors":1}',
        '{"floors":"ab"}',
        '{"floors":[1]}',
        '{"floors":[[]]}',
        '{"rooms": []}',
        '{"floors":[{"rooms":[],"links":[]}]}',  # no floor number
        '{"floors":[{"floor":1,"rooms":[],"links":[["r0","r1"]]}]}',  # malformed link
        '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"living","xy":[1,2]}],'
        '"links":[["r0","r9","door"]]}]}',  # dangling link
    ],
)
def test_malformed_model_output_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        from_compact(text, ImageSize(w=10, h=10))
