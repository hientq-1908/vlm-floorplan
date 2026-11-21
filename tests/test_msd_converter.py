"""MSD converter: safe pickle loading, link translation, entrance flag, geometry."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from msd_fixture import (
    APARTMENT_EDGES,
    APARTMENT_ROOMS,
    BATHROOM,
    CORRIDOR,
    apartment_pickle,
    msd_pickle,
)

from vlm_floorplan.converters.msd import convert_plan
from vlm_floorplan.converters.msd_pickle import parse_msd_graph
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import LinkKind, Room, RoomType


def _by_source(record: PlanRecord) -> dict[str, Room]:
    """Rooms keyed by MSD type name (unique in the fixture, apart from none)."""
    return {record.rooms[room.id].source_type: room for room in record.graph.rooms()}


def _kinds(record: PlanRecord) -> dict[frozenset[str], LinkKind]:
    return {link.pair: link.kind for link in record.graph.floors[0].links}


def test_parse_reads_nodes_and_edges_without_importing_networkx_or_torch() -> None:
    watched = ("networkx", "torch")
    before = {name: sys.modules.get(name) for name in watched}  # torch may be loaded already
    pickled = apartment_pickle()
    graph = parse_msd_graph(pickled)

    assert set(graph.nodes) == set(APARTMENT_ROOMS)
    assert graph.nodes[1].room_type == 0
    assert graph.nodes[1].polygon[0] == (3.2, 0.0)
    assert {(e.a, e.b, e.connectivity) for e in graph.edges} == {
        (min(a, b), max(a, b), kind) for a, b, kind in APARTMENT_EDGES
    }
    assert {name: sys.modules.get(name) for name in watched} == before


def test_parse_never_runs_code_stored_in_the_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return os.system, (f"touch {marker}",)

    import pickle

    with pytest.raises(ValueError, match="not an MSD"):
        parse_msd_graph(pickle.dumps(Evil()))
    assert not marker.exists()


def test_parse_rejects_data_that_is_not_a_pickle() -> None:
    with pytest.raises(ValueError, match="not an MSD"):
        parse_msd_graph(b"definitely not a pickle")


def test_rooms_get_coarse_types_and_keep_msd_name_as_source_type() -> None:
    record = convert_plan(apartment_pickle(), "7")

    assert record.id == "msd/7" and record.source == "msd"
    rooms = _by_source(record)
    assert rooms["Corridor"].type == RoomType.ENTRY
    assert rooms["Livingroom"].type == RoomType.LIVING
    assert rooms["Balcony"].type == RoomType.TERRACE
    assert rooms["Stairs"].type == RoomType.STAIRS
    assert record.quality.unlabeled_fraction == 0.0


def test_edges_become_door_and_opening_links_and_touching_rooms_a_wall_link() -> None:
    record = convert_plan(apartment_pickle(), "7")
    rooms = _by_source(record)
    pair = lambda a, b: frozenset((rooms[a].id, rooms[b].id))  # noqa: E731
    kinds = _kinds(record)

    assert kinds[pair("Corridor", "Bedroom")] == LinkKind.DOOR
    assert kinds[pair("Kitchen", "Livingroom")] == LinkKind.OPENING
    assert kinds[pair("Corridor", "Stairs")] == LinkKind.DOOR  # the entrance edge
    assert kinds[pair("Corridor", "Kitchen")] == LinkKind.WALL  # 0.2 m apart, no edge
    assert kinds[pair("Bedroom", "Livingroom")] == LinkKind.WALL
    assert pair("Bedroom", "Stairs") not in kinds  # far apart


def test_entrance_flag_goes_on_the_apartment_side_of_the_entrance_edge() -> None:
    record = convert_plan(apartment_pickle(), "7")
    entrances = [room for room in record.graph.rooms() if room.entrance]

    # The hall has bedrooms and a kitchen behind it; the stairwell has nothing.
    assert [record.rooms[room.id].source_type for room in entrances] == ["Corridor"]
    assert entrances[0].xy[0] > 0  # the hall, not the stairwell on its left


def test_entrance_between_two_rooms_with_no_dwelling_behind_either_flags_both() -> None:
    data = msd_pickle(
        {0: (CORRIDOR, (0, 0, 3, 2)), 1: (CORRIDOR, (3.2, 0, 6, 2))}, [(0, 1, "entrance")]
    )
    record = convert_plan(data, "1")

    assert [room.entrance for room in record.graph.rooms()] == [True, True]


def test_graph_is_graph_only_with_a_virtual_image_and_y_flipped() -> None:
    record = convert_plan(apartment_pickle(), "7")
    rooms = _by_source(record)

    assert record.image == ""
    assert max(record.graph.image.w, record.graph.image.h) == 1024
    assert all(0 <= v <= 1000 for room in record.graph.rooms() for v in room.xy)
    # Kitchen is at larger y (further "up" in the plan) than the hall: smaller grid y.
    assert rooms["Kitchen"].xy[1] < rooms["Corridor"].xy[1]
    # Reading order: ids run top to bottom, then left to right.
    top_row = sorted(record.graph.rooms(), key=lambda room: (room.xy[1], room.xy[0]))
    assert [room.id for room in top_row] == [f"r{i}" for i in range(len(top_row))]


def test_unknown_connectivity_is_reported_and_not_linked() -> None:
    data = msd_pickle(
        {0: (CORRIDOR, (0, 0, 3, 2)), 1: (BATHROOM, (10, 0, 13, 2))}, [(0, 1, "teleporter")]
    )
    record = convert_plan(data, "1")

    assert not record.graph.floors[0].links
    assert any("teleporter" in warning for warning in record.quality.warnings)


def test_a_node_that_is_not_a_room_class_is_an_error() -> None:
    data = msd_pickle({0: (9, (0, 0, 3, 2))}, [])  # 9 = Structure, image-only

    with pytest.raises(ValueError, match="not a room class"):
        convert_plan(data, "1")


def test_a_graph_without_rooms_is_an_error() -> None:
    with pytest.raises(ValueError, match="no rooms"):
        convert_plan(msd_pickle({}, []), "1")


def test_record_survives_a_json_round_trip() -> None:
    record = convert_plan(apartment_pickle(), "7")

    assert PlanRecord.model_validate_json(record.model_dump_json()) == record
