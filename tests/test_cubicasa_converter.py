"""CubiCasa5K converter, on a synthetic plan (see cubicasa_fixture.py for its layout)."""

from __future__ import annotations

from pathlib import Path

import pytest
from cubicasa_fixture import HEIGHT, WIDTH, write_cubicasa_plan, write_wall_cross_plan

from vlm_floorplan.compact import from_compact, to_compact
from vlm_floorplan.converters.cubicasa import convert_plan
from vlm_floorplan.record import PlanQuality, PlanRecord
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, LinkKind, Room, RoomType


def _by_type(record: PlanRecord) -> dict[str, Room]:
    return {room.type.value: room for room in record.graph.rooms()}


def _links(record: PlanRecord, floor: int = 1) -> dict[frozenset[str], LinkKind]:
    graph_floor = next(f for f in record.graph.floors if f.floor == floor)
    return {link.pair: link.kind for link in graph_floor.links}


def test_rooms_are_typed_and_numbered_in_reading_order(cubicasa_record: PlanRecord) -> None:
    rooms = cubicasa_record.graph.rooms()
    assert [room.id for room in rooms] == [f"r{i}" for i in range(6)]
    assert [room.type.value for room in rooms] == [
        "kitchen", "living", "dining", "bedroom", "unknown", "storage"
    ]  # fmt: skip
    ys = [room.xy[1] for room in rooms]
    assert ys == sorted(ys), "rooms run top to bottom"


def test_points_are_normalized_to_the_scaled_image_and_inside_their_rooms(
    cubicasa_record: PlanRecord,
) -> None:
    assert cubicasa_record.graph.image == ImageSize(w=WIDTH, h=HEIGHT)
    kitchen = _by_type(cubicasa_record)["kitchen"]
    # kitchen spans x 20-290 and y 20-200 of a 600 x 400 image: (33, 50) to (483, 500) on the grid
    x, y = kitchen.xy
    assert 33 <= x <= 483 and 50 <= y <= 500


def test_kitchen_and_living_room_are_joined_by_a_door(cubicasa_record: PlanRecord) -> None:
    rooms = _by_type(cubicasa_record)
    pair = frozenset({rooms["kitchen"].id, rooms["living"].id})
    assert _links(cubicasa_record)[pair] == LinkKind.DOOR


def test_touching_rooms_get_an_opening_link(cubicasa_record: PlanRecord) -> None:
    rooms = _by_type(cubicasa_record)
    pair = frozenset({rooms["living"].id, rooms["dining"].id})
    assert _links(cubicasa_record)[pair] == LinkKind.OPENING


def test_rooms_separated_only_by_a_wall_get_a_wall_link(cubicasa_record: PlanRecord) -> None:
    rooms = _by_type(cubicasa_record)
    pair = frozenset({rooms["kitchen"].id, rooms["bedroom"].id})
    assert _links(cubicasa_record)[pair] == LinkKind.WALL


def test_rooms_meeting_only_diagonally_at_a_wall_cross_are_not_linked(tmp_path: Path) -> None:
    record = convert_plan(write_wall_cross_plan(tmp_path / "p"), "cubicasa5k/x/1")
    rooms = _by_type(record)

    def pair(first: str, second: str) -> frozenset[str]:
        return frozenset({rooms[first].id, rooms[second].id})

    assert _links(record) == {
        pair("kitchen", "living"): LinkKind.WALL,
        pair("kitchen", "bedroom"): LinkKind.WALL,
        pair("living", "dining"): LinkKind.WALL,
        pair("bedroom", "dining"): LinkKind.WALL,
    }


def test_distant_rooms_are_not_linked(cubicasa_record: PlanRecord) -> None:
    rooms = _by_type(cubicasa_record)
    assert frozenset({rooms["kitchen"].id, rooms["dining"].id}) not in _links(cubicasa_record)


def test_exterior_door_sets_the_entrance_flag_on_one_room(cubicasa_record: PlanRecord) -> None:
    entrances = [room.type.value for room in cubicasa_record.graph.rooms() if room.entrance]
    assert entrances == ["living"]


def test_doors_whose_parent_is_not_a_wall_are_ignored(cubicasa_record: PlanRecord) -> None:
    """The fridge door sits inside the kitchen; it must not become a link or an entrance."""
    doors = [kind for kind in _links(cubicasa_record).values() if kind == LinkKind.DOOR]
    assert len(doors) == 1
    assert not _by_type(cubicasa_record)["kitchen"].entrance


def test_invalid_polygon_is_repaired_to_its_largest_part(cubicasa_record: PlanRecord) -> None:
    storage = _by_type(cubicasa_record)["storage"]
    assert any("repaired invalid Space Storage" in w for w in cubicasa_record.quality.warnings)
    assert cubicasa_record.rooms[storage.id].polygon, "the repaired outline is kept"


def test_finnish_printed_name_types_a_userdefined_room(cubicasa_record: PlanRecord) -> None:
    bedroom = _by_type(cubicasa_record)["bedroom"]
    assert bedroom.label == "MH"
    assert cubicasa_record.rooms[bedroom.id].source_type == "Space UserDefined"


def test_undefined_placeholder_is_not_kept_as_a_label(cubicasa_record: PlanRecord) -> None:
    unknown = _by_type(cubicasa_record)["unknown"]
    assert unknown.label is None
    assert cubicasa_record.rooms[unknown.id].source_type == "Space Undefined"


def test_quality_counts_unlabeled_rooms(cubicasa_record: PlanRecord) -> None:
    assert cubicasa_record.quality.unlabeled_fraction == pytest.approx(1 / 6, abs=0.001)
    assert not cubicasa_record.quality.mostly_unlabeled


def test_mostly_unlabeled_plans_are_flagged() -> None:
    rooms = [Room(id=f"r{i}", type=RoomType.UNKNOWN, xy=(10 * i, 10)) for i in range(3)]
    rooms.append(Room(id="r3", type=RoomType.LIVING, xy=(40, 10)))
    graph = FloorplanGraph(floors=[Floor(floor=1, rooms=rooms)], image=ImageSize(w=10, h=10))
    quality = PlanQuality.from_graph(graph, [])
    assert quality.unlabeled_fraction == 0.75
    assert quality.mostly_unlabeled


def test_multi_floor_plan_gives_one_graph_per_floor_without_cross_links(tmp_path: Path) -> None:
    record = convert_plan(write_cubicasa_plan(tmp_path / "p", two_floors=True), "cubicasa5k/x/1")
    assert [floor.floor for floor in record.graph.floors] == [1, 2]
    first, second = record.graph.floors
    assert [room.id for room in second.rooms] == ["r6", "r7"], "ids continue across floors"
    first_ids = {room.id for room in first.rooms}
    assert all({link.a, link.b} <= first_ids for link in first.links)
    assert _links(record, floor=2) == {frozenset({"r6", "r7"}): LinkKind.DOOR}


def test_record_ground_truth_matches_the_graph(cubicasa_record: PlanRecord) -> None:
    assert set(cubicasa_record.rooms) == {room.id for room in cubicasa_record.graph.rooms()}
    for room in cubicasa_record.graph.rooms():
        assert cubicasa_record.rooms[room.id].polygon


def test_converted_graph_round_trips_through_compact_json(cubicasa_record: PlanRecord) -> None:
    graph = cubicasa_record.graph
    sized = [
        floor.model_copy(update={"rooms": [r.model_copy(update={"size": 50}) for r in floor.rooms]})
        for floor in graph.floors
    ]
    graph = graph.model_copy(update={"floors": sized})
    assert from_compact(to_compact(graph), graph.image) == graph


def test_plan_without_rooms_is_an_error(tmp_path: Path) -> None:
    plan = write_cubicasa_plan(tmp_path / "p")
    (plan / "model.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><g class="Floorplan Floor-1"/></svg>'
    )
    with pytest.raises(ValueError, match="no rooms"):
        convert_plan(plan, "cubicasa5k/x/2")
