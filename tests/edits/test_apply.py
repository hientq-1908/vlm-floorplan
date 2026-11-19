from __future__ import annotations

import pytest

from vlm_floorplan.edits import (
    AddLink,
    AddRoom,
    DeleteRoom,
    EditErrorCode,
    MoveRoom,
    RemoveLink,
    SetEntrance,
    SetLinkKind,
    SetRoomLabel,
    SetRoomType,
    validate_graph,
)
from vlm_floorplan.schema import Floor, FloorplanGraph, Room, RoomType

from .helpers import applied, link_kinds, rejected, sample_graph


def pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def room(graph, room_id):
    return next(r for r in graph.rooms() if r.id == room_id)


def test_apply_does_not_mutate_the_input_graph() -> None:
    graph = sample_graph()
    before = graph.model_dump()
    applied(graph, DeleteRoom(room_id="r1"))
    assert graph.model_dump() == before


# --- add_room ---------------------------------------------------------------------------


def test_add_room_assigns_the_next_free_id_and_records_it_in_the_op() -> None:
    result = applied(sample_graph(), AddRoom(type="bedroom", x=50, y=60, floor=2, label="Kids"))
    assert result.op.room_id == "r7"
    assert result.op.floor == 2
    new = room(result.graph, "r7")
    assert (new.type, new.label, new.xy, new.entrance) == (
        RoomType.BEDROOM,
        "Kids",
        (50, 60),
        False,
    )
    assert new in result.graph.floors[1].rooms


def test_add_room_on_a_single_floor_plan_needs_no_floor() -> None:
    single = sample_graph().model_copy(update={"floors": sample_graph().floors[:1]})
    assert applied(single, AddRoom(type="storage", x=1, y=2)).op.floor == 1


def test_add_room_with_several_floors_needs_a_floor() -> None:
    assert rejected(sample_graph(), AddRoom(type="storage", x=1, y=2)).error.code == (
        EditErrorCode.AMBIGUOUS_FLOOR
    )


def test_add_room_on_a_missing_floor_is_rejected() -> None:
    error = rejected(sample_graph(), AddRoom(type="storage", x=1, y=2, floor=9)).error
    assert error.code == EditErrorCode.UNKNOWN_FLOOR
    assert "[1, 2]" in error.message


def test_add_room_with_an_id_in_use_is_rejected() -> None:
    op = AddRoom(type="storage", x=1, y=2, floor=1, room_id="r3")
    assert rejected(sample_graph(), op).error.code == EditErrorCode.ID_IN_USE


def test_add_room_keeps_the_entrance_and_has_no_box() -> None:
    op = AddRoom(type="entry", x=50, y=50, floor=1, entrance=True)
    new = room(applied(sample_graph(), op).graph, "r7")
    assert new.bbox is None
    assert new.entrance


def test_add_room_warns_that_the_new_room_is_isolated() -> None:
    result = applied(sample_graph(), AddRoom(type="storage", x=1, y=2, floor=1))
    assert [w.room_ids for w in result.warnings] == [("r7",)]


def test_add_room_to_an_empty_plan_gives_r0() -> None:
    empty = sample_graph().model_copy(
        update={"floors": [sample_graph().floors[0].model_copy(update={"rooms": [], "links": []})]}
    )
    assert applied(empty, AddRoom(type="living", x=500, y=500)).op.room_id == "r0"


def test_room_ids_after_a_delete_never_collide() -> None:
    graph = applied(sample_graph(), DeleteRoom(room_id="r6")).graph
    assert applied(graph, AddRoom(type="storage", x=1, y=2, floor=1)).op.room_id == "r6"


# --- move_room --------------------------------------------------------------------------


def test_move_room_changes_only_the_point() -> None:
    graph = sample_graph()
    result = applied(graph, MoveRoom(room_id="r2", x=10, y=20))
    assert room(result.graph, "r2").xy == (10, 20)
    assert "bbox" not in result.description
    assert result.graph.floors[0].links == graph.floors[0].links


def test_move_room_drops_the_box() -> None:
    base = sample_graph()
    floor1 = base.floors[0]
    rooms = [
        Room(id="r1", type=RoomType.LIVING, xy=(400, 500), bbox=(300, 400, 500, 600))
        if r.id == "r1"
        else r
        for r in floor1.rooms
    ]
    boxed_floor = Floor(floor=floor1.floor, rooms=rooms, links=floor1.links)
    graph = FloorplanGraph(floors=[boxed_floor, *base.floors[1:]], image=base.image)
    result = applied(graph, MoveRoom(room_id="r1", x=1000, y=1000))
    moved = room(result.graph, "r1")
    assert moved.xy == (1000, 1000)
    assert moved.bbox is None
    assert result.description == "moved r1 from (400, 500) to (1000, 1000) and dropped its bbox"


def test_move_room_to_the_same_place_is_no_change() -> None:
    assert rejected(sample_graph(), MoveRoom(room_id="r2", x=700, y=500)).error.code == (
        EditErrorCode.NO_CHANGE
    )


def test_move_unknown_room_lists_the_rooms_that_exist() -> None:
    error = rejected(sample_graph(), MoveRoom(room_id="r9", x=1, y=1)).error
    assert error.code == EditErrorCode.UNKNOWN_ROOM
    assert error.room_ids == ("r9",)
    assert "r0, r1, r2, r3, r4, r5, r6" in error.message


# --- delete_room ------------------------------------------------------------------------


def test_delete_room_drops_its_links_and_keeps_the_graph_valid() -> None:
    result = applied(sample_graph(), DeleteRoom(room_id="r1"))
    assert {r.id for r in result.graph.rooms()} == {"r0", "r2", "r3", "r4", "r5", "r6"}
    assert link_kinds(result.graph) == {pair("r3", "r4"): "wall", pair("r5", "r6"): "door"}
    assert validate_graph(result.graph).is_valid
    assert "3 link(s) to r0, r2, r3" in result.description


def test_delete_the_entrance_room_warns_that_no_entrance_is_left() -> None:
    result = applied(sample_graph(), DeleteRoom(room_id="r0"))
    assert [w.code.value for w in result.warnings] == ["no_entrance"]


def test_delete_the_last_room_leaves_a_valid_empty_floor() -> None:
    graph = applied(sample_graph(), DeleteRoom(room_id="r5")).graph
    graph = applied(graph, DeleteRoom(room_id="r6")).graph
    assert graph.floors[1].rooms == []
    assert validate_graph(graph).is_valid


def test_delete_unknown_room_is_rejected() -> None:
    assert (
        rejected(sample_graph(), DeleteRoom(room_id="r42")).error.code == EditErrorCode.UNKNOWN_ROOM
    )


# --- set_room_type / set_room_label / set_entrance --------------------------------------


def test_set_room_type() -> None:
    result = applied(sample_graph(), SetRoomType(room_id="r3", type="storage"))
    assert room(result.graph, "r3").type == RoomType.STORAGE
    assert result.description == "changed r3 from bedroom to storage"


def test_set_room_type_to_the_same_type_is_no_change() -> None:
    assert rejected(sample_graph(), SetRoomType(room_id="r3", type="bedroom")).error.code == (
        EditErrorCode.NO_CHANGE
    )


def test_set_room_label_renames_and_clears() -> None:
    renamed = applied(sample_graph(), SetRoomLabel(room_id="r3", label="  Guest room "))
    assert room(renamed.graph, "r3").label == "Guest room"
    cleared = applied(renamed.graph, SetRoomLabel(room_id="r3", label=""))
    assert room(cleared.graph, "r3").label is None


@pytest.mark.parametrize(("room_id", "label"), [("r1", "OH"), ("r3", "")])
def test_set_room_label_to_the_current_value_is_no_change(room_id: str, label: str) -> None:
    assert rejected(sample_graph(), SetRoomLabel(room_id=room_id, label=label)).error.code == (
        EditErrorCode.NO_CHANGE
    )


def test_set_entrance_on_and_off() -> None:
    on = applied(sample_graph(), SetEntrance(room_id="r3", entrance=True))
    assert room(on.graph, "r3").entrance
    off = applied(on.graph, SetEntrance(room_id="r3", entrance=False))
    assert not room(off.graph, "r3").entrance


def test_set_entrance_to_the_current_value_is_no_change() -> None:
    assert rejected(sample_graph(), SetEntrance(room_id="r0", entrance=True)).error.code == (
        EditErrorCode.NO_CHANGE
    )


# --- add_link ---------------------------------------------------------------------------


def test_add_door_between_non_adjacent_rooms_is_rejected_with_a_reason() -> None:
    error = rejected(sample_graph(), AddLink(room_a="r0", room_b="r4", kind="door")).error
    assert error.code == EditErrorCode.NOT_ADJACENT
    assert error.room_ids == ("r0", "r4")
    assert "not linked" in error.message
    assert "'wall'" in error.message
    assert "set_link_kind" in error.message


def test_add_opening_between_non_adjacent_rooms_is_rejected() -> None:
    assert rejected(
        sample_graph(), AddLink(room_a="r2", room_b="r3", kind="opening")
    ).error.code == (EditErrorCode.NOT_ADJACENT)


def test_add_wall_link_declares_adjacency_and_set_link_kind_then_adds_a_door() -> None:
    with_wall = applied(sample_graph(), AddLink(room_a="r2", room_b="r4", kind="wall")).graph
    assert link_kinds(with_wall)[pair("r2", "r4")] == "wall"
    with_door = applied(with_wall, SetLinkKind(room_a="r2", room_b="r4", kind="door")).graph
    assert link_kinds(with_door)[pair("r2", "r4")] == "door"


@pytest.mark.parametrize("pair_ids", [("r1", "r0"), ("r1", "r2"), ("r3", "r1")])
@pytest.mark.parametrize("kind", ["door", "wall", "opening"])
def test_add_link_on_an_already_linked_pair_is_a_duplicate(
    pair_ids: tuple[str, str], kind: str
) -> None:
    room_a, room_b = pair_ids
    error = rejected(sample_graph(), AddLink(room_a=room_a, room_b=room_b, kind=kind)).error
    assert error.code == EditErrorCode.DUPLICATE_LINK
    assert "set_link_kind" in error.message


def test_add_link_to_itself_is_rejected() -> None:
    assert rejected(sample_graph(), AddLink(room_a="r1", room_b="r1", kind="wall")).error.code == (
        EditErrorCode.SELF_LINK
    )


def test_add_link_across_floors_is_rejected() -> None:
    error = rejected(sample_graph(), AddLink(room_a="r1", room_b="r5", kind="wall")).error
    assert error.code == EditErrorCode.CROSS_FLOOR
    assert "floor 1" in error.message
    assert "floor 2" in error.message


def test_add_link_to_unknown_room_is_rejected() -> None:
    assert rejected(sample_graph(), AddLink(room_a="r1", room_b="r9", kind="wall")).error.code == (
        EditErrorCode.UNKNOWN_ROOM
    )


# --- remove_link ------------------------------------------------------------------------


def test_remove_link_in_either_direction() -> None:
    result = applied(sample_graph(), RemoveLink(room_a="r1", room_b="r0"))
    assert pair("r0", "r1") not in link_kinds(result.graph)
    assert [w.room_ids for w in result.warnings] == [("r0",)]  # r0 is now isolated


def test_remove_link_between_unlinked_rooms_is_rejected() -> None:
    assert rejected(sample_graph(), RemoveLink(room_a="r0", room_b="r4")).error.code == (
        EditErrorCode.UNKNOWN_LINK
    )


# --- set_link_kind ----------------------------------------------------------------------


def test_set_link_kind_closes_a_door_into_a_wall() -> None:
    result = applied(sample_graph(), SetLinkKind(room_a="r0", room_b="r1", kind="wall"))
    assert link_kinds(result.graph)[pair("r0", "r1")] == "wall"
    assert result.graph.floors[0].links[0].a == "r0"  # position and direction are kept


def test_set_link_kind_to_the_current_kind_is_no_change() -> None:
    assert rejected(
        sample_graph(), SetLinkKind(room_a="r0", room_b="r1", kind="door")
    ).error.code == (EditErrorCode.NO_CHANGE)


def test_set_link_kind_without_a_link_is_rejected() -> None:
    error = rejected(sample_graph(), SetLinkKind(room_a="r0", room_b="r4", kind="door")).error
    assert error.code == EditErrorCode.UNKNOWN_LINK


def test_a_handler_bug_is_caught_by_the_validator_and_refused(monkeypatch) -> None:
    """The safety net: if a handler ever builds a broken graph, the edit is refused."""
    from vlm_floorplan.edits import apply as apply_module
    from vlm_floorplan.edits.change import Change

    graph = sample_graph()

    def broken_handler(graph: FloorplanGraph, op: MoveRoom) -> Change:
        bad = Room.model_construct(
            id="r2", type=RoomType.KITCHEN, xy=(5000, 5), bbox=None, entrance=False
        )
        floor = graph.floors[0].model_copy(update={"rooms": [*graph.floors[0].rooms[:2], bad]})
        return Change(graph.model_copy(update={"floors": [floor, graph.floors[1]]}), op, "broken")

    monkeypatch.setitem(apply_module._HANDLERS, MoveRoom, broken_handler)
    error = rejected(graph, MoveRoom(room_id="r2", x=1, y=1)).error
    assert error.code == EditErrorCode.INVALID_RESULT
    assert "x=5000" in error.message
    assert "r2" in error.room_ids
