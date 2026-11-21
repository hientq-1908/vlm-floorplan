from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_floorplan.edits import (
    AddLink,
    AddRoom,
    DeleteRoom,
    EditApplied,
    EditErrorCode,
    EditHistory,
    EditRejected,
    MoveRoom,
    SetLinkKind,
    SetRoomType,
    dump_ops,
    load_ops,
    replay,
)

from .helpers import sample_graph, single_floor_graph


def test_a_new_history_has_nothing_to_undo_or_redo() -> None:
    history = EditHistory(sample_graph())
    assert history.graph == history.base == sample_graph()
    assert (history.can_undo, history.can_redo) == (False, False)
    assert history.undo() is None
    assert history.redo() is None


def test_apply_records_only_successful_edits() -> None:
    history = EditHistory(sample_graph())
    assert isinstance(history.apply(MoveRoom(room_id="r2", x=1, y=1)), EditApplied)
    assert isinstance(history.apply(MoveRoom(room_id="r2", x=1, y=1)), EditRejected)  # no change
    assert isinstance(history.apply(DeleteRoom(room_id="r99")), EditRejected)
    assert history.ops == (MoveRoom(room_id="r2", x=1, y=1),)


def test_undo_and_redo_walk_the_snapshots() -> None:
    history = EditHistory(sample_graph())
    history.apply(MoveRoom(room_id="r2", x=1, y=1))
    after_move = history.graph
    history.apply(DeleteRoom(room_id="r1"))
    after_delete = history.graph
    assert history.undo() == DeleteRoom(room_id="r1")
    assert history.graph == after_move
    assert history.undo() == MoveRoom(room_id="r2", x=1, y=1)
    assert history.graph == sample_graph()
    assert history.ops == ()
    assert history.redo() == MoveRoom(room_id="r2", x=1, y=1)
    assert history.redo() == DeleteRoom(room_id="r1")
    assert history.graph == after_delete
    assert history.redo() is None


def test_a_new_edit_after_undo_discards_the_redo_tail() -> None:
    history = EditHistory(sample_graph())
    history.apply(MoveRoom(room_id="r2", x=1, y=1))
    history.undo()
    history.apply(SetRoomType(room_id="r2", type="dining"))
    assert not history.can_redo
    assert history.ops == (SetRoomType(room_id="r2", type="dining"),)


def test_edits_are_judged_against_the_undone_state() -> None:
    history = EditHistory(sample_graph())
    history.apply(DeleteRoom(room_id="r1"))
    history.undo()
    assert isinstance(history.apply(MoveRoom(room_id="r1", x=1, y=1)), EditApplied)


def test_recorded_ops_are_resolved_so_replay_is_deterministic() -> None:
    history = EditHistory(single_floor_graph())
    history.apply(AddRoom(type="storage", x=5, y=5, size=30))
    history.apply(DeleteRoom(room_id="r5"))
    history.apply(AddRoom(type="utility", x=9, y=9, size=30))
    assert [getattr(op, "room_id", None) for op in history.ops] == ["r5", "r5", "r5"]
    assert replay(history.base, history.ops) == history.graph


def test_replay_reports_the_first_refused_op() -> None:
    ops = [MoveRoom(room_id="r2", x=1, y=1), AddLink(room_a="r0", room_b="r4", kind="door")]
    result = replay(sample_graph(), ops)
    assert isinstance(result, EditRejected)
    assert result.error.code == EditErrorCode.NOT_ADJACENT
    assert result.op == ops[1]


def test_ops_survive_a_json_round_trip() -> None:
    history = EditHistory(single_floor_graph())
    history.apply(AddRoom(type="storage", x=5, y=5, size=30, label="Pantry"))
    history.apply(AddLink(room_a="r5", room_b="r2", kind="wall"))
    history.apply(SetLinkKind(room_a="r5", room_b="r2", kind="door"))
    text = dump_ops(history.ops)
    assert "null" not in text
    assert load_ops(text) == list(history.ops)
    assert replay(history.base, load_ops(text)) == history.graph


def test_load_ops_rejects_an_unknown_op() -> None:
    with pytest.raises(ValidationError):
        load_ops('[{"op": "explode_room", "room_id": "r1"}]')
