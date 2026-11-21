from __future__ import annotations

import pytest

from vlm_floorplan.edits import (
    AddLink,
    AddRoom,
    DeleteRoom,
    apply_op,
    format_edit_outcome,
    summarize_graph,
)

from .helpers import sample_graph, single_floor_graph

WALL_LINE = "exterior wall: (50, 400), (50, 900), (900, 900), (900, 400)"
PLAN_LINES = [
    "plan: 5 rooms, 4 links",
    WALL_LINE,
    "r0 entry at (100, 500) size 40 entrance",
    "r1 living 'OH' at (400, 500) size 50",
    "r2 kitchen at (700, 500) size 60",
    "r3 bedroom at (400, 800) size 70",
    "r4 bathroom at (700, 800) size 80",
    "links: r0-r1 door, r1-r2 opening, r1-r3 wall, r3-r4 wall",
]


def test_summarize_graph_is_single_floor_text_with_wall_and_sizes() -> None:
    assert summarize_graph(single_floor_graph()) == "\n".join(PLAN_LINES)


def test_the_summary_names_no_floor_and_indents_nothing() -> None:
    text = summarize_graph(single_floor_graph())
    assert "floor" not in text.replace("exterior wall", "")
    assert not any(line.startswith(" ") for line in text.splitlines())


def test_summarize_graph_marks_an_empty_plan_and_keeps_the_wall() -> None:
    graph = single_floor_graph()
    for room_id in ("r0", "r1", "r2", "r3", "r4"):
        graph = apply_op(graph, DeleteRoom(room_id=room_id)).graph
    assert summarize_graph(graph) == f"plan: empty\n{WALL_LINE}"


def test_a_room_without_a_size_or_a_plan_without_a_wall_leaves_the_part_out() -> None:
    text = summarize_graph(sample_graph().model_copy(update={"floors": sample_graph().floors[:1]}))
    assert "exterior wall" not in text
    assert "r0 entry at (100, 500) entrance" in text
    assert "size" not in text


def test_a_plan_with_several_floors_has_no_text_form() -> None:
    with pytest.raises(ValueError, match="single-floor"):
        summarize_graph(sample_graph())


def test_applied_edit_text_has_status_warnings_and_the_whole_plan() -> None:
    graph = single_floor_graph()
    outcome = apply_op(graph, AddRoom(type="storage", x=10, y=20, size=33, label="Closet"))
    assert format_edit_outcome(outcome, graph) == "\n".join(
        [
            "OK: added storage room r5 'Closet' at (10, 20).",
            "Warning: room r5 has no links to other rooms.",
            "plan: 6 rooms, 4 links",
            WALL_LINE,
            *PLAN_LINES[2:-1],
            "r5 storage 'Closet' at (10, 20) size 33",
            PLAN_LINES[-1],
        ]
    )


def test_rejected_edit_text_says_nothing_changed_and_shows_the_unchanged_plan() -> None:
    graph = single_floor_graph()
    outcome = apply_op(graph, AddLink(room_a="r0", room_b="r4", kind="door"))
    text = format_edit_outcome(outcome, graph)
    assert text.startswith("REJECTED (not_adjacent): Cannot add a door between r0 and r4")
    _, plan = text.split("The plan was not changed.\n")
    assert plan == "\n".join(PLAN_LINES)
