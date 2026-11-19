from __future__ import annotations

from vlm_floorplan.edits import (
    AddLink,
    AddRoom,
    DeleteRoom,
    apply_op,
    format_edit_outcome,
    summarize_graph,
)

from .helpers import sample_graph


def test_summarize_graph_lists_rooms_and_links_per_floor() -> None:
    assert summarize_graph(sample_graph()) == "\n".join(
        [
            "floor 1: 5 rooms, 4 links",
            "  r0 entry at (100, 500) entrance",
            "  r1 living 'OH' at (400, 500)",
            "  r2 kitchen at (700, 500)",
            "  r3 bedroom at (400, 800)",
            "  r4 bathroom at (700, 800)",
            "  links: r0-r1 door, r1-r2 opening, r1-r3 wall, r3-r4 wall",
            "floor 2: 2 rooms, 1 links",
            "  r5 bedroom at (300, 300)",
            "  r6 stairs at (600, 300)",
            "  links: r5-r6 door",
        ]
    )


def test_summarize_graph_marks_an_empty_floor() -> None:
    graph = apply_op(
        apply_op(sample_graph(), DeleteRoom(room_id="r5")).graph, DeleteRoom(room_id="r6")
    ).graph
    assert summarize_graph(graph).endswith("floor 2: empty")


def test_applied_edit_text_has_description_warnings_and_totals() -> None:
    outcome = apply_op(sample_graph(), AddRoom(type="storage", x=10, y=20, floor=1, label="Closet"))
    assert format_edit_outcome(outcome) == "\n".join(
        [
            "OK: added storage room r7 'Closet' at (10, 20) on floor 1.",
            "Warning: room r7 has no links to other rooms on floor 1.",
            "The plan now has 8 rooms and 5 links (2 door, 1 opening, 2 wall).",
        ]
    )


def test_rejected_edit_text_leads_with_the_code_and_says_nothing_changed() -> None:
    outcome = apply_op(sample_graph(), AddLink(room_a="r0", room_b="r4", kind="door"))
    text = format_edit_outcome(outcome)
    assert text.startswith("REJECTED (not_adjacent): Cannot add a door between r0 and r4")
    assert text.endswith("The plan was not changed.")


def test_include_graph_appends_the_plan() -> None:
    outcome = apply_op(sample_graph(), DeleteRoom(room_id="r4"))
    text = format_edit_outcome(outcome, include_graph=True)
    assert "floor 1: 4 rooms, 3 links" in text
    assert "r4 bathroom" not in text
