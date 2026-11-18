from __future__ import annotations

import pytest

from vlm_floorplan.edits import IssueCode, Severity, validate_graph
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room, RoomType

from .helpers import sample_graph

IMAGE = ImageSize(w=10, h=10)


def room(room_id: str, **fields) -> Room:
    return Room(id=room_id, type=RoomType.LIVING, xy=fields.pop("xy", (5, 5)), **fields)


def raw_graph(floors: list[Floor]) -> FloorplanGraph:
    """A graph that skipped pydantic validation, as hand-patched data would."""
    return FloorplanGraph.model_construct(floors=floors, image=IMAGE)


def raw_floor(number: int, rooms: list[Room], links: list[Link] | None = None) -> Floor:
    return Floor.model_construct(floor=number, rooms=rooms, links=links or [])


def codes(graph: FloorplanGraph) -> set[IssueCode]:
    return {issue.code for issue in validate_graph(graph).issues}


def test_the_sample_graph_has_no_issues() -> None:
    report = validate_graph(sample_graph())
    assert report.is_valid
    assert report.issues == ()


def test_self_link() -> None:
    graph = raw_graph([raw_floor(1, [room("r0")], [Link(a="r0", b="r0", kind="door")])])
    assert codes(graph) == {IssueCode.SELF_LINK}


def test_dangling_link_names_the_missing_room() -> None:
    graph = raw_graph([raw_floor(1, [room("r0")], [Link(a="r0", b="r9", kind="wall")])])
    (issue,) = validate_graph(graph).errors
    assert issue.code == IssueCode.DANGLING_LINK
    assert "r9" in issue.message
    assert issue.room_ids == ("r0", "r9")


def test_cross_floor_link_names_the_other_floor() -> None:
    floors = [
        raw_floor(1, [room("r0")], [Link(a="r0", b="r1", kind="door")]),
        raw_floor(2, [room("r1")]),
    ]
    (issue,) = validate_graph(raw_graph(floors)).errors
    assert issue.code == IssueCode.CROSS_FLOOR_LINK
    assert "floor 2" in issue.message


def test_duplicate_link_in_either_direction() -> None:
    links = [Link(a="r0", b="r1", kind="door"), Link(a="r1", b="r0", kind="wall")]
    graph = raw_graph([raw_floor(1, [room("r0"), room("r1")], links)])
    assert codes(graph) == {IssueCode.DUPLICATE_LINK}


def test_duplicate_room_ids_and_floor_numbers() -> None:
    graph = raw_graph([raw_floor(1, [room("r0")]), raw_floor(1, [room("r0")])])
    assert codes(graph) == {IssueCode.DUPLICATE_FLOOR, IssueCode.DUPLICATE_ROOM_ID}


def test_coordinates_out_of_range() -> None:
    bad = Room.model_construct(
        id="r0", type=RoomType.LIVING, xy=(1001, -1), bbox=None, entrance=False
    )
    issues = validate_graph(raw_graph([raw_floor(1, [bad])])).errors
    assert [i.code for i in issues] == [IssueCode.COORD_OUT_OF_RANGE] * 2
    assert "x=1001" in issues[0].message
    assert "y=-1" in issues[1].message


def test_bbox_outside_the_grid_and_inverted_and_missing_the_point() -> None:
    def with_box(box):
        made = Room.model_construct(
            id="r0", type=RoomType.LIVING, xy=(50, 50), bbox=box, entrance=False
        )
        return validate_graph(raw_graph([raw_floor(1, [made])])).errors

    assert [i.code for i in with_box((0, 0, 2000, 100))] == [IssueCode.COORD_OUT_OF_RANGE]
    assert [i.code for i in with_box((60, 0, 40, 100))] == [IssueCode.BAD_BBOX]
    assert [i.code for i in with_box((0, 0, 10, 10))] == [IssueCode.BAD_BBOX]


def test_malformed_room_id() -> None:
    bad = Room.model_construct(
        id="kitchen", type=RoomType.LIVING, xy=(1, 1), bbox=None, entrance=False
    )
    assert codes(raw_graph([raw_floor(1, [bad])])) == {IssueCode.BAD_ROOM_ID}


def test_all_problems_are_reported_together() -> None:
    graph = raw_graph(
        [
            raw_floor(
                1,
                [room("r0")],
                [Link(a="r0", b="r0", kind="door"), Link(a="r0", b="r8", kind="wall")],
            )
        ]
    )
    assert codes(graph) == {IssueCode.SELF_LINK, IssueCode.DANGLING_LINK}


def test_warnings_for_no_entrance_and_isolated_rooms() -> None:
    graph = FloorplanGraph(
        floors=[Floor(floor=1, rooms=[room("r0"), room("r1", xy=(6, 6))], links=[])], image=IMAGE
    )
    report = validate_graph(graph)
    assert report.is_valid
    assert {i.code for i in report.warnings} == {IssueCode.NO_ENTRANCE, IssueCode.ISOLATED_ROOM}
    assert all(i.severity == Severity.WARNING for i in report.issues)


def test_a_single_room_floor_is_not_isolated() -> None:
    graph = FloorplanGraph(floors=[Floor(floor=1, rooms=[room("r0", entrance=True)])], image=IMAGE)
    assert validate_graph(graph).issues == ()


@pytest.mark.parametrize("code", list(IssueCode))
def test_every_issue_code_is_a_snake_case_string(code: IssueCode) -> None:
    assert code.value == code.value.lower()
    assert " " not in code.value
