"""The graph validator: structured, model-readable findings about a `FloorplanGraph`.

`FloorplanGraph` already refuses to be built from bad data, so a graph made through its
constructors is valid. This validator exists for three reasons:

1. It is the safety net after every edit: `apply_op` re-checks the result, so a bug in an
   edit handler becomes a refused edit instead of a silently broken plan.
2. It also accepts graphs that skipped pydantic (`model_construct`, hand-patched objects),
   so it never assumes the constructors ran.
3. It reports every problem at once, as sentences naming the rooms involved,
   where pydantic raises on the first. That is the form a model can act on.

Findings come in two severities.
An `error` means the graph is broken and an edit that causes it is refused.
A `warning` means the graph is valid but probably unfinished (a room with no links);
edits report the warnings they introduce, and never refuse because of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from vlm_floorplan.schema import GRID_MAX, MAX_ROOM_SIZE, Floor, FloorplanGraph, Room

_ROOM_ID = re.compile(r"^r\d+$")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(StrEnum):
    DUPLICATE_FLOOR = "duplicate_floor"
    DUPLICATE_ROOM_ID = "duplicate_room_id"
    BAD_ROOM_ID = "bad_room_id"
    COORD_OUT_OF_RANGE = "coord_out_of_range"
    SIZE_OUT_OF_RANGE = "size_out_of_range"
    SELF_LINK = "self_link"
    DANGLING_LINK = "dangling_link"  # an end is not a room on this floor
    CROSS_FLOOR_LINK = "cross_floor_link"  # an end is a room on another floor
    DUPLICATE_LINK = "duplicate_link"
    NO_ENTRANCE = "no_entrance"
    ISOLATED_ROOM = "isolated_room"


@dataclass(frozen=True)
class Issue:
    """One finding."""

    severity: Severity
    code: IssueCode
    message: str
    """A full sentence that names the rooms and floors involved."""
    room_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors


def _error(code: IssueCode, message: str, *room_ids: str) -> Issue:
    return Issue(Severity.ERROR, code, message, room_ids)


def _warning(code: IssueCode, message: str, *room_ids: str) -> Issue:
    return Issue(Severity.WARNING, code, message, room_ids)


def _check_room(room: Room) -> list[Issue]:
    issues: list[Issue] = []
    if not _ROOM_ID.match(room.id):
        issues.append(
            _error(
                IssueCode.BAD_ROOM_ID, f"room id {room.id!r} is not of the form r<number>", room.id
            )
        )
    for name, value in (("x", room.xy[0]), ("y", room.xy[1])):
        if not 0 <= value <= GRID_MAX:
            issues.append(
                _error(
                    IssueCode.COORD_OUT_OF_RANGE,
                    f"room {room.id} has {name}={value}, outside the 0-{GRID_MAX} grid",
                    room.id,
                )
            )
    if room.size is not None and not 1 <= room.size <= MAX_ROOM_SIZE:
        issues.append(
            _error(
                IssueCode.SIZE_OUT_OF_RANGE,
                f"room {room.id} has size={room.size}, outside 1-{MAX_ROOM_SIZE}",
                room.id,
            )
        )
    return issues


def _check_floor(floor: Floor, floor_of: dict[str, int]) -> list[Issue]:
    issues: list[Issue] = []
    here = {room.id for room in floor.rooms}
    seen_pairs: set[frozenset[str]] = set()
    for link in floor.links:
        name = f"link {link.a}-{link.b}"
        if link.a == link.b:
            issues.append(_error(IssueCode.SELF_LINK, f"{name} connects a room to itself", link.a))
            continue
        for end in (link.a, link.b):
            if end in here:
                continue
            if end in floor_of:
                issues.append(
                    _error(
                        IssueCode.CROSS_FLOOR_LINK,
                        f"{name} reaches {end}, which is on floor {floor_of[end]};"
                        " links never cross floors",
                        link.a,
                        link.b,
                    )
                )
            else:
                issues.append(
                    _error(
                        IssueCode.DANGLING_LINK,
                        f"{name} refers to {end}, which does not exist",
                        link.a,
                        link.b,
                    )
                )
        if link.pair in seen_pairs:
            issues.append(
                _error(
                    IssueCode.DUPLICATE_LINK,
                    f"{link.a} and {link.b} are linked more than once",
                    link.a,
                    link.b,
                )
            )
        seen_pairs.add(link.pair)
    return issues


def _check_ids(graph: FloorplanGraph) -> list[Issue]:
    issues: list[Issue] = []
    floor_numbers = [floor.floor for floor in graph.floors]
    for number in sorted({n for n in floor_numbers if floor_numbers.count(n) > 1}):
        issues.append(
            _error(
                IssueCode.DUPLICATE_FLOOR, f"floor number {number} is used by more than one floor"
            )
        )
    room_ids = [room.id for room in graph.rooms()]
    for room_id in sorted({r for r in room_ids if room_ids.count(r) > 1}):
        issues.append(
            _error(
                IssueCode.DUPLICATE_ROOM_ID, f"room id {room_id} is used more than once", room_id
            )
        )
    return issues


def _warnings(graph: FloorplanGraph) -> list[Issue]:
    issues: list[Issue] = []
    rooms = graph.rooms()
    if rooms and not any(room.entrance for room in rooms):
        issues.append(
            _warning(IssueCode.NO_ENTRANCE, "no room is marked as having an exterior door")
        )
    for floor in graph.floors:
        if len(floor.rooms) < 2:
            continue
        linked = {end for link in floor.links for end in (link.a, link.b)}
        for room in floor.rooms:
            if room.id not in linked:
                issues.append(
                    _warning(
                        IssueCode.ISOLATED_ROOM,
                        f"room {room.id} has no links to other rooms",
                        room.id,
                    )
                )
    return issues


def validate_graph(graph: FloorplanGraph) -> ValidationReport:
    """Check a graph and return every finding (errors first, then warnings)."""
    floor_of = {room.id: floor.floor for floor in graph.floors for room in floor.rooms}
    issues = _check_ids(graph)
    for floor in graph.floors:
        for room in floor.rooms:
            issues.extend(_check_room(room))
        issues.extend(_check_floor(floor, floor_of))
    if not issues:
        issues.extend(_warnings(graph))
    return ValidationReport(tuple(issues))


def new_warnings(before: FloorplanGraph, after: FloorplanGraph) -> tuple[Issue, ...]:
    """Warnings present in `after` that were not already present in `before`.

    An edit result should say what the edit caused, not repeat what was already true.
    """
    known = {(i.code, i.room_ids) for i in validate_graph(before).warnings}
    return tuple(i for i in validate_graph(after).warnings if (i.code, i.room_ids) not in known)
