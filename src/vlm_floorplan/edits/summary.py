"""Plan-state text for the model: the plan summary and the text of a tool response.

The model reads the plan as text, not JSON, so the lines are cheap in tokens and easy to scan.
The plan is single-floor here: no floor header, field or indentation.
Every tool response is a status line (`OK:` or `REJECTED (code):`), then any warnings,
then the whole plan as it is now, so the model never has to track the plan itself.

    OK: moved r3 from (120, 340) to (200, 300).
    Warning: room r5 has no links to other rooms.
    plan: 6 rooms, 7 links
    exterior wall: (60, 264), (60, 736), (568, 736), (568, 264)
    r0 bedroom at (120, 340) size 126
    r1 living 'Open hall' at (200, 300) size 222 entrance
    links: r0-r1 door, r0-r2 wall

    REJECTED (not_adjacent): Cannot add a door between r0 and r4: they are not linked. ...
    The plan was not changed.
    plan: 6 rooms, 7 links
    ...

An empty plan reads `plan: empty`. The `exterior wall:` line is left out when the plan has
no wall, and the `links:` line when it has no links.
"""

from __future__ import annotations

from vlm_floorplan.edits.outcome import EditApplied, EditOutcome
from vlm_floorplan.schema import FloorplanGraph, Room


def _room_line(room: Room) -> str:
    parts = [room.id, room.type.value]
    if room.label:
        parts.append(f"'{room.label}'")
    parts.append(f"at ({room.xy[0]}, {room.xy[1]})")
    if room.size is not None:
        parts.append(f"size {room.size}")
    if room.entrance:
        parts.append("entrance")
    return " ".join(parts)


def summarize_graph(graph: FloorplanGraph) -> str:
    """The whole plan as text: header, exterior wall, one line per room, then the links.

    Raises `ValueError` for a plan with several floors: the text has no way to name a floor.
    """
    if len(graph.floors) != 1:
        raise ValueError(f"the plan text is single-floor; this plan has {len(graph.floors)} floors")
    floor = graph.floors[0]
    header = (
        f"plan: {len(floor.rooms)} rooms, {len(floor.links)} links"
        if floor.rooms
        else "plan: empty"
    )
    lines = [header]
    if graph.exterior_wall is not None:
        lines.append("exterior wall: " + ", ".join(f"({x}, {y})" for x, y in graph.exterior_wall))
    lines.extend(_room_line(room) for room in floor.rooms)
    if floor.links:
        links = ", ".join(f"{link.a}-{link.b} {link.kind.value}" for link in floor.links)
        lines.append(f"links: {links}")
    return "\n".join(lines)


def format_edit_outcome(outcome: EditOutcome, graph: FloorplanGraph) -> str:
    """The text a tool call returns to the model: status, warnings, then the current plan.

    `graph` is the plan before the call; a refused call leaves it unchanged, and the
    response says so and shows it.
    """
    if not isinstance(outcome, EditApplied):
        error = outcome.error
        status = [f"REJECTED ({error.code.value}): {error.message}", "The plan was not changed."]
        return "\n".join([*status, summarize_graph(graph)])
    lines = [f"OK: {outcome.description}."]
    lines.extend(f"Warning: {issue.message}." for issue in outcome.warnings)
    lines.append(summarize_graph(outcome.graph))
    return "\n".join(lines)
