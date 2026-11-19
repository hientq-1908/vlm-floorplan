"""Compact text for tool responses: a graph summary and an edit result.

After each tool call the model reads a short text, not JSON, so the lines are cheap
in tokens and easy to scan.
Rejections lead with a fixed word (`REJECTED`) and the error code, so the model
(and a training-data filter) can tell success from failure at a glance.

    OK: moved r3 from (120, 340) to (200, 300).
    Warning: room r5 has no links to other rooms on floor 1.
    The plan now has 6 rooms and 7 links (3 door, 1 opening, 3 wall).

    REJECTED (not_adjacent): Cannot add a door between r0 and r4: they are not linked. ...
    The plan was not changed.
"""

from __future__ import annotations

from collections import Counter

from vlm_floorplan.edits.outcome import EditApplied, EditOutcome
from vlm_floorplan.schema import Floor, FloorplanGraph, Room


def _room_line(room: Room) -> str:
    parts = [room.id, room.type.value]
    if room.label:
        parts.append(f"'{room.label}'")
    parts.append(f"at ({room.xy[0]}, {room.xy[1]})")
    if room.entrance:
        parts.append("entrance")
    return " ".join(parts)


def _floor_lines(floor: Floor) -> list[str]:
    if not floor.rooms:
        return [f"floor {floor.floor}: empty"]
    lines = [f"floor {floor.floor}: {len(floor.rooms)} rooms, {len(floor.links)} links"]
    lines.extend(f"  {_room_line(room)}" for room in floor.rooms)
    if floor.links:
        links = ", ".join(f"{link.a}-{link.b} {link.kind.value}" for link in floor.links)
        lines.append(f"  links: {links}")
    return lines


def summarize_graph(graph: FloorplanGraph) -> str:
    """The whole plan as text: rooms with type, label, position and entrance, then links."""
    lines: list[str] = []
    for floor in graph.floors:
        lines.extend(_floor_lines(floor))
    return "\n".join(lines)


def count_line(graph: FloorplanGraph) -> str:
    """One line of totals, e.g. 'The plan now has 6 rooms and 7 links (3 door, 4 wall).'"""
    links = [link for floor in graph.floors for link in floor.links]
    kinds = Counter(link.kind.value for link in links)
    detail = f" ({', '.join(f'{n} {kind}' for kind, n in sorted(kinds.items()))})" if kinds else ""
    return f"The plan now has {len(graph.rooms())} rooms and {len(links)} links{detail}."


def format_edit_outcome(outcome: EditOutcome, *, include_graph: bool = False) -> str:
    """The text a tool call returns to the model.

    `include_graph` appends the full plan summary; leave it off when the model
    already has the plan in context and only needs to know what changed.
    """
    if not isinstance(outcome, EditApplied):
        error = outcome.error
        return f"REJECTED ({error.code.value}): {error.message}\nThe plan was not changed."
    lines = [f"OK: {outcome.description}."]
    lines.extend(f"Warning: {issue.message}." for issue in outcome.warnings)
    lines.append(count_line(outcome.graph))
    if include_graph:
        lines.append(summarize_graph(outcome.graph))
    return "\n".join(lines)
