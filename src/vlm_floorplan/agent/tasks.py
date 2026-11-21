"""Corruption tasks: break a real plan, ask for the repair, know the repair.

The captain's idea for tool-use data: take a real plan, damage it, use the damaged plan as the
starting state and the real plan as the target.
"Delete a room" read backwards is the user request "add this room",
and the links the real plan had say which rooms the new room must connect to.

A task knows three things about one real plan:

- `corrupt(plan)`: the damaged starting plan,
- `request(plan)`: what the user types to ask for the repair,
- `solve(plan, damaged)`: the scripted tool calls that repair it, each with a short reason.
  These are the SFT target, and they are the baseline an RL rollout is compared against.

Every task builds its damaged plan with the same tool calls the agent uses (`call_tool`),
so a damaged plan is always one the agent could have produced.

A deleted room that had a door to the outside gets it back with `add_exterior_door`,
as a separate call after its links.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from vlm_floorplan.agent.tools import NEW_PLAN, call_tool
from vlm_floorplan.compact import to_compact
from vlm_floorplan.edits.lookup import next_room_id
from vlm_floorplan.edits.ops import MAX_LABEL_LENGTH
from vlm_floorplan.edits.summary import summarize_graph
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import Floor, FloorplanGraph, LinkKind, Room, RoomType

MAX_DRAW_ROOMS = 12
"""`draw_plan` copies the whole plan into one tool call; bigger plans make a very long call."""


@dataclass(frozen=True)
class ToolCall:
    """One scripted step: the call, and the short reasoning that goes before it."""

    name: str
    arguments: dict[str, Any]
    reasoning: str


@dataclass(frozen=True)
class Reply:
    """The assistant's last turn, after the final tool response."""

    reasoning: str
    text: str


class Task(Protocol):
    kind: str

    def corrupt(self, plan: FloorplanGraph) -> FloorplanGraph: ...
    def request(self, plan: FloorplanGraph) -> str: ...
    def solve(self, plan: FloorplanGraph, damaged: FloorplanGraph) -> list[ToolCall]: ...
    def reply(self, plan: FloorplanGraph) -> Reply: ...


def _room(plan: FloorplanGraph, room_id: str) -> Room:
    return next(room for room in plan.rooms() if room.id == room_id)


def _floor_of(plan: FloorplanGraph, room_id: str) -> Floor:
    return next(f for f in plan.floors if any(r.id == room_id for r in f.rooms))


_TYPE_NOUNS = {
    RoomType.LIVING: "living room",
    RoomType.KITCHEN: "kitchen",
    RoomType.DINING: "dining room",
    RoomType.BEDROOM: "bedroom",
    RoomType.BATHROOM: "bathroom",
    RoomType.ENTRY: "entry",
    RoomType.STORAGE: "storage room",
    RoomType.UTILITY: "utility room",
    RoomType.TERRACE: "terrace",
    RoomType.GARAGE: "garage",
    RoomType.STAIRS: "staircase",
    RoomType.OTHER: "room",
    RoomType.UNKNOWN: "room",
}


def _type_name(room: Room) -> str:
    """The room type as a noun, e.g. 'living' is 'living room'."""
    return _TYPE_NOUNS[room.type]


def _a_type_name(room: Room) -> str:
    """The room type with its indefinite article, e.g. 'an entry'."""
    noun = _type_name(room)
    return f"{'an' if noun[0] in 'aeio' else 'a'} {noun}"


def _article(kind: LinkKind) -> str:
    return "an" if kind == LinkKind.OPENING else "a"


def _name(plan: FloorplanGraph, room_id: str) -> str:
    """How a request names a room: its id and type, e.g. 'r2 (living)'."""
    return f"{room_id} ({_room(plan, room_id).type.value})"


def _apply(plan: FloorplanGraph, name: str, arguments: dict[str, Any]) -> FloorplanGraph:
    result = call_tool(plan, name, arguments)
    if not result.applied:
        raise ValueError(f"cannot damage the plan with {name}({arguments}): {result.text}")
    return result.graph


def _room_number(room_id: str) -> int:
    return int(room_id[1:])


@dataclass(frozen=True)
class DeleteRoomTask:
    """A room was deleted; the user asks for it back. Add the room, then its links."""

    room_id: str
    kind: str = "delete_room"

    def corrupt(self, plan: FloorplanGraph) -> FloorplanGraph:
        return _apply(plan, "delete_room", {"room_id": self.room_id})

    def _neighbours(self, plan: FloorplanGraph) -> list[tuple[str, LinkKind]]:
        floor = _floor_of(plan, self.room_id)
        found = [
            (link.b if link.a == self.room_id else link.a, link.kind)
            for link in floor.links
            if self.room_id in (link.a, link.b)
        ]
        return sorted(found, key=lambda item: _room_number(item[0]))

    def _connections(self, plan: FloorplanGraph) -> str:
        """The sentence that lists the room's links, e.g. ' It has a door to r4 (entry).'"""
        phrases = {
            LinkKind.DOOR: "a door to",
            LinkKind.OPENING: "an opening to",
            LinkKind.WALL: "a shared wall with",
        }
        parts = [f"{phrases[kind]} {_name(plan, other)}" for other, kind in self._neighbours(plan)]
        if _room(plan, self.room_id).entrance:
            parts.append("a door to the outside")
        if not parts:
            return ""
        return f" It has {', '.join(parts[:-1])}{' and ' if len(parts) > 1 else ''}{parts[-1]}."

    def request(self, plan: FloorplanGraph) -> str:
        room = _room(plan, self.room_id)
        label = f" called '{room.label}'" if room.label else ""
        text = f"Add {_a_type_name(room)}{label} at ({room.xy[0]}, {room.xy[1]})."
        return text + self._connections(plan)

    def solve(self, plan: FloorplanGraph, damaged: FloorplanGraph) -> list[ToolCall]:
        room = _room(plan, self.room_id)
        new_id = next_room_id(damaged)
        assert room.size is not None, "candidate_tasks only offers rooms that have a size"
        arguments: dict[str, Any] = {
            "type": room.type.value,
            "x": room.xy[0],
            "y": room.xy[1],
            "size": room.size,
        }
        if room.label:
            arguments["label"] = room.label
        calls = [
            ToolCall(
                "add_room",
                arguments,
                f"The user wants {_a_type_name(room)} at ({room.xy[0]}, {room.xy[1]}). I add the"
                f" room first. It starts with no links, so I connect it afterwards.",
            )
        ]
        for other, kind in self._neighbours(plan):
            if kind == LinkKind.WALL:
                why = f"The request says {new_id} shares a wall with {other}, so I link them."
            else:
                why = (
                    f"The request gives {new_id} {_article(kind)} {kind.value} to {other}."
                    f" {_article(kind).capitalize()} {kind.value} needs a wall link first,"
                    f" so I add the wall."
                )
            calls.append(
                ToolCall("add_link", {"room_a": new_id, "room_b": other, "kind": "wall"}, why)
            )
            if kind != LinkKind.WALL:
                calls.append(
                    ToolCall(
                        "set_link_kind",
                        {"room_a": new_id, "room_b": other, "kind": kind.value},
                        f"Now I turn the wall between {new_id} and {other}"
                        f" into {_article(kind)} {kind.value}.",
                    )
                )
        if room.entrance:
            calls.append(
                ToolCall(
                    "add_exterior_door",
                    {"room_id": new_id},
                    f"The request gives {new_id} a door to the outside, so I add that too.",
                )
            )
        return calls

    def reply(self, plan: FloorplanGraph) -> Reply:
        room = _room(plan, self.room_id)
        text = f"Added the {_type_name(room)} at ({room.xy[0]}, {room.xy[1]})."
        return Reply(
            "The room and everything the request lists are in place.",
            text + self._connections(plan),
        )


@dataclass(frozen=True)
class CloseDoorTask:
    """A door or opening was turned into a plain wall; the user asks for it back."""

    room_a: str
    room_b: str
    link_kind: LinkKind
    kind: str = "close_door"

    def corrupt(self, plan: FloorplanGraph) -> FloorplanGraph:
        arguments = {"room_a": self.room_a, "room_b": self.room_b, "kind": "wall"}
        return _apply(plan, "set_link_kind", arguments)

    def request(self, plan: FloorplanGraph) -> str:
        pair = f"{_name(plan, self.room_a)} and {_name(plan, self.room_b)}"
        if self.link_kind == LinkKind.DOOR:
            return f"Put a door between {pair}."
        return f"Open up the wall between {pair}: make it an opening."

    def solve(self, plan: FloorplanGraph, damaged: FloorplanGraph) -> list[ToolCall]:
        kind = self.link_kind.value
        return [
            ToolCall(
                "set_link_kind",
                {"room_a": self.room_a, "room_b": self.room_b, "kind": kind},
                f"{self.room_a} and {self.room_b} are already linked by a wall, so I change that"
                f" link to {_article(self.link_kind)} {kind}. No new link is needed.",
            )
        ]

    def reply(self, plan: FloorplanGraph) -> Reply:
        article, kind = _article(self.link_kind), self.link_kind.value
        text = f"The link between {self.room_a} and {self.room_b} is now {article} {kind}."
        return Reply("The link has the requested kind.", text)


@dataclass(frozen=True)
class DrawPlanTask:
    """The plan is empty and the user describes a whole plan: one `new_plan` call draws it."""

    kind: str = "draw_plan"

    def corrupt(self, plan: FloorplanGraph) -> FloorplanGraph:
        floors = [Floor(floor=floor.floor, rooms=[], links=[]) for floor in plan.floors]
        return FloorplanGraph(floors=floors, image=plan.image, exterior_wall=plan.exterior_wall)

    def request(self, plan: FloorplanGraph) -> str:
        return f"Draw this plan from scratch:\n{summarize_graph(plan)}"

    def solve(self, plan: FloorplanGraph, damaged: FloorplanGraph) -> list[ToolCall]:
        reasoning = "The plan is empty and the user describes all of it, so I draw it in one call."
        return [ToolCall(NEW_PLAN, {"plan": to_compact(plan)}, reasoning)]

    def reply(self, plan: FloorplanGraph) -> Reply:
        links = sum(len(floor.links) for floor in plan.floors)
        text = f"Drew the plan with {len(plan.rooms())} rooms and {links} links."
        return Reply("The new plan matches the description.", text)


TASK_KINDS = ("delete_room", "close_door", "draw_plan")


def candidate_tasks(record: PlanRecord, kind: str) -> list[Task]:
    """Every task of this kind that makes sense on this plan, in plan order."""
    plan = record.graph
    if len(plan.floors) != 1:  # the model-facing text and tools are single-floor
        return []
    if kind == "delete_room":
        linked = {end for floor in plan.floors for link in floor.links for end in (link.a, link.b)}
        return [
            DeleteRoomTask(room.id)
            for room in plan.rooms()
            if room.id in linked
            and room.size is not None  # add_room needs a size to put the room back
            and room.type.value != "unknown"
            and len(room.label or "") <= MAX_LABEL_LENGTH
        ]
    if kind == "close_door":
        return [
            CloseDoorTask(link.a, link.b, link.kind)
            for floor in plan.floors
            for link in floor.links
            if link.kind != LinkKind.WALL
        ]
    if kind == "draw_plan":
        return [DrawPlanTask()] if 0 < len(plan.rooms()) <= MAX_DRAW_ROOMS else []
    raise ValueError(f"unknown task kind {kind!r}; kinds are {', '.join(TASK_KINDS)}")
