"""The stage-1 behavior test: single requests where the right answer is not "draw".

- edit: a plan exists and the user asks for one change ("put a door between ..."); the right
  response is edit tools that make the change, never `new_plan`.
- restart: a plan exists and the user says to start over, asking for what another plan has
  and the current plan lacks; the right response is a `new_plan` that meets the new request.
- contradiction: "a one-bedroom flat with three bedrooms"; no call, one question naming the
  conflict.
- unsupported: a request plus square metres, a compass direction and the like; no call, one
  question naming what the plan cannot hold.

`expect` holds what the scorer needs: the link or room an edit must produce, or the groups of
words a question must contain (at least one word of every group).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Collection

from vlm_floorplan.requirements import all_met, derive_requirements, requirements_to_json
from vlm_floorplan.schema import FloorplanGraph, LinkKind, Room, RoomType
from vlm_floorplan.text2graph.leak import normalize_text
from vlm_floorplan.text2graph.rows import PromptRow, empty_start
from vlm_floorplan.text2graph.templates import choose_kind, render_text
from vlm_floorplan.text2graph.words import NOUNS, NUMBER_WORDS

BEHAVIOR_QUOTAS = {"edit": 40, "restart": 15, "contradiction": 20, "unsupported": 25}
EDIT_KINDS = ("door", "open", "add")
RESTART_TRIES = 10
_OPENABLE = {RoomType.LIVING, RoomType.KITCHEN, RoomType.DINING, RoomType.ENTRY}
_ADDABLE = (RoomType.STORAGE, RoomType.UTILITY, RoomType.TERRACE, RoomType.DINING)
_ANCHORS = (RoomType.LIVING, RoomType.ENTRY, RoomType.BEDROOM, RoomType.KITCHEN)
_RESTARTS = (
    "Forget this plan and draw a new one.",
    "Start again from scratch.",
    "Scrap this one and start over.",
    "Throw this plan away and begin again.",
)
_UNSUPPORTED_ASKS = {
    "area": (
        "of about {m} square metres",
        ["square metres", "square meters", "square metre", "square meter", "m2", "sqm"],
    ),
    "compass": (
        "with the living room facing south",
        ["south", "facing", "direction", "orientation", "compass"],
    ),
    "storeys": (
        "on two storeys",
        ["storey", "storeys", "story", "stories", "two floors", "second floor", "two levels"],
    ),
    "ceiling": ("with 3 metre high ceilings", ["ceiling", "ceilings"]),
    "building_floor": (
        "on the 5th floor of the building",
        ["5th", "fifth", "5th floor", "floor of the building"],
    ),
}


def _noun(room_type: RoomType) -> str:
    return NOUNS[room_type][0][0]


def _name(room: Room, graph: FloorplanGraph) -> str:
    """ "the kitchen", or "bedroom r2" when the plan has several of the type."""
    repeated = sum(r.type == room.type for r in graph.rooms()) > 1
    return f"{_noun(room.type)} {room.id}" if repeated else f"the {_noun(room.type)}"


def _row(index: int, kind: str, behavior: str, text: str, **extra) -> dict:
    return {
        "id": f"B{index:03d}",
        "set": "behavior_test",
        "kind": kind,
        "behavior": behavior,
        "text": text,
        **extra,
    }


def edit_row(
    index: int, record_id: str, source: str, graph: FloorplanGraph, kind: str, rng: random.Random
) -> PromptRow | None:
    """One edit request on an existing plan, or None when the plan has no place for it."""
    rooms = {room.id: room for room in graph.rooms()}
    links = graph.floors[0].links
    start = graph.model_dump(mode="json")
    if kind in ("door", "open"):
        walls = [
            link
            for link in links
            if link.kind == LinkKind.WALL
            and (
                kind == "door"
                or {rooms[link.a].type, rooms[link.b].type} <= _OPENABLE
                and rooms[link.a].type != rooms[link.b].type
            )
            and rooms[link.a].type != RoomType.TERRACE
            and rooms[link.b].type != RoomType.TERRACE
        ]
        if not walls:
            return None
        link = rng.choice(walls)
        a, b = _name(rooms[link.a], graph), _name(rooms[link.b], graph)
        if kind == "door":
            text, wanted = f"Put a door between {a} and {b}.", "door"
        else:
            text = f"Remove the wall between {a} and {b} so they flow into each other."
            wanted = "opening"
        expect = {"edit": "link", "a": link.a, "b": link.b, "link": wanted}
    else:
        present = {room.type for room in rooms.values()}
        new_types = [t for t in _ADDABLE if t not in present]
        anchors = [r for r in rooms.values() if r.type in _ANCHORS]
        if not new_types or not anchors:
            return None
        new_type, anchor = rng.choice(new_types), rng.choice(anchors)
        text = f"Add a {_noun(new_type)} off {_name(anchor, graph)}."
        expect = {"edit": "add", "type": new_type.value, "near": anchor.id}
    return PromptRow(
        **_row(index, f"edit_{kind}", "edit", text),
        source=source,
        record_id=record_id,
        start=start,
        expect=expect,
    )


def restart_row(
    index: int,
    graph: FloorplanGraph,
    other: tuple[str, str, FloorplanGraph],
    rng: random.Random,
) -> PromptRow | None:
    """ "Start over" on `graph`, asking for what the plan `other` has.

    The row names `other` as its record, the plan that meets the request. Requirements are
    drawn until `graph` fails at least one of them, so keeping the old plan is never right;
    None when `other` is `graph` or no draw does that.
    """
    record_id, source, target = other
    if target == graph:
        return None
    for _ in range(RESTART_TRIES):
        reqs = derive_requirements(target, 2, rng, source)
        if not all_met(graph, reqs):
            break
    else:
        return None
    kind = choose_kind(2, reqs, rng)
    request = render_text(target, reqs, kind, source, rng)
    return PromptRow(
        **_row(index, "restart", "restart", f"{rng.choice(_RESTARTS)} {request}"),
        source=source,
        record_id=record_id,
        requirements=requirements_to_json(reqs),
        start=graph.model_dump(mode="json"),
        expect={"new_plan": True},
    )


def _words(n: int) -> list[str]:
    return [NUMBER_WORDS[n], str(n)]


def contradiction_row(
    index: int, record_id: str, source: str, graph: FloorplanGraph, rng: random.Random
) -> PromptRow:
    n = rng.randint(2, 4)
    choice = rng.randrange(4)
    if choice == 0:
        text = f"A one-bedroom flat with {NUMBER_WORDS[n]} bedrooms and a small kitchen."
        groups = [
            ["one-bedroom", "one bedroom", "1-bedroom", "1 bedroom", "single bedroom"],
            _words(n),
        ]
    elif choice == 1:
        text = f"A studio with {NUMBER_WORDS[n]} bedrooms."
        groups = [["studio"], _words(n)]
    elif choice == 2:
        text = (
            "I need a flat with no bathroom at all, and an ensuite bathroom for the main bedroom."
        )
        groups = [["no bathroom", "without a bathroom", "no bath"], ["ensuite", "en-suite"]]
    else:
        text = f"Exactly two bedrooms please, and also {NUMBER_WORDS[n + 1]} bedrooms in total."
        groups = [["two", "2"], _words(n + 1)]
    return PromptRow(
        **_row(index, "contradiction", "contradiction", text),
        source=source,
        record_id=record_id,
        start=empty_start(graph),
        expect={"groups": groups, "need": "all"},
    )


def unsupported_row(
    index: int, record_id: str, source: str, graph: FloorplanGraph, rng: random.Random
) -> PromptRow:
    bedrooms = rng.randint(1, 3)
    asks = rng.sample(sorted(_UNSUPPORTED_ASKS), 2)
    phrases = [_UNSUPPORTED_ASKS[a][0].format(m=rng.choice([55, 70, 85, 110])) for a in asks]
    text = (
        f"I want a {NUMBER_WORDS[bedrooms]}-bedroom flat with a balcony off the living room, "
        f"{phrases[0]} and {phrases[1]}."
    )
    groups = [_UNSUPPORTED_ASKS[a][1] for a in asks]
    return PromptRow(
        **_row(index, "unsupported", "unsupported", text),
        source=source,
        record_id=record_id,
        start=empty_start(graph),
        expect={"groups": groups, "need": "any"},
    )


def build_behavior_rows(
    plans: list[tuple[str, str, FloorplanGraph]],
    rng: random.Random,
    quotas: dict[str, int] | None = None,
    forbidden: Collection[str] = (),
) -> list[PromptRow]:
    """The behavior test set from a pool of (record id, source, canonical graph) plans.

    Plans are used round-robin; an edit is retried on later plans when a plan has no place for
    it. A prompt whose text equals one in `forbidden` (train prompts) is drawn again.
    Raises `ValueError` when the pool cannot fill a quota.
    """
    quotas = quotas or BEHAVIOR_QUOTAS
    plans = list(plans)
    rng.shuffle(plans)
    rows: list[PromptRow] = []
    cursor = 0

    def next_plan() -> tuple[str, str, FloorplanGraph]:
        nonlocal cursor
        plan = plans[cursor % len(plans)]
        cursor += 1
        return plan

    for number in range(quotas["edit"]):
        kind = EDIT_KINDS[number % len(EDIT_KINDS)]
        for attempt in range(len(plans)):
            index = (cursor + attempt) % len(plans)
            record_id, source, graph = plans[index]
            row = edit_row(len(rows), record_id, source, graph, kind, rng)
            if row and normalize_text(row.text) not in forbidden:
                rows.append(row)
                cursor = index + 1
                break
        else:
            raise ValueError(f"no plan in the pool can take an edit of kind {kind}")

    def fresh(make: Callable[[str, str, FloorplanGraph], PromptRow | None]) -> PromptRow:
        for _attempt in range(50):
            row = make(*next_plan())
            if row and normalize_text(row.text) not in forbidden:
                return row
        raise ValueError("no drawn behavior prompt is usable and differs from every train prompt")

    for _ in range(quotas["restart"]):
        rows.append(fresh(lambda _i, _s, g: restart_row(len(rows), g, rng.choice(plans), rng)))
    for _ in range(quotas["contradiction"]):
        rows.append(fresh(lambda i, s, g: contradiction_row(len(rows), i, s, g, rng)))
    for _ in range(quotas["unsupported"]):
        rows.append(fresh(lambda i, s, g: unsupported_row(len(rows), i, s, g, rng)))
    return rows


def behavior_counts(rows: list[PromptRow]) -> dict[str, int]:
    return dict(Counter(row.kind for row in rows))
