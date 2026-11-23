"""Template text: a requirement set (or a whole plan) put into words.

Templates alone are the val and test text of the template sets, so what they say is exactly
the requirements: nothing is added and nothing is left out. `describe` turns one requirement
into a noun phrase ("2 ensuite bathrooms", "the kitchen at the top"), a frame joins the
phrases into a request, and the input kind decides the frame:

| Kind         | Shape                                                              |
|--------------|--------------------------------------------------------------------|
| minimal      | "Draw a two-bedroom flat."                                         |
| free         | a sentence: "We're looking for a place with ..."                   |
| list         | bullets, a semicolon list or a comma list                          |
| partial      | a short request, then "Make sure it also has ..."                  |
| layout       | positions first: "The living room at the top, the kitchen ..."     |
| full         | the whole plan as rooms with points and a links line               |
| full_natural | the whole plan as prose with names and connections                 |
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from vlm_floorplan.requirements import (
    Adjacent,
    Ensuites,
    EntranceInto,
    Entrances,
    HasLabel,
    InRegion,
    ReachableAvoiding,
    Region,
    RelativePosition,
    Requirement,
    RoomCount,
)
from vlm_floorplan.schema import FloorplanGraph, LinkKind, RoomType
from vlm_floorplan.text2graph.words import Style

INPUT_KINDS = ("minimal", "free", "list", "partial", "layout", "full", "full_natural")
KINDS_BY_LEVEL: dict[int, tuple[tuple[str, int], ...]] = {
    0: (("minimal", 1),),
    1: (("free", 4), ("list", 2), ("partial", 4)),
    2: (("free", 4), ("list", 4), ("partial", 3)),
    3: (("free", 3), ("list", 3), ("layout", 3), ("partial", 1)),
    4: (("full", 1), ("full_natural", 1)),
}
_REGION_PHRASES = {
    Region.TOP: ["at the top", "along the top"],
    Region.BOTTOM: ["at the bottom", "along the bottom"],
    Region.LEFT: ["on the left side", "on the left"],
    Region.RIGHT: ["on the right side", "on the right"],
    Region.TOP_LEFT: ["in the top-left corner", "at the top left"],
    Region.TOP_RIGHT: ["in the top-right corner", "at the top right"],
    Region.BOTTOM_LEFT: ["in the bottom-left corner", "at the bottom left"],
    Region.BOTTOM_RIGHT: ["in the bottom-right corner", "at the bottom right"],
}
_POSITIONAL = (InRegion, RelativePosition)
_HOME_WORDS = ("apartment", "flat", "home")
_FREE_FRAMES = (
    "I'd like {home} with {np}.",
    "We're looking for {home} with {np}.",
    "Could you draw {home} that has {np}?",
    "Design {home} with {np}.",
    "I need a floor plan for {home} with {np}.",
    "Please draw {home} that has {np}.",
)
_LAYOUT_FRAMES = (
    "{np}.",
    "Draw a plan with {np}.",
    "Floor plan: {np}.",
    "Lay it out like this: {np}.",
)
_MINIMAL_FRAMES = (
    "Draw {np}.",
    "Sketch {np}.",
    "Can you make me a floor plan for {np}?",
    "I need {np}.",
)


def join_phrases(phrases: Sequence[str]) -> str:
    """ "a, b and c"."""
    if len(phrases) <= 1:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


class Describer:
    """Noun phrases for the requirements of one prompt, in one style, about one plan."""

    def __init__(self, style: Style, plan_counts: Counter[RoomType], rng: random.Random) -> None:
        self.style, self.counts, self.rng = style, plan_counts, rng

    def _ref(self, room_type: RoomType) -> str:
        """ "the kitchen" for a room the plan has once, "a bedroom" for one of several."""
        noun = self.style.noun(room_type)
        return f"the {noun}" if self.counts[room_type] == 1 else f"{article(noun)} {noun}"

    def describe(self, req: Requirement) -> str:
        if isinstance(req, RoomCount):
            return self._count(req)
        if isinstance(req, Adjacent):
            return self._adjacent(req)
        if isinstance(req, Ensuites):
            return self._ensuites(req)
        if isinstance(req, Entrances):
            word = self.rng.choice(["entrance", "front door"])
            noun = word if req.n == 1 else word + "s"
            return f"{self.style.number(req.n)} {noun}" if req.n != 1 else f"a single {word}"
        if isinstance(req, EntranceInto):
            noun = self.style.noun(req.type)
            return self.rng.choice(
                [
                    f"the front door opening into {self._ref(req.type)}",
                    f"the entrance into the {noun}",
                ]
            )
        if isinstance(req, InRegion):
            return self._region(req)
        if isinstance(req, RelativePosition):
            first, second = self._ref(req.a), self._ref(req.b)
            if req.relation == "above":
                return f"{first} above {second}"
            if self.rng.random() < 0.5:
                return f"{first} to the left of {second}"
            return f"{second} to the right of {first}"
        if isinstance(req, ReachableAvoiding):
            target, avoid = self.style.noun(req.type), self.style.noun(req.avoid)
            return (
                f"{article(target)} {target} you can reach without walking through "
                f"{article(avoid)} {avoid}"
            )
        if isinstance(req, HasLabel):
            noun = self.style.noun(req.type)
            return f"one {noun} labeled '{req.text}'"
        raise TypeError(f"cannot describe {req!r}")

    def _count(self, req: RoomCount) -> str:
        noun = self.style.noun(req.type)
        if req.op == "eq" and req.n == 0:
            return f"no {noun}"
        if req.op == "eq":
            return self.style.counted(req.type, req.n)
        prefix = "at least" if req.op == "ge" else "at most"
        return f"{prefix} {self.style.counted(req.type, req.n)}"

    def _ensuites(self, req: Ensuites) -> str:
        word = self.rng.choice(["ensuite", "en-suite"])
        bath = self.style.noun(RoomType.BATHROOM, plural=req.n != 1)
        prefix = {"eq": "", "ge": "at least ", "le": "at most "}[req.op]
        return f"{prefix}{self.style.number(req.n)} {word} {bath}"

    def _adjacent(self, req: Adjacent) -> str:
        first, second = req.a, req.b
        if second == RoomType.TERRACE or (first != RoomType.TERRACE and self.rng.random() < 0.5):
            first, second = second, first
        one, two = self._ref(first), self._ref(second)
        if req.link == "opening":
            return f"{one} open to {two}"
        if req.link == "door":
            return f"a door between {one} and {two}"
        if RoomType.TERRACE in (first, second) and first == RoomType.TERRACE:
            return f"{one} off {two}"
        return self.rng.choice([f"{one} connected to {two}", f"{one} that connects to {two}"])

    def _region(self, req: InRegion) -> str:
        where = self.rng.choice(_REGION_PHRASES[req.region])
        if self.counts[req.type] > 1:
            return f"the {self.style.noun(req.type, plural=True)} {where}"
        return f"the {self.style.noun(req.type)} {where}"


def home_word(graph: FloorplanGraph, source: str, rng: random.Random) -> str:
    types = {room.type for room in graph.rooms()}
    if source == "cubicasa5k" and types & {RoomType.GARAGE, RoomType.STAIRS}:
        return "house"
    return rng.choice(_HOME_WORDS)


def _with_article(home: str) -> str:
    return f"{article(home)} {home}"


def _size_word(graph: FloorplanGraph) -> str:
    rooms = len(graph.rooms())
    return "small " if rooms <= 6 else "large " if rooms >= 12 else ""


def choose_kind(level: int, requirements: Sequence[Requirement], rng: random.Random) -> str:
    """An input kind for the level, one that the requirement set can fill."""
    kinds, weights = zip(*KINDS_BY_LEVEL[level], strict=True)
    kind = rng.choices(kinds, weights)[0]
    positional = any(isinstance(req, _POSITIONAL) for req in requirements)
    if kind == "layout" and not positional:
        kind = "free"
    if kind in {"free", "list", "partial"} and not requirements:
        kind = "minimal"
    return kind


def render_text(
    graph: FloorplanGraph,
    requirements: Sequence[Requirement],
    kind: str,
    source: str,
    rng: random.Random,
) -> str:
    """The request text for a requirement set (or, for the two full kinds, a whole plan)."""
    if kind == "full":
        return full_schedule_text(graph)
    if kind == "full_natural":
        return full_natural_text(graph, rng)
    style = Style.draw(rng)
    counts = Counter(room.type for room in graph.rooms())
    describer = Describer(style, counts, rng)
    home = _with_article(home_word(graph, source, rng))
    if kind == "minimal":
        return _minimal(graph, requirements, style, home_word(graph, source, rng), rng)
    phrases = [describer.describe(req) for req in requirements]
    if kind == "layout":
        positional = [
            p for p, r in zip(phrases, requirements, strict=True) if isinstance(r, _POSITIONAL)
        ]
        rest = [
            p for p, r in zip(phrases, requirements, strict=True) if not isinstance(r, _POSITIONAL)
        ]
        ordered = positional + rest
        return capitalize(rng.choice(_LAYOUT_FRAMES).format(np=join_phrases(ordered)))
    if kind == "list":
        return _list(phrases, home_word(graph, source, rng), rng)
    if kind == "partial":
        return _partial(phrases, home, rng)
    return capitalize(rng.choice(_FREE_FRAMES).format(home=home, np=join_phrases(phrases)))


def _minimal(
    graph: FloorplanGraph,
    requirements: Sequence[Requirement],
    style: Style,
    home: str,
    rng: random.Random,
) -> str:
    bedrooms = [r for r in requirements if isinstance(r, RoomCount) and r.type == RoomType.BEDROOM]
    size = f"{style.number(bedrooms[0].n)}-bedroom " if bedrooms else _size_word(graph)
    noun = f"{article(size or home)} {size}{home}"
    return rng.choice(_MINIMAL_FRAMES).format(np=noun)


def _list(phrases: list[str], home: str, rng: random.Random) -> str:
    shape = rng.choice(["bullets", "semicolons", "commas"])
    if shape == "bullets":
        return "Requirements:\n" + "\n".join(f"- {capitalize(p)}" for p in phrases)
    if shape == "semicolons":
        return f"{capitalize(home)}: " + "; ".join(phrases) + "."
    return f"{capitalize(home)} requirements: " + ", ".join(phrases) + "."


def _partial(phrases: list[str], home: str, rng: random.Random) -> str:
    if len(phrases) == 1:
        return rng.choice(
            [
                f"Draw {home}. Make sure it has {phrases[0]}.",
                f"Make sure the plan has {phrases[0]}.",
            ]
        )
    head = join_phrases(phrases[: max(1, len(phrases) // 2)])
    tail = join_phrases(phrases[max(1, len(phrases) // 2) :])
    return f"Draw {home} with {head}. Make sure it also has {tail}."


def full_schedule_text(graph: FloorplanGraph) -> str:
    """The whole plan, one room per line with its point, and one links line."""
    floor = graph.floors[0]
    lines = ["Draw this plan from scratch:"]
    for room in floor.rooms:
        entrance = " entrance" if room.entrance else ""
        lines.append(f"{room.id} {room.type.value} at ({room.xy[0]}, {room.xy[1]}){entrance}")
    if floor.links:
        lines.append("links: " + ", ".join(f"{k.a}-{k.b} {k.kind.value}" for k in floor.links))
    return "\n".join(lines)


def full_natural_text(graph: FloorplanGraph, rng: random.Random) -> str:
    """The whole plan as a list of named rooms and a sentence of connections."""
    style = Style.draw(rng)
    floor = graph.floors[0]
    totals = Counter(room.type for room in floor.rooms)
    seen: Counter[RoomType] = Counter()
    names: dict[str, str] = {}
    for room in floor.rooms:
        seen[room.type] += 1
        noun = style.noun(room.type)
        names[room.id] = f"{noun} {seen[room.type]}" if totals[room.type] > 1 else f"the {noun}"
    lines = [f"Draw these {len(floor.rooms)} rooms (positions on the 0-1000 grid):"]
    for room in floor.rooms:
        door = ", with the front door" if room.entrance else ""
        lines.append(f"- {capitalize(names[room.id])} at ({room.xy[0]}, {room.xy[1]}){door}")
    kinds = {
        LinkKind.DOOR: "a door",
        LinkKind.OPENING: "an opening",
        LinkKind.WALL: "a shared wall",
    }
    links = [f"{kinds[k.kind]} between {names[k.a]} and {names[k.b]}" for k in floor.links]
    if links:
        lines.append("Connections: " + "; ".join(links) + ".")
    return "\n".join(lines)
