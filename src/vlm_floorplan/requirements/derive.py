"""Draw a requirement set that a given plan meets, at one of five specificity levels.

The text-to-graph prompts are generated from a real plan: its facts become requirements
(`candidate_pool`), a level decides how many and of which kind (`derive_requirements`), and a
template turns them into words. Because every requirement is a true fact of the source plan,
the source plan meets its own prompt by construction; `tests/requirements/test_derive.py` checks
that on every level.

| Level | Requirements                                                                  |
|-------|-------------------------------------------------------------------------------|
| L0    | none, or the bedroom count                                                    |
| L1    | bedroom count and at most one more                                            |
| L2    | bedroom and bathroom counts and 1 to 4 of counts, adjacency, ensuites, doors  |
| L3    | the L2 base and 3 to 7 more, with layout regions, labels, reachability       |
| L4    | every room type's count and the entrance count (the whole plan is described)  |
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from vlm_floorplan.requirements.facts import PASSAGE_KINDS, Facts, compute_facts
from vlm_floorplan.requirements.types import (
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

LEVELS = (0, 1, 2, 3, 4)
MIN_BOX_EXTENT = 300
"""A plan narrower than this on an axis has no meaningful left/right or top/bottom."""
MIN_SEPARATION = 150
"""Two unique rooms are only compared by position when their points are this far apart."""
MASTER_LABEL = "Master bedroom"
_NEGATABLE = (
    RoomType.DINING,
    RoomType.STORAGE,
    RoomType.UTILITY,
    RoomType.TERRACE,
    RoomType.GARAGE,
    RoomType.ENTRY,
)
_COUNTED = (
    RoomType.KITCHEN,
    RoomType.LIVING,
    RoomType.DINING,
    RoomType.STORAGE,
    RoomType.UTILITY,
    RoomType.TERRACE,
    RoomType.GARAGE,
    RoomType.ENTRY,
)
_PLAIN = {RoomType.OTHER, RoomType.UNKNOWN, RoomType.STAIRS}
"""Types a person does not ask for by name in a requirement."""
_RELATIONAL = (Adjacent, InRegion, RelativePosition)

Pool = dict[str, list[Requirement]]


def _count(room_type: RoomType, n: int, rng: random.Random) -> RoomCount:
    """A count requirement the plan meets: usually exact, sometimes "at least"."""
    roll = rng.random()
    if n >= 1 and roll < 0.15:
        return RoomCount(type=room_type, op="ge", n=max(1, n - rng.choice([0, 1])))
    if n >= 1 and roll < 0.22:
        return RoomCount(type=room_type, op="le", n=n + rng.choice([0, 1]))
    return RoomCount(type=room_type, op="eq", n=n)


def _adjacent_candidates(facts: Facts) -> list[Requirement]:
    found: dict[tuple[RoomType, RoomType, str], Adjacent] = {}
    for floor in facts.graph.floors:
        for link in floor.links:
            if link.kind not in PASSAGE_KINDS:
                continue
            types = sorted(
                {facts.rooms[link.a].type, facts.rooms[link.b].type}, key=lambda t: t.value
            )
            if len(types) != 2 or any(t in _PLAIN for t in types):
                continue
            kinds = ["connected", link.kind.value if link.kind == LinkKind.OPENING else ""]
            for kind in filter(None, kinds):
                found[(types[0], types[1], kind)] = Adjacent(a=types[0], b=types[1], link=kind)
    return list(found.values())


def _region_candidates(facts: Facts, rng: random.Random) -> list[Requirement]:
    min_x, min_y, max_x, max_y = facts.box
    if max_x - min_x < MIN_BOX_EXTENT or max_y - min_y < MIN_BOX_EXTENT:
        return []
    found: list[Requirement] = []
    for room_type, rooms in facts.by_type.items():
        if room_type in _PLAIN:
            continue
        shared = set.intersection(*(set(facts.regions_of(room)) for room in rooms))
        if not shared:
            continue
        corners = [r for r in Region if "_" in r.value and r in shared]
        sides = [r for r in Region if "_" not in r.value and r in shared]
        # One region per type, a corner when there is one, so the text names a clear place.
        choice = rng.choice(corners if corners and rng.random() < 0.5 else corners + sides)
        found.append(InRegion(type=room_type, region=choice))
    return found


def _relative_candidates(facts: Facts) -> list[Requirement]:
    unique = [rooms[0] for t, rooms in facts.by_type.items() if len(rooms) == 1 and t not in _PLAIN]
    found: list[Requirement] = []
    for index, first in enumerate(unique):
        for second in unique[index + 1 :]:
            for axis, relation in ((0, "left_of"), (1, "above")):
                gap = second.xy[axis] - first.xy[axis]
                if abs(gap) <= MIN_SEPARATION:
                    continue
                near, far = (first, second) if gap > 0 else (second, first)
                found.append(RelativePosition(a=near.type, b=far.type, relation=relation))
    return found


def candidate_pool(graph: FloorplanGraph, source: str, rng: random.Random) -> Pool:
    """Every requirement family with the facts of this plan that can state it."""
    facts = compute_facts(graph)
    pool: Pool = defaultdict(list)
    for room_type in (RoomType.BEDROOM, RoomType.BATHROOM):
        pool[f"main_{room_type.value}"].append(_count(room_type, facts.count(room_type), rng))
    for room_type in _COUNTED:
        if facts.count(room_type) and not _never_negated(room_type, source):
            pool["count"].append(_count(room_type, facts.count(room_type), rng))
    for room_type in _NEGATABLE:
        if facts.count(room_type) == 0 and not _never_negated(room_type, source):
            pool["negative"].append(RoomCount(type=room_type, op="eq", n=0))
    pool["adjacent"] = _adjacent_candidates(facts)
    if facts.ensuite_ids:
        pool["ensuites"].append(Ensuites(op="eq", n=len(facts.ensuite_ids)))
    entrances = len(facts.entrance_rooms)
    pool["entrances"].append(Entrances(op="eq", n=entrances))
    for room_type in sorted({room.type for room in facts.entrance_rooms}, key=lambda t: t.value):
        if room_type not in _PLAIN:
            pool["entrance_into"].append(EntranceInto(type=room_type))
    pool["region"] = _region_candidates(facts, rng)
    pool["relative"] = _relative_candidates(facts)
    if facts.count(RoomType.BEDROOM) and facts.count(RoomType.BATHROOM):
        reach = ReachableAvoiding(type=RoomType.BATHROOM, avoid=RoomType.BEDROOM)
        if facts.reachable(avoid=RoomType.BEDROOM) & {
            r.id for r in facts.by_type[RoomType.BATHROOM]
        }:
            pool["reachable"].append(reach)
    if facts.count(RoomType.BEDROOM) >= 2:
        pool["label"].append(HasLabel(type=RoomType.BEDROOM, text=MASTER_LABEL))
    return dict(pool)


def _never_negated(room_type: RoomType, source: str) -> bool:
    """Types whose absence says nothing about the source (ResPlan merges halls into living)."""
    if room_type == RoomType.ENTRY and source == "resplan":
        return True
    return room_type == RoomType.GARAGE and source != "cubicasa5k"


def _types_of(req: Requirement) -> set[RoomType]:
    if isinstance(req, Adjacent | RelativePosition):
        return {req.a, req.b}
    if isinstance(req, InRegion):
        return {req.type}
    return set()


class _Picker:
    """Collects requirements while keeping each room type in at most one relation."""

    def __init__(self, pool: Pool, rng: random.Random) -> None:
        self.pool, self.rng = pool, rng
        self.chosen: list[Requirement] = []
        self.related: set[RoomType] = set()
        self.counted: set[RoomType] = set()
        self.families: set[str] = set()

    def add(self, req: Requirement) -> bool:
        if isinstance(req, RoomCount):
            if req.type in self.counted:
                return False
            self.counted.add(req.type)
        if isinstance(req, _RELATIONAL):
            types = _types_of(req)
            if types & self.related:
                return False
            self.related |= types
        if req in self.chosen:
            return False
        self.chosen.append(req)
        return True

    def take(self, family: str, count: int = 1) -> int:
        """Add up to `count` requirements of one family; returns how many were added."""
        options = list(self.pool.get(family, []))
        self.rng.shuffle(options)
        added = 0
        for req in options:
            if added >= count:
                break
            if self.add(req):
                added += 1
                self.families.add(family)
        return added

    def fill(self, families: Sequence[str], target: int) -> None:
        """Add requirements from the families, in a random order, until `target` are added."""
        order = list(families)
        added = 0
        for _attempt in range(4 * target + 4):
            if added >= target:
                break
            added += self.take(self.rng.choice(order))


def _full_schedule(graph: FloorplanGraph) -> list[Requirement]:
    facts = compute_facts(graph)
    counts = [
        RoomCount(type=room_type, op="eq", n=len(rooms))
        for room_type, rooms in sorted(facts.by_type.items(), key=lambda item: item[0].value)
    ]
    return [*counts, Entrances(op="eq", n=len(facts.entrance_rooms))]


def derive_requirements(
    graph: FloorplanGraph, level: int, rng: random.Random, source: str = ""
) -> list[Requirement]:
    """A requirement set of this level that `graph` meets. Counts come first in the list."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level}")
    if level == 4:
        return _full_schedule(graph)
    pool = candidate_pool(graph, source, rng)
    picker = _Picker(pool, rng)
    bedrooms = pool["main_bedroom"][0]
    if level == 0:
        if rng.random() < 0.6 and isinstance(bedrooms, RoomCount) and bedrooms.n >= 1:
            picker.add(bedrooms)
        return picker.chosen
    picker.add(bedrooms)
    extras = ("adjacent", "entrances", "negative", "ensuites", "count")
    if level == 1:
        if rng.random() < 0.7:
            picker.fill(("main_bathroom", *extras), 1)
        return picker.chosen
    picker.take("main_bathroom")
    if level == 2:
        picker.fill(extras, rng.randint(1, 4))
        return picker.chosen
    layout = ("region", "relative", "label", "reachable", "entrance_into")
    picker.fill((*extras, *layout), rng.randint(3, 7))
    # Layout requirements are what sets L3 apart; make sure at least one made it in.
    if not picker.families & set(layout):
        picker.fill(layout, 1)
    return picker.chosen[:9]


def apply_labels(graph: FloorplanGraph, requirements: Sequence[Requirement]) -> FloorplanGraph:
    """The plan with each `HasLabel` put on its room: the largest room of that type.

    Source plans carry no labels (the canonical frame drops them), so the reference plan of a
    prompt that names a room gets the user's name here.
    """
    wanted = [req for req in requirements if isinstance(req, HasLabel)]
    if not wanted:
        return graph
    floors = [floor.model_copy(deep=True) for floor in graph.floors]
    for req in wanted:
        rooms = [room for floor in floors for room in floor.rooms if room.type == req.type]
        if rooms:
            max(rooms, key=lambda room: room.size or 0).label = req.text
    return graph.model_copy(update={"floors": floors})
