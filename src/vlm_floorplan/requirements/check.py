"""Does a plan meet a requirement? One function decides it, for every use.

`check_requirements(graph, requirements)` is the ground truth of the text-to-graph data
generator (a sample is kept only if its target meets what the text asks), the metric of the
held-out evaluation, and the reward a later reinforcement-learning run can use.
It reads only the graph, so it works on a model's plan as it does on a source plan.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vlm_floorplan.requirements.facts import PASSAGE_KINDS, Facts, compute_facts
from vlm_floorplan.requirements.types import (
    Adjacent,
    Ensuites,
    EntranceInto,
    Entrances,
    HasLabel,
    InRegion,
    ReachableAvoiding,
    RelativePosition,
    Requirement,
    RoomCount,
)
from vlm_floorplan.schema import FloorplanGraph, LinkKind

_COMPARE: dict[str, Callable[[int, int], bool]] = {
    "eq": operator.eq,
    "ge": operator.ge,
    "le": operator.le,
}


@dataclass(frozen=True)
class RequirementResult:
    requirement: Requirement
    met: bool


def _adjacent(req: Adjacent, facts: Facts) -> bool:
    for floor in facts.graph.floors:
        for link in floor.links:
            if not _link_matches(req.link, link.kind):
                continue
            pair = {facts.rooms[link.a].type, facts.rooms[link.b].type}
            if pair == {req.a, req.b} or (req.a == req.b and pair == {req.a}):
                return True
    return False


def _link_matches(wanted: str, kind: LinkKind) -> bool:
    if wanted == "touching":
        return True
    if wanted == "connected":
        return kind in PASSAGE_KINDS
    return kind.value == wanted


def _in_region(req: InRegion, facts: Facts) -> bool:
    rooms = facts.by_type.get(req.type, [])
    return bool(rooms) and all(req.region in facts.regions_of(room) for room in rooms)


def _relative(req: RelativePosition, facts: Facts) -> bool:
    first, second = facts.by_type.get(req.a, []), facts.by_type.get(req.b, [])
    if not first or not second:
        return False
    axis = 0 if req.relation == "left_of" else 1
    return all(a.xy[axis] < b.xy[axis] for a in first for b in second)


def _reachable(req: ReachableAvoiding, facts: Facts) -> bool:
    reached = facts.reachable(avoid=req.avoid)
    return any(room.id in reached for room in facts.by_type.get(req.type, []))


def _holds(req: Requirement, facts: Facts) -> bool:
    if isinstance(req, RoomCount):
        return _COMPARE[req.op](facts.count(req.type), req.n)
    if isinstance(req, Adjacent):
        return _adjacent(req, facts)
    if isinstance(req, Ensuites):
        return _COMPARE[req.op](len(facts.ensuite_ids), req.n)
    if isinstance(req, Entrances):
        return _COMPARE[req.op](len(facts.entrance_rooms), req.n)
    if isinstance(req, EntranceInto):
        return any(room.type == req.type for room in facts.entrance_rooms)
    if isinstance(req, InRegion):
        return _in_region(req, facts)
    if isinstance(req, RelativePosition):
        return _relative(req, facts)
    if isinstance(req, ReachableAvoiding):
        return _reachable(req, facts)
    if isinstance(req, HasLabel):
        wanted = req.text.strip().casefold()
        return any(
            room.label is not None and room.label.strip().casefold() == wanted
            for room in facts.by_type.get(req.type, [])
        )
    raise TypeError(f"unknown requirement {req!r}")


def holds(requirement: Requirement, graph: FloorplanGraph) -> bool:
    """Whether one requirement is met by the plan."""
    return _holds(requirement, compute_facts(graph))


def check_requirements(
    graph: FloorplanGraph, requirements: Sequence[Requirement]
) -> list[RequirementResult]:
    """One result per requirement, in order."""
    facts = compute_facts(graph)
    return [RequirementResult(req, _holds(req, facts)) for req in requirements]


def all_met(graph: FloorplanGraph, requirements: Sequence[Requirement]) -> bool:
    """True when every requirement holds (a request with no requirements is met by any plan)."""
    return all(result.met for result in check_requirements(graph, requirements))
