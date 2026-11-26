"""Plausibility of a drawn plan: does it look like a plan a person would live in?

A plan can meet every requirement and still be nonsense (a bedroom reachable only through
the bathroom, links that cross, forty rooms added until every "at least" holds). These
metrics compare a plan with real ones, so the same numbers are computed on the source plans
of the test prompts (the "real reference") and on what the model draws.

Each plan yields counts; `aggregate_plausibility` sums them and divides once, and a ratio with
an empty denominator is `None`.

| Metric             | Per plan                                                                    |
|--------------------|-----------------------------------------------------------------------------|
| circulation        | every room is reachable from an entrance over doors and openings            |
| link crossings     | two links whose straight segments cross (links sharing a room never count)  |
| link length        | linked-pair distance over the smaller nearest-neighbour distance of the two |
| validator warnings | `validate_graph` warnings: isolated room, no entrance                       |
| typed-link rarity  | links whose (type pair, kind) is below 0.5 % of real links                  |
| room-count fit     | total rooms within the real 5th to 95th percentile for the bedroom count    |
| wall fit           | indoor rooms whose point lies inside the exterior wall                      |

Rarity and the room-count band need real plans; `build_reference_tables` makes them from the
train plans and the set builder saves them next to the held-out sets, so a run needs no
training data.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Point, Polygon

from vlm_floorplan.edits.validator import validate_graph
from vlm_floorplan.requirements.facts import compute_facts
from vlm_floorplan.schema import FloorplanGraph, RoomType

RARE_LINK_SHARE = 0.005
"""A (type pair, kind) below this share of real links is rare."""
MAX_BEDROOM_BUCKET = 5
"""Bedroom counts from this up share one room-count band."""
MIN_PLANS_FOR_BAND = 20
LOW_PERCENTILE, HIGH_PERCENTILE = 5, 95


class ReferenceTables(BaseModel):
    """What real plans look like, from the train split."""

    model_config = ConfigDict(extra="forbid")

    link_share: dict[str, float] = Field(default_factory=dict)
    """(type pair, kind) key to its share of all real links."""
    room_count_band: dict[str, tuple[int, int]] = Field(default_factory=dict)
    """Bedroom bucket to the (5th, 95th) percentile of total rooms among real plans."""
    n_plans: int = 0
    n_links: int = 0


class PlanPlausibility(BaseModel):
    """Raw counts for one plan."""

    model_config = ConfigDict(extra="forbid")

    rooms: int
    unreachable_rooms: int
    has_crossing: bool
    links: int
    link_ratios: list[float]
    warnings: dict[str, int]
    rare_links: int
    in_band: bool | None
    indoor_rooms: int
    indoor_inside_wall: int


def link_key(type_a: RoomType, type_b: RoomType, kind: str) -> str:
    first, second = sorted((type_a.value, type_b.value))
    return f"{first}-{second}:{kind}"


def bedroom_bucket(count: int) -> str:
    return str(min(count, MAX_BEDROOM_BUCKET))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0-100), None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def build_reference_tables(graphs: Iterable[FloorplanGraph]) -> ReferenceTables:
    links: Counter[str] = Counter()
    totals_by_bucket: dict[str, list[int]] = {}
    n_plans = 0
    for graph in graphs:
        facts = compute_facts(graph)
        n_plans += 1
        for floor in graph.floors:
            for link in floor.links:
                key = link_key(facts.rooms[link.a].type, facts.rooms[link.b].type, link.kind.value)
                links[key] += 1
        bucket = bedroom_bucket(facts.count(RoomType.BEDROOM))
        totals_by_bucket.setdefault(bucket, []).append(len(facts.rooms))
    n_links = sum(links.values())
    band = {}
    for bucket, totals in totals_by_bucket.items():
        if len(totals) >= MIN_PLANS_FOR_BAND:
            low, high = percentile(totals, LOW_PERCENTILE), percentile(totals, HIGH_PERCENTILE)
            band[bucket] = (math.floor(low or 0), math.ceil(high or 0))
    return ReferenceTables(
        link_share={key: count / n_links for key, count in links.items()} if n_links else {},
        room_count_band=band,
        n_plans=n_plans,
        n_links=n_links,
    )


def _orientation(p: tuple[int, int], q: tuple[int, int], r: tuple[int, int]) -> int:
    value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (value > 0) - (value < 0)


def segments_cross(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    """True when segment ab and segment cd cross at a point inside both (a proper crossing)."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _nearest_neighbor(facts_rooms: list) -> dict[str, float]:
    nearest = {}
    for room in facts_rooms:
        distances = [math.dist(room.xy, other.xy) for other in facts_rooms if other.id != room.id]
        nearest[room.id] = min(distances) if distances else 0.0
    return nearest


def plan_plausibility(graph: FloorplanGraph, tables: ReferenceTables) -> PlanPlausibility:
    facts = compute_facts(graph)
    rooms = list(facts.rooms.values())
    reached = facts.reachable()
    links = [link for floor in graph.floors for link in floor.links]
    segments = [(link, facts.rooms[link.a].xy, facts.rooms[link.b].xy) for link in links]
    crossing = any(
        not {first.a, first.b} & {second.a, second.b} and segments_cross(p1, p2, q1, q2)
        for (first, p1, p2), (second, q1, q2) in combinations(segments, 2)
    )
    nearest = _nearest_neighbor(rooms)
    ratios = []
    for link, p1, p2 in segments:
        scale = min(nearest[link.a], nearest[link.b])
        if scale > 0:
            ratios.append(round(math.dist(p1, p2) / scale, 3))
    rare = sum(
        tables.link_share.get(
            link_key(facts.rooms[link.a].type, facts.rooms[link.b].type, link.kind.value), 0.0
        )
        < RARE_LINK_SHARE
        for link in links
    )
    band = tables.room_count_band.get(bedroom_bucket(facts.count(RoomType.BEDROOM)))
    indoor = [room for room in rooms if room.type != RoomType.TERRACE]
    inside = len(indoor)
    if graph.exterior_wall is not None and len(graph.exterior_wall) >= 3:
        wall = Polygon(graph.exterior_wall)
        inside = sum(wall.buffer(0).contains(Point(room.xy)) for room in indoor)
    return PlanPlausibility(
        rooms=len(rooms),
        unreachable_rooms=len(rooms) - len(reached),
        has_crossing=crossing,
        links=len(links),
        link_ratios=ratios,
        warnings=dict(Counter(issue.code.value for issue in validate_graph(graph).warnings)),
        rare_links=rare if tables.link_share else 0,
        in_band=None if band is None else band[0] <= len(rooms) <= band[1],
        indoor_rooms=len(indoor),
        indoor_inside_wall=inside,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_plausibility(plans: Sequence[PlanPlausibility]) -> dict[str, object]:
    """Counts first, ratios last; `None` where there is nothing to divide by."""
    n = len(plans)
    warned = [plan for plan in plans if plan.warnings]
    ratios = [value for plan in plans for value in plan.link_ratios]
    banded = [plan for plan in plans if plan.in_band is not None]
    codes: Counter[str] = Counter()
    for plan in plans:
        codes.update(plan.warnings.keys())
    return {
        "n_plans": n,
        "circulation_rate": _ratio(sum(plan.unreachable_rooms == 0 for plan in plans), n),
        "crossing_rate": _ratio(sum(plan.has_crossing for plan in plans), n),
        "link_length_ratio_p95": percentile(ratios, 95),
        "link_length_ratio_median": percentile(ratios, 50),
        "warning_rate": _ratio(len(warned), n),
        "warning_plans_by_code": dict(codes),
        "rare_link_rate": _ratio(
            sum(plan.rare_links for plan in plans), sum(p.links for p in plans)
        ),
        "room_count_in_band_rate": _ratio(sum(bool(p.in_band) for p in banded), len(banded)),
        "indoor_inside_wall_rate": _ratio(
            sum(plan.indoor_inside_wall for plan in plans), sum(p.indoor_rooms for p in plans)
        ),
    }
