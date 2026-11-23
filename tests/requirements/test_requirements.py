import random

import pytest
from t2g_fixtures import make_record

from vlm_floorplan.agent.sample_plan import sample_record
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
    RoomCount,
    apply_labels,
    check_requirements,
    derive_requirements,
    requirements_from_json,
    requirements_to_json,
)
from vlm_floorplan.requirements.derive import LEVELS
from vlm_floorplan.schema import RoomType
from vlm_floorplan.text2graph.frame import canonicalize

GRAPH = sample_record().graph  # r0 bed, r1 bath, r2 kitchen, r3 living, r4 entry*, r5 storage


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        (RoomCount(type="bedroom", op="eq", n=1), True),
        (RoomCount(type="bedroom", op="ge", n=2), False),
        (RoomCount(type="garage", op="eq", n=0), True),
        (RoomCount(type="bathroom", op="le", n=0), False),
        (Adjacent(a="kitchen", b="living", link="opening"), True),
        (Adjacent(a="living", b="kitchen", link="door"), False),
        (Adjacent(a="bedroom", b="bathroom", link="touching"), True),
        (Adjacent(a="bedroom", b="bathroom", link="connected"), False),
        (Entrances(op="eq", n=1), True),
        (EntranceInto(type="entry"), True),
        (EntranceInto(type="living"), False),
        (Ensuites(op="eq", n=0), True),
        (InRegion(type="bedroom", region=Region.TOP_LEFT), True),
        (InRegion(type="storage", region=Region.TOP), False),
        (RelativePosition(a="bedroom", b="kitchen", relation="left_of"), True),
        (RelativePosition(a="kitchen", b="bedroom", relation="left_of"), False),
        (ReachableAvoiding(type="bathroom", avoid="bedroom"), True),
        (ReachableAvoiding(type="bathroom", avoid="living"), False),
        (HasLabel(type="bedroom", text="Master bedroom"), False),
    ],
)
def test_requirement_is_decided_from_the_graph(requirement, expected):
    assert check_requirements(GRAPH, [requirement])[0].met is expected


def test_a_bathroom_that_opens_only_to_a_bedroom_is_ensuite():
    graph = GRAPH.model_copy(deep=True)
    link_type = type(graph.floors[0].links[0])
    kept = [link for link in graph.floors[0].links if "r1" not in (link.a, link.b)]
    graph.floors[0].links = [*kept, link_type(a="r0", b="r1", kind="door")]
    assert check_requirements(graph, [Ensuites(op="eq", n=1)])[0].met
    graph.floors[0].links.append(link_type(a="r1", b="r3", kind="door"))
    assert not check_requirements(graph, [Ensuites(op="eq", n=1)])[0].met


def test_a_label_requirement_needs_the_label_on_a_room_of_the_type():
    requirement = HasLabel(type=RoomType.BEDROOM, text="Master bedroom")
    labeled = apply_labels(GRAPH, [requirement])
    assert check_requirements(labeled, [requirement])[0].met
    assert not check_requirements(GRAPH, [requirement])[0].met


def test_requirements_round_trip_through_json():
    reqs = [
        RoomCount(type="bedroom", op="eq", n=2),
        Adjacent(a="kitchen", b="living", link="opening"),
        HasLabel(type="bedroom", text="Master bedroom"),
    ]
    assert requirements_from_json(requirements_to_json(reqs)) == reqs


@pytest.mark.parametrize("level", LEVELS)
def test_every_source_plan_meets_every_requirement_derived_from_it(level):
    for seed in range(40):
        canon = canonicalize(make_record(seed)).record.graph
        rng = random.Random(seed)
        reqs = derive_requirements(canon, level, rng, "resplan")
        labeled = apply_labels(canon, reqs)
        unmet = [r.requirement for r in check_requirements(labeled, reqs) if not r.met]
        assert not unmet, (seed, level, unmet)


def test_level_three_asks_for_more_than_level_two():
    canon = canonicalize(make_record(3)).record.graph
    sizes = {
        level: sum(
            len(derive_requirements(canon, level, random.Random(s), "resplan")) for s in range(30)
        )
        for level in (1, 2, 3)
    }
    assert sizes[1] < sizes[2] < sizes[3]
