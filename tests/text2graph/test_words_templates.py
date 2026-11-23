import random

import pytest
from t2g_fixtures import make_record

from vlm_floorplan.requirements import Adjacent, Ensuites, HasLabel, RoomCount, derive_requirements
from vlm_floorplan.text2graph.frame import canonicalize
from vlm_floorplan.text2graph.templates import choose_kind, render_text
from vlm_floorplan.text2graph.words import slot_check

REQS = [
    RoomCount(type="bedroom", op="eq", n=3),
    RoomCount(type="bathroom", op="ge", n=2),
    Adjacent(a="kitchen", b="living", link="opening"),
]


@pytest.mark.parametrize(
    ("text", "problems"),
    [
        ("3 bedrooms, at least 2 bathrooms and a kitchen open to the living room", []),
        ("three bedrooms, two shower rooms and a lounge open to the kitchen", []),
        ("3 bedrooms and 2 bathrooms, kitchen open to the living room", []),
        ("3 bedrooms, 2 bathrooms, a kitchen open to the living room", []),
        (
            "3 bedrooms, kitchen open to the living room",
            ["missing room type bathroom", "missing number 2"],
        ),
        (
            "3 bedrooms, 2 bathrooms, a dining room, kitchen and living room",
            ["extra room type dining"],
        ),
        (
            "4 bedrooms, 2 bathrooms, kitchen open to the living room",
            ["missing number 3", "extra number 4"],
        ),
    ],
)
def test_slot_check(text, problems):
    assert slot_check(text, REQS) == problems


def test_quoted_labels_do_not_count_as_room_types():
    reqs = [RoomCount(type="bedroom", op="eq", n=2), HasLabel(type="bedroom", text="Study")]
    assert slot_check("We're after 2 bedrooms, one labeled 'Study'.", reqs) == []
    assert slot_check("Two bedrooms, one called 'Master bedroom'", reqs) == []


def test_ensuite_counts_the_bathroom_type():
    assert slot_check("2 ensuite bathrooms", [Ensuites(op="eq", n=2)]) == []


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_template_text_passes_its_own_slot_check(level):
    for seed in range(40):
        graph = canonicalize(make_record(seed)).record.graph
        rng = random.Random(seed * 7 + level)
        reqs = derive_requirements(graph, level, rng, "resplan")
        kind = choose_kind(level, reqs, rng)
        text = render_text(graph, reqs, kind, "resplan", rng)
        assert text and not slot_check(text, reqs), (kind, text, reqs)


def test_full_kinds_list_every_room_and_link():
    graph = canonicalize(make_record(3)).record.graph
    rng = random.Random(0)
    schedule = render_text(graph, [], "full", "resplan", rng)
    natural = render_text(graph, [], "full_natural", "resplan", rng)
    for room in graph.rooms():
        assert f"{room.id} {room.type.value} at ({room.xy[0]}, {room.xy[1]})" in schedule
        assert f"({room.xy[0]}, {room.xy[1]})" in natural
    assert "links:" in schedule and "Connections:" in natural


def test_the_same_seed_gives_the_same_text():
    graph = canonicalize(make_record(8)).record.graph
    texts = []
    for _ in range(2):
        rng = random.Random(5)
        reqs = derive_requirements(graph, 3, rng, "resplan")
        texts.append(render_text(graph, reqs, choose_kind(3, reqs, rng), "resplan", rng))
    assert texts[0] == texts[1]
