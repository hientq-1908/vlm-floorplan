"""Property-style tests: random op sequences never break a graph.

There is no hypothesis dependency; each test draws ops from a seeded `random.Random`,
so a failure names its seed and reproduces exactly.
The generator deliberately mixes valid and invalid requests (unknown ids, cross-floor
links, doors between strangers, no-ops), because a model will do the same.
"""

from __future__ import annotations

import random

import pytest

from vlm_floorplan.compact import from_compact, to_compact
from vlm_floorplan.edits import (
    AddExteriorDoor,
    AddLink,
    AddRoom,
    DeleteRoom,
    EditApplied,
    EditHistory,
    EditOp,
    EditRejected,
    MoveRoom,
    RemoveExteriorDoor,
    RemoveLink,
    SetLinkKind,
    SetRoomLabel,
    SetRoomSize,
    SetRoomType,
    apply_op,
    dump_ops,
    format_edit_outcome,
    load_ops,
    replay,
    validate_graph,
)
from vlm_floorplan.schema import GRID_MAX, MAX_ROOM_SIZE, FloorplanGraph, LinkKind, RoomType

from .helpers import single_floor_graph

SEEDS = range(40)
STEPS = 60
LABELS = ["", "Hall", "Guest room", "OH", "  padded  "]


def random_op(rng: random.Random, graph: FloorplanGraph) -> EditOp:
    ids = [room.id for room in graph.rooms()]

    def pick() -> str:
        # Mostly live ids, sometimes a stranger.
        return rng.choice(ids) if ids and rng.random() < 0.9 else f"r{rng.randint(90, 99)}"

    def coord() -> int:
        return rng.randint(0, GRID_MAX)

    def size() -> int:
        return rng.randint(1, MAX_ROOM_SIZE)

    kind = rng.choice(list(LinkKind))
    makers = [
        lambda: AddRoom(type=rng.choice(list(RoomType)), x=coord(), y=coord(), size=size()),
        lambda: AddRoom(
            type=rng.choice(list(RoomType)),
            x=coord(),
            y=coord(),
            size=size(),
            label=rng.choice(LABELS),
        ),
        lambda: SetRoomSize(room_id=pick(), size=size()),
        lambda: MoveRoom(room_id=pick(), x=coord(), y=coord()),
        lambda: DeleteRoom(room_id=pick()),
        lambda: SetRoomType(room_id=pick(), type=rng.choice(list(RoomType))),
        lambda: SetRoomLabel(room_id=pick(), label=rng.choice(LABELS)),
        lambda: AddExteriorDoor(room_id=pick()),
        lambda: RemoveExteriorDoor(room_id=pick()),
        lambda: AddLink(room_a=pick(), room_b=pick(), kind=kind),
        lambda: RemoveLink(room_a=pick(), room_b=pick()),
        lambda: SetLinkKind(room_a=pick(), room_b=pick(), kind=kind),
    ]
    return rng.choice(makers)()


def run(seed: int) -> tuple[EditHistory, list[tuple[FloorplanGraph, EditApplied | EditRejected]]]:
    rng = random.Random(seed)
    history = EditHistory(single_floor_graph())
    trace = []
    for _ in range(STEPS):
        before = history.graph
        trace.append((before, history.apply(random_op(rng, before))))
    return history, trace


@pytest.mark.parametrize("seed", SEEDS)
def test_every_graph_along_a_random_run_is_valid(seed: int) -> None:
    _, trace = run(seed)
    for before, outcome in trace:
        if isinstance(outcome, EditApplied):
            assert validate_graph(outcome.graph).is_valid, (seed, outcome.op)
            assert FloorplanGraph.model_validate(outcome.graph.model_dump()) == outcome.graph
            assert (
                from_compact(
                    to_compact(outcome.graph), outcome.graph.image, outcome.graph.exterior_wall
                )
                == outcome.graph
            )
            assert outcome.graph != before  # applied edits are never no-ops


@pytest.mark.parametrize("seed", SEEDS)
def test_inputs_are_never_mutated_and_refusals_change_nothing(seed: int) -> None:
    _, trace = run(seed)
    for before, outcome in trace:
        snapshot = before.model_dump()
        again = apply_op(before, outcome.op if outcome.op is not None else DeleteRoom(room_id="r0"))
        assert before.model_dump() == snapshot
        assert type(again) is type(outcome)


@pytest.mark.parametrize("seed", SEEDS)
def test_replaying_the_recorded_ops_reproduces_the_graph(seed: int) -> None:
    history, _ = run(seed)
    assert replay(history.base, history.ops) == history.graph
    assert replay(history.base, load_ops(dump_ops(history.ops))) == history.graph


@pytest.mark.parametrize("seed", SEEDS)
def test_undoing_everything_restores_the_base_and_redo_restores_the_end(seed: int) -> None:
    history, _ = run(seed)
    end = history.graph
    steps = len(history.ops)
    for _ in range(steps):
        assert history.undo() is not None
    assert history.graph == history.base
    for _ in range(steps):
        assert history.redo() is not None
    assert history.graph == end


@pytest.mark.parametrize("seed", SEEDS)
def test_links_never_leave_a_floor_and_ids_stay_unique(seed: int) -> None:
    history, _ = run(seed)
    graph = history.graph
    ids = [room.id for room in graph.rooms()]
    assert len(ids) == len(set(ids))
    for floor in graph.floors:
        here = {room.id for room in floor.rooms}
        assert all({link.a, link.b} <= here for link in floor.links)
        assert len({link.pair for link in floor.links}) == len(floor.links)


@pytest.mark.parametrize("seed", SEEDS)
def test_add_link_only_ever_adds_a_wall_between_unlinked_rooms(seed: int) -> None:
    _, trace = run(seed)
    for before, outcome in trace:
        if not (isinstance(outcome, EditApplied) and isinstance(outcome.op, AddLink)):
            continue
        pair = frozenset((outcome.op.room_a, outcome.op.room_b))
        assert outcome.op.kind == LinkKind.WALL
        assert not any(link.pair == pair for floor in before.floors for link in floor.links)


@pytest.mark.parametrize("seed", SEEDS[:10])
def test_the_text_for_the_model_is_always_produced(seed: int) -> None:
    _, trace = run(seed)
    for before, outcome in trace:
        text = format_edit_outcome(outcome, before)
        assert text.startswith("OK: " if isinstance(outcome, EditApplied) else "REJECTED (")


def test_the_random_runs_exercise_both_outcomes_and_every_op_kind() -> None:
    applied_ops, rejected_codes = set(), set()
    for seed in SEEDS:
        for _, outcome in run(seed)[1]:
            if isinstance(outcome, EditApplied):
                applied_ops.add(type(outcome.op).__name__)
            else:
                rejected_codes.add(outcome.error.code)
    assert len(applied_ops) == 11
    assert len(rejected_codes) >= 6
