from __future__ import annotations

import pytest

from vlm_floorplan.agent import (
    TASK_KINDS,
    CloseDoorTask,
    DeleteRoomTask,
    FloorplanEditEnv,
    Task,
    candidate_tasks,
)
from vlm_floorplan.agent.env import REJECTION_PENALTY, plan_score
from vlm_floorplan.agent.sample_plan import sample_record
from vlm_floorplan.schema import FloorplanGraph

RECORD = sample_record()
ALL_TASKS = [
    pytest.param(task, id=f"{task.kind}-{index}")
    for kind in TASK_KINDS
    for index, task in enumerate(candidate_tasks(RECORD, kind))
]


def solve(env: FloorplanEditEnv) -> None:
    for call in env.solution():
        env.step(call.name, call.arguments)


def test_every_task_kind_has_candidates_on_the_sample_plan() -> None:
    assert {param.values[0].kind for param in ALL_TASKS} == set(TASK_KINDS)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_the_scripted_solution_reaches_full_reward_without_a_rejection(task: Task) -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, task)
    solve(env)
    assert env.rejected_calls == 0
    assert env.reward() == pytest.approx(1.0)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_the_damaged_plan_scores_below_the_scripted_solution(task: Task) -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, task)
    before = env.reward()
    solve(env)
    assert before < env.reward()


def test_reset_returns_the_damaged_plan_and_the_request() -> None:
    env = FloorplanEditEnv()
    prompt = env.reset(RECORD, DeleteRoomTask("r1"))
    assert prompt.startswith(
        "Current plan:\nplan: 5 rooms, 6 links\nexterior wall: (0, 0), (1000, 0)"
    )
    assert "r1 bathroom" not in prompt.split("Request:")[0]
    assert "Request: Add a bathroom at (500, 200)." in prompt
    assert "a door to r3 (living)" in prompt
    assert "a shared wall with r0 (bedroom)" in prompt


def test_a_request_for_an_entry_uses_an() -> None:
    env = FloorplanEditEnv()
    prompt = env.reset(RECORD, DeleteRoomTask("r4"))
    assert "Request: Add an entry at (" in prompt
    assert env.solution()[0].reasoning.startswith("The user wants an entry at (")


def test_the_target_stays_the_real_plan_while_the_damaged_plan_changes() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r1"))
    solve(env)
    assert env.target is RECORD
    assert len(env.initial_graph.rooms()) == 5
    assert len(env.graph.rooms()) == 6


def test_a_deleted_entrance_room_gets_its_exterior_door_back_with_its_own_call() -> None:
    env = FloorplanEditEnv()
    prompt = env.reset(RECORD, DeleteRoomTask("r4"))
    assert "a door to the outside" in prompt
    assert [call.name for call in env.solution()][-1] == "add_exterior_door"
    solve(env)
    assert env.reward() == pytest.approx(1.0)


def test_skipping_the_exterior_door_call_scores_below_full_reward() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r4"))
    for call in env.solution():
        if call.name != "add_exterior_door":
            env.step(call.name, call.arguments)
    assert env.rejected_calls == 0
    assert env.reward() < 1.0


def test_every_partial_repair_of_a_deleted_entrance_room_beats_the_damaged_plan() -> None:
    # Oracle: the docs promise every missing room, link or exterior door counts against
    # the plan, so a plan with the room back is closer to the target than one without it.
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r4"))
    damaged = env.reward()
    partial = []
    for call in env.solution():
        env.step(call.name, call.arguments)
        partial.append(env.reward())
    assert env.rejected_calls == 0
    assert all(reward > damaged for reward in partial)
    assert partial[-1] == pytest.approx(1.0)


def without_entrances(graph: FloorplanGraph, *, keep_links: bool = True) -> FloorplanGraph:
    return graph.model_copy(
        update={
            "floors": [
                floor.model_copy(
                    update={
                        "links": floor.links if keep_links else [],
                        "rooms": [
                            room.model_copy(update={"entrance": False}) for room in floor.rooms
                        ],
                    }
                )
                for floor in graph.floors
            ]
        }
    )


def test_a_plan_with_the_right_rooms_but_no_links_or_entrance_scores_below_full() -> None:
    bare = without_entrances(RECORD.graph, keep_links=False)
    assert plan_score(bare, RECORD) == pytest.approx(0.5)


def test_a_spurious_exterior_door_on_an_entrance_free_target_costs_reward() -> None:
    target = RECORD.model_copy(update={"graph": without_entrances(RECORD.graph)})
    env = FloorplanEditEnv()
    env.reset(target, CloseDoorTask("r0", "r3", target.graph.floors[0].links[2].kind))
    solve(env)
    assert env.reward() == pytest.approx(1.0)
    env.step("add_exterior_door", {"room_id": "r0"})
    assert env.rejected_calls == 0
    assert env.reward() == pytest.approx(0.75)


def test_a_refused_call_changes_nothing_and_costs_the_penalty() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, CloseDoorTask("r0", "r3", RECORD.graph.floors[0].links[2].kind))
    before = env.reward()
    text = env.step("add_link", {"room_a": "r0", "room_b": "r3", "kind": "door"})
    assert text.startswith("REJECTED (duplicate_link):")
    assert env.rejected_calls == 1
    assert env.reward() == pytest.approx(before - REJECTION_PENALTY)


def test_an_unknown_tool_or_bad_arguments_are_refused_not_raised() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r1"))
    assert env.step("fly", {}).startswith("REJECTED (unknown_tool):")
    assert env.step("move_room", {"room_id": "r0"}).startswith("REJECTED (invalid_arguments):")
    assert env.rejected_calls == 2


def test_reset_clears_the_counters() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r1"))
    env.step("fly", {})
    env.reset(RECORD, DeleteRoomTask("r1"))
    assert (env.calls, env.rejected_calls) == (0, 0)


def test_a_wrong_room_type_scores_below_the_right_one() -> None:
    env = FloorplanEditEnv()
    env.reset(RECORD, DeleteRoomTask("r1"))
    for call in env.solution():
        arguments = (
            {**call.arguments, "type": "kitchen"} if call.name == "add_room" else call.arguments
        )
        env.step(call.name, arguments)
    assert env.rejected_calls == 0
    assert env.reward() < 1.0
