from __future__ import annotations

import json
from typing import Any

import pytest

from vlm_floorplan.agent import (
    TASK_KINDS,
    DeleteRoomTask,
    DrawPlanTask,
    build_episode,
)
from vlm_floorplan.agent.episode import sample_episodes
from vlm_floorplan.agent.sample_plan import sample_record

RECORD = sample_record()


def roles(episode: dict[str, Any]) -> list[str]:
    return [message["role"] for message in episode["messages"]]


def test_a_delete_room_episode_alternates_one_call_and_one_response() -> None:
    episode = build_episode(RECORD, DeleteRoomTask("r1"))
    middle = roles(episode)[2:-1]
    assert roles(episode)[:2] == ["system", "user"]
    assert middle == ["assistant", "tool"] * (len(middle) // 2)
    assert roles(episode)[-1] == "assistant"
    assert episode["reward"] == pytest.approx(1.0)


def test_every_tool_turn_has_reasoning_and_exactly_one_call() -> None:
    episode = build_episode(RECORD, DeleteRoomTask("r1"))
    calling = [m for m in episode["messages"] if m.get("tool_calls")]
    assert len(calling) == 5  # add_room, three add_link, one set_link_kind for the door
    for message in calling:
        assert message["content"] == ""
        assert message["reasoning_content"]
        assert len(message["tool_calls"]) == 1
        assert isinstance(message["tool_calls"][0]["function"]["arguments"], dict)


def test_tool_messages_carry_the_real_environment_response() -> None:
    episode = build_episode(RECORD, DeleteRoomTask("r1"))
    tools = [m for m in episode["messages"] if m["role"] == "tool"]
    assert tools[0]["name"] == "add_room"
    assert tools[0]["content"].startswith("OK: added bathroom room r6 at (500, 200)")
    assert all(m["content"].startswith("OK:") for m in tools)


def test_the_final_turn_is_a_reply_without_a_call() -> None:
    final = build_episode(RECORD, DeleteRoomTask("r1"))["messages"][-1]
    assert "tool_calls" not in final
    assert final["content"]
    assert final["reasoning_content"]


def test_the_draw_plan_episode_is_one_new_plan_call() -> None:
    episode = build_episode(RECORD, DrawPlanTask())
    calls = [m["tool_calls"][0]["function"] for m in episode["messages"] if m.get("tool_calls")]
    assert [call["name"] for call in calls] == ["new_plan"]
    assert json.loads(calls[0]["arguments"]["plan"])["floors"]
    assert episode["reward"] == pytest.approx(1.0)


def test_the_record_carries_all_eleven_tools_and_is_json() -> None:
    episode = build_episode(RECORD, DeleteRoomTask("r1"))
    assert len(episode["tools"]) == 11
    assert json.loads(json.dumps(episode)) == episode


def test_sample_episodes_covers_every_task_kind_and_is_reproducible() -> None:
    first = sample_episodes([RECORD], per_kind=2, seed=3)
    assert {episode["task"] for episode in first} == set(TASK_KINDS)
    assert first == sample_episodes([RECORD], per_kind=2, seed=3)
    assert all(episode["reward"] == pytest.approx(1.0) for episode in first)
