"""Stage-1 samples of the agreed text-to-graph SFT set replay through the environment unchanged.

`tests/fixtures/text2graph_stage1.jsonl` holds four real records from the 1,000-sample set
(a draw, a draw with a label, a link edit, an add-room chain), reasoning removed.
For each one the plan in the user message is parsed into a graph, every assistant tool call is
played through `call_tool`, and the response must equal the recorded tool message character for
character.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from vlm_floorplan.agent import agent_tool_schemas, call_tool
from vlm_floorplan.agent.episode import SYSTEM_PROMPT
from vlm_floorplan.edits import summarize_graph
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

FIXTURES = Path(__file__).parent.parent / "fixtures"
RECORDS = [
    json.loads(line) for line in (FIXTURES / "text2graph_stage1.jsonl").read_text().splitlines()
]
TOOLS = json.loads((FIXTURES / "text2graph_tools.json").read_text())

_ROOM = re.compile(
    r"^(?P<id>r\d+) (?P<type>\w+)(?: '(?P<label>[^']*)')? at \((?P<x>\d+), (?P<y>\d+)\)"
    r"(?: size (?P<size>\d+))?(?P<entrance> entrance)?$"
)
_POINT = re.compile(r"\((\d+), (\d+)\)")


def parse_plan_state(text: str) -> FloorplanGraph:
    """The plan text the model reads, back into a graph (the inverse of the summary)."""
    rooms: list[Room] = []
    links: list[Link] = []
    wall = None
    for line in text.splitlines()[1:]:  # the first line is the `plan:` header
        if line.startswith("exterior wall: "):
            wall = [(int(x), int(y)) for x, y in _POINT.findall(line)]
        elif line.startswith("links: "):
            for item in line.removeprefix("links: ").split(", "):
                pair, kind = item.split(" ")
                a, b = pair.split("-")
                links.append(Link(a=a, b=b, kind=kind))
        else:
            match = _ROOM.match(line)
            assert match, line
            rooms.append(
                Room(
                    id=match["id"],
                    type=match["type"],
                    label=match["label"],
                    xy=(int(match["x"]), int(match["y"])),
                    size=int(match["size"]),
                    entrance=bool(match["entrance"]),
                )
            )
    floor = Floor(floor=1, rooms=rooms, links=links)
    return FloorplanGraph(floors=[floor], image=ImageSize(w=1000, h=1000), exterior_wall=wall)


def current_plan(user_message: str) -> str:
    return user_message.removeprefix("Current plan:\n").split("\n\nRequest:")[0]


@pytest.mark.parametrize("record", RECORDS, ids=[r["id"] for r in RECORDS])
def test_the_plan_text_in_the_user_message_is_what_the_summary_writes(
    record: dict[str, Any],
) -> None:
    state = current_plan(record["messages"][1]["content"])
    assert summarize_graph(parse_plan_state(state)) == state


@pytest.mark.parametrize("record", RECORDS, ids=[r["id"] for r in RECORDS])
def test_every_tool_response_is_reproduced_by_the_environment(record: dict[str, Any]) -> None:
    messages = record["messages"]
    graph = parse_plan_state(current_plan(messages[1]["content"]))
    calls = 0
    for message, reply in zip(messages[2:], messages[3:], strict=False):
        if not message.get("tool_calls"):
            continue
        function = message["tool_calls"][0]["function"]
        result = call_tool(graph, function["name"], function["arguments"])
        assert result.applied
        assert result.text == reply["content"], (record["id"], function["name"])
        assert result.text.endswith(summarize_graph(result.graph))
        graph = result.graph
        calls += 1
    assert calls >= 1


def test_the_agent_tools_are_the_agreed_tools_plus_set_room_size() -> None:
    ours = {schema["function"]["name"]: schema for schema in agent_tool_schemas()}
    agreed = {schema["function"]["name"]: schema for schema in TOOLS}
    assert {name: ours[name] for name in agreed} == agreed
    assert set(ours) - set(agreed) == {"set_room_size"}


def test_the_system_prompt_is_the_agreed_one() -> None:
    assert RECORDS[0]["messages"][0]["content"] == SYSTEM_PROMPT


def test_no_message_names_a_floor() -> None:
    for record in RECORDS:
        for message in record["messages"]:
            assert not re.search(r"\bfloors?\b", (message.get("content") or "").lower())
    for schema in agent_tool_schemas():
        assert "floor" not in schema["function"]["parameters"]["properties"]
    new_plan = next(s for s in agent_tool_schemas() if s["function"]["name"] == "new_plan")
    assert "floor" not in json.dumps(new_plan)
