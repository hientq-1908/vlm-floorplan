"""FloorplanEditEnv: one repair episode, usable for SFT data, rollouts and GRPO.

    env = FloorplanEditEnv()
    prompt = env.reset(record, task)        # the user message: damaged plan + request
    text = env.step("add_room", {...})      # the tool response, exactly what the model reads
    env.reward()                            # how close the plan is to the real one, minus penalties
    env.solution()                          # the scripted tool calls that repair the plan

The same object serves three jobs:

- SFT data: run `solution()` through `step()` and record the conversation (`episode.py`).
- Rollouts: let a model call `step()` and read `reward()` at the end.
- GRPO: `reset` / `step` / `reward` are the whole interface a trainer needs.

The target is the real `PlanRecord`. The reward compares the final plan to it with
`evaluate_plan`, which matches rooms by position (a point inside the real room's polygon),
so a re-added room does not need the id it had before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlm_floorplan.agent.tasks import Task, ToolCall
from vlm_floorplan.agent.tools import call_tool
from vlm_floorplan.edits.summary import summarize_graph
from vlm_floorplan.evaluation.metrics import PrfCounts, evaluate_plan
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import FloorplanGraph, LinkKind

REJECTION_PENALTY = 0.05
"""Subtracted from the reward for every refused tool call, so a clean run beats a lucky one."""


def plan_score(final: FloorplanGraph | None, target: PlanRecord) -> float:
    """0 to 1: how much of the real plan the final plan has, and nothing else.

    The mean of four scores, each skipped when the target gives it nothing to measure:
    room F1, room-type accuracy among matched rooms, link F1 where a link only counts
    when its kind (door, opening, wall) is right too, and exterior-door F1.
    """
    metrics = evaluate_plan(final, target)
    if not metrics.valid:
        return 0.0
    rooms = PrfCounts(
        tp=metrics.matched_rooms,
        fp=metrics.pred_rooms - metrics.matched_rooms,
        fn=metrics.gt_rooms - metrics.matched_rooms,
    )
    links = PrfCounts()
    for kind in LinkKind:
        links = links + metrics.links.end_to_end[kind.value]
    type_accuracy = metrics.type_correct / metrics.type_labeled if metrics.type_labeled else None
    parts = [
        score
        for score in (rooms.f1, type_accuracy, links.f1, metrics.entrance.f1)
        if score is not None
    ]
    return sum(parts) / len(parts) if parts else 0.0


class FloorplanEditEnv:
    """One episode at a time: `reset` starts it, `step` plays a tool call."""

    def __init__(self, rejection_penalty: float = REJECTION_PENALTY) -> None:
        self.rejection_penalty = rejection_penalty
        self._record: PlanRecord | None = None
        self._task: Task | None = None
        self._initial: FloorplanGraph | None = None
        self._graph: FloorplanGraph | None = None
        self.calls = 0
        self.rejected_calls = 0

    def reset(self, record: PlanRecord, task: Task) -> str:
        """Damage the real plan with `task` and return the user message that asks for the repair."""
        self._record, self._task = record, task
        self._initial = self._graph = task.corrupt(record.graph)
        self.calls = self.rejected_calls = 0
        return f"Current plan:\n{summarize_graph(self._initial)}\n\nRequest: {self.request}"

    @property
    def request(self) -> str:
        return self._task_and_record()[0].request(self._record.graph)  # type: ignore[union-attr]

    @property
    def graph(self) -> FloorplanGraph:
        """The plan as it is now."""
        assert self._graph is not None, "call reset() first"
        return self._graph

    @property
    def initial_graph(self) -> FloorplanGraph:
        assert self._initial is not None, "call reset() first"
        return self._initial

    @property
    def target(self) -> PlanRecord:
        return self._task_and_record()[1]

    @property
    def task(self) -> Task:
        return self._task_and_record()[0]

    def _task_and_record(self) -> tuple[Task, PlanRecord]:
        assert self._task is not None and self._record is not None, "call reset() first"
        return self._task, self._record

    def step(self, tool: str, arguments: Mapping[str, Any]) -> str:
        """Play one tool call and return the tool response text. Bad calls never raise."""
        result = call_tool(self.graph, tool, arguments)
        self._graph = result.graph
        self.calls += 1
        self.rejected_calls += not result.applied
        return result.text

    def reward(self) -> float:
        """`plan_score` of the current plan against the real plan, minus the rejection penalty."""
        return plan_score(self.graph, self.target) - self.rejection_penalty * self.rejected_calls

    def solution(self) -> list[ToolCall]:
        """The scripted repair: the SFT target, and a baseline for rollouts."""
        task, record = self._task_and_record()
        return task.solve(record.graph, self.initial_graph)
