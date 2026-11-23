"""Removing the project a job leaves empty on ClearML, tested with a fake client."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_eval_collect import FakeTask, good_task

from vlm_floorplan.evaluation.clearml_cleanup import remove_empty_projects
from vlm_floorplan.evaluation.collect import collect_and_delete


@dataclass
class FakeServer:
    """Projects (id to name) with the tasks and models filed under them."""

    projects: dict[str, str]
    tasks: dict[str, list[str]] = field(default_factory=dict)
    models: dict[str, list[str]] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)

    @property
    def client(self) -> Any:
        server = self

        class Projects:
            def get_by_id(self, project: str) -> Any:
                return SimpleNamespace(id=project, name=server.projects[project])

            def get_all(self, name: str, **_: Any) -> list[Any]:
                return [
                    SimpleNamespace(id=i, name=n)
                    for i, n in server.projects.items()
                    if re.search(name, n)
                ]

            def delete(self, project: str, force: bool) -> None:
                assert force is False, "never force-delete a project"
                server.deleted.append(server.projects.pop(project))

        class Items:
            def __init__(self, table: dict[str, list[str]]) -> None:
                self.table = table

            def get_all(self, project: list[str], **_: Any) -> list[Any]:
                return [SimpleNamespace(id=i) for i in self.table.get(project[0], [])]

        return SimpleNamespace(
            projects=Projects(), tasks=Items(server.tasks), models=Items(server.models)
        )


def test_empty_project_is_deleted() -> None:
    server = FakeServer({"p1": "vlm-floorplan"})
    report = remove_empty_projects(server.client, "p1")
    assert server.deleted == ["vlm-floorplan"] and report.deleted == ["vlm-floorplan"]


def test_hidden_subprojects_go_first_then_the_project() -> None:
    server = FakeServer(
        {"p1": "vlm-floorplan", "p2": "vlm-floorplan/.datasets", "p3": "other-project"}
    )
    remove_empty_projects(server.client, "p1")
    assert server.deleted == ["vlm-floorplan/.datasets", "vlm-floorplan"]
    assert server.projects == {"p3": "other-project"}


def test_project_holding_someone_elses_task_is_left_alone_and_reported() -> None:
    server = FakeServer({"p1": "vlm-floorplan"}, tasks={"p1": ["t-other"]})
    report = remove_empty_projects(server.client, "p1")
    assert server.deleted == []
    assert "1 task(s)" in report.kept["vlm-floorplan"]


def test_project_holding_a_model_is_left_alone() -> None:
    server = FakeServer({"p1": "vlm-floorplan"}, models={"p1": ["m1", "m2"]})
    remove_empty_projects(server.client, "p1")
    assert server.deleted == []


def test_nonempty_subproject_keeps_its_parent_too() -> None:
    server = FakeServer(
        {"p1": "vlm-floorplan", "p2": "vlm-floorplan/.datasets"}, tasks={"p2": ["dataset-task"]}
    )
    report = remove_empty_projects(server.client, "p1")
    assert server.deleted == []
    assert set(report.kept) == {"vlm-floorplan", "vlm-floorplan/.datasets"}


def test_similarly_named_project_is_not_a_subproject() -> None:
    server = FakeServer({"p1": "vlm-floorplan", "p2": "vlm-floorplan-extra"})
    remove_empty_projects(server.client, "p1")
    assert server.deleted == ["vlm-floorplan"]


# --- the collector calls it only after the verified delete -------------------------------------


class ProjectTask(FakeTask):
    def __init__(self, *args: Any, fail: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.fail = fail

    def delete(self) -> None:
        self.events.append("delete task")
        super().delete()

    def remove_empty_project(self) -> list[str]:
        self.events.append("remove project")
        if self.fail:
            raise RuntimeError("server unreachable")
        return ["deleted empty ClearML project 'vlm-floorplan'"]


def test_collector_removes_the_project_after_deleting_the_task(tmp_path: Path) -> None:
    base = good_task(tmp_path)
    task = ProjectTask(base._artifacts)
    report = collect_and_delete(task, tmp_path / "out")
    assert task.events == ["delete task", "remove project"]
    assert "deleted empty ClearML project 'vlm-floorplan'" in report.checks


def test_project_is_never_touched_when_verification_fails(tmp_path: Path) -> None:
    task = ProjectTask({})  # completed task without its result archive
    with pytest.raises(Exception, match="no 'baseline_run' artifact"):
        collect_and_delete(task, tmp_path / "out")
    assert task.events == []


def test_a_failing_project_cleanup_is_reported_not_raised(tmp_path: Path) -> None:
    task = ProjectTask(good_task(tmp_path)._artifacts, fail=True)
    report = collect_and_delete(task, tmp_path / "out")
    assert task.deleted
    assert any("project cleanup failed" in c for c in report.checks)
