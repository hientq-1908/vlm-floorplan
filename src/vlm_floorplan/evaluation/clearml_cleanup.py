"""Remove a ClearML project once our job has left it empty.

Captain policy: ClearML is compute only, so after a verified download nothing of ours stays on the
server. Deleting the task removes its artifacts and models, but the project the task was filed under
would stay behind. This deletes that project, and its hidden subprojects (for example the
`.datasets` one the SDK creates), but only when it is empty. A project that still holds anything
(tasks, models) is never touched; it is reported instead, because it is not ours to delete.

The client is the ClearML `APIClient`, passed in so the logic is tested with a fake.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanupReport:
    deleted: list[str] = field(default_factory=list)
    """Names of the projects that were empty and are now deleted."""
    kept: dict[str, str] = field(default_factory=dict)
    """Project name to what it still holds, for projects left alone."""

    def lines(self) -> list[str]:
        lines = [f"deleted empty ClearML project {name!r}" for name in self.deleted]
        lines += [f"kept ClearML project {name!r}: {why}" for name, why in self.kept.items()]
        return lines


def _contents(client: Any, project_id: str) -> str:
    tasks = client.tasks.get_all(project=[project_id], search_hidden=True, only_fields=["id"])
    models = client.models.get_all(project=[project_id], only_fields=["id"])
    parts = []
    if tasks:
        parts.append(f"{len(tasks)} task(s)")
    if models:
        parts.append(f"{len(models)} model(s)")
    return " and ".join(parts)


def remove_empty_projects(client: Any, project_id: str) -> CleanupReport:
    """Delete the project and its subprojects that are empty, deepest first."""
    report = CleanupReport()
    project = client.projects.get_by_id(project=project_id)
    children = client.projects.get_all(
        search_hidden=True, name=f"^{re.escape(project.name)}/", only_fields=["id", "name"]
    )
    family = sorted(children, key=lambda p: len(p.name), reverse=True)
    family.append(project)
    for candidate in family:
        held = _contents(client, candidate.id)
        blocked_by_child = any(name.startswith(candidate.name + "/") for name in report.kept)
        if held or blocked_by_child:
            report.kept[candidate.name] = held or "a subproject that is not empty"
            continue
        client.projects.delete(project=candidate.id, force=False)
        report.deleted.append(candidate.name)
    return report


FAILED_URIS_HEADER = "Failed deleting the following URIs:"


def delete_tolerating_missing_files(
    delete: Callable[[], Any],
    error_type: type[Exception],
    still_exists: Callable[[], bool],
    is_gone: Callable[[str], bool],
) -> list[str]:
    """Run a task delete that may raise because artifact files were already gone (HTTP 404).

    ClearML deletes the task first and then its files, and raises one error listing every file it
    could not delete. A listed file that is already gone (`is_gone`, HTTP 404) is fine. Any other
    listed file is still on the server, so it is returned as a report line instead of being hidden.
    An error that is not about files, or a task that still exists, is raised as before.
    """
    try:
        deleted = delete()
    except error_type as err:
        message = str(err)
        if still_exists() or not message.startswith(FAILED_URIS_HEADER):
            raise
        failed = [line.strip() for line in message.splitlines()[1:] if line.strip()]
        left = [uri for uri in failed if not is_gone(uri)]
        if not left:
            return []
        return [
            "ClearML task deleted, but these files could not be removed, delete them by hand: "
            + ", ".join(left)
        ]
    if not deleted:
        raise RuntimeError("ClearML did not delete the task")
    return []
