"""Package the repository at HEAD so a ClearML agent can run it without any git access.

The agent has no credentials for the private repository, so the launcher uploads the source
with the task instead of letting the agent clone it. The archive is `git archive HEAD`: exactly
the committed files, no untracked files and no local edits. It is then filtered again, so a file
that was committed by mistake (data, an environment file, a credential) still never leaves the
machine.
"""

from __future__ import annotations

import fnmatch
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

EXCLUDED_DIRS = ("data", "outputs", ".venv", ".git", ".pytest_cache", ".ruff_cache")
"""Top-level directories that never travel: datasets, run outputs, environments, caches."""
EXCLUDED_NAMES = (".env", ".env.*", "*.conf", "*.pem", "*.key", "*.p12", "credentials*", "id_rsa*")
"""File names (in any directory) that look like secrets."""


def is_excluded(path: str) -> bool:
    """Whether a path inside the repository must stay out of the archive."""
    posix = PurePosixPath(path)
    if posix.parts and posix.parts[0] in EXCLUDED_DIRS:
        return True
    return any(fnmatch.fnmatch(posix.name, pattern) for pattern in EXCLUDED_NAMES)


def pack_repo_source(repo_root: Path, archive: Path) -> list[str]:
    """Write `archive` (a .tar.gz) with the committed, non-excluded files; returns their paths."""
    produced = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", "HEAD"],
        capture_output=True,
        check=True,
    )
    import io

    kept: list[str] = []
    with (
        tarfile.open(fileobj=io.BytesIO(produced.stdout)) as source,
        tarfile.open(archive, "w:gz") as target,
    ):
        for member in source:
            if is_excluded(member.name):
                continue
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
            if member.isfile():
                kept.append(member.name)
    return kept
