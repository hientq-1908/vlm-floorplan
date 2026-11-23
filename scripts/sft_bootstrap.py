"""What the ClearML agent runs first for an SFT task: unpack the source, build the env, run the job.

The launcher creates the task from this single file (no git repository), because the agent has no
credentials to clone the private repo. The repository source travels as the task artifact
`repo_source` (`vlm_floorplan/training/source_pack.py`). This script then:

1. unpacks that archive,
2. installs `uv` and builds the project environment from `uv.lock` with Python 3.12 (the agent's
   docker image only ships an older Python), so the environment is the one the tests ran in,
3. runs `scripts/sft_job.py` inside it, which does the training and uploads the result.

It must stay standalone and run on the agent's old Python: standard library only, plus `clearml`
(already installed by the agent). Its exit code is the job's, so a failure marks the task failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SOURCE_ARTIFACT = "repo_source"
PYTHON_VERSION = "3.12"


def unpack_source(archive: Path, dest: Path) -> Path:
    """Unpack the source archive into `dest`.

    Refuses absolute paths, `..` parts and anything but regular files and directories. The check
    is done here because `extractall(filter=...)` is missing on some of the agent's old Pythons.
    """
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if member.name.startswith(("/", "\\")) or ".." in parts:
                raise tarfile.TarError(f"refusing path outside the destination: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise tarfile.TarError(f"refusing non-regular member: {member.name}")
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, members=members, **extra)
    return dest


def uv_commands(source: Path, env_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """The environment build command, the job command, and the environment variables for both."""
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(env_dir)}
    flags = ["--frozen", "--no-dev", "--group", "ml", "--python", PYTHON_VERSION]
    build = ["uv", "sync", *flags]
    run = ["uv", "run", *flags, "python", str(source / "scripts" / "sft_job.py")]
    return build, run, env


def main() -> int:
    from clearml import Task

    task = Task.get_task(task_id=os.environ["CLEARML_TASK_ID"])
    archive = Path(task.artifacts[SOURCE_ARTIFACT].get_local_copy(extract_archive=False))
    work = Path(tempfile.mkdtemp(prefix="sft-src-"))
    source = unpack_source(archive, work / "source")

    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    build, run, env = uv_commands(source, work / "venv")
    subprocess.run(build, cwd=source, env=env, check=True)
    return subprocess.run(run, cwd=source, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
