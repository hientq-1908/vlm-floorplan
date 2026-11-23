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

import glob
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

SOURCE_ARTIFACT = "repo_source"
PYTHON_VERSION = "3.12"
AGENT_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
CUDA_COMPAT_PATTERN = "/usr/local/cuda*/compat"
"""CUDA forward-compatibility libraries shipped in the agent's docker image."""
GPU_CHECK = """
import sys
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("GPU CHECK FAILED: torch cannot use the GPU (driver too old for this torch build?)")
print("gpu:", torch.cuda.get_device_name(0))
x = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
if float((x @ x)[0, 0]) != 64.0:
    sys.exit("GPU CHECK FAILED: a bf16 matmul on the GPU gave a wrong result")
print("bf16 matmul on the GPU ok")
"""


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


def newest_first(compat_dirs: list[str]) -> list[str]:
    """The compat directories ordered by their CUDA version, newest first."""

    def version(path: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", Path(path).parent.name))

    return sorted(compat_dirs, key=version, reverse=True)


def with_compat_libraries(env: dict[str, str], compat_dirs: list[str]) -> dict[str, str]:
    """`env` with the CUDA compat directories first on LD_LIBRARY_PATH."""
    existing = [env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []
    return {**env, "LD_LIBRARY_PATH": ":".join([*compat_dirs, *existing])}


def choose_gpu_env(
    env: dict[str, str], compat_dirs: list[str], gpu_works: Callable[[dict[str, str]], bool]
) -> dict[str, str] | None:
    """The first environment in which `gpu_works`: `env` as is, then with the compat libraries.

    The agent's driver (CUDA 12.8) is older than the CUDA 13 wheel of torch in `uv.lock`. On a
    data-center GPU such as the A10G, the newer user-space driver libraries in the image's compat
    directory let it run anyway. They break a GeForce card or a newer host driver, so they are
    only tried when the plain environment fails. None when neither works.
    """
    candidates = [env]
    if compat_dirs:
        candidates.append(with_compat_libraries(env, newest_first(compat_dirs)))
    return next((candidate for candidate in candidates if gpu_works(candidate)), None)


def uv_commands(source: Path, env_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """The environment build command, the job command, and the environment variables for both."""
    # The agent's own interpreter settings must not leak into the Python 3.12 builds: its
    # PYTHONPATH made a 3.12 build load the agent venv's 3.10 setuptools and crash.
    env = {k: v for k, v in os.environ.items() if k not in AGENT_PYTHON_VARS}
    env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
    flags = ["--frozen", "--no-dev", "--group", "ml", "--python", PYTHON_VERSION]
    uv = [sys.executable, "-m", "uv"]
    build = [*uv, "sync", *flags]
    run = [*uv, "run", *flags, "python", str(source / "scripts" / "sft_job.py")]
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

    # Fail in seconds with a clear message instead of after the model download.
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found on PATH", flush=True)
    compat_dirs = glob.glob(CUDA_COMPAT_PATTERN)
    print("CUDA compat directories:", compat_dirs or "none found", flush=True)
    check = [*run[: run.index("python")], "python", "-c", GPU_CHECK]

    def gpu_works(candidate: dict[str, str]) -> bool:
        print("GPU check, LD_LIBRARY_PATH:", candidate.get("LD_LIBRARY_PATH", ""), flush=True)
        return subprocess.run(check, cwd=source, env=candidate).returncode == 0

    gpu_env = choose_gpu_env(env, compat_dirs, gpu_works)
    if gpu_env is None:
        return 3
    return subprocess.run(run, cwd=source, env=gpu_env).returncode


if __name__ == "__main__":
    sys.exit(main())
