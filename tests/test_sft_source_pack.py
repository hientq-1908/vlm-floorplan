"""The repository source that travels to the agent, and the bootstrap that unpacks it.

The agent cannot clone the private repository, so the launcher ships `git archive HEAD`. What
must hold: the package and the lockfile are in it, and data, secrets and uncommitted files are not.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from vlm_floorplan.evaluation.launcher import REPO_ROOT
from vlm_floorplan.training.job import SOURCE_ARTIFACT
from vlm_floorplan.training.source_pack import is_excluded, pack_repo_source

COMMITTED = {
    "pyproject.toml": "[project]\n",
    "uv.lock": "version = 1\n",
    "src/vlm_floorplan/__init__.py": "",
    "src/vlm_floorplan/training/train.py": "x = 1\n",
    "scripts/sft_job.py": "",
    "configs/sft_stage1.toml": "[model]\n",
    # committed by mistake: still must not travel
    "data/sft/train.jsonl": '{"id": "A"}\n',
    "outputs/run/adapter.bin": "weights",
    ".env": "TOKEN=secret",
    "deploy/clearml.conf": "api { secret_key: abc }",
    "notes/id_rsa": "private key",
}


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    for name, text in COMMITTED.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "add", "-f", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def names(archive: Path) -> set[str]:
    with tarfile.open(archive) as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


def test_archive_holds_the_package_the_lockfile_and_the_config(repo: Path, tmp_path: Path) -> None:
    archive = tmp_path / "src.tar.gz"
    kept = pack_repo_source(repo, archive)
    assert {
        "pyproject.toml",
        "uv.lock",
        "src/vlm_floorplan/training/train.py",
        "scripts/sft_job.py",
        "configs/sft_stage1.toml",
    } <= names(archive)
    assert set(kept) == names(archive)


def test_archive_leaves_out_data_outputs_and_secrets_even_when_committed(
    repo: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "src.tar.gz"
    pack_repo_source(repo, archive)
    present = names(archive)
    for forbidden in (
        "data/sft/train.jsonl",
        "outputs/run/adapter.bin",
        ".env",
        "deploy/clearml.conf",
        "notes/id_rsa",
    ):
        assert forbidden not in present


def test_archive_is_head_not_the_working_tree(repo: Path, tmp_path: Path) -> None:
    (repo / "src/vlm_floorplan/training/train.py").write_text("x = 2  # uncommitted edit\n")
    (repo / "untracked.py").write_text("print('new')\n")
    archive = tmp_path / "src.tar.gz"
    pack_repo_source(repo, archive)
    with tarfile.open(archive) as tar:
        member = tar.extractfile("src/vlm_floorplan/training/train.py")
        assert member is not None and member.read() == b"x = 1\n"
    assert "untracked.py" not in names(archive)


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        ("data/x.jsonl", True),
        ("src/data/loader.py", False),
        (".env.local", True),
        ("a/b/secrets.pem", True),
        ("src/vlm_floorplan/training/config.py", False),
        ("uv.lock", False),
    ],
)
def test_exclusion_rules(path: str, excluded: bool) -> None:
    assert is_excluded(path) is excluded


# --- the bootstrap the agent runs ----------------------------------------------------------------


@pytest.fixture(scope="module")
def bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sft_bootstrap", REPO_ROOT / "scripts" / "sft_bootstrap.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["sft_bootstrap"] = module
    spec.loader.exec_module(module)
    return module


def test_bootstrap_unpacks_what_the_launcher_packs(
    bootstrap: ModuleType, repo: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "src.tar.gz"
    pack_repo_source(repo, archive)
    source = bootstrap.unpack_source(archive, tmp_path / "unpacked")
    assert (source / "uv.lock").is_file()
    assert (source / "src/vlm_floorplan/training/train.py").read_text() == "x = 1\n"
    assert not (source / "data").exists()


@pytest.fixture(params=["with_data_filter", "old_python_without_filter"])
def tar_filter_support(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "old_python_without_filter":
        monkeypatch.delattr(tarfile, "data_filter")


def test_bootstrap_unpacks_on_old_python_without_the_tar_filter(
    bootstrap: ModuleType, repo: Path, tmp_path: Path, tar_filter_support: None
) -> None:
    archive = tmp_path / "src.tar.gz"
    pack_repo_source(repo, archive)
    source = bootstrap.unpack_source(archive, tmp_path / "unpacked")
    assert (source / "src/vlm_floorplan/training/train.py").read_text() == "x = 1\n"


def test_bootstrap_refuses_a_path_that_escapes_the_destination(
    bootstrap: ModuleType, tmp_path: Path, tar_filter_support: None
) -> None:
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")
    with pytest.raises(tarfile.TarError):
        bootstrap.unpack_source(evil, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_bootstrap_refuses_a_link_that_points_outside_the_destination(
    bootstrap: ModuleType, tmp_path: Path, tar_filter_support: None
) -> None:
    evil = tmp_path / "evil.tar.gz"
    link = tarfile.TarInfo("src/outside")
    link.type = tarfile.SYMTYPE
    link.linkname = str(tmp_path)
    with tarfile.open(evil, "w:gz") as tar:
        tar.addfile(link)
    with pytest.raises(tarfile.TarError):
        bootstrap.unpack_source(evil, tmp_path / "dest")
    assert not (tmp_path / "dest" / "src" / "outside").exists()


def test_bootstrap_builds_the_env_from_the_lockfile_with_python_3_12(
    bootstrap: ModuleType, tmp_path: Path
) -> None:
    build, run, env = bootstrap.uv_commands(tmp_path / "src", tmp_path / "venv")
    uv = [sys.executable, "-m", "uv"]  # the uv that pip just installed, not whatever is on PATH
    assert build[:4] == [*uv, "sync"] and run[:4] == [*uv, "run"]
    for command in (build, run):
        assert "--frozen" in command  # never re-resolve: the lockfile is the environment
        assert command[command.index("--python") + 1] == "3.12"
        assert command[command.index("--group") + 1] == "ml"
    assert run[-1] == str(tmp_path / "src" / "scripts" / "sft_job.py")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "venv")


def test_bootstrap_and_job_agree_on_the_source_artifact_name(bootstrap: ModuleType) -> None:
    assert bootstrap.SOURCE_ARTIFACT == SOURCE_ARTIFACT


def test_bootstrap_keeps_the_agents_python_settings_out_of_the_uv_env(
    bootstrap: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live failure: the agent venv's PYTHONPATH made a 3.12 build import 3.10 setuptools."""
    monkeypatch.setenv("PYTHONPATH", "/root/.clearml/venvs-builds/3.10/lib/python3.10")
    monkeypatch.setenv("VIRTUAL_ENV", "/root/.clearml/venvs-builds/3.10")
    monkeypatch.setenv("CLEARML_TASK_ID", "abc")
    _, _, env = bootstrap.uv_commands(tmp_path / "src", tmp_path / "venv")
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert env["CLEARML_TASK_ID"] == "abc"  # the job still needs to find its task


def test_compat_libraries_go_first_on_the_library_path(bootstrap: ModuleType) -> None:
    env = {"LD_LIBRARY_PATH": "/usr/lib/host", "OTHER": "1"}
    result = bootstrap.with_compat_libraries(env, ["/usr/local/cuda-13.0/compat"])
    assert result["LD_LIBRARY_PATH"] == "/usr/local/cuda-13.0/compat:/usr/lib/host"
    assert result["OTHER"] == "1" and env["LD_LIBRARY_PATH"] == "/usr/lib/host"


def test_compat_libraries_work_without_an_existing_library_path(bootstrap: ModuleType) -> None:
    result = bootstrap.with_compat_libraries({}, ["/a/compat", "/b/compat"])
    assert result["LD_LIBRARY_PATH"] == "/a/compat:/b/compat"


def test_gpu_env_stays_plain_when_the_gpu_works_without_compat_libraries(
    bootstrap: ModuleType,
) -> None:
    env = {"LD_LIBRARY_PATH": "/usr/lib/host"}
    tried: list[dict[str, str]] = []
    chosen = bootstrap.choose_gpu_env(
        env, ["/usr/local/cuda-13.0/compat"], lambda e: tried.append(e) or True
    )
    assert chosen == env
    assert tried == [env]


def test_gpu_env_falls_back_to_the_newest_compat_libraries(bootstrap: ModuleType) -> None:
    env = {"LD_LIBRARY_PATH": "/usr/lib/host"}
    compat = [
        "/usr/local/cuda-12.10/compat",
        "/usr/local/cuda-13.0/compat",
        "/usr/local/cuda-12.4/compat",
    ]
    chosen = bootstrap.choose_gpu_env(env, compat, lambda e: e != env)
    assert chosen["LD_LIBRARY_PATH"] == (
        "/usr/local/cuda-13.0/compat:/usr/local/cuda-12.10/compat:"
        "/usr/local/cuda-12.4/compat:/usr/lib/host"
    )


@pytest.mark.parametrize("compat", [[], ["/usr/local/cuda-13.0/compat"]])
def test_gpu_env_is_none_when_no_environment_reaches_the_gpu(
    bootstrap: ModuleType, compat: list[str]
) -> None:
    assert bootstrap.choose_gpu_env({}, compat, lambda e: False) is None


def test_gpu_check_fails_fast_with_a_clear_message_when_torch_sees_no_gpu(
    bootstrap: ModuleType,
) -> None:
    pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, "-c", bootstrap.GPU_CHECK],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode != 0
    assert "GPU CHECK FAILED" in result.stderr
