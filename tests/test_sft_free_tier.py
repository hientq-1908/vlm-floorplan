"""The input directory for the free-tier notebook: the committed source and the training file."""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from vlm_floorplan.training.free_tier import (
    METADATA_NAME,
    SOURCE_NAME,
    TRAIN_NAME,
    build_inputs,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
        "src/vlm_floorplan/training/train.py": "x = 1\n",
        "data/sft/train.jsonl": "committed by mistake",
        ".env": "TOKEN=secret",
        "notes/id_rsa": "private key",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
        ["add", "-f", "."],
        ["commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.fixture
def train_file(tmp_path: Path) -> Path:
    path = tmp_path / "samples.jsonl"
    path.write_text('{"id": "A"}\n')
    return path


def test_inputs_hold_the_source_and_the_training_file_and_metadata(
    repo: Path,
    train_file: Path,
    tmp_path: Path,
) -> None:
    out = build_inputs(repo, train_file, tmp_path / "out", "me/vlmfp-sft-inputs")
    assert (out / TRAIN_NAME).read_text() == '{"id": "A"}\n'
    with tarfile.open(out / SOURCE_NAME) as tar:
        names = set(tar.getnames())
    assert {"pyproject.toml", "uv.lock", "src/vlm_floorplan/training/train.py"} <= names
    assert not {"data/sft/train.jsonl", ".env", "notes/id_rsa"} & names
    metadata = json.loads((out / METADATA_NAME).read_text())
    assert metadata["id"] == "me/vlmfp-sft-inputs"


def test_no_metadata_without_a_dataset_id(repo: Path, train_file: Path, tmp_path: Path) -> None:
    out = build_inputs(repo, train_file, tmp_path / "out", None)
    assert not (out / METADATA_NAME).exists()


def test_a_non_empty_output_directory_is_refused(
    repo: Path,
    train_file: Path,
    tmp_path: Path,
) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "precious").write_text("keep")
    with pytest.raises(FileExistsError, match="already has files"):
        build_inputs(repo, train_file, tmp_path / "out", None)


def test_a_missing_training_file_and_a_bad_dataset_id_are_refused(
    repo: Path,
    train_file: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        build_inputs(repo, tmp_path / "missing.jsonl", tmp_path / "a", None)
    with pytest.raises(ValueError, match="owner/slug"):
        build_inputs(repo, train_file, tmp_path / "b", "just-a-slug")
