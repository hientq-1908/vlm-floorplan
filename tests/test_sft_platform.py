"""The notebook helpers: GPU choice, finding the inputs, carrying a run into a new session."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from vlm_floorplan.evaluation.launcher import REPO_ROOT

spec = importlib.util.spec_from_file_location("sft_platform", REPO_ROOT / "scripts/sft_platform.py")
assert spec and spec.loader
platform = importlib.util.module_from_spec(spec)
sys.modules["sft_platform"] = platform
spec.loader.exec_module(platform)


@pytest.mark.parametrize(
    ("name", "mib", "capability"),
    [("Tesla T4", 15360, (7, 5)), ("Tesla P100-PCIE-16GB", 16384, (6, 0))],
)
def test_16_gb_cards_without_bf16_get_qlora(
    name: str, mib: int, capability: tuple[int, int]
) -> None:
    plan = platform.plan_for_gpu(name, mib, capability)
    assert plan.qlora
    assert "QLoRA" in plan.reason and name in plan.reason


def test_a_24_gb_ampere_card_keeps_the_default_recipe() -> None:
    plan = platform.plan_for_gpu("NVIDIA A10G", 23028, (8, 6))
    assert not plan.qlora


def test_a_16_gb_ampere_card_is_too_small_for_bf16_lora() -> None:
    assert platform.plan_for_gpu("RTX A4000", 16376, (8, 6)).qlora


@pytest.mark.parametrize("mib", [4096, 8192, 12288])
def test_a_gpu_under_16_gb_is_refused_with_its_size(mib: int) -> None:
    with pytest.raises(platform.UnsupportedGpuError, match=r"needs a 16 GB GPU"):
        platform.plan_for_gpu("Small GPU", mib, (7, 5))


def test_find_file_returns_the_one_match_anywhere_below_the_roots(tmp_path: Path) -> None:
    target = tmp_path / "input" / "datasets" / "me" / "inputs" / "train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert platform.find_file([tmp_path / "missing", tmp_path / "input"], "train.jsonl") == target


def test_find_file_refuses_zero_or_several_matches(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "train.jsonl").write_text("{}")
    with pytest.raises(FileNotFoundError, match="found 2"):
        platform.find_file([tmp_path], "train.jsonl")
    with pytest.raises(FileNotFoundError, match="found 0"):
        platform.find_file([tmp_path], "nothing.txt")


def make_earlier_run(root: Path) -> Path:
    run = root / "my-notebook" / "sft-stage1"
    (run / "checkpoint-50").mkdir(parents=True)
    (run / "run_config.json").write_text("{}")
    (run / "checkpoint-50" / "optimizer.pt").write_bytes(b"state")
    return run


def test_stage_resume_copies_the_attached_run_into_the_working_directory(tmp_path: Path) -> None:
    make_earlier_run(tmp_path / "input")
    staged = platform.stage_resume(
        [tmp_path / "input"], "sft-stage1", tmp_path / "work" / "sft-stage1"
    )
    assert staged == tmp_path / "work" / "sft-stage1"
    assert (staged / "checkpoint-50" / "optimizer.pt").read_bytes() == b"state"


def test_stage_resume_without_an_attached_run_does_nothing(tmp_path: Path) -> None:
    assert platform.stage_resume([tmp_path], "sft-stage1", tmp_path / "work" / "sft-stage1") is None


def test_stage_resume_never_overwrites_a_run_in_the_working_directory(tmp_path: Path) -> None:
    make_earlier_run(tmp_path / "input")
    (tmp_path / "work" / "sft-stage1").mkdir(parents=True)
    assert (
        platform.stage_resume([tmp_path / "input"], "sft-stage1", tmp_path / "work" / "sft-stage1")
        is None
    )
    assert not list((tmp_path / "work" / "sft-stage1").iterdir())


def test_stage_resume_refuses_two_candidate_runs(tmp_path: Path) -> None:
    make_earlier_run(tmp_path / "input")
    make_earlier_run(tmp_path / "other")
    with pytest.raises(FileExistsError, match="several earlier runs"):
        platform.stage_resume(
            [tmp_path / "input", tmp_path / "other"], "sft-stage1", tmp_path / "work" / "sft-stage1"
        )
