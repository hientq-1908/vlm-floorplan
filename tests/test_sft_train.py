"""The whole SFT run on a tiny random Qwen3.5, on the CPU.

The model has the real architecture (linear and full attention, vision tower) but 4 small
layers, so TRL's `SFTTrainer`, the LoRA wrapping, the data path, the adapter save and the smoke
mode all run for real in a few seconds. Real speed and memory are what the GPU smoke run is for.

Needs the ml group and the Qwen3.5-4B tokenizer files (`uv run --group ml pytest -m hf`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sft_fixture import make_sample, tiny_qwen3_5_model, write_samples

from vlm_floorplan.training.config import TrainConfig, load_config
from vlm_floorplan.training.data import Sample, render_sample, training_template
from vlm_floorplan.training.train import run_training

pytestmark = [pytest.mark.hf, pytest.mark.slow]

ACCUMULATION = 2
OVERLONG_ROOMS = 80


@pytest.fixture(scope="module", autouse=True)
def offline_hub(qwen_tokenizer: Any) -> Any:
    """Stop PEFT and transformers asking the Hub about the base model name of the tiny model.

    The tokenizer fixture has already fetched (or found cached) everything the test needs; the
    extra lookups only add network timeouts.
    """
    from huggingface_hub import constants

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(constants, "HF_HUB_OFFLINE", True)
        yield


@pytest.fixture(scope="module")
def train_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    samples = [make_sample(f"T{i}", rooms=1 + i % 3) for i in range(7)]
    samples.append(make_sample("Q", tool_call=False))
    samples.append(make_sample("OVERLONG", rooms=OVERLONG_ROOMS))
    return write_samples(tmp_path_factory.mktemp("data") / "train.jsonl", samples)


@pytest.fixture(scope="module")
def max_length(qwen_tokenizer: Any) -> int:
    """Longer than every normal sample and shorter than the overlong one."""
    template = training_template(qwen_tokenizer)
    lengths = {
        rooms: render_sample(
            qwen_tokenizer,
            Sample("x", (s := make_sample(rooms=rooms))["messages"], s["tools"]),
            template,
        ).length
        for rooms in (3, OVERLONG_ROOMS)
    }
    assert lengths[3] < lengths[OVERLONG_ROOMS]
    return lengths[3] + 1


def tiny_config(tmp_path: Path, train_file: Path, max_length: int, *extra: str) -> TrainConfig:
    return load_config(
        overrides=[
            f"data.train_file={json.dumps(str(train_file))}",
            f"data.max_length={max_length}",
            f"train.output_dir={json.dumps(str(tmp_path / 'run'))}",
            f"train.gradient_accumulation_steps={ACCUMULATION}",
            "train.bf16=false",
            "train.logging_steps=1",
            "lora.r=4",
            "lora.alpha=8",
            "smoke.steps=2",
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory, train_file: Path, max_length: int) -> Path:
    config = tiny_config(tmp_path_factory.mktemp("full"), train_file, max_length)
    return run_training(config, model=tiny_qwen3_5_model())


def test_full_run_saves_the_adapter(full_run: Path) -> None:
    assert (full_run / "adapter" / "adapter_config.json").is_file()
    assert (full_run / "adapter" / "adapter_model.safetensors").is_file()
    config = json.loads((full_run / "adapter" / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == (4, 8)


def test_full_run_drops_the_overlong_sample_and_says_so(full_run: Path) -> None:
    report = json.loads((full_run / "data_report.json").read_text())
    assert list(report["dropped"]) == ["OVERLONG"]
    assert (report["total"], report["kept"]) == (9, 8)
    summary = json.loads((full_run / "train_summary.json").read_text())
    assert summary["samples"] == 8


def test_full_run_logs_a_loss_for_every_optimizer_step(full_run: Path) -> None:
    summary = json.loads((full_run / "train_summary.json").read_text())
    steps = [row for row in summary["log_history"] if "loss" in row]
    assert len(steps) == 8 // ACCUMULATION
    assert all(row["loss"] > 0 for row in steps)


def test_full_run_keeps_adapter_snapshots_and_the_resolved_config(full_run: Path) -> None:
    assert list(full_run.glob("checkpoint-*"))
    assert json.loads((full_run / "run_config.json").read_text())["lora"]["r"] == 4


def test_saved_adapter_loads_into_the_base_model_and_generates(
    full_run: Path, qwen_tokenizer: Any
) -> None:
    from peft import PeftModel

    from vlm_floorplan.training.generate import first_request, generate_reply

    model = PeftModel.from_pretrained(tiny_qwen3_5_model(), full_run / "adapter").eval()
    raw = make_sample()
    sample = Sample(raw["id"], raw["messages"], raw["tools"])
    reply = generate_reply(model, qwen_tokenizer, sample, max_new_tokens=4)
    assert isinstance(reply, str)
    assert [m["role"] for m in first_request(sample)] == ["system", "user"]


def test_smoke_run_writes_speed_and_memory_and_no_adapter(
    tmp_path: Path, train_file: Path, max_length: int
) -> None:
    config = tiny_config(tmp_path, train_file, max_length)
    run_dir = run_training(config, smoke=True, model=tiny_qwen3_5_model())

    smoke = json.loads((run_dir / "smoke.json").read_text())
    assert smoke["optimizer_steps"] == config.smoke.steps
    assert smoke["tokens_per_second"] > 0
    assert "gpu" in smoke  # null without a GPU, peak memory figures with one
    assert not (run_dir / "adapter").exists()
    report = json.loads((run_dir / "data_report.json").read_text())
    assert report["dropped_count"] == 1


def test_existing_run_directory_is_never_reused(
    tmp_path: Path, train_file: Path, max_length: int
) -> None:
    config = tiny_config(tmp_path, train_file, max_length)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "precious.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="train.output_dir"):
        run_training(config, model=tiny_qwen3_5_model())
    assert (tmp_path / "run" / "precious.txt").read_text() == "keep"


def test_everything_over_the_limit_is_an_error(tmp_path: Path, train_file: Path) -> None:
    config = tiny_config(tmp_path, train_file, 10)
    with pytest.raises(ValueError, match="longer than 10 tokens"):
        run_training(config, model=tiny_qwen3_5_model())


def test_unreadable_data_fails_before_a_run_directory_exists(
    tmp_path: Path, max_length: int
) -> None:
    config = tiny_config(tmp_path, tmp_path / "missing.jsonl", max_length)
    with pytest.raises(FileNotFoundError):
        run_training(config, model=tiny_qwen3_5_model())
    assert not (tmp_path / "run").exists()


def test_load_base_model_reads_a_saved_model_in_the_configured_dtype(tmp_path: Path) -> None:
    import torch

    from vlm_floorplan.training.config import ModelConfig
    from vlm_floorplan.training.model import load_base_model

    tiny_qwen3_5_model().save_pretrained(tmp_path / "tiny")
    model = load_base_model(ModelConfig(name=str(tmp_path / "tiny"), dtype="bfloat16"))
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert next(model.parameters()).dtype == torch.bfloat16
