"""Smoke run of the real model on a real GPU. Opt-in: `uv run --group ml pytest -m gpu`.

It downloads Qwen3.5-4B (about 9 GB) and needs a 24 GB GPU, so it is deselected by default and
also skips itself when the GPU is missing or too small.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sft_fixture import make_sample, write_samples

from vlm_floorplan.training.config import load_config
from vlm_floorplan.training.train import run_training

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MIN_MEMORY_GIB = 22


@pytest.fixture
def gpu() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib < MIN_MEMORY_GIB:
        pytest.skip(f"the GPU has {total_gib:.0f} GiB, the 4B model needs about {MIN_MEMORY_GIB}")


def test_smoke_run_on_the_real_model(gpu: None, tmp_path: Path) -> None:
    train_file = write_samples(tmp_path / "t.jsonl", [make_sample(f"T{i}") for i in range(8)])
    config = load_config(
        overrides=[
            f"data.train_file={json.dumps(str(train_file))}",
            f"train.output_dir={json.dumps(str(tmp_path / 'run'))}",
            "train.gradient_accumulation_steps=2",
            "smoke.steps=2",
        ]
    )
    run_dir = run_training(config, smoke=True)
    smoke = json.loads((run_dir / "smoke.json").read_text())
    assert smoke["tokens_per_second"] > 0
    assert smoke["gpu"]["peak_memory_allocated_gib"] > 5
