"""The SFT config file is the single source of the recipe (plan section 6.1 to 6.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_floorplan.training.config import DEFAULT_CONFIG, apply_overrides, load_config


def test_defaults_are_the_plan_recipe() -> None:
    config = load_config()
    assert config.model.name == "Qwen/Qwen3.5-4B"
    assert config.model.dtype == "bfloat16"
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (32, 64, 0.05)
    assert set(config.lora.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj",  # full-attention layers
        "in_proj_qkv", "in_proj_z", "out_proj",  # linear-attention layers
        "gate_proj", "up_proj", "down_proj",  # MLPs
    }  # fmt: skip
    train = config.train
    assert train.learning_rate == 1e-4
    assert train.lr_scheduler == "cosine"
    assert train.warmup_ratio == 0.03
    assert train.weight_decay == 0.0
    assert (train.per_device_batch_size, train.gradient_accumulation_steps) == (1, 16)
    assert train.bf16 and train.gradient_checkpointing and train.group_by_length


def test_max_length_defaults_to_5120() -> None:
    assert load_config().data.max_length == 5120


def test_overrides_change_only_the_named_value() -> None:
    config = load_config(overrides=["train.learning_rate=5e-5", "data.max_length=8192"])
    assert config.train.learning_rate == 5e-5
    assert config.data.max_length == 8192
    assert config.lora.r == 32


def test_override_accepts_bools_lists_and_bare_strings() -> None:
    config = load_config(
        overrides=["train.bf16=false", 'lora.target_modules=["q_proj"]', "train.output_dir=out/x"]
    )
    assert config.train.bf16 is False
    assert config.lora.target_modules == ["q_proj"]
    assert config.train.output_dir == Path("out/x")


@pytest.mark.parametrize(
    "override", ["train.nonsense=1", "nowhere.key=1", "train.learning_rate", "lora=3"]
)
def test_bad_override_is_rejected(override: str) -> None:
    with pytest.raises(ValueError):
        load_config(overrides=[override])


def test_unknown_key_in_the_file_is_rejected(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text().replace("seed = 42", "seed = 42\nsede = 1")
    path = tmp_path / "typo.toml"
    path.write_text(text)
    with pytest.raises(ValidationError, match="sede"):
        load_config(path)


def test_missing_key_in_the_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "short.toml"
    path.write_text(DEFAULT_CONFIG.read_text().replace("max_length = 5120\n", ""))
    with pytest.raises(ValidationError, match="max_length"):
        load_config(path)


def test_apply_overrides_edits_the_tree_in_place() -> None:
    tree = {"a": {"b": 1}}
    apply_overrides(tree, ["a.b=2"])
    assert tree == {"a": {"b": 2}}
