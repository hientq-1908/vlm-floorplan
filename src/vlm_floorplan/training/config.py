"""Every training setting, in one TOML file.

`configs/sft_stage1.toml` is the single source of the defaults: the classes below declare
fields without defaults, so a setting that is missing from the file is an error instead of a
silent fallback. Unknown keys are errors too, which catches typos.

One run can change values without editing the file, with `--set section.key=value`
(`apply_overrides`); the value is read as TOML, so `5e-5`, `true` and `"text"` all work.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sft_stage1.toml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    name: str
    dtype: str


class LoraSettings(_Section):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class DataConfig(_Section):
    train_file: Path
    max_length: int = Field(gt=0)


class TrainSettings(_Section):
    output_dir: Path
    learning_rate: float = Field(gt=0)
    lr_scheduler: str
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    num_epochs: float = Field(gt=0)
    gradient_checkpointing: bool
    bf16: bool
    group_by_length: bool
    logging_steps: int = Field(gt=0)
    report_to: str
    save_steps: float = Field(gt=0)
    seed: int


class SmokeConfig(_Section):
    steps: int = Field(gt=0)


class TrainConfig(_Section):
    model: ModelConfig
    lora: LoraSettings
    data: DataConfig
    train: TrainSettings
    smoke: SmokeConfig


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = tree
    for name in parents:
        if not isinstance(node.get(name), dict):
            raise ValueError(f"unknown config section {name!r} in override {dotted!r}")
        node = node[name]
    if leaf not in node:
        raise ValueError(f"unknown config key in override {dotted!r}")
    node[leaf] = value


def apply_overrides(tree: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` strings to a config tree (a copy is not made)."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # a bare word such as a path is a string
        _set_path(tree, key.strip(), value)
    return tree


def load_config(path: Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> TrainConfig:
    """Read the TOML file, apply `--set` overrides, and validate."""
    with path.open("rb") as handle:
        tree = tomllib.load(handle)
    return TrainConfig.model_validate(apply_overrides(tree, overrides or []))
