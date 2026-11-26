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
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sft_stage1.toml"

QLORA_FP16_OVERRIDES = [
    "model.dtype=float16",
    'model.quantization="nf4"',
    "train.bf16=false",
    "train.fp16=true",
]
"""`--set` overrides that switch the recipe to QLoRA with fp16 compute (`sft_train.py --qlora`).

For 16 GB GPUs without bf16 (T4, P100): the base model is loaded 4-bit and the rest of the
recipe is unchanged. The default stays bf16 LoRA on a 24 GB GPU.
"""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    name: str
    dtype: Literal["float32", "float16", "bfloat16"]
    """Dtype of the weights that stay unquantized, and the compute dtype of 4-bit layers."""
    quantization: Literal["none", "nf4"]
    """`nf4` loads the language model's linear layers 4-bit (QLoRA, bitsandbytes)."""

    @model_validator(mode="after")
    def _quantized_weights_compute_in_half_precision(self) -> Self:
        if self.quantization == "nf4" and self.dtype == "float32":
            raise ValueError("model.quantization = 'nf4' needs model.dtype float16 or bfloat16")
        return self


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
    fp16: bool
    group_by_length: bool
    logging_steps: int = Field(gt=0)
    report_to: str
    save_steps: float = Field(gt=0)
    resumable: bool
    """Snapshots also keep the optimizer, scheduler and RNG state (last 2 kept), for `--resume`."""
    time_limit_hours: float = Field(ge=0)
    """Stop cleanly with a resumable snapshot after this many hours; 0 means no limit."""
    seed: int


class SmokeConfig(_Section):
    steps: int = Field(gt=0)


class TrainConfig(_Section):
    model: ModelConfig
    lora: LoraSettings
    data: DataConfig
    train: TrainSettings
    smoke: SmokeConfig

    @model_validator(mode="after")
    def _one_mixed_precision_mode(self) -> Self:
        if self.train.bf16 and self.train.fp16:
            raise ValueError("train.bf16 and train.fp16 cannot both be true")
        if self.train.fp16 and self.model.dtype != "float16":
            raise ValueError("train.fp16 needs model.dtype = 'float16'")
        if self.train.time_limit_hours and not self.train.resumable:
            raise ValueError("train.time_limit_hours needs train.resumable = true")
        return self


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
