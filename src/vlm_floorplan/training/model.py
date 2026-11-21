"""The base model, its frozen vision tower, and the LoRA adapter.

Qwen3.5-4B is a vision-language model: a vision tower plus a merger (which maps image features
into the language model's embedding space) sit next to the language model. Stage 1 has no images.
Two things keep it that way:

- `freeze_vision` switches off gradients for every vision parameter. PEFT freezes the whole
  base model anyway, but TRL issue 6617 reports that the vision tower stays trainable on
  text-only data in some setups, so the freeze is explicit.
- The LoRA target names (`config.toml` has the list) match only language-model layers. The test
  `tests/test_sft_model.py` proves it on the real architecture.

`check_trainable` is the last guard: after the adapter is attached, the only trainable
parameters must be LoRA weights of the language model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vlm_floorplan.training.config import LoraSettings, ModelConfig

if TYPE_CHECKING:
    from peft import LoraConfig
    from torch import nn

log = logging.getLogger(__name__)

VISION_MARKER = ".visual."
"""Parameter names of the vision tower and its merger contain this (`model.visual.merger...`)."""
LANGUAGE_MARKER = ".language_model."


def load_base_model(config: ModelConfig) -> nn.Module:
    """Load the pretrained model in bf16 on the CPU; the trainer moves it to the GPU."""
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        config.name, dtype=getattr(torch, config.dtype)
    )


def freeze_vision(model: nn.Module) -> int:
    """Turn off gradients for the vision tower and merger; returns how many tensors were frozen."""
    frozen = 0
    for name, param in model.named_parameters():
        if VISION_MARKER in f".{name}":
            param.requires_grad_(False)
            frozen += 1
    return frozen


def make_lora_config(settings: LoraSettings) -> LoraConfig:
    from peft import LoraConfig

    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def check_trainable(model: nn.Module) -> int:
    """Fail unless every trainable parameter is a LoRA weight of the language model.

    Returns the number of trainable parameters.
    """
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters: the LoRA adapter was not attached")
    stray = [
        name
        for name, _ in trainable
        if "lora_" not in name or LANGUAGE_MARKER not in f".{name}" or VISION_MARKER in f".{name}"
    ]
    if stray:
        raise RuntimeError(
            f"{len(stray)} trainable parameters are not language-model LoRA weights, "
            f"for example {stray[:3]}"
        )
    return sum(count for _, count in trainable)
