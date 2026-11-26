"""The base model, its frozen vision tower, and the LoRA adapter.

Qwen3.5-4B is a vision-language model: a vision tower plus a merger (which maps image features
into the language model's embedding space) sit next to the language model. Stage 1 has no images.
Two things keep it that way:

- `freeze_vision` switches off gradients for every vision parameter. PEFT freezes the whole
  base model anyway, but TRL issue 6617 reports that the vision tower stays trainable on
  text-only data in some setups, so the freeze is explicit.
- The LoRA target names (`config.toml` has the list) match only language-model layers. The test
  `tests/test_sft_model.py` proves it on the real architecture.

With `model.quantization = "nf4"` (QLoRA, for 16 GB GPUs without bf16) the language model's linear
layers are loaded 4-bit through bitsandbytes and computed in `model.dtype`. The vision tower and the
tiny `in_proj_a` and `in_proj_b` projections stay unquantized, and the LoRA adapter stays fp32.

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
    from transformers import BitsAndBytesConfig

log = logging.getLogger(__name__)

VISION_MARKER = ".visual."
"""Parameter names of the vision tower and its merger contain this (`model.visual.merger...`)."""
LANGUAGE_MARKER = ".language_model."
NOT_QUANTIZED = ("model.visual", "in_proj_a", "in_proj_b")
"""Modules that stay in full precision under 4-bit loading (transformers matches a module name
that starts with, or ends with, one of these; the output head is kept by transformers itself)."""


def make_quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    """The 4-bit loading settings for `model.quantization = "nf4"`, else None."""
    if config.quantization == "none":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, config.dtype),
        llm_int8_skip_modules=list(NOT_QUANTIZED),
    )


def load_base_model(config: ModelConfig) -> nn.Module:
    """Load the pretrained model in `config.dtype`.

    Unquantized, it is on the CPU and the trainer moves it to the GPU. A 4-bit model is placed on
    the current GPU while it loads, because bitsandbytes quantizes on the GPU.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    quantization = make_quantization_config(config)
    kwargs: dict[str, object] = {}
    if quantization is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("model.quantization = 'nf4' needs a CUDA GPU to load the model")
        kwargs = {
            "quantization_config": quantization,
            "device_map": {"": torch.cuda.current_device()},
        }
    return AutoModelForImageTextToText.from_pretrained(
        config.name, dtype=getattr(torch, config.dtype), **kwargs
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


def upcast_trainable_to_fp32(model: nn.Module) -> int:
    """Make every trainable parameter fp32; returns how many tensors were changed.

    fp16 mixed precision needs fp32 master weights: its gradient scaler refuses fp16 gradients.
    TRL casts the LoRA weights of a quantized model to bf16, which a T4 only emulates, so the
    adapter is put back to fp32 before the optimizer is built.
    """
    import torch

    changed = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            changed += 1
    return changed


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
