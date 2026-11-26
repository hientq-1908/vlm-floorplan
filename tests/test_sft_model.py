"""LoRA reaches the layers the plan names, and only those; the vision tower stays frozen.

The checks run on the real Qwen3.5-4B architecture built on the `meta` device: no weights are
downloaded or allocated, only the config is read.

Needs the ml group and the Qwen3.5-4B config (`uv run --group ml pytest -m hf`).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from vlm_floorplan.training.config import QLORA_FP16_OVERRIDES, load_config
from vlm_floorplan.training.model import (
    NOT_QUANTIZED,
    check_trainable,
    freeze_vision,
    make_lora_config,
    make_quantization_config,
    upcast_trainable_to_fp32,
)

pytestmark = pytest.mark.hf

FULL_ATTENTION_LAYERS = 8
LINEAR_ATTENTION_LAYERS = 24
LAYERS = 32


def build_meta_model(dtype: str = "bfloat16") -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    try:
        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
    except OSError as err:
        pytest.skip(f"cannot load the Qwen3.5-4B config: {err}")
    with torch.device("meta"):
        return transformers.AutoModelForImageTextToText.from_config(
            config, dtype=getattr(torch, dtype)
        )


@pytest.fixture(scope="module")
def peft_model() -> Any:
    model = build_meta_model()  # skips without the ml group
    from peft import get_peft_model

    freeze_vision(model)
    return get_peft_model(model, make_lora_config(load_config().lora))


def adapted_modules(model: Any) -> Counter[tuple[str, str]]:
    """(tower, layer type) to count, for every module that got a LoRA adapter."""
    found: Counter[tuple[str, str]] = Counter()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and len(module.lora_A) > 0:
            tower = "visual" if ".visual." in name else "language_model"
            found[(tower, name.rsplit(".", 1)[-1])] += 1
    return found


def test_lora_hits_the_planned_projections_of_every_layer_type(peft_model: Any) -> None:
    found = adapted_modules(peft_model)
    expected = {
        "q_proj": FULL_ATTENTION_LAYERS,
        "k_proj": FULL_ATTENTION_LAYERS,
        "v_proj": FULL_ATTENTION_LAYERS,
        "o_proj": FULL_ATTENTION_LAYERS,
        "in_proj_qkv": LINEAR_ATTENTION_LAYERS,
        "in_proj_z": LINEAR_ATTENTION_LAYERS,
        "out_proj": LINEAR_ATTENTION_LAYERS,
        "gate_proj": LAYERS,
        "up_proj": LAYERS,
        "down_proj": LAYERS,
    }
    assert dict(found) == {("language_model", name): n for name, n in expected.items()}


def test_lora_leaves_the_small_linear_attention_inputs_alone(peft_model: Any) -> None:
    names = {name for _, name in adapted_modules(peft_model)}
    assert not names & {"in_proj_a", "in_proj_b", "conv1d"}


def test_no_adapter_in_the_vision_tower_or_merger(peft_model: Any) -> None:
    assert not [key for key in adapted_modules(peft_model) if key[0] == "visual"]


def test_only_language_model_lora_weights_are_trainable(peft_model: Any) -> None:
    assert check_trainable(peft_model) > 0
    vision = [n for n, p in peft_model.named_parameters() if ".visual." in n and p.requires_grad]
    assert vision == []


def test_freeze_vision_covers_the_merger() -> None:
    model = build_meta_model()
    assert all(p.requires_grad for p in model.parameters())
    assert freeze_vision(model) > 0
    merger = [p for n, p in model.named_parameters() if ".visual.merger." in n]
    assert merger and not any(p.requires_grad for p in merger)
    language = [p for n, p in model.named_parameters() if ".language_model." in n]
    assert all(p.requires_grad for p in language)


def test_check_trainable_rejects_a_trainable_vision_weight(peft_model: Any) -> None:
    visual = next(p for n, p in peft_model.named_parameters() if ".visual.merger." in n)
    visual.requires_grad_(True)
    try:
        with pytest.raises(RuntimeError, match="not language-model LoRA"):
            check_trainable(peft_model)
    finally:
        visual.requires_grad_(False)


def test_check_trainable_rejects_a_model_without_adapter() -> None:
    model = build_meta_model()
    for param in model.parameters():
        param.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        check_trainable(model)


QLORA_CONFIG = load_config(overrides=QLORA_FP16_OVERRIDES)


@pytest.fixture(scope="module")
def qlora_peft_model() -> Any:
    """The QLoRA model: 4-bit layers swapped in on the meta device, then the same LoRA setup."""
    pytest.importorskip("bitsandbytes")
    from peft import get_peft_model
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    model = build_meta_model("float16")
    quantization = make_quantization_config(QLORA_CONFIG.model)
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=quantization.llm_int8_skip_modules,
        quantization_config=quantization,
    )
    freeze_vision(model)
    return get_peft_model(model, make_lora_config(QLORA_CONFIG.lora))


def test_default_config_has_no_quantization() -> None:
    assert make_quantization_config(load_config().model) is None


def test_qlora_config_loads_nf4_with_fp16_compute() -> None:
    pytest.importorskip("torch")
    quantization = make_quantization_config(QLORA_CONFIG.model)
    assert quantization.load_in_4bit
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert str(quantization.bnb_4bit_compute_dtype) == "torch.float16"
    assert list(quantization.llm_int8_skip_modules) == list(NOT_QUANTIZED)


def test_qlora_quantizes_the_language_model_and_leaves_the_rest(qlora_peft_model: Any) -> None:
    kinds: dict[str, set[str]] = {}
    for name, module in qlora_peft_model.named_modules():
        base = getattr(module, "base_layer", None)
        if hasattr(module, "lora_A") and base is not None:
            kinds.setdefault(type(base).__name__, set()).add(name.rsplit(".", 1)[-1])
    assert set(kinds) == {"Linear4bit"}
    assert kinds["Linear4bit"] == {
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    }  # fmt: skip
    small = {
        type(m).__name__
        for n, m in qlora_peft_model.named_modules()
        if n.endswith(("in_proj_a", "in_proj_b")) or ".visual." in n and hasattr(m, "weight")
    }
    assert "Linear4bit" not in small


def test_qlora_trains_only_fp32_lora_weights_of_the_language_model(qlora_peft_model: Any) -> None:
    import torch

    assert check_trainable(qlora_peft_model) > 0
    trainable = [p for p in qlora_peft_model.parameters() if p.requires_grad]
    assert {p.dtype for p in trainable} == {torch.float32}
    vision = [
        n for n, p in qlora_peft_model.named_parameters() if ".visual." in n and p.requires_grad
    ]
    assert vision == []


def test_qlora_trainable_check_still_rejects_a_trainable_vision_weight(
    qlora_peft_model: Any,
) -> None:
    visual = next(p for n, p in qlora_peft_model.named_parameters() if ".visual.merger." in n)
    visual.requires_grad_(True)
    try:
        with pytest.raises(RuntimeError, match="not language-model LoRA"):
            check_trainable(qlora_peft_model)
    finally:
        visual.requires_grad_(False)


def test_upcast_puts_trainable_bf16_weights_back_to_fp32_and_leaves_frozen_ones() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(2, 2).to(torch.bfloat16)
    frozen = torch.nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)
    model = torch.nn.Sequential(layer, frozen)
    assert upcast_trainable_to_fp32(model) == 2  # weight and bias of the trainable layer
    assert {p.dtype for p in layer.parameters()} == {torch.float32}
    assert {p.dtype for p in frozen.parameters()} == {torch.bfloat16}
