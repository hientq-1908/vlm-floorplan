"""LoRA reaches the layers the plan names, and only those; the vision tower stays frozen.

The checks run on the real Qwen3.5-4B architecture built on the `meta` device: no weights are
downloaded or allocated, only the config is read.

Needs the ml group and the Qwen3.5-4B config (`uv run --group ml pytest -m hf`).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from vlm_floorplan.training.config import load_config
from vlm_floorplan.training.model import check_trainable, freeze_vision, make_lora_config

pytestmark = pytest.mark.hf

FULL_ATTENTION_LAYERS = 8
LINEAR_ATTENTION_LAYERS = 24
LAYERS = 32


def build_meta_model() -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    try:
        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
    except OSError as err:
        pytest.skip(f"cannot load the Qwen3.5-4B config: {err}")
    with torch.device("meta"):
        return transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)


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
