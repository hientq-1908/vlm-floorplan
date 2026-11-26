"""Quick generation check of a saved LoRA adapter, with Hugging Face PEFT.

    uv run --group ml python scripts/sft_generate.py \\
        --adapter outputs/sft-stage1/adapter --samples data/sft/train.jsonl --index 0

It loads the base model named in the adapter's `adapter_config.json`, attaches the adapter,
and answers the first request of one sample: the messages up to (not including) the first
assistant turn, with the sample's tools. The prompt is built with the template shipped with the
model and ends with `<think>`, so the model thinks first, as it will when served.

This is a sanity check, not an evaluation: does the adapter load, and does the reply look like
the training data (reasoning, then a tool call)? It prints the reply next to the sample's own
reference reply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_floorplan.training.data import Sample, read_samples

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TOOL_CALL_OPEN = "<tool_call>"


def first_request(sample: Sample) -> list[dict[str, Any]]:
    """The messages before the assistant's first turn."""
    for index, message in enumerate(sample.messages):
        if message["role"] == "assistant":
            return sample.messages[:index]
    raise ValueError(f"sample {sample.id} has no assistant turn")


def generate_reply(
    model: Any, tokenizer: PreTrainedTokenizerBase, sample: Sample, max_new_tokens: int
) -> str:
    """Greedy reply to the sample's first request (the text after the opening `<think>`)."""
    import torch

    inputs = tokenizer.apply_chat_template(
        first_request(sample),
        tools=sample.tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def load_adapter_model(adapter_dir: Path, base_name: str | None = None) -> Any:
    """The base model with the adapter attached (bf16, on the GPU when there is one)."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText

    base = base_name or PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16).to(device)
    return PeftModel.from_pretrained(model, adapter_dir).eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--adapter", type=Path, required=True, help="the saved `adapter/` directory"
    )
    parser.add_argument("--samples", type=Path, required=True, help="training-format JSONL")
    parser.add_argument("--index", type=int, default=0, help="which sample of the file")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--base", help="base model, when not the one named in the adapter")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    sample = read_samples(args.samples)[args.index]
    model = load_adapter_model(args.adapter, args.base)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base or model.peft_config["default"].base_model_name_or_path
    )
    reply = generate_reply(model, tokenizer, sample, args.max_new_tokens)

    expected = sample.messages[len(first_request(sample))]
    print(f"sample {sample.id}")
    print("--- model reply (after the opening <think>) ---")
    print(reply)
    print("--- reference reply ---")
    print(json.dumps(expected, indent=2, ensure_ascii=False))
    print(
        f"tool call written: {TOOL_CALL_OPEN in reply}; "
        f"tool call expected: {bool(expected.get('tool_calls'))}"
    )
    return 0
