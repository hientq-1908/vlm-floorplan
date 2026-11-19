"""Show an episode as the text the model is trained on, through Qwen3.5's own chat template.

This is for review: you read exactly what the tokenizer would see, including the tool list in
the system block and the `<think>` span in front of every assistant turn.
It needs `transformers` (the `ml` dependency group) and the model's tokenizer files,
a few MB from the Hugging Face Hub, cached after the first download.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "Qwen/Qwen3.5-4B"


def load_tokenizer(model_id: str = MODEL_ID) -> Any:
    from transformers import AutoTokenizer  # heavy, so only imported when rendering

    return AutoTokenizer.from_pretrained(model_id)


def render_episode(episode: dict[str, Any], tokenizer: Any) -> str:
    """The episode as one string, tool list included, with no generation prompt."""
    return tokenizer.apply_chat_template(
        episode["messages"], tools=episode["tools"], tokenize=False
    )
