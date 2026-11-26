"""Small stand-ins for the SFT training data and model, so the training code runs on a CPU.

`make_sample` builds a conversation in the training-file format with the same shape as the real
stage-1 samples: a system prompt, a user request with the current plan state, an assistant turn
with reasoning and one tool call, the tool response, and a short reply. The tool schemas are a
cut-down pair; the code under test never looks at tool names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "new_plan",
            "description": "Replace the plan with a new one.",
            "parameters": {
                "type": "object",
                "properties": {"plan": {"type": "string", "description": "The plan as JSON."}},
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_room",
            "description": "Add a room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["living", "bedroom"]},
                    "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                },
                "required": ["type", "x", "y"],
            },
        },
    },
]

SYSTEM = "You edit a floorplan room graph with tools."
STATE = (
    "plan: 1 rooms, 0 links\n"
    "exterior wall: (0, 0), (0, 500), (500, 500), (500, 0)\n"
    "r0 living at (250, 250) size 90"
)


def make_sample(
    sample_id: str = "T0001", rooms: int = 2, *, tool_call: bool = True
) -> dict[str, Any]:
    """One stage-1 style conversation; more `rooms` make it longer."""
    plan = json.dumps(
        {
            "rooms": [
                {"id": f"r{i}", "type": "bedroom", "xy": [100 + i, 200], "size": 40}
                for i in range(rooms)
            ]
        },
        separators=(",", ":"),
    )
    user = {"role": "user", "content": f"Current plan:\n{STATE}\n\nRequest: Draw {rooms} bedrooms."}
    if not tool_call:
        return {
            "id": sample_id,
            "messages": [
                {"role": "system", "content": SYSTEM},
                user,
                {
                    "role": "assistant",
                    "reasoning_content": "The request mixes two units, so I ask first.",
                    "content": "Do you want square metres or relative sizes?",
                },
            ],
            "tools": TOOLS,
        }
    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": SYSTEM},
            user,
            {
                "role": "assistant",
                "reasoning_content": f"Copy the {rooms} rooms into one new_plan call.",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "new_plan", "arguments": {"plan": plan}},
                    }
                ],
            },
            {"role": "tool", "name": "new_plan", "content": f"OK: replaced the plan.\n{STATE}"},
            {
                "role": "assistant",
                "reasoning_content": "The tool accepted the plan.",
                "content": f"Drew {rooms} bedrooms.",
            },
        ],
        "tools": TOOLS,
    }


def make_two_call_sample(sample_id: str = "T0002") -> dict[str, Any]:
    """One request answered with two tool calls (like the stage-1 edit samples)."""
    sample = make_sample(sample_id)
    call = {
        "type": "function",
        "function": {"name": "add_room", "arguments": {"type": "living", "x": 10, "y": 20}},
    }
    sample["messages"][3:3] = [
        {
            "role": "assistant",
            "reasoning_content": "First add the living room.",
            "content": "",
            "tool_calls": [call],
        },
        {"role": "tool", "name": "add_room", "content": f"OK: added r2.\n{STATE}"},
    ]
    return sample


def write_samples(path: Path, samples: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    return path


def tiny_qwen3_5_model() -> Any:
    """A random Qwen3.5 with the real architecture but 4 small layers (3 linear, 1 full attention).

    It keeps the real vocabulary so the real tokenizer's ids are valid.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 4
    text.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 4
    text.linear_key_head_dim = 16
    text.linear_value_head_dim = 16
    text.rope_parameters["mrope_section"] = [2, 1, 1]
    vision = config.vision_config
    vision.depth = 1
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 2
    vision.out_hidden_size = 64
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
