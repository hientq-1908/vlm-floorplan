"""Smoke mode: how fast is training and how much GPU memory does it need?

Plan section 6.3 estimates 750 to 1,000 tokens per second and says the real figure must be
measured first, because the torch fallback kernels for the linear-attention layers may be much
slower. Smoke mode runs a few optimizer steps on a small slice and writes `smoke.json`:

- `tokens_per_second`: non-padding tokens per second over the steps after the first (the first
  step pays for warm-up). With a single step it falls back to that step.
- `peak_memory_allocated_gib` and `peak_memory_reserved_gib`: the GPU's high-water marks.
- `projected_hours_per_epoch`: what one epoch over the full set would take at the measured speed.

The slice always contains the longest kept sample, so the memory figure is the worst case.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from vlm_floorplan.training.data import Rendered

GIB = 1024**3


def select_smoke_slice(items: list[Rendered], count: int) -> list[Rendered]:
    """`count` samples: the longest one plus the first `count - 1` of the others."""
    if count >= len(items):
        return list(items)
    longest = max(range(len(items)), key=lambda i: items[i].length)
    others = [item for i, item in enumerate(items) if i != longest]
    return [items[longest], *others[: count - 1]]


def cuda_peak_memory() -> dict[str, float | str] | None:
    """The GPU's peak memory since the last reset, or None without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "device": torch.cuda.get_device_name(),
        "peak_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
        "peak_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
        "total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
    }


def reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class SmokeMeter(TrainerCallback):
    """Times every optimizer step and writes `smoke.json` when training ends.

    The trainer must run with `include_num_input_tokens_seen="non_padding"`, which keeps
    `state.num_input_tokens_seen` up to date.
    """

    def __init__(
        self,
        out_path: Path,
        epoch_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
        memory: Callable[[], dict[str, Any] | None] = cuda_peak_memory,
    ):
        self.out_path = out_path
        self.epoch_tokens = epoch_tokens
        self._clock = clock
        self._memory = memory
        self._start = 0.0
        self._marks: list[tuple[float, int]] = []
        """(seconds since start, tokens seen) at the end of each optimizer step."""

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        reset_cuda_peak()
        self._start = self._clock()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._marks.append((self._clock() - self._start, int(state.num_input_tokens_seen)))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.out_path.write_text(json.dumps(self.result(), indent=2) + "\n")

    def result(self) -> dict[str, Any]:
        if not self._marks:
            raise RuntimeError("smoke run finished without a single optimizer step")
        if len(self._marks) > 1:
            (t0, n0), (t1, n1) = self._marks[0], self._marks[-1]
        else:
            (t0, n0), (t1, n1) = (0.0, 0), self._marks[0]
        tokens_per_second = (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0
        projected = self.epoch_tokens / tokens_per_second / 3600 if tokens_per_second else None
        return {
            "optimizer_steps": len(self._marks),
            "tokens_total": self._marks[-1][1],
            "seconds_total": round(self._marks[-1][0], 2),
            "step_seconds": [
                round(b[0] - a[0], 2)
                for a, b in zip([(0.0, 0), *self._marks], self._marks, strict=False)
            ],
            "tokens_per_second": round(tokens_per_second, 1),
            "epoch_tokens": self.epoch_tokens,
            "projected_hours_per_epoch": None if projected is None else round(projected, 1),
            "gpu": self._memory(),
        }
