"""Helpers for `notebooks/kaggle_sft.ipynb` (Kaggle and Colab free tiers).

Standard library only: the notebook imports this file straight from the unpacked source, before
the training environment exists, and the tests load it by path.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

MIN_GPU_GIB = 14.0
"""Smallest GPU memory accepted. A 16 GB card (T4, P100) reports about 15 GiB."""
BF16_LORA_MIN_GIB = 22.0
"""The default recipe (bf16 LoRA) is built for one 24 GB GPU; a 24 GB card reports about 22 GiB."""
BF16_MIN_CAPABILITY = (8, 0)
"""Ampere and newer compute bf16 in hardware; a T4 (7.5) or P100 (6.0) does not."""


class UnsupportedGpuError(RuntimeError):
    """The GPU cannot train this model; the message says why."""


@dataclass(frozen=True)
class TrainingPlan:
    qlora: bool
    """True: pass `--qlora` (4-bit weights, fp16 compute). False: the default bf16 LoRA."""
    reason: str


def plan_for_gpu(name: str, total_mib: int, capability: tuple[int, int]) -> TrainingPlan:
    """Pick the precision for one GPU, or refuse it.

    `total_mib` and `capability` come from `nvidia-smi --query-gpu=memory.total,compute_cap`.
    """
    gib = total_mib / 1024
    if gib < MIN_GPU_GIB:
        raise UnsupportedGpuError(
            f"{name} has {gib:.1f} GiB, but this training needs a 16 GB GPU or larger "
            "(T4 or P100 is the smallest that works, with QLoRA). Pick another accelerator."
        )
    if capability >= BF16_MIN_CAPABILITY and gib >= BF16_LORA_MIN_GIB:
        return TrainingPlan(False, f"{name} ({gib:.0f} GiB, bf16): the default bf16 LoRA recipe")
    reason = "no bf16" if capability < BF16_MIN_CAPABILITY else "less than 24 GB"
    return TrainingPlan(True, f"{name} ({gib:.0f} GiB, {reason}): QLoRA, 4-bit weights, fp16")


def find_file(roots: list[Path], name: str) -> Path:
    """The only file called `name` under `roots`; an error that names the roots otherwise."""
    found = sorted({path for root in roots if root.is_dir() for path in root.rglob(name)})
    if len(found) != 1:
        where = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"expected exactly one {name} under {where}, found {len(found)}")
    return found[0]


def stage_resume(input_roots: list[Path], run_name: str, run_dir: Path) -> Path | None:
    """Copy an earlier session's run directory next to this one so `--resume` can continue it.

    On Kaggle the output of a previous notebook version is added to the notebook as an input,
    where it is read-only; training needs it in the working directory. Returns the copy, or None
    when `run_dir` already exists (a rerun inside one session) or no earlier run is attached.
    """
    if run_dir.exists():
        return None
    earlier = sorted(
        {
            path.parent
            for root in input_roots
            if root.is_dir()
            for path in root.rglob("run_config.json")
            if path.parent.name == run_name
        }
    )
    if len(earlier) > 1:
        raise FileExistsError(f"several earlier runs named {run_name} are attached: {earlier}")
    if not earlier:
        return None
    shutil.copytree(earlier[0], run_dir)
    return run_dir
