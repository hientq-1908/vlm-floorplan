"""The SFT run: `uv run --group ml python scripts/sft_train.py`.

Steps, in order:

1. load the config (and `--set` overrides) and refuse a run directory that already has files,
2. read the JSONL, render every sample with the training chat template, drop samples over
   `data.max_length` tokens (`data.py`); bad data stops the run before it creates anything,
3. load Qwen3.5-4B, freeze the vision tower, attach the LoRA adapter (`model.py`),
4. train with TRL's `SFTTrainer` on the pre-tokenized samples,
5. save the adapter to `<run dir>/adapter` and a summary to `<run dir>/train_summary.json`.

`--qlora` switches to 4-bit QLoRA with fp16 compute for 16 GB GPUs without bf16 (T4, P100).

`--resume` continues an unfinished run in its existing run directory from the last snapshot, for
free notebook sessions that end after 12 h. The run must have been started with
`train.resumable = true` (snapshots then keep the optimizer state) and the same settings.
`train.time_limit_hours` ends a session cleanly with a snapshot before the platform cuts it.

`--smoke` does the same on a slice of a few optimizer steps and writes `smoke.json` with tokens
per second and peak GPU memory (`smoke.py`). It never saves a model, and it writes to the sibling
directory `<output_dir>-smoke`, so a smoke run never blocks the full run that follows it.

The run directory holds everything a run produces: `run_config.json` (the resolved settings),
`data_report.json`, `adapter/`, `checkpoint-*/` snapshots, `train_summary.json`, `smoke.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_floorplan.training.config import (
    DEFAULT_CONFIG,
    QLORA_FP16_OVERRIDES,
    TrainConfig,
    load_config,
)

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

log = logging.getLogger("sft")

ADAPTER_DIR = "adapter"
SUMMARY_FILE = "train_summary.json"
DATA_REPORT_FILE = "data_report.json"
SMOKE_FILE = "smoke.json"
CONFIG_FILE = "run_config.json"


def run_dir_for(output_dir: Path, smoke: bool) -> Path:
    return output_dir.with_name(f"{output_dir.name}-smoke") if smoke else output_dir


def check_run_dir_is_free(run_dir: Path) -> None:
    """An existing run directory with files is never reused (`--resume` is the exception)."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} already has files; pick another one with --set train.output_dir=..."
        )


def make_sft_config(config: TrainConfig, run_dir: Path, *, smoke: bool) -> Any:
    """TRL's `SFTConfig` from our settings.

    `max_length=None` turns TRL's own truncation off: the samples were length-filtered already,
    and a sample is never cut. Packing is off in v1 (plan 6.2); the sampler groups samples of
    similar length instead, so a batch wastes little on padding.
    """
    from trl import SFTConfig

    t = config.train
    return SFTConfig(
        output_dir=str(run_dir),
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_steps=t.warmup_ratio,  # a float below 1 is a share of all steps
        weight_decay=t.weight_decay,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=1.0 if smoke else t.num_epochs,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length" if t.group_by_length else "random",
        max_length=None,
        packing=False,
        logging_steps=1 if smoke else t.logging_steps,
        report_to="none" if smoke else t.report_to,
        save_strategy="no" if smoke else "steps",
        save_steps=t.save_steps,
        save_only_model=not t.resumable,
        save_total_limit=2 if t.resumable else None,
        include_num_input_tokens_seen="non_padding",
        seed=t.seed,
    )


def set_initial_loss_scale(accelerator: Any, scale: float) -> None:
    """Replace the fp16 loss scaler with one that starts at `scale`.

    `Trainer` offers no setting for it. This must run before `train()`, which hands the scaler to
    the optimizer; a resumed run then loads the scaler state saved in its snapshot.
    """
    import torch

    if accelerator.scaler is None:
        raise RuntimeError("train.fp16 is set, but the trainer made no fp16 loss scaler")
    accelerator.scaler = torch.amp.GradScaler(accelerator.device.type, init_scale=scale)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def run_training(
    config: TrainConfig,
    *,
    smoke: bool = False,
    resume: bool = False,
    model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Run one SFT job and return its run directory.

    With `resume`, an unfinished run in its existing directory continues from its last snapshot.
    A run stopped by `train.time_limit_hours` returns without a final adapter; resume it.

    `model` and `tokenizer` are for tests (a tiny random model); a real run leaves them out.
    """
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    from vlm_floorplan.training.data import load_training_data, to_dataset
    from vlm_floorplan.training.model import (
        check_trainable,
        freeze_vision,
        load_base_model,
        make_lora_config,
        upcast_trainable_to_fp32,
    )
    from vlm_floorplan.training.session import TimeLimit, check_resume
    from vlm_floorplan.training.smoke import SmokeMeter, select_smoke_slice

    if resume and smoke:
        raise ValueError("--resume cannot be combined with --smoke")
    run_dir = run_dir_for(config.train.output_dir, smoke)
    settings = {"smoke": smoke, **config.model_dump(mode="json")}
    if resume:
        snapshot = check_resume(run_dir, settings, CONFIG_FILE, ADAPTER_DIR)
        log.info("resuming %s from %s", run_dir, snapshot.name)
    else:
        check_run_dir_is_free(run_dir)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.name)
    # Bad data fails here, before a run directory exists.
    items, report = load_training_data(config.data.train_file, tokenizer, config.data.max_length)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / CONFIG_FILE, settings)
    _write_json(run_dir / DATA_REPORT_FILE, report.to_json())
    if smoke:
        t = config.train
        count = config.smoke.steps * t.gradient_accumulation_steps * t.per_device_batch_size
        items = select_smoke_slice(items, count)
        log.info("smoke mode: %d samples, %d optimizer steps", len(items), config.smoke.steps)
    longest = max(item.length for item in items)
    if longest > config.data.max_length:  # the filter makes this impossible; trust but check
        raise RuntimeError(f"a sample of {longest} tokens got past the length filter")

    model = model or load_base_model(config.model)
    log.info("froze %d vision tensors", freeze_vision(model))

    callbacks: list[Any] = []
    if smoke:
        callbacks.append(SmokeMeter(run_dir / SMOKE_FILE, report.kept_tokens))
    limit = TimeLimit(config.train.time_limit_hours) if config.train.time_limit_hours else None
    if limit is not None and not smoke:
        callbacks.append(limit)
    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(config, run_dir, smoke=smoke),
        train_dataset=to_dataset(items),
        processing_class=tokenizer,
        peft_config=make_lora_config(config.lora),
        callbacks=callbacks,
    )
    if config.train.fp16:
        log.info("moved %d LoRA tensors to fp32", upcast_trainable_to_fp32(trainer.model))
        set_initial_loss_scale(trainer.accelerator, config.train.fp16_init_scale)
    trainable = check_trainable(trainer.model)
    log.info("trainable parameters: %s", f"{trainable:,}")

    started = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    if smoke:
        log.info("wrote %s", run_dir / SMOKE_FILE)
        return run_dir
    if limit is not None and limit.expired and trainer.state.global_step < trainer.state.max_steps:
        log.info(
            "time limit reached at step %d of %d; continue with --resume",
            trainer.state.global_step,
            trainer.state.max_steps,
        )
        return run_dir

    trainer.model.save_pretrained(run_dir / ADAPTER_DIR)
    _write_json(
        run_dir / SUMMARY_FILE,
        {
            "train_seconds": round(time.time() - started, 1),
            "trainable_parameters": trainable,
            "samples": len(items),
            "log_history": trainer.state.log_history,
        },
    )
    log.info("saved the adapter to %s", run_dir / ADAPTER_DIR)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value for this run, e.g. --set train.learning_rate=5e-5",
    )
    parser.add_argument("--smoke", action="store_true", help="a few steps, write smoke.json")
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="4-bit QLoRA with fp16 compute, for 16 GB GPUs without bf16 (T4); --set still wins",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue an unfinished run from its last snapshot"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    overrides = [*QLORA_FP16_OVERRIDES, *args.overrides] if args.qlora else args.overrides
    config = load_config(args.config, overrides)
    run_training(config, smoke=args.smoke, resume=args.resume)
    return 0
