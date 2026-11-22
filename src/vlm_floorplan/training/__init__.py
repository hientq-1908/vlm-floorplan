"""SFT stage 1: LoRA fine-tuning of Qwen3.5-4B on one-turn text-to-graph conversations.

Read in this order:

- `config.py`   every setting, loaded from `configs/sft_stage1.toml`
- `data.py`     JSONL samples to token ids and an assistant-only loss mask, with the length filter
- `model.py`    base model, frozen vision tower, LoRA adapter
- `smoke.py`    tokens per second and peak GPU memory
- `train.py`    the training run itself (`scripts/sft_train.py`)
- `generate.py` quick generation check of a saved adapter
- `job.py`, `launcher.py`   the ClearML side

`docs/sft-training.md` explains the whole flow. Importing this package needs only the core
dependencies; torch, transformers, TRL and PEFT are imported inside the functions that use them.
"""
