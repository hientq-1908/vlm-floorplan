# SFT stage 1 training

This page explains the training code for SFT stage 1 (text to graph) and how to run it.
The code lives in `src/vlm_floorplan/training/` and the settings in `configs/sft_stage1.toml`.
The recipe is section 6 of the text-to-graph plan; the sample format is the one agreed in the sample report (7.11 to 7.19).

The model is Qwen3.5-4B with thinking on, trained with a LoRA adapter on one 24 GB GPU.
Stage 1 trains one-turn conversations: a user request that carries the current plan and exterior wall, the model's reasoning, one tool call, the tool response, and a short reply.
Single-request behavior samples (edit, restart, clarifying question) use the same format.
General replay data is generated later on the GPU and is just more lines in the same file.

## The parts

| File | What it does |
|---|---|
| `configs/sft_stage1.toml` | Every setting, with the recipe values as defaults. |
| `training/config.py` | Loads and validates the file, applies `--set` overrides. |
| `training/data.py` | JSONL to token ids and loss mask, with the length filter. |
| `training/model.py` | Loads the base model, freezes the vision tower, builds the LoRA config. |
| `training/train.py` | The run itself, and the `--smoke` mode. |
| `training/smoke.py` | Measures tokens per second and peak GPU memory. |
| `training/generate.py` | Quick generation check of a saved adapter with PEFT. |
| `training/job.py`, `training/launcher.py` | The ClearML job and launcher. |
| `scripts/sft_train.py`, `sft_generate.py`, `clearml_sft.py`, `sft_job.py` | Thin entry points. |

## The training file

One JSON object per line:

```json
{"id": "T0001", "messages": [...], "tools": [...]}
```

`messages` are chat messages with the roles system, user, assistant and tool.
An assistant message holds its thinking in `reasoning_content` and its tool calls in `tool_calls`.
`tools` is the tool schema block the chat template writes into the prompt.
The code never looks at tool names or room types, so those can change without touching it.
Each sample must end with an assistant message, and ids must be unique.
Each sample must hold exactly one user message, because stage 1 is one-turn only (see [The chat template](#the-chat-template)).
Tool calls and tool responses inside that one turn are fine; a sample with a second user turn is rejected, not dropped.
A bad line fails with its file name and line number.

## How the data becomes a loss

### Tokenizing here, not inside TRL

TRL can tokenize a conversational dataset by itself.
This code renders each sample from its raw dicts and hands TRL ready token ids instead.
The reason is the Hugging Face `Dataset`: it merges the tool calls of all samples into one record type and fills the gaps with nulls, so a call to one tool would gain empty arguments of every other tool.
The length filter also needs the exact token count before training starts.

### The chat template

The tokenizer is called with TRL's Qwen3.5 think training template.
It renders exactly the text of the template shipped with the model, and adds `{% generation %}` markers around every assistant turn so the tokenizer can report which tokens the assistant wrote.
`tests/test_sft_template.py` checks that a one-turn sample renders identically in both templates, with and without tool calls, and has a control that shows the two do differ once a second user turn exists.
That difference is why stage 1 is one-turn only: the shipped template drops the reasoning of earlier turns, and the training template keeps it.

### Assistant-only loss

`labels` equal `input_ids` on assistant tokens and are -100 (ignored by the loss) everywhere else.
Trained: the reasoning, the tool call, the reply and the end-of-turn token.
Masked: the system prompt, the tool schemas, the user request and the tool responses.
The `<think>` that opens the reasoning is part of the prompt, so thinking is kept without training the opening tag.
This is the mask TRL computes for `assistant_only_loss=True`; `tests/test_sft_masking.py` compares it with TRL's own path for a vision-language model and checks every masked part.

### Length filter, no truncation, no packing

Samples longer than `data.max_length` tokens (default 5120, use 8192 for stage 2) are dropped and counted.
The count and the dropped ids are logged and written to `data_report.json` in the run directory.
A sample is never cut, because the end of a sample is the tool call or the reply the model must learn.
TRL's own truncation is switched off (`max_length=None`), and the code checks that nothing longer got through.

Packing is off.
The torch fallbacks for the linear-attention layers do not respect sample boundaries, so packed samples would leak state into each other.
Samples of similar length share a batch instead (`train.group_by_length`).
Packing can be reconsidered after a smoke run shows packed and unpacked logits match with the fast kernels installed.

## Model and LoRA

`model.py` loads Qwen3.5-4B in bf16, freezes every vision-tower and merger parameter, and the trainer attaches the adapter from the config:

- rank 32, alpha 64, dropout 0.05,
- `q_proj`, `k_proj`, `v_proj`, `o_proj` on the 8 full-attention layers,
- `in_proj_qkv`, `in_proj_z`, `out_proj` on the 24 linear-attention layers,
- `gate_proj`, `up_proj`, `down_proj` on all 32 layers,
- `in_proj_a`, `in_proj_b` and `conv1d` stay frozen.

After the adapter is attached, `check_trainable` fails the run unless every trainable parameter is a LoRA weight of the language model.
`tests/test_sft_model.py` checks the module counts above on the real architecture without downloading any weights.

The optimizer is AdamW with learning rate 1e-4, cosine decay, 3 % warmup and no weight decay.
Batch size is 1 per device with 16 gradient accumulation steps, bf16, and gradient checkpointing.
`train.save_steps = 0.25` saves adapter-only snapshots at 25, 50, 75 and 100 % of the steps.

On a 16 GB GPU without bf16 (T4, Kaggle, Colab free) the same recipe runs as QLoRA, chosen with `--qlora` or the four settings in `QLORA_FP16_OVERRIDES`:

- `model.quantization = "nf4"`: the base weights load 4-bit (bitsandbytes, double quantization), and the vision tower, `in_proj_a` and `in_proj_b` stay 16-bit,
- `model.dtype = "float16"` and `train.fp16 = true` instead of bf16, so the compute is fp16 with a loss scaler,
- the LoRA weights stay fp32 (TRL would otherwise cast them to bf16), and `check_trainable` runs unchanged, so the vision tower is still frozen and only LoRA weights train.
- the fp16 loss scaler starts at `train.fp16_init_scale = 1024` instead of torch's 65536.
  From 65536, every one of the first 4 optimizer steps overflowed on a T4 and was skipped (`grad_norm` NaN) while the scale halved down to 4096, and one more step overflowed later at 4096.
  1024 is below every scale seen to overflow.

The default stays bf16 LoRA, with `model.quantization = "none"`.
The config refuses a mix that cannot work, such as bf16 together with fp16, or fp16 with a bf16 model.
`tests/test_sft_config.py` and `tests/test_sft_model.py` cover the selection and the trainable-parameter check under QLoRA without downloading weights.

## Config

All settings are in `configs/sft_stage1.toml`, and the class fields in `config.py` have no defaults, so a missing key is an error and so is an unknown key.
Change one value for one run with `--set`:

```bash
--set train.learning_rate=5e-5 --set data.max_length=8192 --set data.train_file=/path/to/train.jsonl
```

The value is read as TOML: numbers, `true`, quoted strings and lists work, and a bare word is a string.

Three keys exist for runs that must survive a session limit:

- `train.resumable = true` keeps full checkpoints (with optimizer state, the last two) instead of adapter-only snapshots,
- `train.time_limit_hours` stops the run cleanly with a checkpoint after that many hours (needs `resumable`),
- `scripts/sft_train.py --resume` continues the run in `train.output_dir`, after checking that the settings and the config file are the ones the run started with (`time_limit_hours` may change).

`--resume` cannot be combined with `--smoke`.

## Run it locally

You need the `ml` dependency group, a GPU with 24 GB, and access to the Qwen3.5-4B weights on the Hugging Face Hub (about 9 GB, cached after the first run).

```bash
uv sync --group ml
TRAIN=/path/to/train.jsonl

# 1. Smoke run: 4 optimizer steps on a slice, no model is saved.
#    It writes to outputs/sft-stage1-smoke, next to the full run's directory.
uv run --group ml python scripts/sft_train.py --smoke --set data.train_file=$TRAIN

# 2. Read the result.
cat outputs/sft-stage1-smoke/smoke.json

# 3. Full run.
uv run --group ml python scripts/sft_train.py --set data.train_file=$TRAIN

# 4. Check the adapter.
uv run --group ml python scripts/sft_generate.py \
  --adapter outputs/sft-stage1/adapter --samples $TRAIN --index 0
```

On a GPU with 16 GB or without bf16, add `--qlora` to the smoke and full commands (see "Model and LoRA"; the Kaggle section below uses it).

Smoke mode trains on steps x accumulation x per-device batch samples: the longest kept sample plus the first ones.
The longest sample makes the memory figure the worst case.
`smoke.json` holds:

- `tokens_per_second`: non-padding tokens per second over the steps after the first, which pays for warm-up,
- `peak_memory_allocated_gib` and `peak_memory_reserved_gib` (under `gpu`; `null` without a GPU),
- `step_seconds`, and `projected_hours_per_epoch` for one epoch over the whole file at the measured speed.

Plan section 6.3 estimates 750 to 1,000 tokens per second and says the real figure must be measured first.
This is that measurement.
Without `flash-linear-attention` and `causal-conv1d` installed, transformers uses slow torch fallbacks for the linear-attention layers and says so in the log.

A run directory is never reused: if it already has files, the run stops and asks for another `train.output_dir`.
It holds:

| File | Content |
|---|---|
| `run_config.json` | The resolved settings of this run. |
| `data_report.json` | Samples kept and dropped, token counts. |
| `adapter/` | The final LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`). |
| `checkpoint-*/` | Adapter snapshots. |
| `train_summary.json` | The loss log and the run time. |

The generation check loads the base model named in `adapter_config.json`, attaches the adapter with PEFT, and answers the first request of one sample.
It prints the model's reply next to the reference reply and says whether a tool call was written.
It is a sanity check, not an evaluation.

## Run it on ClearML

ClearML is compute only, as in the baseline job: the launcher attaches the training file to the task, the agent trains, and after the job every output is downloaded and the task is deleted.
The tree must be clean, because the agent receives the source at HEAD (not your working tree).
Credentials come from `~/clearml.conf` or the `CLEARML_*` environment variables and never enter the repository.

```bash
# Show the plan; reads and validates the training file, touches neither ClearML nor the network.
uv run --group ml python scripts/clearml_sft.py launch --train-file $TRAIN --queue QUEUE --dry-run

# Smoke run first, then collect.
uv run --group ml python scripts/clearml_sft.py launch --train-file $TRAIN --queue QUEUE --smoke --wait

# Full run.
uv run --group ml python scripts/clearml_sft.py launch --train-file $TRAIN --queue QUEUE --wait

# Or collect a finished task later.
uv run --group ml python scripts/clearml_sft.py collect --task-id TASK_ID --run-name RUN_NAME
```

`--set section.key=value` is accepted here too and is checked before anything is sent.
Results go to `~/Documents/pets/vlm-floorplan-runs/<run-name>/` (change with `--runs-dir` or `VLM_FLOORPLAN_RUNS_DIR`).

The collector is the baseline's: it deletes the task only after verification.

1. Every artifact exists locally with the byte size ClearML reports.
2. The result archive is a readable zip.
3. The job's `MANIFEST.json` matches the extracted files byte for byte (SHA-256).
4. A successful job left its outputs: `smoke.json` for a smoke run, otherwise `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `train_summary.json` and `data_report.json`.
5. Task details, scalars and the console log are saved.

After the task is deleted, the collector also deletes its empty ClearML project, as described in `docs/evaluation.md`.

A failed task is downloaded the same way, since its logs are the point, and then deleted.
The job writes `job.log` into the result, which is the complete log.
The agent cannot clone the private repository, so the launcher uploads the source with the task: `git archive HEAD` filtered again so that data, outputs and secret-looking files never leave the machine (`source_pack.py`).
The task itself is the single script `scripts/sft_bootstrap.py`, which unpacks that archive, builds the environment from `uv.lock` with Python 3.12 (the agent image only has an older Python), and runs `scripts/sft_job.py` in it.
Before training, the bootstrap prints `nvidia-smi` and runs a tiny bf16 matmul to check that torch can use the GPU.
If that fails, it retries with the docker image's CUDA compat libraries (`/usr/local/cuda*/compat`, newest version first) first on `LD_LIBRARY_PATH`, so the CUDA 13 torch wheel in `uv.lock` can run on an agent with an older driver (data-center GPUs only).
If both fail, it stops with a clear message.
`--queue` is required; list the workspace queues and their live workers first (for example with ClearML's `APIClient().workers.get_all()`).
On the agent, a full run reports its loss curve to ClearML (`train.report_to = "clearml"`); a smoke run and a local run do not.
Only metrics go there: the job sets `CLEARML_LOG_MODEL=false`, so checkpoints are not uploaded, since the result archive already carries the adapter.

## Run it on Kaggle (free tier)

The same training runs on a free Kaggle GPU with `notebooks/kaggle_sft.ipynb`.
Kaggle gives 30 GPU hours a week, sessions of up to 12 hours, and keeps a notebook's output.
The notebook also runs on Colab free, with Google Drive instead of Kaggle Datasets, but a Colab session is often 4 to 5 hours and not guaranteed.

### What changes, and what it costs

A T4 has 16 GB and no bf16, so the notebook trains the QLoRA variant described under "Model and LoRA".
Everything else is the repository's recipe: rank 32, the same target modules, `max_length` 5120, batch 1 with 16 accumulation steps, gradient checkpointing, no truncation, assistant-only loss.

QLoRA is a quality caveat, not a free lunch.
The base weights are rounded to 4 bits, so the adapter learns against a slightly different base than the bf16 one, and fp16 has a narrower range than bf16 (the loss scaler skips the first few steps).
The adapter loads on the bf16 base for evaluation, but it was trained against the 4-bit one.
Treat a QLoRA adapter as a good first result and a way to iterate on data, and compare it with a bf16 run on the held-out sets before relying on it.
The smoke run below shows a sane loss, not that the final quality matches bf16.

### Measured on Kaggle

One smoke run on a Kaggle `Tesla T4` (2 x T4 machine, only GPU 0 is used), 8 optimizer steps (128 samples, the longest one included) on a test file of 1,000 samples, `max_length` 5120:

| Measure | Value |
|---|---|
| Precision | QLoRA, 4-bit weights, fp16 |
| Fast linear-attention kernels | yes (flash-linear-attention and causal-conv1d) |
| Tokens per second | 305 (non-padding tokens, after the first step) |
| Peak GPU memory | 6.31 GiB allocated, 6.70 GiB reserved of 14.56 GiB |
| Loss | finite on all 8 steps, 0.87 to 0.73 |
| Gradient norm | NaN on the first 4 steps (fp16 loss scaler settling), finite on the next 4 |
| Projected time for one epoch | 3.4 h over the test file's 3.74 M tokens, so about 0.9 h per million tokens |

For a file of that size one epoch is about 11 % of the weekly 30 GPU hours and fits in one session, so a few epochs fit in a week.
The notebook prints this projection for your own file at the end of every smoke run.
For comparison, the bf16 LoRA run on a 24 GB A10G measured 449 tokens per second, which is about 2.3 h for the same number of tokens, so the T4 is roughly a third slower.
The memory is far below the card, so the longest sample fits with room to spare.

Installing `flash-linear-attention` is quick, but `causal-conv1d` compiles CUDA code and took about 25 minutes on Kaggle, in every session.
Set `BUILD_CAUSAL_CONV1D = False` in the first cell to skip it and use the torch path for the convolution (not measured).
If neither kernel installs, the notebook says so and continues on the slow path, which is slower.
Only one smoke run was measured, so read the figures as one sample, not a benchmark.

### Steps

1. Pack the inputs.
   This needs a clean tree, because the source is `git archive HEAD` (filtered, no data or secrets).

   ```bash
   uv run python scripts/sft_pack_inputs.py --train-file $TRAIN --out-dir /tmp/sft-inputs --dataset-id YOUR_KAGGLE_USER/vlmfp-sft-inputs
   ```

   It writes `vlm-floorplan-source.tar.gz`, `train.jsonl` and `dataset-metadata.json`.
2. Upload them as a private Kaggle Dataset, either on the website (New Dataset) or with the Kaggle CLI: `kaggle datasets create -p /tmp/sft-inputs`.
   Kaggle unpacks the tarball into a folder; the notebook accepts both forms.
3. Import `notebooks/kaggle_sft.ipynb` as a new notebook, set it to private, add the Dataset as an input, and in the session options choose the GPU accelerator (T4 x2) and turn Internet on (the weights and the Python packages are downloaded).
4. Run with `MODE = "smoke"` first.
   It builds the environment, checks the GPU, trains 8 steps and prints the table above for your data, including the projected hours per epoch.
5. Set `MODE = "train"` and use Save Version, "Save and Run All (Commit)", so the run continues in the background for up to 12 hours.

The notebook refuses a GPU with less than 14 GB.
A T4 gets QLoRA with fp16; a card with bf16 and 22 GB or more gets the repository default.
Kaggle has retired the P100, so in practice the choice is a T4.
It sets `CUDA_VISIBLE_DEVICES=0`, because with two GPUs the trainer would split every batch of 1 over both cards, which 4-bit loading does not support.

### Outputs and resuming

Everything is written under `/kaggle/working`, which Kaggle keeps as the notebook's output:

| Path | Content |
|---|---|
| `sft-stage1/adapter/` | The final LoRA adapter, when the run finishes. |
| `sft-stage1/checkpoint-*/` | The last two full checkpoints, with optimizer state. |
| `sft-stage1/data_report.json`, `run_config.json`, `train_summary.json` | As in a local run. |
| `sft-stage1-smoke/smoke.json` | The smoke numbers, in a smoke run. |
| `logs/` | The complete log of each session. |

Free sessions end, so the full run is resumable.
The notebook stops training `SAVE_MARGIN_HOURS` (0.5) before `SESSION_HOURS` (12) and saves a checkpoint.
To continue, save a version of the notebook, start a new session of the same notebook, add its previous output as an input (Add Input, Notebook Output Files), and run it again.
The notebook copies the earlier run from the read-only input and continues with `--resume`, after the same checks as above, so it does not silently start over.
Only a run that stopped before its first checkpoint (it holds just `run_config.json` and `data_report.json`) is removed and started over.
Download the finished `sft-stage1/adapter/` from the notebook's output page.

On Colab the same thing happens in the Drive folder `vlm-floorplan-sft` (the two input files, and the runs in `vlm-floorplan-sft/runs`), so reconnecting and running the notebook again resumes the run.

The notebook never needs a Kaggle token.
Do not paste any token into a notebook, and keep the Dataset and the notebook private, since the notebook output holds your training data report.

## Tests

```bash
uv run pytest                      # config, data, filter, job and launcher tests; the rest skip without the ml group
uv run --group ml pytest           # adds the tokenizer, template, masking, LoRA and tiny-model tests
uv run --group ml pytest -m gpu    # opt-in: a real smoke run on a 24 GB GPU
```

The tiny-model tests run the whole `SFTTrainer` path on the CPU with a random Qwen3.5 of four small layers: LoRA wrapping, loss, adapter save, smoke mode and reloading the adapter for generation.
They need the Qwen3.5-4B tokenizer files, downloaded once and cached.

## What was not run

The bf16 path and ClearML were not run: the bf16 code was written on a machine with an 8 GB GPU and there was no ClearML server.
The QLoRA path ran on the real 4B model, in smoke mode on a Kaggle T4 (above) and on that 8 GB GPU; a full run, a resume across two real sessions and the Colab path were not run.
Check these on the first real run:

- peak memory and speed at `max_length` 5120 (the smoke run measures both),
- that the torch wheel locked in `uv.lock` matches the agent's CUDA driver,
- that `report_to = "clearml"` attaches to the task created by `Task.create` on the agent,
- the size of the result archive with the adapter snapshots (each adapter is fp32 and several hundred MB).
