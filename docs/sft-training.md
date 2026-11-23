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

## Config

All settings are in `configs/sft_stage1.toml`, and the class fields in `config.py` have no defaults, so a missing key is an error and so is an unknown key.
Change one value for one run with `--set`:

```bash
--set train.learning_rate=5e-5 --set data.max_length=8192 --set data.train_file=/path/to/train.jsonl
```

The value is read as TOML: numbers, `true`, quoted strings and lists work, and a bare word is a string.

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

Smoke mode trains on steps x accumulation samples: the longest kept sample plus the first ones.
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
Before training, the bootstrap puts the docker image's CUDA compat libraries (`/usr/local/cuda*/compat`) first on `LD_LIBRARY_PATH` so the CUDA 13 torch wheel in `uv.lock` can run on an agent with an older driver (data-center GPUs only), then prints `nvidia-smi` and runs a tiny bf16 matmul, and stops with a clear message if torch cannot use the GPU.
`--queue` is required; list the workspace queues and their live workers first (for example with ClearML's `APIClient().workers.get_all()`).
On the agent, a full run reports its loss curve to ClearML (`train.report_to = "clearml"`); a smoke run and a local run do not.
Only metrics go there: the job sets `CLEARML_LOG_MODEL=false`, so checkpoints are not uploaded, since the result archive already carries the adapter.

## Tests

```bash
uv run pytest                      # config, data, filter, job and launcher tests; the rest skip without the ml group
uv run --group ml pytest           # adds the tokenizer, template, masking, LoRA and tiny-model tests
uv run --group ml pytest -m gpu    # opt-in: a real smoke run on a 24 GB GPU
```

The tiny-model tests run the whole `SFTTrainer` path on the CPU with a random Qwen3.5 of four small layers: LoRA wrapping, loss, adapter save, smoke mode and reloading the adapter for generation.
They need the Qwen3.5-4B tokenizer files, downloaded once and cached.

## What was not run

Nothing here was run on the real 4B model or on ClearML: this code was written on a machine with an 8 GB GPU and no ClearML server.
Check these on the first real run:

- peak memory and speed at `max_length` 5120 (the smoke run measures both),
- that the torch wheel locked in `uv.lock` matches the agent's CUDA driver,
- that `report_to = "clearml"` attaches to the task created by `Task.create` on the agent,
- the size of the result archive with the adapter snapshots (each adapter is fp32 and several hundred MB).
