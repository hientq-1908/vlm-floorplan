# Evaluation and the zero-shot baseline

This document explains how a model is scored and how the zero-shot baseline (milestone M2) is run.
The code lives in `src/vlm_floorplan/evaluation/`.
Nothing here needs a GPU except the baseline job itself.

## The question the baseline answers

How far is base Qwen3.5-4B, with no training, from a usable floorplan-to-graph model?
Two knobs are compared: image resolution (768 px versus 1280 px on the long side) and nothing else.
The prompt, sampling settings and metrics are fixed, so a difference comes from the resolution.

## Pipeline

1. `harness.py` sends each plan image and the fixed prompt to an OpenAI-compatible endpoint and saves the raw reply.
2. `parser.py` turns the reply into a `FloorplanGraph`, or into one error category.
3. `matching.py` pairs predicted rooms with ground-truth rooms.
4. `metrics.py` counts what matched, per plan.
5. `aggregate.py` sums the counts into dataset scores, overall and by plan size.
6. `scoring.py` writes `per_plan.jsonl` and `metrics.json`.

Steps 2 to 6 read only files.
`python -m vlm_floorplan.evaluation rescore <run-dir>` repeats them on saved outputs, so improving the parser never costs a model call.

## Running the harness

```bash
uv run python -m vlm_floorplan.evaluation run \
  --split data/splits/test.jsonl --run-name base-768 --out-dir runs \
  --model qwen3.5-4b --base-url http://localhost:8000/v1 --max-long-side 768
```

The endpoint can be vLLM serving the base model, vLLM serving our LoRA, or a hosted frontier model.
For a hosted endpoint pass `--api-key-env NAME_OF_VARIABLE`; the key is read from the environment and never written to disk.
Use `--thinking server-default` for endpoints that reject `chat_template_kwargs`.

### Run directory

| File | Content |
|------|---------|
| `config.json` | All settings, without secrets. |
| `prompt.txt` | The exact prompt. |
| `inputs/images/` | The exact pixels sent, one PNG per plan. |
| `inputs.jsonl` | Per plan: sizes before and after resizing, and the request (image elided). |
| `outputs.jsonl` | Per plan: content, reasoning, finish reason, token counts, latency, or an `api_error` or `input_error`. |
| `per_plan.jsonl` | Per plan: parse result, parsed graph, metric counts. |
| `metrics.json` | Aggregate scores, scores by plan size, parse-error counts, token and latency statistics. |
| `MANIFEST.json` | SHA-256 of every other file. |

Re-running a command on the same run directory resumes it: plans that already have a reply in `outputs.jsonl` are skipped.
Plans that failed with an `api_error` or `input_error` are tried again, and scoring uses the last row per plan, so a retry replaces the failure.
A run directory with different model or prompt settings is refused, so two experiments never mix.
The server URL, concurrency and limit may change between resumes.

## Design choices

### Resolution is a client-side resize

`--max-long-side` shrinks the image before it is sent and never enlarges it.
This makes every endpoint see the same pixels, and the run directory stores exactly those pixels.
The server's own `max_pixels` default is far above 1280 px, so it never interferes.
Images are encoded as PNG, because thin lines and small labels are what the model must read.

### Rooms are matched by point-in-polygon

The model outputs a point per room, not a box, so there is no IoU to compute.
A predicted room matches the ground-truth room whose polygon contains its point.
This needs no threshold and is easy to explain: the point landed inside the kitchen.

- Matching is one-to-one. If two predicted points land in one room, the one nearest the room's own point matches and the other is a duplicate (a false positive).
- A ground-truth room with no polygon is matched by nearest point instead, within 80 grid units of 1000.
- Floors are ignored while matching. One image has one coordinate frame, so a point identifies a room whichever floor graph the model put it in.

### Counts first, ratios last

Each plan yields raw counts.
Aggregation sums counts and divides once (micro averaging), so large plans weigh more, as they do in the data.
A ratio with a zero denominator is `null`, never 0 or 1, so a missing score cannot pass for a real one.

### What is measured

| Metric | Meaning |
|--------|---------|
| `validity_rate` | Share of outputs that parse into a valid graph. `strict_validity_rate` counts only outputs that needed no leniency. |
| `room_recall`, `room_precision` | Over valid outputs. `room_recall_incl_invalid` treats an invalid output as missing every room. |
| `room_count_mae`, `room_count_bias` | Absolute and signed room-count error. Positive bias means too many rooms. |
| `floor_count_accuracy` | Share of plans with the right number of floors. |
| `type_accuracy` | Among matched rooms whose true type is known. Rooms the source left `unknown` are excluded, because their truth is missing, not wrong. |
| `links.<kind>` | Precision, recall and F1 for `door`, `opening`, `wall` and `any` (is there a connection at all). Two views, below. |
| `entrance_accuracy`, `entrance` | Accuracy of the flag on matched rooms, plus precision and recall, because most rooms are not entrances and accuracy alone is inflated. |

Link scores come in two views.
`end_to_end` counts every ground-truth link, so a link at a missed room is a miss.
`matched` counts only links between rooms that were matched, which separates connection quality from room detection quality.
With a weak zero-shot model the first view will be near zero and uninformative, which is why the second exists.

Plans are also grouped by ground-truth room count: 1-6, 7-10, 11-15 and 16+ rooms.
The CubiCasa5K median is 9 rooms per floor.

## The parser

A thinking model produces messy text, so the parser tolerates what is harmless and names what is not.

Tolerated:

- A `<think>` block, or only `</think>` (Qwen3.5's template pre-fills the opening tag).
- Prose and markdown fences around the JSON.
- Several JSON objects, where the last valid graph wins.
- The answer sitting in the reasoning field with an empty content field.
- A tool call as XML text inside the reasoning (TRL issue 8050): the parameter bodies are searched for the graph.
- A single floor without the `floors` wrapper, floors without numbers, and links written as objects instead of triples.

Leniency is recorded in `ParseResult.lenient`, so the strict validity rate shows how often the model follows the requested format exactly.

Failures get exactly one category: `empty`, `no_json`, `truncated`, `invalid_json`, `wrong_shape` or `schema_invalid`.
A reply that stopped because of `max_tokens` and holds no graph is reported as `truncated`, whatever its text looks like.
A failed API call is its own category, `api_error`, and an image that cannot be loaded is `input_error`.
All of these count as invalid outputs.

## The zero-shot baseline job

### Serving

`deploy/vllm/serve.sh` serves Qwen3.5-4B with vLLM pinned to 0.33.0.
That version has the fix for Qwen3.5 LoRA adapters being silently ignored, and is the newest one the pinned TRL accepts.
The script enables `--reasoning-parser qwen3`, `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder`.
See `deploy/vllm/README.md` for the settings.

### ClearML launcher

ClearML is compute only.
The launcher enforces the captain's policy: after every job, all outputs are downloaded and the task is deleted from ClearML.

```bash
# Show what would happen. Touches neither ClearML nor the network.
uv run python scripts/clearml_baseline.py launch --split data/splits/test.jsonl --dry-run

# Launch, wait, download, verify, delete.
uv run --group ml python scripts/clearml_baseline.py launch --split data/splits/test.jsonl --wait

# Or collect a finished task later.
uv run --group ml python scripts/clearml_baseline.py collect --task-id TASK_ID --run-name RUN_NAME
```

Results go to `~/Documents/pets/vlm-floorplan-runs/<run-name>/`.
Each collected run's `config.json` names the split path the agent used, a temporary copy that no longer exists.
To rescore it locally, point `--split` at your own copy of the split, either the JSONL or the unpacked split directory:

```bash
uv run python -m vlm_floorplan.evaluation rescore \
  ~/Documents/pets/vlm-floorplan-runs/<run-name>/result/long-side-768 --split data/splits/test.jsonl
```

The override applies to that rescore only; `config.json` is left as it was, so it still matches the run.
Change the root with `--runs-dir` or the `VLM_FLOORPLAN_RUNS_DIR` environment variable.

Credentials are never in the repository.
ClearML reads them from `~/clearml.conf` or the `CLEARML_*` environment variables.

How a launch works:

1. The split (JSONL plus the images it names) is packed into one archive and attached to the task as an artifact, so it is deleted with the task.
2. The task is created from the current git commit, so the commit must be pushed and the tree clean (`--allow-dirty` overrides the second check).
3. On the agent, `scripts/baseline_job.py` starts vLLM, runs the harness once per resolution, stops the server, and uploads one result archive.
   A failed resolution does not stop the other.
   The job exits non-zero if anything failed, so ClearML marks the task failed.

### Verify, then delete

The collector deletes the task only after every check passes:

1. Every artifact exists locally with the byte size ClearML reports.
2. The result archive is a readable zip.
3. The job's own `MANIFEST.json` matches the extracted files, byte for byte (SHA-256).
4. A completed task has the result archive and one `metrics.json` per expected resolution.
5. Task details, scalars and the console log are saved.

A task that is still queued or running is refused before anything is downloaded, because deleting it would kill the job.
If any check fails, the task is kept and the command exits with status 2.
A failed or stopped task is downloaded the same way (its logs are the point) and deleted afterwards.
A download directory that already has files is never overwritten.

The ClearML console log is fetched as the last 10,000 log events.
The job also writes `job.log` and `vllm.log` into the result archive, and those are the complete logs.

## What was not run

The ClearML launcher, the job script and `serve.sh` were written without a GPU or a ClearML server.
The tests cover everything that does not need them: the dry run, split packing, the harness against a fake client and a local HTTP server, the manifest, and the verify-then-delete order against a fake task.
Check these on the first real run: the vLLM flags against `vllm serve --help` for 0.33.0, and `Task.create` with `repo`, `branch` and `commit` on the agent.
