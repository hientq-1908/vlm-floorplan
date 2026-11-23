# Text-to-graph evaluation

This page explains how a model is measured on the first fine-tune's task: a description of an apartment in, one `new_plan` call out, then a short reply.
The same harness scores the base model and a fine-tuned one, so "with and without fine-tuning" is one comparison.
Code: `src/vlm_floorplan/requirements/`, `src/vlm_floorplan/text2graph/`, and the `t2g_*` and `plausibility` modules of `src/vlm_floorplan/evaluation/`.
The image-to-graph evaluation is in [evaluation.md](evaluation.md) and is unchanged.

## What the model reads and writes

The harness sends exactly what training uses: `agent.episode.SYSTEM_PROMPT`, the twelve tools of `agent_tool_schemas()`, and one user turn (`agent.env.user_message`, the same function that builds the training user turn).

```text
Current plan:
plan: empty
exterior wall: (60, 264), (60, 736), (568, 736), ...

Request: We're looking for a place with three bedrooms, at least two bathrooms ...
```

Thinking is on, with the sampling defaults of the model card (temperature 0.6, top-p 0.95, top-k 20), and `max_tokens` is 8192.
One request is the agent loop the model is trained on.
Every structured tool call in a reply runs in the real environment (`agent.tools.call_tool`) on the running plan, the assistant turn and one tool message per call with the real response are appended, and the conversation is sent again, until the model replies without a tool call.
The loop stops at 8 tool calls (`max_tool_calls`); a prompt that reaches it is flagged `hit_call_cap`, counted in `call_cap_rate`, and is not a clean finish: it has no strict plan and no behavior of it is correct.
`outputs.jsonl` keeps the whole transcript per prompt: the first reply's fields at the top level, `turns` (content, reasoning, tool calls, finish reason, token counts and tool responses of every turn), `calls` (the ordered call sequence), `final_content` (the last reply) and `final_graph`.
The final reply is sampled, so reply faithfulness can be scored from it, but it is not scored yet (see follow-ups).

## Held-out sets

`python -m vlm_floorplan.text2graph build-sets` writes them into one directory, from a fixed seed, and nothing in it is committed.

```bash
uv run python -m vlm_floorplan.text2graph build-sets --out DIR \
  --resplan-dir PROCESSED/resplan --cubicasa-file PROCESSED/cubicasa5k.jsonl \
  --swiss-file PROCESSED/swiss_dwellings.jsonl --paraphrase \
  --train-prompts data/t2g/sft1000/train.jsonl
```

`--paraphrase` reads `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` from the environment (and optionally `DEEPSEEK_MODEL`); they are never printed, logged or stored.
Without it the paraphrased set stays empty.

| File | Size | Content |
|------|------|---------|
| `template_test.jsonl` | 600 | Exact template prompts from test-split plans: 200 ResPlan, 250 Swiss Dwellings, 150 CubiCasa5K. Levels L0 to L4 in a 5 / 20 / 30 / 30 / 15 % mix. |
| `template_val.jsonl` | 600 | The same recipe on val-split plans, for choosing a checkpoint only. |
| `paraphrase_test.jsonl` | 200 | Template test prompts (L0 to L3) rewritten by DeepSeek, each passing both paraphrase checks. |
| `behavior_test.jsonl` | 100 | 40 edits (not `new_plan`), 15 restarts (asking for what the current plan lacks), 20 contradictions, 25 unsupported asks. |
| `handwritten_test.jsonl` | 0 | An empty slot for the captain's 60 prompts (below). |
| `references.jsonl` | | The canonical source plan of every record id, with polygons. |
| `reference_tables.json` | | Real link shares and room-count bands from the train plans, for two plausibility metrics. |
| `manifest.json` | | Seed, git commit, counts per set, leak-check result, paraphrase statistics, real reference numbers. |

### Splits

ResPlan and CubiCasa5K use their official splits.
Swiss Dwellings has none, so `text2graph/splits.py` splits by site: the SHA-1 of the site id modulo 20 gives a bucket, 0 is val, 1 is test, 2 to 19 are train.
The stage-1 training set was drawn with this function, so no test site can be in it.
The train data generator must use `split_of` from the same module.

### The leak check

A held-out plan is kept only if none of these hold against the train plans:

- its record id is a train record id,
- its Swiss Dwellings site holds a train apartment,
- its structure (a Weisfeiler-Lehman hash, 3 rounds, over typed rooms, typed links and entrance flags) equals a train structure,
- its prompt text equals a train prompt (when `--train-prompts` is given).

The check runs while the pools are built, and again over every final prompt: a violation raises `LeakError` and no set is written.
The result is in `manifest.json`.
The structure rule is strict on purpose, so the test set measures generalization and not recall, but it removes many plans of the common apartment types.
Fewer CubiCasa5K test plans survive than the plan's 58 (every plan must also be one floor, 3 to 16 rooms, fully typed, with no isolated room), so that source gives up to four levels per plan and may still fall short of 150 prompts.
The manifest has the real counts.

### The canonical frame

A source plan is rotated to its dominant wall direction, scaled into a 1000 x 1000 canvas with a 60 unit margin, renumbered in reading order, given a size per room (the radius of the equal-area circle), stripped of source labels, and given an exterior wall (the outline of the indoor rooms).
The user turn shows that wall.
`plan_score` against the canonical plan is the L4 closeness metric.

### Paraphrase checks

A paraphrase replaces its template only if it passes both of the generator's checks:

1. the lexical slot check: every room type and every number of 2 or more in the requirements appears, and no other room type or number does;
2. a blind round trip: a second call sees only the paraphrase and extracts the requirements as JSON, and they must equal the intended set exactly.

A prompt whose paraphrase fails three tries is skipped and the next one is tried, until 200 pass.

### The hand-written slot

The captain writes 60 prompts (free descriptions and requirement lists, about an hour).
Put them in `handwritten_test.jsonl`, one `PromptRow` per line (`text/rows.py`): `text_source` is `handwritten`, `source` is `handwritten`, `start` is a plan with no rooms and an exterior wall (copy one from a template prompt), and `requirements` are extracted by `Teacher.extract` and confirmed by hand.
The harness accepts the file as it does the others; the job runs it only when `handwritten_test` is named in `--prompt-sets`, and skips it while it is empty.

## Requirements

`requirements/` holds the one function that decides whether a plan meets a request.

| Requirement | Example text |
|-------------|--------------|
| `RoomCount(type, op, n)` | "3 bedrooms", "at least 2 bathrooms", "no garage" |
| `Adjacent(a, b, link)` | "a kitchen open to the living room", "a balcony off the living room" |
| `Ensuites(op, n)` | "2 ensuite bathrooms" |
| `Entrances(op, n)` | "2 front doors" |
| `EntranceInto(type)` | "the front door opens into the hall" |
| `InRegion(type, region)` | "the bedrooms along the top" |
| `RelativePosition(a, b, relation)` | "the kitchen to the left of the living room" |
| `ReachableAvoiding(type, avoid)` | "a bathroom you can reach without walking through a bedroom" |
| `HasLabel(type, text)` | "one bedroom labeled 'Master bedroom'" |

`check_requirements(graph, requirements)` reads only the graph, so it is the same for a source plan and a model's plan.
Choices that matter when reading a result:

- An ensuite bathroom has at least one door or opening, one of them to a bedroom, and none to a room other than a bedroom, storage or a bathroom. A bathroom that also opens to the hall is shared.
- A region is a third of the bounding box of the room points; a type is in a region when every room of the type is.
- An entrance count is the number of rooms with an exterior door.
- Reachability walks over door and opening links from the entrance rooms.

`derive_requirements(graph, level, rng)` draws a set the plan meets by construction (a test checks this on every level).
`references.jsonl` keeps the source plans unlabeled, so a source plan meets its own prompt once `apply_labels` puts the label of a `HasLabel` requirement on it.

## Metrics

Every metric is a count first and a ratio last, and a ratio with an empty denominator is `null`.
`metrics.json` holds two readings of each requirement metric: `strict` scores the plan the environment holds after the loop, counted only when an accepted structured `new_plan` call made it, and `lenient` also counts a plan found in the first reply's reasoning or answer text by the perception baseline's parser.
The gap between them is how often the model follows the format.

| Group | Metric | Definition |
|-------|--------|------------|
| Format | `tool_call_rate` | Share of transcripts with a structured `new_plan` call. |
| | `validity_rate` | Share whose first `new_plan` call the environment accepts. Refusals are counted by reason in `invalid_reasons`. |
| | `strict_plan_rate` | Share that end with a plan from an accepted `new_plan`, a retry after a refusal included, and did not reach the call cap. |
| | `call_cap_rate` | Share of prompts that reached the 8-call cap, overall and per behavior kind. |
| Requirements | `requirement_satisfaction` | Met over stated requirements, micro-averaged; a reply with no plan meets none. By level and by requirement type. |
| | `all_met_rate` | Share of prompts with every requirement met. The headline. By level, source and input kind. |
| | `l4_plan_score_*` | `plan_score` against the source plan, for L4 prompts only. |
| Plausibility | `circulation_rate` | Every room reachable from an entrance. |
| | `crossing_rate` | Two links whose segments cross. |
| | `link_length_ratio_p95` | Linked-pair distance over the smaller nearest-neighbour distance of the two rooms. |
| | `warning_rate` | The validator warns: isolated room, no entrance. |
| | `rare_link_rate` | Links whose (type pair, kind) is under 0.5 % of real train links. |
| | `room_count_in_band_rate` | Total rooms within the real 5th to 95th percentile for the plan's bedroom count. |
| | `indoor_inside_wall_rate` | Indoor rooms whose point is inside the exterior wall. |
| Behavior | `correct_rate` per kind | Edits: edit tools only, and the final plan has the link asked for, or a new room of the type linked to the anchor. Restarts: an accepted `new_plan` whose final plan meets every requirement. Questions, read from the first reply: no tool call, a question, and every conflicting term (contradiction) or any unsupported term (unsupported) named. |

Plausibility is computed on strict plans, and `plausibility_incl_lenient` adds the lenient ones.
`real_reference` is the same plausibility computed on the source plans of the prompts: the band a generated plan should fall in.
A model that returns the source plan scores 1.0 everywhere, which a test checks.

## Running it

Serve the base model, or the base model with a LoRA adapter, then run the same command with the served name.

```bash
MAX_MODEL_LEN=88064 deploy/vllm/serve.sh                       # base, served as qwen3.5-4b
MAX_MODEL_LEN=88064 LORA_MODULES="floorplan=/path/to/adapter" deploy/vllm/serve.sh  # adds --enable-lora

uv run python -m vlm_floorplan.evaluation t2g-run --prompts DIR/template_test.jsonl \
  --run-name base-template --model qwen3.5-4b
uv run python -m vlm_floorplan.evaluation t2g-run --prompts DIR/template_test.jsonl \
  --run-name sft1-template --model floorplan
uv run python -m vlm_floorplan.evaluation t2g-rescore runs/base-template
```

The served context must hold the longest transcript, not one request: the last of up to 8 requests resends the first prompt, every earlier turn and every tool response, and asks for `max_tokens` more.
`t2g_harness.context_tokens(max_tokens)` gives the size, 88064 for the default `max_tokens` of 8192; vLLM refuses a longer request with HTTP 400 and the prompt is recorded as a failed call.
The ClearML job sets it this way by itself.

`--variant one-example` appends one worked example to the system prompt.
The run directory is as for the perception harness (`config.json`, `outputs.jsonl`, `per_prompt.jsonl`, `metrics.json`, `MANIFEST.json`), plus `system_prompt.txt`; it resumes and can be rescored without calling the model.
For a hosted endpoint pass `--api-key-env`, on the CC BY sources only.

## The zero-shot baseline job

The job reuses the ClearML launcher and the verify-then-delete collector with a task axis: `--task text2graph` runs every variant (`as-is`, `one-example`) on every set (`template_test`, `paraphrase_test`, `behavior_test`) against one vLLM server.
The sets directory is packed into the task as an artifact, so deleting the task removes it from ClearML.

```bash
uv run --group ml python scripts/clearml_baseline.py launch --task text2graph --sets-dir DIR --dry-run
uv run --group ml python scripts/clearml_baseline.py launch --task text2graph --sets-dir DIR --wait
uv run --group ml python scripts/clearml_baseline.py collect --task-id TASK_ID --run-name NAME
```

Results land in `~/Documents/pets/vlm-floorplan-runs/<run-name>/result/<variant>-<set>/`.

## Gates for the SFT run

These thresholds are proposed in the plan (section 5.5) and are not enforced by any code.
They will be set with the captain after the zero-shot baseline numbers exist.

- Strict tool-call rate at least 98 %, and validity at least 98 %.
- All-met at least 90 % on L1 and L2, and at least 75 % on L3, measured on the template test.
- The paraphrased and hand-written sets within 10 points of the template set; a larger gap means the model learned the templates.
- Every plausibility metric within the real band, for example crossings at most 1.5 times the real rate.
- Distinct structures per prompt at least 4 of 8 on L1 and L2, which needs the diversity metrics.
- General retention: no benchmark more than 2 points below base.

## Not measured yet

- Diversity (distinct structures, layout spread, copy rate) needs 8 samples per prompt.
- Reply faithfulness: the final reply is in the transcript but not scored.
- Multi-turn refinement is stage 2.
- General-retention benchmarks.
- The 60 hand-written prompts.
