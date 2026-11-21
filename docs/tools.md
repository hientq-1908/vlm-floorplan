# Agent tools and the repair environment

This document explains `vlm_floorplan.agent`: the tools the editing model is served with, how a call looks on the wire, and the environment that turns real plans into tool-use data and RL episodes.
The package has no torch or web dependencies, so the app backend, the data generator and a trainer can all import it.
The edit tools themselves are described in `docs/edit-operations.md`.

## The tool set

There are eleven tools: the ten edit tools, plus `new_plan` for drawing a whole plan from scratch.
`agent_tool_schemas()` returns them in the OpenAI function-calling format, which is what vLLM takes as `tools` and what the Qwen3.5 chat template renders into the system block.

| Tool | Arguments | What it does |
|------|-----------|--------------|
| `add_room` | `type`, `x`, `y`, optional `label`, `floor` | Adds a room with no links and no exterior door. The next free id is assigned. |
| `move_room` | `room_id`, `x`, `y` | Moves a room's point. |
| `delete_room` | `room_id` | Deletes a room and every link that touches it. |
| `set_room_type` | `room_id`, `type` | Changes the room type. |
| `set_room_label` | `room_id`, `label` | Renames a room. An empty label removes the name. |
| `add_link` | `room_a`, `room_b`, `kind` | Links two unlinked rooms. Only `wall` is allowed. |
| `remove_link` | `room_a`, `room_b` | Removes a link completely. |
| `set_link_kind` | `room_a`, `room_b`, `kind` | Changes a link to `door`, `opening` or `wall`. |
| `add_exterior_door` | `room_id` | Marks that a door to the outside opens into the room. |
| `remove_exterior_door` | `room_id` | Removes that mark. |
| `new_plan` | `plan` | Replaces the whole plan with a new one, given as compact JSON in one string. |

Every argument is a flat scalar: a string or an integer.
`new_plan` takes the plan as a string on purpose.
It is the one place a nested structure would otherwise be needed, and compact JSON (`compact.py`) is already the text the perception model writes, so one format serves both.
The ten edit tools stay available after `new_plan`, for every later change.

Why `new_plan` is a separate tool and not a series of `add_room` calls: drawing ten rooms and twenty links one call at a time is many turns, and the user asked for the result, not the steps.
`new_plan` keeps the current image size, and the plan it replaces is gone, so its description tells the model to use it only when there is nothing to keep.

## The call format

The model is served with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` (see `deploy/vllm/README.md`).
One assistant turn holds at most one tool call, and its short reasoning goes in `reasoning_content`.

What the model writes, as the chat template renders it (the template adds `<think>` itself):

```text
<|im_start|>assistant
<think>
The request gives r5 a door to r4. A door needs a wall link first, so I add the wall.
</think>

<tool_call>
<function=add_link>
<parameter=room_a>
r5
</parameter>
<parameter=room_b>
r4
</parameter>
<parameter=kind>
wall
</parameter>
</function>
</tool_call><|im_end|>
```

What the vLLM server returns for that turn, and what a training record stores:

| Where | `arguments` is | Shape |
|-------|----------------|-------|
| vLLM response (OpenAI API) | a JSON string | `{"name": "add_link", "arguments": "{\"room_a\": \"r5\", ...}"}`, and `reasoning_content` next to `content` |
| Training record (TRL) | a dict | `{"name": "add_link", "arguments": {"room_a": "r5", ...}}` |
| `FloorplanEditEnv.step` | a dict | decode the server's string with `json.loads` first |

The `qwen3_coder` parser types each parameter from the tool schema, so `x` arrives as an integer and not as the text `"850"`.
That is the reason every argument is a flat scalar with a declared type.
One quirk to know about: the chat template renders a Python `True` as `True`, not `true`, so a boolean argument looks capitalised in training text.
No current tool has a boolean argument, which keeps the quirk out of the data.

The tool response comes back as a `tool` message, which the template wraps in `<tool_response>` inside a `user` turn.
It is the text from `format_edit_outcome`, so the app and the training data show the model the same sentences.

```text
OK: linked r5 and r4 with a wall.
The plan now has 6 rooms and 9 links (4 door, 1 opening, 4 wall).
```

A refused call starts with `REJECTED` and a stable error code, and says the plan was not changed:

```text
REJECTED (not_adjacent): Cannot add a door between r0 and r2: they are not linked. If they do touch, first add a link of kind 'wall' between them, then change it to a door with set_link_kind.
The plan was not changed.
```

### `new_plan` and invalid plans

`new_plan` parses the text with `from_compact`, checks it with `validate_graph`, and refuses an empty plan.
A refusal uses the code `invalid_plan`, names the first problem in plain words, and leaves the plan alone:

```text
REJECTED (invalid_plan): The plan is not valid: floor 1: link r0-r9 refers to r9, which is not a room on this floor.
The plan was not changed.

REJECTED (invalid_plan): The plan is not valid: it is not JSON (Expecting value at character 0).
The plan was not changed.

REJECTED (invalid_plan): The plan has no rooms.
The plan was not changed.
```

An accepted plan answers like an edit, including warnings:

```text
OK: replaced the plan with a new one on 1 floor.
Warning: no room is marked as having an exterior door.
The plan now has 6 rooms and 9 links (5 door, 1 opening, 3 wall).
```

A call with the wrong arguments (a missing or extra argument, a non-string `plan`) is refused with `invalid_arguments`.
No tool call ever raises: a bad call is an event the model reads and recovers from.

## The environment

`FloorplanEditEnv` runs one repair episode.
The idea is to damage a real plan, ask for the repair, and let the real plan be the target.

```python
env = FloorplanEditEnv()
prompt = env.reset(record, DeleteRoomTask("r5"))  # user message: damaged plan + request
text = env.step("add_room", {"type": "storage", "x": 850, "y": 850})  # tool response text
env.reward()  # plan_score(final, target) - penalties
env.solution()  # scripted ToolCalls that repair the plan
```

- `reset(record, task)` damages `record.graph` with the task and returns the user message: `summarize_graph` of the damaged plan, then the request.
  The record stays the target.
- `step(tool, arguments)` plays the call through `call_tool`, which uses `apply_tool_call` for the edit tools, and returns the tool response text.
  A refused call changes nothing and is counted.
- `reward()` is `plan_score(current plan, target)` minus `0.05` per refused call.
  `plan_score` is the mean of four scores from `evaluate_plan`, each skipped only when neither the target nor the plan has anything to measure (a component where the plan misses every real item, or has only spurious ones, scores 0): room F1, room-type accuracy among matched rooms, link F1 where the kind must be right too, and exterior-door F1 over every room, so a real entrance room that is missing also counts as a missed exterior door.
  Rooms are matched by position (a point inside the real room's polygon), so a re-added room does not need its old id.
- `solution()` returns the scripted inverse edit as `ToolCall(name, arguments, reasoning)` values.

### Tasks

| Task | Damage | Request | Scripted solution |
|------|--------|---------|-------------------|
| `delete_room` | A room and its links are deleted. | "Add a storage room at (850, 850). It has a shared wall with r3 (living) and a door to r4 (entry)." | `add_room`, then per neighbour `add_link` wall and, for a door or opening, `set_link_kind`, then `add_exterior_door` if the room had one. |
| `close_door` | A door or opening becomes a wall. | "Put a door between r2 (kitchen) and r4 (entry)." | One `set_link_kind`. |
| `draw_plan` | The plan is emptied. | "Draw this plan from scratch:" and the plan summary. | One `new_plan`. |

`delete_room` is the captain's idea read backwards: the links the real plan had say which rooms the new room must connect to, and that is the reasoning the model learns to write.
The request lists the links explicitly.
A harder request that leaves them out is a later change to `request()` only, because the target and the reward do not change.

## One full episode

`DeleteRoomTask("r5")` on the built-in sample plan (`agent/sample_plan.py`), as a TRL record (`tools` omitted here, it holds the eleven schemas):

```json
{"id": "sample/six-rooms#delete_room", "task": "delete_room", "reward": 1.0, "messages": [
 {"role": "system", "content": "You edit a floorplan room graph with tools. ..."},
 {"role": "user", "content": "Current plan:\nfloor 1: 5 rooms, 7 links\n  r0 bedroom at (200, 200)\n  ...\n  links: r0-r1 wall, r1-r2 wall, ...\n\nRequest: Add a storage room at (850, 850). It has a shared wall with r3 (living) and a door to r4 (entry)."},
 {"role": "assistant", "reasoning_content": "The user wants a storage room at (850, 850). I add the room first. It starts with no links, so I connect it afterwards.", "content": "",
  "tool_calls": [{"type": "function", "function": {"name": "add_room", "arguments": {"type": "storage", "x": 850, "y": 850}}}]},
 {"role": "tool", "name": "add_room", "content": "OK: added storage room r5 at (850, 850) on floor 1.\nWarning: room r5 has no links to other rooms on floor 1.\nThe plan now has 6 rooms and 7 links (4 door, 1 opening, 2 wall)."},
 {"role": "assistant", "reasoning_content": "The request says r5 shares a wall with r3, so I link them.", "content": "",
  "tool_calls": [{"type": "function", "function": {"name": "add_link", "arguments": {"room_a": "r5", "room_b": "r3", "kind": "wall"}}}]},
 {"role": "tool", "name": "add_link", "content": "OK: linked r5 and r3 with a wall.\nThe plan now has 6 rooms and 8 links (4 door, 1 opening, 3 wall)."},
 {"role": "assistant", "reasoning_content": "The request gives r5 a door to r4. A door needs a wall link first, so I add the wall.", "content": "",
  "tool_calls": [{"type": "function", "function": {"name": "add_link", "arguments": {"room_a": "r5", "room_b": "r4", "kind": "wall"}}}]},
 {"role": "tool", "name": "add_link", "content": "OK: linked r5 and r4 with a wall.\nThe plan now has 6 rooms and 9 links (4 door, 1 opening, 4 wall)."},
 {"role": "assistant", "reasoning_content": "Now I turn the wall between r5 and r4 into a door.", "content": "",
  "tool_calls": [{"type": "function", "function": {"name": "set_link_kind", "arguments": {"room_a": "r5", "room_b": "r4", "kind": "door"}}}]},
 {"role": "tool", "name": "set_link_kind", "content": "OK: changed the link between r5 and r4 from wall to door.\nThe plan now has 6 rooms and 9 links (5 door, 1 opening, 3 wall)."},
 {"role": "assistant", "reasoning_content": "The room and everything the request lists are in place.", "content": "Added the storage room at (850, 850). It has a shared wall with r3 (living) and a door to r4 (entry)."}
]}
```

The tool responses are not scripted text.
They are what the environment returned when the scripted calls were played, so they always match what the app would say.

## Building examples

```bash
uv run --group ml python scripts/build_tool_episodes.py \
    --records data/processed/train.jsonl --out-dir data/tool_episodes --per-kind 2
```

The script reads the first plans of a processed JSONL file (ResPlan or CubiCasa5K), picks `--per-kind` random tasks of each kind, and writes:

- `episodes.jsonl`: one TRL conversational record per line.
- `episodes.txt`: the same episodes rendered through the Qwen3.5 chat template, for review.
  It needs `transformers` (the `ml` group) and the tokenizer files; without them only the JSONL is written.

Without `--records` it uses the built-in sample plan, so it also runs with no data on disk.

## How the environment is used

- **SFT data.**
  `build_episode(record, task)` plays `solution()` through `step()` and records the conversation.
  It raises `ValueError` if the environment refuses a scripted call, so no episode teaches a repair that fails.
  The episodes train the tool-calling behaviour: reading the plan, writing short reasoning, one call per turn, reading the response.
  Only the assistant turns are trained on; the user turn and the tool responses are masked by `assistant_only_loss`.
- **Rollouts.**
  Give the model the `reset` prompt and the tool list, play each tool call it makes through `step` (decode the server's JSON `arguments` first), and stop when it answers without a call.
  `reward()` then scores the whole rollout, and `solution()` gives the baseline to compare it with.
  A rollout of a model that has not been trained yet shows which calls it gets wrong, as `REJECTED` counts.
- **GRPO, later.**
  `reset`, `step` and `reward` are the whole interface a trainer needs.
  The reward is dense enough to rank rollouts from the same prompt: partial repairs score below 1.0, since every missing room, link or exterior door counts against the plan, and each refused call costs a little.
  Wiring the environment into a trainer is not part of this prototype.

## Limits of this prototype

- Requests come from one template per task.
  Paraphrasing, multi-turn sessions and the mixture of task kinds are the session generator's job.
- The user message is text only.
  The floorplan image is not attached yet, so the model reads the plan from `summarize_graph`.
- The rendered view and the tool schemas were checked against the Qwen3.5-4B chat template.
  The call format was not run against a live vLLM server here.
- `delete_room` needs a room with at least one link, a known type and a label short enough for `add_room`, and `draw_plan` is limited to plans of twelve rooms or fewer, because it copies the whole plan into one call.
